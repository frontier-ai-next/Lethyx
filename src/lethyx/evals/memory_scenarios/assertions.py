"""Memory contract assertions for scenario harness."""

import re
from pathlib import Path
from typing import Any

from lethyx.evals.memory_scenarios.models import ScenarioStepResult
from lethyx.evals.memory_scenarios.state_assertions import (
    evaluate_prompt_section_assertions,
    evaluate_state_assert_block,
)


def assert_single_always_loaded_block(prompt: str) -> list[str]:
    errors: list[str] = []
    count = len(re.findall(r"^## \[ALWAYS-LOADED MEMORY\]", prompt, flags=re.MULTILINE))
    if count != 1:
        errors.append(f"expected exactly one ALWAYS-LOADED block, got {count}")
    if "## [/ALWAYS-LOADED MEMORY]" not in prompt:
        errors.append("missing [/ALWAYS-LOADED MEMORY] closing tag")
    return errors


def assert_memory_injection_contract(task: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    overlays = task.get("context_overlays") or {}
    contract = overlays.get("memory_injection_contract")
    if not isinstance(contract, dict):
        return {}, ["missing memory_injection_contract"]
    if contract.get("mode") != "LETHYX_owned":
        errors.append(f"contract.mode={contract.get('mode')}")
    if not contract.get("proactive_overlay_injected"):
        errors.append("proactive_overlay_injected not true")
    if not contract.get("proactive_overlay_hash"):
        errors.append("missing proactive_overlay_hash")
    if not contract.get("retrieval_is_supplemental_only"):
        errors.append("retrieval_is_supplemental_only not true")
    if not overlays.get("prevent_host_auto_core_overlay"):
        errors.append("prevent_host_auto_core_overlay not set")
    return contract, errors


def prompt_line_index(prompt: str, marker: str) -> int:
    idx = prompt.find(marker)
    return idx if idx >= 0 else -1


def assert_prompt_order(prompt: str, before_pairs: list[list[str]]) -> list[str]:
    errors: list[str] = []
    for left, right in before_pairs:
        li = prompt_line_index(prompt, left)
        ri = prompt_line_index(prompt, right)
        if li < 0 or ri < 0:
            if li < 0:
                errors.append(f"prompt missing marker: {left}")
            if ri < 0:
                errors.append(f"prompt missing marker: {right}")
            continue
        if li >= ri:
            errors.append(f"prompt order: expected '{left}' before '{right}'")
    return errors


def skipped_bkb_ids(report: dict[str, Any]) -> set[str]:
    return {
        str(row.get("id"))
        for row in (report.get("skipped") or [])
        if isinstance(row, dict) and row.get("id")
    }


def included_bkb_ids_from_audit(task: dict[str, Any]) -> set[str]:
    proactive = (task.get("context_overlays") or {}).get("proactive_memory") or {}
    audit = proactive.get("injection_audit") or proactive.get("telemetry", {}).get(
        "injection_audit"
    )
    if not isinstance(audit, dict):
        return set()
    return {str(x) for x in (audit.get("included_bkb_ids") or [])}


def evaluate_assert_block(
    assert_key: str,
    spec: dict[str, Any],
    step: ScenarioStepResult,
    *,
    task: dict[str, Any] | None = None,
    repo: Path | None = None,
    workspace_id: str = "",
    drive: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    prompt = step.prompt or str((step.task or {}).get("input") or "")
    task = task or step.task
    report = step.injection_report

    prompt_spec = spec.get("prompt") or {}
    if prompt_spec:
        if "contains_once" in prompt_spec:
            for needle in prompt_spec["contains_once"]:
                if prompt.count(str(needle)) != 1:
                    errors.append(f"{assert_key}.prompt: expected once {needle!r}")
        for needle in prompt_spec.get("contains") or []:
            if str(needle) not in prompt:
                errors.append(f"{assert_key}.prompt: missing {needle!r}")
        for needle in prompt_spec.get("not_contains") or []:
            if str(needle) in prompt:
                errors.append(f"{assert_key}.prompt: must not contain {needle!r}")
        errors.extend(assert_prompt_order(prompt, prompt_spec.get("before") or []))
        errors.extend(evaluate_prompt_section_assertions(assert_key, prompt_spec, prompt))

    bkb_spec = spec.get("bkb") or {}
    if bkb_spec and task and not bkb_spec.get("unchanged") and not bkb_spec.get("changed"):
        included = included_bkb_ids_from_audit(task)
        skipped = skipped_bkb_ids(report) if report else set()
        for rid in bkb_spec.get("included_ids") or []:
            if str(rid) not in included:
                errors.append(f"{assert_key}.bkb: expected included {rid}")
        for rid in bkb_spec.get("skipped_ids") or []:
            rid = str(rid)
            if rid not in skipped:
                errors.append(
                    f"{assert_key}.bkb: expected skipped {rid}, "
                    f"got skipped={sorted(skipped)} included={sorted(included)}"
                )
        for rid in bkb_spec.get("not_included_ids") or []:
            if str(rid) in included:
                errors.append(f"{assert_key}.bkb: must not be included {rid}")

    contract_spec = spec.get("contract") or {}
    if contract_spec and task:
        contract, cerrs = assert_memory_injection_contract(task)
        errors.extend(f"{assert_key}.contract: {e}" for e in cerrs)
        if contract_spec.get("present") and not contract:
            errors.append(f"{assert_key}.contract: missing")

    if spec.get("single_always_loaded"):
        errors.extend(f"{assert_key}: {e}" for e in assert_single_always_loaded_block(prompt))

    if spec.get("supplemental_non_directive") and "Supplemental" in prompt:
        if "NON-DIRECTIVE" not in prompt:
            errors.append(f"{assert_key}: supplemental recall missing NON-DIRECTIVE label")

    if repo is not None:
        errors.extend(
            evaluate_state_assert_block(
                assert_key,
                spec,
                step,
                before=step.snapshot_before,
                after=step.snapshot_after,
                repo=repo,
                workspace_id=workspace_id,
                drive=drive,
            )
        )

    health_spec = spec.get("memory_health") or {}
    if health_spec:
        health = (step.overlays or {}).get("memory_health") or {}
        if health_spec.get("expect_unavailable"):
            if health.get("volatile_stub"):
                errors.append(f"{assert_key}.memory_health: volatile stub still enabled")
            elif health.get("ok") and not health.get("stores_fail"):
                pass
            elif health.get("ok"):
                errors.append(f"{assert_key}.memory_health: backend failure masked as ok")
            elif not health.get("stores_fail"):
                errors.append(f"{assert_key}.memory_health: missing stores_fail detail")

    return errors


def structured_facts(step: ScenarioStepResult, task: dict[str, Any]) -> dict[str, Any]:
    prompt = step.prompt or str(task.get("input") or "")
    contract, _ = assert_memory_injection_contract(task)
    phase = str(task.get("phase_id") or report_phase(step) or "")
    report = step.injection_report
    return {
        "phase": phase,
        "included_bkb_ids": sorted(included_bkb_ids_from_audit(task)),
        "skipped_bkb_ids": sorted(skipped_bkb_ids(report)) if report else [],
        "prompt_order": {
            "always_loaded_memory_before_phase_instructions": _order_ok(
                prompt, "[ALWAYS-LOADED MEMORY]", "Phase instructions"
            ),
            "always_loaded_memory_before_supplemental_recall": _order_ok(
                prompt, "[ALWAYS-LOADED MEMORY]", "Supplemental"
            ),
        },
        "proactive_overlay_hash": contract.get("proactive_overlay_hash"),
    }


def report_phase(step: ScenarioStepResult) -> str:
    return str((step.injection_report or {}).get("phase_id") or "")


def _order_ok(prompt: str, left: str, right: str) -> bool:
    li, ri = prompt.find(left), prompt.find(right)
    if li < 0 or ri < 0:
        return True
    return li < ri
