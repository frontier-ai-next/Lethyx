"""Compile typed phase-plan contracts into PlanIR."""

import shlex
from typing import Any

from lethyx.contracts.models import (
    ContractIssue,
    PlanIR,
    ProofSpec,
    SubtaskIR,
)
from lethyx.contracts.proof_templates import instantiate_proof_template


def _tuple_str(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


# Documented `success_test` aliases that compile into a typed proof. Mirrors the
# plan-policy layer so a leaf is not rejected as `missing_proof` merely because it
# used an accepted alias key instead of the literal `success_test`.
_LEGACY_SUCCESS_TEST_ALIAS_KEYS = (
    "success_test",
    "success_check",
    "success_checks",
    "acceptance_command",
    "verification_command",
    "verification_commands",
    "verification",
)


def _success_text_from_value(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for name in ("value", "command", "commands", "cmd", "command_line", "pytest_id"):
            value = raw.get(name)
            if isinstance(value, (list, tuple)):
                text = "; ".join(str(v).strip() for v in value if str(v).strip())
            else:
                text = str(value or "").strip()
            if text:
                return text
        return ""
    if isinstance(raw, (list, tuple)):
        return "; ".join(str(v).strip() for v in raw if str(v).strip())
    return ""


# Bare verification-tool names are not concrete proof commands; when a leaf uses
# one as its `success_test` it should be overridden by a concrete alias (e.g.
# `verification_command`), mirroring the plan-policy layer.
_GENERIC_SUCCESS_TOOLS = frozenset(
    {
        "run_workspace_verify",
        "run_unit_tests",
        "harness_run",
        "http_boot",
        "behavioral_http",
        "run_real_e2e",
    }
)


def _is_bare_generic_tool(text: str) -> bool:
    return str(text or "").strip().strip("`").lower() in _GENERIC_SUCCESS_TOOLS


def _legacy_success_test_text(item: dict[str, Any]) -> str:
    bare_fallback = ""
    for key in _LEGACY_SUCCESS_TEST_ALIAS_KEYS:
        if key not in item:
            continue
        text = _success_text_from_value(item.get(key))
        if not text:
            continue
        if _is_bare_generic_tool(text):
            bare_fallback = bare_fallback or text
            continue
        return text
    return bare_fallback


def _legacy_success_test_command(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    try:
        return tuple(shlex.split(text, posix=False))
    except ValueError:
        return tuple(part for part in text.split() if part)


def _path_looks_like_test(path: str) -> bool:
    rel = str(path or "").replace("\\", "/").lower()
    name = rel.rsplit("/", 1)[-1]
    return (
        rel.startswith("tests/")
        or "/tests/" in rel
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.ts")
        or name.endswith(".test.tsx")
        or name.endswith(".spec.ts")
        or name.endswith(".spec.tsx")
    )


def _legacy_success_test_targets(command: tuple[str, ...]) -> tuple[str, ...]:
    targets: list[str] = []
    for item in command:
        text = str(item or "").strip().strip("\"'")
        if not text or text.startswith("-"):
            continue
        if ".py" in text or ".test." in text or ".spec." in text:
            targets.append(text)
    return tuple(dict.fromkeys(targets))


def _legacy_success_test_proof_payload(
    item: dict[str, Any],
    *,
    subtask_id: str,
) -> dict[str, Any] | None:
    command = _legacy_success_test_command(_legacy_success_test_text(item))
    if not command:
        return None
    files_changed = _tuple_str(
        item.get("files_to_change") or item.get("files_to_modify") or item.get("files_affected")
    )
    files_created = _tuple_str(item.get("files_to_create") or item.get("new_files"))
    changed_files = tuple(dict.fromkeys((*files_changed, *files_created)))
    test_targets = _legacy_success_test_targets(command)
    changes_tests = any(_path_looks_like_test(path) for path in changed_files)
    uses_pytest = any(part == "pytest" or part.endswith("pytest") for part in command)
    proof: dict[str, Any] = {
        "execution": {
            "kind": "pytest" if uses_pytest else "command",
            "command": list(command),
            "shell": False,
        },
        "oracle": {
            "oracle_type": "unit_assertions" if uses_pytest else "build",
            "required_properties": (["no_test_tampering"] if changes_tests else ["build_succeeds"]),
        },
        "scope": {
            "files_under_test": [path for path in changed_files if not _path_looks_like_test(path)],
            "changed_files_expected": list(changed_files),
            "pytest_targets": list(test_targets) if uses_pytest else [],
        },
        "anti_gaming": {"requires_real_runtime": True, "allows_mock": False},
    }
    if changes_tests:
        proof["generated_test_contract"] = {
            "interface_model": {"legacy_success_test": True, "subtask_id": subtask_id},
            "proof_budget": {
                "max_generated_tests_per_subtask": 3,
                "allow_expanded_generated_tests": False,
            },
            "oracle_claims": [
                {
                    "claim_id": f"{subtask_id}_required_behavior",
                    "source": "task_requirement",
                    "subject": subtask_id,
                    "expected_behavior": str(
                        item.get("goal") or item.get("description") or item.get("title") or ""
                    ),
                    "test_refs": list(test_targets)
                    or [path for path in changed_files if _path_looks_like_test(path)],
                }
            ],
        }
    return proof


_PLAN_CHILD_KEYS = {"subtasks", "steps", "phases", "tasks", "items", "children"}
_PLAN_META_KEYS = ("plan_id", "run_id", "workspace_id")
_PROOF_TOP_LEVEL_KEYS = {
    "template_id",
    "template_args",
    "execution",
    "oracle",
    "scope",
    "anti_gaming",
    "harness",
    "harness_profile",
    "harness_id",
    "harness_options",
    "generated_test_contract",
    "required_capabilities",
    "human_claims",
    "evidence_refs",
    # Legacy planner output occasionally nested this here; execute still knows
    # how to lift it, but new plans should prefer the subtask-level field.
    "memory_scope",
    # Descriptive/lifted fields that planners commonly add but that are silently
    # consumed by ProofSpec.from_mapping — blocking on them is pure friction.
    "title",
    "description",
    "name",
    "label",
    "execution_environment_id",
    "environment_id",
    "env_id",
}


def _child_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    return []


def _leaf_payloads(item: dict[str, Any]) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []
    for key, value in item.items():
        if str(key).lower() in _PLAN_CHILD_KEYS:
            for child in _child_dicts(value):
                children.extend(_leaf_payloads(child))
    if children:
        return children
    return [item]


def _iter_subtask_payloads(raw_plan: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("subtasks", "steps", "phases"):
        value = raw_plan.get(key)
        if isinstance(value, (list, dict)):
            leaves: list[dict[str, Any]] = []
            for item in _child_dicts(value):
                leaves.extend(_leaf_payloads(item))
            return leaves
    plan_obj = raw_plan.get("plan")
    if isinstance(plan_obj, dict):
        return _iter_subtask_payloads(plan_obj)
    return []


def canonicalize_phase_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    """Return a storage-safe phase plan with a canonical `subtasks` array."""

    if not isinstance(raw_plan, dict):
        return {}
    source = raw_plan.get("plan") if isinstance(raw_plan.get("plan"), dict) else raw_plan
    canonical: dict[str, Any] = {
        str(key): value
        for key, value in source.items()
        if str(key).lower() not in _PLAN_CHILD_KEYS and str(key) != "plan"
    }
    for key in _PLAN_META_KEYS:
        if key not in canonical and raw_plan.get(key) is not None:
            canonical[key] = raw_plan[key]
    canonical["subtasks"] = [dict(item) for item in _iter_subtask_payloads(source)]
    return canonical


def _proof_shape_issues(
    proof_payload: dict[str, Any],
    *,
    subtask_id: str,
) -> list[ContractIssue]:
    unknown = sorted(str(key) for key in proof_payload if str(key) not in _PROOF_TOP_LEVEL_KEYS)
    if not unknown:
        return []
    misplaced_harness = [
        key for key in unknown if key in {"assert_command", "interaction_command", "driver_command"}
    ]
    harness_hint = (
        " Move "
        + ", ".join(f"`{key}`" for key in misplaced_harness)
        + " under `proof.harness_options`; these command slots are not top-level proof fields."
        if misplaced_harness
        else ""
    )
    return [
        ContractIssue(
            code="invalid_plan_contract",
            severity="blocking",
            subtask_id=subtask_id,
            message=(
                "Unknown proof field(s) "
                + ", ".join(f"`{key}`" for key in unknown[:8])
                + "; use the typed proof contract fields exactly."
                + harness_hint
            ),
        )
    ]


def _ensure_no_test_tampering_property(
    proof_payload: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    """Auto-attach the runtime `no_test_tampering` guard for test-touching subtasks.

    Creating/changing tests is normal execute work. Rather than block the plan and
    force the planner to remember a magic proof property (a brittle plan-time gate),
    we attach the property automatically. The actual protection is runtime: the
    proof still enforces `no_test_tampering` when it runs, so weakened tests are
    caught by evidence, not by lexical plan policing.
    """
    if not isinstance(proof_payload, dict):
        return proof_payload
    scope = proof_payload.get("scope") if isinstance(proof_payload.get("scope"), dict) else {}
    changed_paths = [
        *_tuple_str(item.get("files_to_create") or item.get("new_files")),
        *_tuple_str(
            item.get("files_to_change") or item.get("files_to_modify") or item.get("files_affected")
        ),
        *_tuple_str(scope.get("changed_files_expected")),
    ]
    if not any(_path_looks_like_test(path) for path in changed_paths):
        return proof_payload
    oracle = (
        dict(proof_payload.get("oracle")) if isinstance(proof_payload.get("oracle"), dict) else {}
    )
    props = list(oracle.get("required_properties") or [])
    if "no_test_tampering" not in {str(p).strip() for p in props}:
        props.append("no_test_tampering")
        oracle["required_properties"] = props
        proof_payload = {**proof_payload, "oracle": oracle}
    return proof_payload


def _coerce_headless_runtime_claim(proof_payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve the headless-vs-real-runtime contradiction at the source.

    A subtask cannot be both a ``desktop_gui_headless`` proof (controller/model/view
    behaviour without a display) and claim ``anti_gaming.requires_real_runtime=true``
    (real window evidence). Planners routinely mislabel headless view tests this way
    while a separate ``desktop_gui_runtime`` smoke leaf carries the real runtime
    evidence. Rather than approve it in plan_review and then fatally abort in execute
    (two layers disagreeing), we normalise the headless leaf to
    ``requires_real_runtime=false`` so it runs as the headless proof it actually is.
    """
    if not isinstance(proof_payload, dict):
        return proof_payload
    profile = str(proof_payload.get("harness_profile") or "").strip().lower()
    if "headless" not in profile:
        return proof_payload
    anti = proof_payload.get("anti_gaming")
    if isinstance(anti, dict) and anti.get("requires_real_runtime"):
        proof_payload = {
            **proof_payload,
            "anti_gaming": {**anti, "requires_real_runtime": False},
        }
    return proof_payload


def compile_phase_plan(
    raw_plan: dict[str, Any], *, run_id: str = "", workspace_id: str = ""
) -> tuple[PlanIR | None, list[ContractIssue]]:
    """Compile the v1 phase plan into PlanIR."""

    if not isinstance(raw_plan, dict):
        return None, [
            ContractIssue(
                code="invalid_plan_contract",
                severity="blocking",
                message="Phase plan contract must be an object.",
            )
        ]
    issues: list[ContractIssue] = []
    subtasks: list[SubtaskIR] = []
    effective_run_id = str(raw_plan.get("run_id") or run_id or "")
    effective_workspace_id = str(raw_plan.get("workspace_id") or workspace_id or "")
    for idx, item in enumerate(_iter_subtask_payloads(raw_plan), start=1):
        subtask_id = str(
            item.get("id") or item.get("subtask_id") or item.get("name") or f"subtask_{idx}"
        )
        proof_payload = item.get("proof")
        proof = None
        generated_contract = _dict_value(
            item.get("generated_test_contract")
            or (
                proof_payload.get("generated_test_contract")
                if isinstance(proof_payload, dict)
                else None
            )
        )
        if isinstance(proof_payload, dict):
            proof_payload = instantiate_proof_template(
                proof_payload,
                subtask_payload=item,
            )
            proof_payload = _ensure_no_test_tampering_property(proof_payload, item)
            proof_payload = _coerce_headless_runtime_claim(proof_payload)
            issues.extend(_proof_shape_issues(proof_payload, subtask_id=subtask_id))
            proof = ProofSpec.from_mapping(proof_payload)
        elif _legacy_success_test_text(item):
            proof_payload = _legacy_success_test_proof_payload(
                item,
                subtask_id=subtask_id,
            )
            if proof_payload is not None:
                proof = ProofSpec.from_mapping(proof_payload)
                generated_contract = _dict_value(proof_payload.get("generated_test_contract"))
            issues.append(
                ContractIssue(
                    code="legacy_contract_used",
                    severity="warning",
                    subtask_id=subtask_id,
                    message=(
                        "Legacy `success_test` was compiled into a typed proof; "
                        "new plans should provide `proof` or `proof.template_id`."
                    ),
                )
            )
        else:
            issues.append(
                ContractIssue(
                    code="missing_proof",
                    severity="blocking",
                    subtask_id=subtask_id,
                    message="Subtask must provide a typed `proof` object.",
                )
            )
        subtasks.append(
            SubtaskIR(
                id=subtask_id,
                title=str(item.get("title") or item.get("name") or subtask_id),
                goal=str(item.get("goal") or item.get("description") or ""),
                files_to_change=_tuple_str(
                    item.get("files_to_change")
                    or item.get("files_to_modify")
                    or item.get("files_affected")
                ),
                files_to_create=_tuple_str(item.get("files_to_create") or item.get("new_files")),
                dependencies=_tuple_str(item.get("dependencies")),
                proof=proof,
                generated_test_contract=generated_contract,
                acceptance_claims=_tuple_str(
                    item.get("acceptance_claims") or item.get("acceptance_criteria")
                ),
                memory_scope=_dict_value(item.get("memory_scope")),
                allowed_tools=_tuple_str(item.get("allowed_tools") or item.get("tools")),
                allowed_skills=_tuple_str(item.get("allowed_skills") or item.get("skills")),
                codeptr_refs=_tuple_str(item.get("codeptr_refs")),
                mcp_refs=_tuple_str(item.get("mcp_refs")),
            )
        )
    if not subtasks:
        issues.append(
            ContractIssue(
                code="missing_plan_subtasks",
                severity="blocking",
                message="Phase plan must contain a non-empty typed subtask list.",
            )
        )
    return (
        PlanIR(
            run_id=effective_run_id,
            workspace_id=effective_workspace_id,
            subtasks=tuple(subtasks),
            notes=str(
                raw_plan.get("notes") or raw_plan.get("rationale") or raw_plan.get("summary") or ""
            ),
        ),
        issues,
    )
