"""State-delta assertions (BKB, palace, proposal queue, overlay, memory health)."""

import fnmatch
import json
from pathlib import Path
from typing import Any

import yaml

from lethyx.evals.memory_scenarios.models import ScenarioStepResult


def extract_prompt_section(prompt: str, start: str, end: str) -> str:
    si = prompt.find(start)
    if si < 0:
        return ""
    ei = prompt.find(end, si + len(start))
    if ei < 0:
        return prompt[si:]
    return prompt[si : ei + len(end)]


def palace_store_counts(snapshot: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in (snapshot.get("palace") or {}).get("nodes") or []:
        if not isinstance(node, dict):
            continue
        store = str(node.get("store") or "unknown")
        counts[store] = counts.get(store, 0) + 1
    return counts


def _bkb_rule_ids(bkb_yaml: str) -> set[str]:
    if not bkb_yaml.strip():
        return set()
    try:
        data = yaml.safe_load(bkb_yaml) or {}
    except yaml.YAMLError:
        return set()
    return {str(r.get("id")) for r in (data.get("rules") or []) if r.get("id")}


def _proposal_queue_stats(drive: Path) -> dict[str, Any]:
    proposals = drive / "state" / "bkb_proposals"
    if not proposals.is_dir():
        return {"queued": 0, "accepted": 0, "rejected": 0, "needs_evidence": 0, "files": []}
    files = list(proposals.glob("*"))
    accepted = [f for f in files if f.suffix == ".json" and "accepted" in f.name]
    rejected = [f for f in files if "rejected" in f.name]
    candidates = [f for f in files if f.name.endswith(".candidate.json")]
    return {
        "queued": len(candidates),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "needs_evidence": 0,
        "files": [f.name for f in files],
    }


def estimate_overlay_tokens(overlays: dict[str, Any]) -> int:
    proactive = overlays.get("proactive_memory") or {}
    if isinstance(proactive, dict):
        telemetry = (
            proactive.get("telemetry") if isinstance(proactive.get("telemetry"), dict) else {}
        )
        tokens = telemetry.get("memory_overlay_tokens")
        if tokens is not None:
            try:
                return int(tokens)
            except (TypeError, ValueError):
                pass
    text = json.dumps(proactive, ensure_ascii=False)
    return len(text) // 4


def evaluate_state_assert_block(
    assert_key: str,
    spec: dict[str, Any],
    step: ScenarioStepResult,
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    repo: Path,
    workspace_id: str,
    drive: Path | None,
) -> list[str]:
    errors: list[str] = []

    bkb_spec = spec.get("bkb") or {}
    if bkb_spec:
        before_yaml = str(before.get("bkb_yaml") or "")
        after_yaml = str(after.get("bkb_yaml") or "")
        if bkb_spec.get("unchanged") and before_yaml != after_yaml:
            errors.append(f"{assert_key}.bkb: expected unchanged bkb.yaml")
        if bkb_spec.get("changed") and before_yaml == after_yaml:
            errors.append(f"{assert_key}.bkb: expected bkb.yaml to change")
        after_ids = _bkb_rule_ids(after_yaml)
        for rid in bkb_spec.get("contains_rule_ids") or []:
            if str(rid) not in after_ids:
                errors.append(f"{assert_key}.bkb: missing rule id {rid}")
        for rid in bkb_spec.get("not_contains_rule_ids") or []:
            if str(rid) in after_ids:
                errors.append(f"{assert_key}.bkb: must not contain rule {rid}")

    palace_spec = spec.get("palace") or {}
    if palace_spec:
        before_counts = palace_store_counts(before)
        after_counts = palace_store_counts(after)
        for store, expected in (palace_spec.get("new_store_nodes") or {}).items():
            delta = after_counts.get(store, 0) - before_counts.get(store, 0)
            if delta != int(expected):
                errors.append(
                    f"{assert_key}.palace: expected {expected} new {store} nodes, delta={delta}"
                )
        for store in palace_spec.get("no_new_store_nodes") or []:
            delta = after_counts.get(str(store), 0) - before_counts.get(str(store), 0)
            if delta != 0:
                errors.append(f"{assert_key}.palace: expected no new {store} nodes, delta={delta}")
        meta_spec = palace_spec.get("contains_metadata") or {}
        if meta_spec:
            nodes = (after.get("palace") or {}).get("nodes") or []
            durable = [n for n in nodes if n.get("store") == "palace.durable"]
            if not durable:
                errors.append(f"{assert_key}.palace: no palace.durable nodes for metadata check")
            else:
                node = durable[-1]
                for key, val in meta_spec.items():
                    if key == "evidence_refs_json_contains":
                        raw = str(
                            node.get("evidence_refs_json")
                            or node.get("metadata", {}).get("evidence_refs_json")
                            or ""
                        )
                        if str(val) not in raw:
                            errors.append(f"{assert_key}.palace: evidence_refs missing {val!r}")
                    elif str(node.get(key) or "") != str(val):
                        errors.append(
                            f"{assert_key}.palace: metadata {key} expected {val!r}, "
                            f"got {node.get(key)!r}"
                        )

    pq_spec = spec.get("proposal_queue") or {}
    if pq_spec and drive:
        stats = _proposal_queue_stats(drive)
        if "queued" in pq_spec and stats["queued"] != int(pq_spec["queued"]):
            errors.append(
                f"{assert_key}.proposal_queue: expected queued={pq_spec['queued']}, got {stats['queued']}"
            )
        if "accepted" in pq_spec and stats["accepted"] != int(pq_spec["accepted"]):
            errors.append(f"{assert_key}.proposal_queue: accepted mismatch")
        if "rejected" in pq_spec and stats["rejected"] != int(pq_spec["rejected"]):
            errors.append(f"{assert_key}.proposal_queue: rejected mismatch")
        for pattern in pq_spec.get("files_exist") or []:
            if not any(fnmatch.fnmatch(name, str(pattern)) for name in stats["files"]):
                errors.append(f"{assert_key}.proposal_queue: missing file {pattern!r}")
        for pattern in pq_spec.get("no_files") or []:
            if any(fnmatch.fnmatch(name, str(pattern)) for name in stats["files"]):
                errors.append(f"{assert_key}.proposal_queue: forbidden file {pattern!r}")

    overlay_spec = spec.get("overlay") or {}
    if overlay_spec and step.overlays:
        tokens = estimate_overlay_tokens(step.overlays)
        max_t = overlay_spec.get("max_tokens")
        if max_t is not None and tokens > int(max_t):
            errors.append(f"{assert_key}.overlay: tokens {tokens} > max {max_t}")
        proactive = step.overlays.get("proactive_memory") or {}
        rendered = str(proactive.get("rendered_markdown") or proactive.get("markdown") or "")
        if not rendered:
            sections = proactive.get("sections") or []
            rendered = "\n".join(
                str(s.get("name") or s.get("title") or "") for s in sections if isinstance(s, dict)
            )
        for section in overlay_spec.get("required_sections") or []:
            if str(section) not in rendered:
                errors.append(f"{assert_key}.overlay: missing section {section!r}")
        for marker in overlay_spec.get("forbidden_truncation_markers") or []:
            if str(marker) in rendered:
                errors.append(f"{assert_key}.overlay: forbidden marker {marker!r}")

    return errors


def evaluate_prompt_section_assertions(
    assert_key: str,
    prompt_spec: dict[str, Any],
    prompt: str,
) -> list[str]:
    errors: list[str] = []
    for block in prompt_spec.get("not_in_section") or []:
        if not isinstance(block, dict):
            continue
        section = str(block.get("section") or "always_loaded_memory")
        start = "## [ALWAYS-LOADED MEMORY]"
        end = "## [/ALWAYS-LOADED MEMORY]"
        if section != "always_loaded_memory":
            start = f"## [{section}]"
            end = f"## [/{section}]"
        section_text = extract_prompt_section(prompt, start, end)
        for needle in block.get("text") or []:
            if str(needle) in section_text:
                errors.append(f"{assert_key}.prompt: forbidden {needle!r} inside {section}")
    return errors
