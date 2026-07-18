"""Typed runtime control decisions consumed before another LLM round."""

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from lethyx.contracts.hashing import hash_value
from lethyx.contracts.models import json_ready

CONTROL_DECISION_FILENAME = "control_decision.json"
CONTROL_DECISION_LEDGER_FILENAME = "control_decisions.jsonl"


def _state_dir(drive_root: Path) -> Path:
    path = Path(drive_root) / "state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def control_decision_from_recovery_decision(
    recovery_decision: Mapping[str, Any],
    *,
    run_id: str = "",
    task_id: str = "",
    phase_id: str = "",
    target_phase: str = "",
    target_work_item_kind: str = "",
    allowed_next_tools: list[str] | None = None,
    state_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonicalize a RecoveryDecision payload into a runner-owned decision."""

    decision = dict(recovery_decision)
    kind = str(decision.get("kind") or "").strip()
    reason_code = str(decision.get("trigger_code") or kind).strip()
    active_subtask_id = str(decision.get("active_subtask_id") or "").strip()
    blocker_fingerprint = str(decision.get("blocker_fingerprint") or "").strip()
    issues = [
        json_ready(item) for item in (decision.get("issues") or []) if isinstance(item, Mapping)
    ]
    required_changes = [
        json_ready(item)
        for item in (
            decision.get("required_plan_changes") or decision.get("required_changes") or []
        )
        if isinstance(item, Mapping)
    ]
    evidence_refs = _string_list(decision.get("evidence_refs"))
    payload = {
        "control_decision_id": hash_value(
            {
                "kind": kind,
                "reason_code": reason_code,
                "active_subtask_id": active_subtask_id,
                "blocker_fingerprint": blocker_fingerprint,
                "recovery_decision_id": decision.get("decision_id"),
                "task_id": task_id,
            }
        )[:16],
        "kind": kind,
        "target_phase": target_phase or str(decision.get("loop_back_target") or "none"),
        "target_work_item_kind": target_work_item_kind,
        "reason_code": reason_code,
        "blocker_fingerprint": blocker_fingerprint,
        "active_subtask_id": active_subtask_id,
        "allowed_next_actions": _string_list(decision.get("allowed_next_actions")),
        "forbidden_next_actions": _string_list(decision.get("forbidden_next_actions")),
        "allowed_next_tools": list(allowed_next_tools or []),
        "issues": issues,
        "required_changes": required_changes,
        "evidence_refs": evidence_refs,
        "recovery_decision": json_ready(decision),
        "source": "RecoveryDecision",
        "run_id": run_id,
        "source_task_id": task_id,
        "phase_id": phase_id,
        "created_at": time.time(),
        "state_signature": json_ready(dict(state_signature or {})),
    }
    return payload


def write_control_decision(drive_root: Path, decision: Mapping[str, Any]) -> Path:
    """Persist the active ControlDecision and append it to the decision ledger."""

    state = _state_dir(Path(drive_root))
    payload = json_ready(dict(decision))
    latest = state / CONTROL_DECISION_FILENAME
    tmp = latest.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, latest)
    with (state / CONTROL_DECISION_LEDGER_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return latest


def load_latest_control_decision(drive_root: Path | None) -> dict[str, Any]:
    if drive_root is None:
        return {}
    path = Path(drive_root) / "state" / CONTROL_DECISION_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def clear_control_decision(drive_root: Path | None) -> None:
    if drive_root is None:
        return
    try:
        (Path(drive_root) / "state" / CONTROL_DECISION_FILENAME).unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


__all__ = [
    "CONTROL_DECISION_FILENAME",
    "CONTROL_DECISION_LEDGER_FILENAME",
    "clear_control_decision",
    "control_decision_from_recovery_decision",
    "load_latest_control_decision",
    "write_control_decision",
]
