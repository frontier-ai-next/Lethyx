"""Subtask recovery timestamps for superseding stale revise reviews."""

from typing import Any

from lethyx.contracts.models import ReviewContract


def _signal_created_at(row: dict[str, Any]) -> float:
    try:
        return float(row.get("created_at") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _completion_passed(completion: dict[str, Any] | None) -> bool:
    if not isinstance(completion, dict):
        return False
    report = completion.get("verification_report")
    if isinstance(report, dict) and report.get("passed") is True:
        return True
    contract = completion.get("completion_contract")
    if isinstance(contract, dict):
        nested = contract.get("verification_report")
        if isinstance(nested, dict) and nested.get("passed") is True:
            return True
    return False


def _recovery_from_completion_dict(completion: dict[str, Any] | None) -> float:
    if not _completion_passed(completion):
        return 0.0
    if not isinstance(completion, dict):
        return 0.0
    try:
        return float(completion.get("completed_at") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _recovery_from_plan_dict(plan_data: dict[str, Any] | None) -> dict[str, float]:
    recovery: dict[str, float] = {}
    if not isinstance(plan_data, dict):
        return recovery
    for node in plan_data.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if str(node.get("manifest_id") or node.get("id") or "") != "execute":
            continue
        for card in node.get("subtasks") or []:
            if not isinstance(card, dict):
                continue
            if str(card.get("status") or "") != "done":
                continue
            subtask_id = str(card.get("id") or "").strip()
            if not subtask_id:
                continue
            stamp = _recovery_from_completion_dict(card.get("completion"))
            if stamp > 0:
                recovery[subtask_id] = max(recovery.get(subtask_id, 0.0), stamp)
    return recovery


def _recovery_from_signals(signal_rows: list[dict[str, Any]]) -> dict[str, float]:
    recovery: dict[str, float] = {}
    for row in signal_rows:
        if str(row.get("kind") or "") != "mark_subtask_complete":
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        contract = payload.get("completion_contract")
        if not isinstance(contract, dict):
            continue
        subtask_id = str(contract.get("subtask_id") or payload.get("subtask_id") or "").strip()
        if not subtask_id:
            continue
        report = contract.get("verification_report")
        if not (isinstance(report, dict) and report.get("passed") is True):
            continue
        created = _signal_created_at(row)
        if created > 0:
            recovery[subtask_id] = max(recovery.get(subtask_id, 0.0), created)
    return recovery


def subtask_passing_recovery_at(
    *,
    plan_data: dict[str, Any] | None = None,
    signal_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    recovery = _recovery_from_plan_dict(plan_data)
    for subtask_id, stamp in _recovery_from_signals(signal_rows or ()).items():
        recovery[subtask_id] = max(recovery.get(subtask_id, 0.0), stamp)
    return recovery


def review_subtask_id(review: ReviewContract) -> str:
    for issue in review.issues:
        subtask_id = str(issue.subtask_id or "").strip()
        if subtask_id:
            return subtask_id
    return ""


def review_superseded_by_recovery(
    review: ReviewContract,
    *,
    recovery_at: dict[str, float],
    review_created_at: float,
) -> bool:
    if review.verdict not in {"revise", "abort"}:
        return False
    subtask_id = review_subtask_id(review)
    if not subtask_id:
        return False
    recovered = recovery_at.get(subtask_id, 0.0)
    return recovered > review_created_at > 0
