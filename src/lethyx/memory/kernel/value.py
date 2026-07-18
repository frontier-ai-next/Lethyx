"""Directive memory utility tracking.

This module deliberately measures utility separately from authority.  Low value can
rank or suppress non-safety rules under budget, but it cannot delete, quarantine,
or supersede durable memory.
"""

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lethyx.memory.proactive.models import BeliefRule
else:
    BeliefRule = Any


@dataclass
class MemoryUseRecord:
    schema_version: str = "1"
    rule_id: str = ""
    memory_event_id: str = ""
    workspace_id: str = ""
    run_id: str = ""
    phase_id: str = ""
    subtask_id: str = ""
    agent_kind: str = ""
    injection_id: str = ""
    injected_at: str = ""
    surface: str = "directive"
    token_cost: int = 0
    reason: str = ""
    first_action_observed: str = ""
    first_action_aligned: bool | None = None
    used_by_agent: bool | None = None
    phase_outcome: str = "unknown"
    verifier_outcome: str = "unknown"
    success_score: float | None = None
    failure_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryValueStats:
    rule_id: str
    uses_count: int = 0
    eligible_count: int = 0
    injected_count: int = 0
    success_after_use: int = 0
    failure_after_use: int = 0
    first_action_aligned_count: int = 0
    first_action_total: int = 0
    contradiction_count: int = 0
    token_cost_total: int = 0
    token_cost_avg: float = 0.0
    estimated_value: float = 0.0
    confidence: float = 0.0
    last_used_at: str = ""
    last_helped_at: str = ""
    last_hurt_at: str = ""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _state_dir(drive_root: Path) -> Path:
    path = Path(drive_root) / "state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def estimate_memory_value(stats: MemoryValueStats) -> float:
    success_rate = (stats.success_after_use + 1) / (stats.uses_count + 2)
    failure_rate = (stats.failure_after_use + 1) / (stats.uses_count + 2)
    alignment_rate = stats.first_action_aligned_count / max(1, stats.first_action_total)
    confidence = min(1.0, math.log(1 + stats.uses_count) / math.log(20))
    normalized_token_cost = stats.token_cost_avg / 1000
    contradiction_penalty = math.log(1 + stats.contradiction_count)
    value = (
        confidence * (success_rate - failure_rate)
        + 0.8 * alignment_rate
        - 0.15 * normalized_token_cost
        - 1.2 * contradiction_penalty
    )
    stats.confidence = confidence
    stats.estimated_value = value
    return value


def record_directive_injection(
    drive_root: Path | None,
    rule: BeliefRule,
    context: dict[str, Any] | None = None,
    token_cost: int = 0,
    reason: str = "eligible_directive",
) -> MemoryUseRecord:
    context = dict(context or {})
    record = MemoryUseRecord(
        rule_id=rule.id,
        memory_event_id=str(context.get("memory_event_id") or ""),
        workspace_id=str(context.get("workspace_id") or ""),
        run_id=str(context.get("run_id") or ""),
        phase_id=str(context.get("phase_id") or ""),
        subtask_id=str(context.get("subtask_id") or ""),
        agent_kind=str(context.get("agent_kind") or "generic_agent"),
        injection_id=str(context.get("injection_id") or f"inj_{uuid.uuid4().hex[:12]}"),
        injected_at=_now(),
        surface="directive",
        token_cost=max(0, int(token_cost or 0)),
        reason=reason,
        metadata={"rule_type": rule.rule_type, **dict(context.get("metadata") or {})},
    )
    if drive_root is not None:
        state = _state_dir(Path(drive_root))
        with (state / "memory_use_records.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
    return record


def _stats_path(drive_root: Path) -> Path:
    return _state_dir(Path(drive_root)) / "memory_value_stats.json"


def load_memory_value_stats(drive_root: Path | None) -> dict[str, MemoryValueStats]:
    if drive_root is None:
        return {}
    path = _stats_path(Path(drive_root))
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = data.get("rules") if isinstance(data, dict) else data
    stats: dict[str, MemoryValueStats] = {}
    if isinstance(rows, dict):
        iterable = rows.values()
    elif isinstance(rows, list):
        iterable = rows
    else:
        iterable = []
    for row in iterable:
        if not isinstance(row, dict) or not row.get("rule_id"):
            continue
        stats[str(row["rule_id"])] = MemoryValueStats(
            **{k: v for k, v in row.items() if k in MemoryValueStats.__dataclass_fields__}
        )
    return stats


def save_memory_value_stats(drive_root: Path, stats: dict[str, MemoryValueStats]) -> None:
    payload = {"schema_version": "1", "rules": {k: asdict(v) for k, v in sorted(stats.items())}}
    _stats_path(Path(drive_root)).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _load_records(drive_root: Path) -> list[MemoryUseRecord]:
    path = _state_dir(Path(drive_root)) / "memory_use_records.jsonl"
    records: list[MemoryUseRecord] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            records.append(
                MemoryUseRecord(
                    **{k: v for k, v in row.items() if k in MemoryUseRecord.__dataclass_fields__}
                )
            )
    return records


def update_memory_value_stats(
    drive_root: Path | None,
    records: list[MemoryUseRecord] | None = None,
    phase_outcome: str | None = None,
    verifier_outcome: str | None = None,
) -> dict[str, MemoryValueStats]:
    if drive_root is None:
        return {}
    root = Path(drive_root)
    # Recompute from the JSONL source of truth when records are not supplied so
    # repeated report refreshes are idempotent. Explicit records represent a new
    # incremental update from the current overlay build.
    if records is None:
        stats: dict[str, MemoryValueStats] = {}
        records_to_apply = _load_records(root)
    else:
        stats = load_memory_value_stats(root)
        records_to_apply = records
    for record in records_to_apply:
        item = stats.setdefault(record.rule_id, MemoryValueStats(rule_id=record.rule_id))
        item.eligible_count += 1
        item.injected_count += 1
        if record.used_by_agent is not False:
            item.uses_count += 1
            item.last_used_at = record.injected_at
        if record.first_action_aligned is not None:
            item.first_action_total += 1
            if record.first_action_aligned:
                item.first_action_aligned_count += 1
        outcome = phase_outcome or record.phase_outcome
        verifier = verifier_outcome or record.verifier_outcome
        if outcome in {"success", "passed"} or verifier in {"success", "passed"}:
            item.success_after_use += 1
            item.last_helped_at = record.injected_at
        elif outcome in {"failure", "failed"} or verifier in {"failure", "failed"}:
            item.failure_after_use += 1
            item.last_hurt_at = record.injected_at
        item.token_cost_total += max(0, record.token_cost)
        item.token_cost_avg = item.token_cost_total / max(1, item.injected_count)
        estimate_memory_value(item)
    save_memory_value_stats(root, stats)
    return stats


def _safety_critical(rule: BeliefRule) -> bool:
    return str(rule.rule_type).lower() == "invariant" or str(rule.scope).lower() in {
        "constitution",
        "identity",
    }


def should_suppress_low_value(
    rule: BeliefRule, stats: MemoryValueStats | None, safety_critical: bool = False
) -> tuple[bool, str]:
    if safety_critical or _safety_critical(rule):
        return False, "safety_critical_never_suppressed_by_value"
    if stats is None or stats.confidence < 0.4:
        return False, "insufficient_value_confidence"
    if stats.estimated_value < -0.5:
        return True, "low_estimated_value_under_budget"
    return False, "value_ok"


def rank_directives_by_value(
    rules: list[BeliefRule],
    stats: dict[str, MemoryValueStats] | None,
    token_budget: int,
) -> list[BeliefRule]:
    stats = stats or {}

    def key(rule: BeliefRule) -> tuple[int, float, float]:
        stat = stats.get(rule.id)
        value = stat.estimated_value if stat else 0.0
        safety = 0 if _safety_critical(rule) else 1
        return (safety, -value, -float(rule.strength or 0.0))

    return sorted(rules, key=key)
