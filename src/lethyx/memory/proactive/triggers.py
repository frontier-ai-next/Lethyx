"""Deterministic prospective memory triggers."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lethyx.memory.proactive.models import BeliefRule


@dataclass(frozen=True)
class TriggerContext:
    phase_id: str = ""
    workspace_id: str = ""
    run_id: str = ""
    agent_kind: str = "generic_agent"
    changed_paths: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    run_state_flags: tuple[str, ...] = ()
    previous_phase: str = ""
    failure_signatures: tuple[str, ...] = ()
    tools_available: tuple[str, ...] = ()
    env_flags: tuple[str, ...] = ()
    retry_count: int = 0


@dataclass(frozen=True)
class TriggerMatchResult:
    matched: bool
    reason: str
    expected_first_actions: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


def _values(rule: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = rule.get(key) or ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple, set)):
        return tuple(str(v) for v in raw if str(v).strip())
    return ()


def _any_match(
    candidates: tuple[str, ...], values: tuple[str, ...], *, prefix: bool = False
) -> bool:
    if not values:
        return True
    if "*" in values:
        return True
    if prefix:
        return any(any(c == v or c.startswith(v) for c in candidates) for v in values)
    return any(v in candidates for v in values)


def _all_match(
    candidates: tuple[str, ...], values: tuple[str, ...], *, prefix: bool = False
) -> bool:
    if not values:
        return True
    if prefix:
        return all(any(c == v or c.startswith(v) for c in candidates) for v in values)
    return all(v in candidates for v in values)


def trigger_matches(rule: BeliefRule, context: TriggerContext) -> TriggerMatchResult:
    trigger = dict(rule.trigger or {})
    expected = tuple(
        str(v) for v in (rule.rule or {}).get("expected_first_action_any", ()) if str(v).strip()
    )
    if str(rule.rule_type or "") != "prospective_trigger":
        return TriggerMatchResult(True, "not_a_trigger", expected)
    if not trigger:
        return TriggerMatchResult(False, "missing_trigger", expected)
    phase_any = _values(trigger, "phase_any")
    if phase_any and context.phase_id not in phase_any and "*" not in phase_any:
        return TriggerMatchResult(False, "phase_mismatch", expected)
    phase_not_any = _values(trigger, "phase_not_any")
    if phase_not_any and context.phase_id in phase_not_any:
        return TriggerMatchResult(False, "phase_excluded", expected)
    agents = _values(trigger, "agent_kinds_any")
    if agents and context.agent_kind not in agents and "*" not in agents:
        return TriggerMatchResult(False, "agent_mismatch", expected)
    if not _any_match(context.changed_paths, _values(trigger, "changed_paths_any"), prefix=True):
        return TriggerMatchResult(False, "changed_path_mismatch", expected)
    if not _all_match(context.changed_paths, _values(trigger, "changed_paths_all"), prefix=True):
        return TriggerMatchResult(False, "changed_path_all_mismatch", expected)
    if not _any_match(context.artifacts, _values(trigger, "artifacts_exist_any"), prefix=True):
        return TriggerMatchResult(False, "artifact_mismatch", expected)
    if not _all_match(context.artifacts, _values(trigger, "artifacts_exist_all"), prefix=True):
        return TriggerMatchResult(False, "artifact_all_mismatch", expected)
    if not _any_match(context.run_state_flags, _values(trigger, "run_state_flags_any")):
        return TriggerMatchResult(False, "run_state_flag_mismatch", expected)
    if not _all_match(context.run_state_flags, _values(trigger, "run_state_flags_all")):
        return TriggerMatchResult(False, "run_state_flag_all_mismatch", expected)
    prev = _values(trigger, "previous_phase_any")
    if prev and context.previous_phase not in prev and "*" not in prev:
        return TriggerMatchResult(False, "previous_phase_mismatch", expected)
    if not _any_match(context.failure_signatures, _values(trigger, "failure_signatures_any")):
        return TriggerMatchResult(False, "failure_signature_mismatch", expected)
    if not _any_match(context.tools_available, _values(trigger, "tools_available_any")):
        return TriggerMatchResult(False, "tool_mismatch", expected)
    if not _any_match(context.env_flags, _values(trigger, "env_flags_any")):
        return TriggerMatchResult(False, "env_flag_mismatch", expected)
    min_retry = int(trigger.get("min_retry_count") or 0)
    max_retry = int(trigger.get("max_retry_count") or 999)
    if context.retry_count < min_retry or context.retry_count > max_retry:
        return TriggerMatchResult(False, "retry_count_mismatch", expected)
    return TriggerMatchResult(True, "matched", expected, {"trigger": trigger})


def _read_json_list(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if isinstance(data, list):
        return tuple(str(v) for v in data)
    if isinstance(data, dict):
        for key in ("changed_paths", "artifacts", "flags", "failure_signatures"):
            value = data.get(key)
            if isinstance(value, list):
                return tuple(str(v) for v in value)
    return ()


def build_trigger_context(
    *,
    drive_root: Path | None = None,
    phase_id: str = "",
    workspace_id: str = "",
    run_id: str = "",
    agent_kind: str = "generic_agent",
    changed_paths: tuple[str, ...] = (),
    run_state_flags: tuple[str, ...] = (),
    tools_available: tuple[str, ...] = (),
    retry_count: int = 0,
) -> TriggerContext:
    artifacts: list[str] = []
    previous_phase = ""
    failures: tuple[str, ...] = ()
    if drive_root is not None:
        root = Path(drive_root)
        changed_paths = changed_paths or _read_json_list(root / "state" / "changed_paths.json")
        run_state_flags = run_state_flags or _read_json_list(
            root / "state" / "run_state_flags.json"
        )
        failures = _read_json_list(root / "state" / "failure_signatures.json")
        phase_plan = root / "state" / "phase_plan.json"
        if phase_plan.is_file():
            try:
                data = json.loads(phase_plan.read_text(encoding="utf-8"))
                previous_phase = (
                    str(data.get("previous_phase") or "") if isinstance(data, dict) else ""
                )
            except (OSError, json.JSONDecodeError):
                previous_phase = ""
        for base in (root / "state", root / "artifacts"):
            if base.is_dir():
                for item in base.rglob("*"):
                    if item.is_file():
                        artifacts.append(str(item.relative_to(root)).replace("\\", "/"))
    env_flags = tuple(
        k for k, v in os.environ.items() if k.startswith("LETHYX_") and str(v).strip()
    )
    return TriggerContext(
        phase_id=phase_id,
        workspace_id=workspace_id,
        run_id=run_id,
        agent_kind=agent_kind,
        changed_paths=tuple(changed_paths),
        artifacts=tuple(artifacts),
        run_state_flags=tuple(run_state_flags),
        previous_phase=previous_phase,
        failure_signatures=failures,
        tools_available=tuple(tools_available),
        env_flags=env_flags,
        retry_count=retry_count,
    )
