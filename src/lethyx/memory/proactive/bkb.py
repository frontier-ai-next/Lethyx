"""BKB parsing, lifecycle filtering, and conflict resolution."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from lethyx.memory.proactive.models import BeliefRule, BkbConflictError

log = logging.getLogger(__name__)

_PLATFORM_BKB_PATH = Path(__file__).with_name("platform_bkb_rules.yaml")

_SCOPE_PRECEDENCE = {
    "constitution": 0,
    "identity": 0,
    "manager": 1,
    "workspace": 2,
    "phase": 3,
    "run": 4,
    "preference": 5,
    "agent": 5,
}

_RULE_TYPE_PRECEDENCE = {
    "invariant": 0,
    "prospective_trigger": 1,
    "anti_pattern": 2,
    "behavior": 3,
    "risk": 4,
    "preference": 5,
    "belief": 3,
    "capability": 4,
}


def _debug_inject_deprecated() -> bool:
    return str(os.environ.get("LETHYX_MEMORY_DEBUG", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_bkb_rules(path: Path) -> list[BeliefRule]:
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    raw_rules = data.get("rules") if isinstance(data, dict) else data
    if not isinstance(raw_rules, list):
        return []
    rules: list[BeliefRule] = []
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        rules.append(
            BeliefRule(
                id=str(item.get("id") or ""),
                title=str(item.get("title") or ""),
                scope=str(item.get("scope") or "manager"),
                rule_type=str(item.get("type") or item.get("rule_type") or "behavior"),
                status=str(item.get("status") or "candidate"),
                trust=str(item.get("trust") or "candidate"),
                strength=float(item.get("strength") or 0.5),
                rule=item.get("rule") if isinstance(item.get("rule"), dict) else {},
                applies_to=item.get("applies_to")
                if isinstance(item.get("applies_to"), dict)
                else {},
                source_evidence=list(item.get("source_evidence") or []),
                supersedes=list(item.get("supersedes") or []),
                superseded_by=item.get("superseded_by"),
                confidence=float(item.get("confidence") or 0.0),
                support_count=int(item.get("support_count") or 0),
                contradiction_count=int(item.get("contradiction_count") or 0),
                created_at=str(item.get("created_at") or ""),
                last_verified_at=str(item.get("last_verified_at") or ""),
                expires_at=str(item.get("expires_at") or ""),
                source_backend=str(item.get("source_backend") or ""),
                lifecycle_reason=str(item.get("lifecycle_reason") or ""),
                trigger=item.get("trigger") if isinstance(item.get("trigger"), dict) else {},
                metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            )
        )
    return rules


def _scope_rank(scope: str) -> int:
    return _SCOPE_PRECEDENCE.get(str(scope or "").lower(), 99)


def _type_rank(rule_type: str) -> int:
    return _RULE_TYPE_PRECEDENCE.get(str(rule_type or "").lower(), 99)


def _rule_applies(
    rule: BeliefRule,
    *,
    workspace_id: str,
    phase_id: str,
    agent_name: str = "generic_agent",
) -> bool:
    applies = rule.applies_to or {}
    workspaces = applies.get("workspaces") or ["*"]
    phases = applies.get("phases") or ["*"]
    agents = applies.get("agents") or ["*"]
    if "*" not in workspaces and workspace_id not in workspaces:
        return False
    if "*" not in phases and phase_id not in phases:
        return False
    if "*" not in agents and agent_name not in agents:
        return False
    return True


def _rule_expired(rule: BeliefRule) -> bool:
    raw = str(rule.expires_at or "").strip()
    if not raw:
        return False
    try:
        from datetime import datetime, timezone

        stamp = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(stamp) < datetime.now(timezone.utc)
    except ValueError:
        return False


def _bkb_skip_reason(
    rule: BeliefRule,
    *,
    workspace_id: str,
    phase_id: str,
    agent_name: str = "generic_agent",
    require_verified: bool = True,
) -> str | None:
    status = str(rule.status or "").lower()
    if status in {"quarantined", "superseded", "candidate", "retracted"}:
        return f"status_{status}"
    if status == "deprecated" and not _debug_inject_deprecated():
        return "status_deprecated"
    if status != "active":
        return f"status_{status or 'unknown'}"
    if _rule_expired(rule):
        return "expired"
    if (
        rule.contradiction_count > rule.support_count
        and str(rule.rule_type or "").lower() != "invariant"
    ):
        return "contradiction_heuristic"
    if require_verified and rule.trust != "verified":
        return "trust_not_verified"
    if not _rule_applies(rule, workspace_id=workspace_id, phase_id=phase_id, agent_name=agent_name):
        return "phase_or_workspace_mismatch"
    return None


def partition_bkb_rules(
    rules: list[BeliefRule],
    *,
    workspace_id: str,
    phase_id: str,
    agent_name: str = "generic_agent",
    require_verified: bool = True,
) -> tuple[list[BeliefRule], list[dict[str, Any]]]:
    """Return active BKB rules and skipped entries with reasons for injection audit."""
    active: list[BeliefRule] = []
    skipped: list[dict[str, Any]] = []
    for rule in rules:
        reason = _bkb_skip_reason(
            rule,
            workspace_id=workspace_id,
            phase_id=phase_id,
            agent_name=agent_name,
            require_verified=require_verified,
        )
        if reason:
            skipped.append(
                {
                    "id": rule.id,
                    "title": rule.title,
                    "reason": reason,
                    "surface": "directive",
                    "source_backend": "bkb",
                }
            )
            continue
        active.append(rule)
    return active, skipped


def filter_active_rules(
    rules: list[BeliefRule],
    *,
    workspace_id: str,
    phase_id: str,
    agent_name: str = "generic_agent",
    require_verified: bool = True,
) -> list[BeliefRule]:
    active, _skipped = partition_bkb_rules(
        rules,
        workspace_id=workspace_id,
        phase_id=phase_id,
        agent_name=agent_name,
        require_verified=require_verified,
    )
    return active


def _rules_conflict(a: BeliefRule, b: BeliefRule) -> bool:
    """Heuristic: same trigger, different forbidden/behavior."""
    ra = a.rule or {}
    rb = b.rule or {}
    trigger_a = str(ra.get("trigger") or "").strip().lower()
    trigger_b = str(rb.get("trigger") or "").strip().lower()
    if not trigger_a or trigger_a != trigger_b:
        return False
    forbidden_a = str(ra.get("forbidden") or ra.get("behavior") or "").strip().lower()
    forbidden_b = str(rb.get("forbidden") or rb.get("behavior") or "").strip().lower()
    return bool(forbidden_a and forbidden_b and forbidden_a != forbidden_b)


def resolve_bkb_conflicts(
    rules: list[BeliefRule],
) -> tuple[list[BeliefRule], list[dict[str, Any]]]:
    """Return winning rules and conflict log. Fail on unresolved invariant clash."""
    conflicts: list[dict[str, Any]] = []
    winners: list[BeliefRule] = []
    pool = list(rules)
    while pool:
        pool.sort(key=lambda r: (_scope_rank(r.scope), _type_rank(r.rule_type), -r.strength))
        winner = pool.pop(0)
        losers = [o for o in pool if _rules_conflict(winner, o)]
        for other in losers:
            conflicts.append(
                {
                    "winner_id": winner.id,
                    "omitted_id": other.id,
                    "winner_scope": winner.scope,
                    "omitted_scope": other.scope,
                }
            )
        if losers:
            hard = winner.rule_type == "invariant" and any(
                loser.rule_type == "invariant"
                and _scope_rank(loser.scope) <= _scope_rank(winner.scope)
                for loser in losers
            )
            if hard:
                raise BkbConflictError(
                    f"Unresolved invariant conflict: {winner.id} vs {losers[0].id}"
                )
            pool = [r for r in pool if r not in losers]
        winners.append(winner)
    return winners, conflicts


def _belief_rule_to_mapping(rule: BeliefRule) -> dict[str, Any]:
    payload = {
        "id": rule.id,
        "title": rule.title,
        "scope": rule.scope,
        "type": rule.rule_type,
        "status": rule.status,
        "trust": rule.trust,
        "applies_to": rule.applies_to or {},
        "rule": rule.rule or {},
    }
    if rule.source_evidence:
        payload["source_evidence"] = rule.source_evidence
    if rule.supersedes:
        payload["supersedes"] = rule.supersedes
    if rule.superseded_by:
        payload["superseded_by"] = rule.superseded_by
    for key, value in (
        ("strength", rule.strength),
        ("confidence", rule.confidence),
        ("support_count", rule.support_count),
        ("contradiction_count", rule.contradiction_count),
        ("created_at", rule.created_at),
        ("last_verified_at", rule.last_verified_at),
        ("expires_at", rule.expires_at),
        ("source_backend", rule.source_backend),
        ("lifecycle_reason", rule.lifecycle_reason),
    ):
        if value not in ("", None, 0, 0.0):
            payload[key] = value
    if rule.trigger:
        payload["trigger"] = dict(rule.trigger)
    if rule.metadata:
        payload["metadata"] = dict(rule.metadata)
    return payload


def _raw_bkb_rules(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    raw_rules = data.get("rules") if isinstance(data, dict) else data
    if not isinstance(raw_rules, list):
        return []
    return [dict(item) for item in raw_rules if isinstance(item, dict)]


def merge_bkb_rules(path: Path, rules: list[dict[str, Any]]) -> None:
    """Merge raw BKB rule mappings into ``path`` by stable rule id."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for item in [*_raw_bkb_rules(target), *rules]:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("id") or "").strip()
        if not rule_id:
            anonymous.append(dict(item))
            continue
        merged[rule_id] = dict(item)
    payload = {"rules": [*anonymous, *merged.values()]}
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def ensure_platform_bkb_rules(manager_core_root: Path) -> None:
    """Merge bundled platform rules into manager core bkb.yaml by rule id."""
    platform_rules = load_bkb_rules(_PLATFORM_BKB_PATH)
    if not platform_rules:
        return
    target = Path(manager_core_root) / "bkb.yaml"
    try:
        merge_bkb_rules(target, [_belief_rule_to_mapping(rule) for rule in platform_rules])
    except OSError:
        log.debug("failed to write merged platform BKB rules", exc_info=True)


def format_bkb_section(rules: list[BeliefRule], *, max_chars: int) -> tuple[str, list[str]]:
    lines: list[str] = []
    refs: list[str] = []
    for rule in rules:
        refs.append(f"bkb:{rule.scope}:{rule.id}")
        lines.append(f"- **{rule.title}** ({rule.rule_type})")
        body = rule.rule or {}
        for key in ("behavior", "forbidden", "trigger"):
            val = str(body.get(key) or "").strip()
            if val:
                lines.append(f"  - {key}: {val}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n...[BKB truncated]"
    return text, refs
