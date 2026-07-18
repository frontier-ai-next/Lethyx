"""BKB promotion gate — controlled updates to core files."""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lethyx.contracts import EvidenceRef
from lethyx.contracts.evidence import ALLOWED_EVIDENCE_REF_TYPES, EvidenceResolver
from lethyx.contracts.models import WorkspaceContext
from lethyx.memory.backends.base import DurableLesson
from lethyx.memory.backends.contracts import durable_lesson_to_block_record
from lethyx.memory.backends.factory import create_durable_backend
from lethyx.memory.hindsight.candidates import (
    add_accepted_fingerprints,
    candidate_fingerprint,
)
from lethyx.memory.hindsight.mapping import derived_tags, stable_hash

from lethyx.memory.hindsight.payloads import render_bkb_rule_for_hindsight
from lethyx.memory.paths import (
    manager_core_root,
    workspace_core_root,
    workspace_root as workspace_project_root,
)
from lethyx.memory.proactive.bkb import load_bkb_rules, resolve_bkb_conflicts
from lethyx.memory.proactive.models import BeliefRule, BkbConflictError
from lethyx.memory.tribunal.models import TribunalInput
from lethyx.memory.tribunal.report import write_tribunal_report
from lethyx.memory.tribunal.runner import run_tribunal
from lethyx.memory.proactive.io import (
    _atomic_write_unlocked,
    append_audit,
    append_rule_md_slice,
    bkb_transaction,
)

_TRUSTED_BKB_ACTORS = frozenset({"supervisor", "verifier", "harness", "watcher"})
_TRUSTED_BKB_PRODUCERS = frozenset({"supervisor", "verifier", "harness"})
_LEDGER_BACKED_REF_TYPES = frozenset(
    {
        "ledger_event",
        "verification_report",
        "test_run",
        "mutation_report",
        "input_sensitivity_report",
    }
)


@dataclass
class ProposedBkbPatch:
    patch_id: str
    rules: list[dict[str, Any]] = field(default_factory=list)
    source_evidence: list[dict[str, Any]] = field(default_factory=list)
    actor: str = "Lethyx"
    run_id: str = ""
    phase_id: str = ""
    workspace_id: str = ""


def _artifact_ref_path_exists(
    ref_id: str,
    *,
    repo_root: Path,
    ws_project_root: Path,
    drive_root: Path | None = None,
) -> bool:
    candidates = [
        ws_project_root / ref_id,
        repo_root / ref_id,
    ]
    if drive_root is not None:
        candidates.append(drive_root / ref_id)
    for path in candidates:
        try:
            if path.resolve().is_file():
                return True
        except OSError:
            continue
    return False


def validate_patch_evidence(
    patch: ProposedBkbPatch,
    *,
    repo_root: Path,
    workspace_id: str = "",
    drive_root: Path | None = None,
) -> None:
    """Reject string refs, agent-only producers, or unresolvable ledger refs."""
    if not patch.source_evidence:
        raise ValueError("BKB patch requires non-empty typed source_evidence")
    if str(patch.actor or "").strip().lower() not in _TRUSTED_BKB_ACTORS:
        raise ValueError(
            f"BKB patch actor must be one of {_TRUSTED_BKB_ACTORS}, got {patch.actor!r}"
        )

    for raw in patch.source_evidence:
        if isinstance(raw, str):
            raise ValueError("source_evidence must be typed EvidenceRef objects, not strings")

    ws = workspace_id or patch.workspace_id or ""
    if ws:
        ws_project = workspace_project_root(repo_root, ws)
    else:
        ws_project = repo_root.resolve()
    ctx = WorkspaceContext(
        repo_root=str(repo_root.resolve()),
        workspace_id=ws,
        workspace_root=str(ws_project),
    )
    resolver = EvidenceResolver(ctx)

    for raw in patch.source_evidence:
        if not isinstance(raw, dict):
            raise ValueError("source_evidence entries must be objects")
        ref = EvidenceRef.from_mapping(raw)
        if ref.produced_by not in _TRUSTED_BKB_PRODUCERS:
            raise ValueError("BKB evidence must be produced by supervisor, verifier, or harness")
        issues = resolver.validate_ref(ref, phase=patch.phase_id)
        blocking = [issue for issue in issues if issue.severity in {"blocking", "error"}]
        if blocking:
            raise ValueError(blocking[0].message)

    has_strong_evidence = False
    for raw in patch.source_evidence:
        if not isinstance(raw, dict):
            continue
        ref = EvidenceRef.from_mapping(raw)
        if ref.ref_type not in ALLOWED_EVIDENCE_REF_TYPES:
            continue
        if ref.ref_type in _LEDGER_BACKED_REF_TYPES:
            row_issues = resolver.validate_ref(ref, phase=patch.phase_id)
            if not any(i.severity in {"blocking", "error"} for i in row_issues):
                has_strong_evidence = True
                break
        if ref.ref_type == "artifact":
            if _artifact_ref_path_exists(
                ref.ref_id,
                repo_root=repo_root.resolve(),
                ws_project_root=ws_project,
                drive_root=drive_root,
            ):
                has_strong_evidence = True
                break
    if not has_strong_evidence:
        raise ValueError("BKB patch requires ledger-backed evidence or an existing artifact path")


def accept_bkb_patch(
    repo_root: Path,
    patch: ProposedBkbPatch,
    *,
    target: str = "manager",
    drive_root: Path | None = None,
) -> dict[str, Any]:
    """Accept patch: merge rules as active/verified after conflict check."""
    ws = patch.workspace_id.strip()
    if target == "workspace" and ws:
        core_root = workspace_core_root(repo_root, ws)
    else:
        core_root = manager_core_root(repo_root)
        target = "manager"

    source_backend = str(
        patch.actor
        or (
            patch.rules[0].get("source_backend")
            if patch.rules and isinstance(patch.rules[0], dict)
            else ""
        )
        or "Lethyx"
    )
    tribunal_decision = run_tribunal(
        TribunalInput(
            candidate_id=patch.patch_id,
            rules=tuple(dict(rule) for rule in patch.rules if isinstance(rule, dict)),
            source_evidence=tuple(
                dict(ref) for ref in patch.source_evidence if isinstance(ref, dict)
            ),
            actor=patch.actor,
            source_backend=source_backend,
            workspace_id=ws,
            target_workspace_id=ws if target == "workspace" else "",
            drive_root=str(drive_root or ""),
        )
    )
    tribunal_decision = write_tribunal_report(drive_root, patch.patch_id, tribunal_decision)
    if tribunal_decision.decision != "accept":
        raise ValueError(
            f"memory tribunal {tribunal_decision.decision}: {tribunal_decision.reason}"
        )

    validate_patch_evidence(
        patch,
        repo_root=repo_root,
        workspace_id=ws,
        drive_root=drive_root,
    )

    proposed: list[BeliefRule] = []
    for raw in patch.rules:
        if not isinstance(raw, dict):
            continue
        proposed.append(
            BeliefRule(
                id=str(raw.get("id") or f"bkb_{uuid.uuid4().hex[:8]}"),
                title=str(raw.get("title") or ""),
                scope=str(raw.get("scope") or target),
                rule_type=str(raw.get("type") or raw.get("rule_type") or "behavior"),
                status="active",
                trust="verified",
                strength=float(raw.get("strength") or 0.85),
                rule=raw.get("rule") if isinstance(raw.get("rule"), dict) else {},
                applies_to=raw.get("applies_to") if isinstance(raw.get("applies_to"), dict) else {},
                source_evidence=list(patch.source_evidence or raw.get("source_evidence") or []),
                confidence=float(raw.get("confidence") or 0.85),
                support_count=max(1, int(raw.get("support_count") or 1)),
                source_backend=str(patch.actor or raw.get("source_backend") or "supervisor"),
            )
        )
    if not proposed:
        raise ValueError("BKB patch contains no valid rules")

    bkb_path = core_root / "bkb.yaml"
    old_hash = "sha256:empty"
    if bkb_path.is_file():
        old_hash = f"sha256:{hashlib.sha256(bkb_path.read_bytes()).hexdigest()}"

    with bkb_transaction(core_root) as locked_bkb:
        existing = load_bkb_rules(locked_bkb)
        active = [r for r in existing + proposed if r.status == "active"]
        try:
            resolve_bkb_conflicts(active)
        except BkbConflictError as exc:
            raise ValueError(str(exc)) from exc

        data = (
            yaml.safe_load(locked_bkb.read_text(encoding="utf-8")) if locked_bkb.is_file() else {}
        )
        raw_rules = list(data.get("rules") or []) if isinstance(data, dict) else []

        for rule in proposed:
            entry = {
                "id": rule.id,
                "title": rule.title,
                "scope": rule.scope,
                "type": rule.rule_type,
                "status": "active",
                "trust": "verified",
                "strength": rule.strength,
                "rule": rule.rule,
                "applies_to": rule.applies_to,
                "source_evidence": rule.source_evidence,
                "confidence": rule.confidence or rule.strength,
                "support_count": max(1, rule.support_count or 1),
                "source_backend": rule.source_backend or patch.actor,
            }
            replaced = False
            for i, existing_rule in enumerate(raw_rules):
                if isinstance(existing_rule, dict) and existing_rule.get("id") == rule.id:
                    raw_rules[i] = entry
                    replaced = True
                    break
            if not replaced:
                raw_rules.append(entry)

        new_content = yaml.safe_dump({"rules": raw_rules}, allow_unicode=True)
        new_hash = _atomic_write_unlocked(locked_bkb, new_content)

        for rule in proposed:
            body = json.dumps(rule.rule, ensure_ascii=False) if rule.rule else rule.title
            append_rule_md_slice(
                core_root,
                rule_type=rule.rule_type,
                title=rule.title,
                body=body[:500],
                rule_id=rule.id,
                target=target,
            )

        append_audit(
            core_root / "audit.jsonl",
            {
                "patch_id": patch.patch_id,
                "action": "accepted",
                "source_evidence": patch.source_evidence,
                "old_hash": old_hash,
                "new_hash": new_hash,
                "actor": patch.actor,
                "run_id": patch.run_id,
                "phase_id": patch.phase_id,
            },
        )

    _retain_bkb_rules_in_selected_backend(repo_root, patch, proposed)
    _retain_accepted_bkb_in_selected_backend(repo_root, patch, proposed, target=target)
    _record_accepted_bkb_fingerprints(repo_root, patch, proposed)

    return {"accepted": True, "patch_id": patch.patch_id, "rules": [r.id for r in proposed]}


def reject_bkb_patch(
    repo_root: Path,
    patch: ProposedBkbPatch,
    *,
    reason: str = "",
    quarantine: bool = False,
    target: str = "manager",
) -> dict[str, Any]:
    if target == "workspace" and patch.workspace_id.strip():
        core_root = workspace_core_root(repo_root, patch.workspace_id)
    else:
        core_root = manager_core_root(repo_root)

    append_audit(
        core_root / "audit.jsonl",
        {
            "patch_id": patch.patch_id,
            "action": "rejected",
            "reason": reason,
            "quarantine": quarantine,
            "source_evidence": patch.source_evidence,
            "actor": patch.actor,
            "run_id": patch.run_id,
            "phase_id": patch.phase_id,
        },
    )
    return {"accepted": False, "patch_id": patch.patch_id, "reason": reason}


def _retain_bkb_rules_in_selected_backend(
    repo_root: Path,
    patch: ProposedBkbPatch,
    rules: list[BeliefRule],
) -> None:
    import logging

    from lethyx.memory.kernel.models import memory_event_from_tool_write
    from lethyx.memory.kernel.writer import write_memory_event
    from lethyx.memory.palace.tiers import Tier

    log = logging.getLogger(__name__)
    ws = patch.workspace_id or ""
    for rule in rules:
        content = f"{rule.title}\n\n{json.dumps(rule.rule, ensure_ascii=False)}"
        tier = Tier.ALWAYS_ON if rule.rule_type in {"invariant", "behavior"} else Tier.WARM
        try:
            event = memory_event_from_tool_write(
                content=content,
                title=rule.title,
                memory_kind="lesson",
                workspace_id=ws,
                tags=["core_lesson", "bkb", rule.rule_type],
                scope="cross_run_durable",
                tier=tier,
                trust_level="public_verified",
                evidence_refs=list(patch.source_evidence or []),
                lifecycle="active",
                surface="supplemental_evidence",
                source_backend="bkb_promotion",
                verified=True,
                phase_id=patch.phase_id,
                run_id=patch.run_id,
                palace_store="palace.lesson",
                metadata={"bkb_rule_id": rule.id, "patch_id": patch.patch_id},
            )
            result = write_memory_event(repo_root, event, workspace_id=ws)
            if not result.saved:
                log.debug(
                    "BKB selected-backend retention skipped for %s: %s",
                    rule.id,
                    list(result.policy_issues) or result.error,
                )
        except Exception:
            log.debug("BKB selected-backend retention failed for %s", rule.id, exc_info=True)


def _rule_dict(rule: BeliefRule) -> dict[str, Any]:
    return {
        "id": rule.id,
        "title": rule.title,
        "scope": rule.scope,
        "type": rule.rule_type,
        "status": rule.status,
        "trust": rule.trust,
        "strength": rule.strength,
        "rule": rule.rule,
        "applies_to": rule.applies_to,
        "source_evidence": rule.source_evidence,
        "confidence": rule.confidence,
        "support_count": rule.support_count,
        "source_backend": rule.source_backend,
    }


def _retain_accepted_bkb_in_selected_backend(
    repo_root: Path,
    patch: ProposedBkbPatch,
    rules: list[BeliefRule],
    *,
    target: str,
) -> None:
    for rule in rules:
        raw_rule = _rule_dict(rule)
        fingerprint = candidate_fingerprint(
            {
                "kind": rule.rule_type,
                "scope": rule.scope,
                "title": rule.title,
                "content": rule.rule,
                "source_evidence": patch.source_evidence,
            }
        )
        lesson = DurableLesson(
            lesson_id=rule.id,
            kind="accepted_bkb_rule",
            title=rule.title or rule.id,
            content=render_bkb_rule_for_hindsight(raw_rule, patch),
            workspace_id=patch.workspace_id,
            run_id=patch.run_id,
            phase_id=patch.phase_id,
            trust_level="supervisor_verified",
            evidence_refs=list(patch.source_evidence),
            tags=derived_tags(
                kind="bkb_rule",
                workspace_id=patch.workspace_id,
                run_id=patch.run_id,
                phase_id=patch.phase_id,
                trust_level="supervisor_verified",
                scope=target,
                store="bkb",
            ),
            metadata={
                "bkb_rule_id": rule.id,
                "patch_id": patch.patch_id,
                "target": target,
                "source_kind": "accepted_bkb_rule",
                "candidate_fingerprint": fingerprint,
                "source_hash": stable_hash(raw_rule),
            },
        )
        record = durable_lesson_to_block_record(lesson, backend_id="")
        create_durable_backend(repo_root, workspace_id=patch.workspace_id).retain_block(record)


def _record_accepted_bkb_fingerprints(
    repo_root: Path,
    patch: ProposedBkbPatch,
    rules: list[BeliefRule],
) -> None:
    fingerprints: list[str] = []
    for rule in rules:
        fingerprints.append(
            candidate_fingerprint(
                {
                    "kind": rule.rule_type,
                    "scope": rule.scope,
                    "title": rule.title,
                    "content": rule.rule,
                    "source_evidence": patch.source_evidence,
                }
            )
        )
    try:
        add_accepted_fingerprints(
            repo_root=repo_root,
            workspace_id=patch.workspace_id,
            fingerprints=fingerprints,
        )
    except OSError:
        return
