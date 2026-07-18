"""Policy helpers for agent-authored memory routing."""

from lethyx.memory.authoring.models import MemoryRouteDecision, MemoryWriteProposal
from lethyx.memory.blocks.models import MemoryBlock

_DIRECTIVE_KINDS = {"bkb_rule", "behavior", "invariant", "prospective_trigger", "core_lesson"}
_VERIFIED_EVIDENCE_KINDS = {
    "verified_outcome",
    "test_run",
    "verification_report",
    "mutation_report",
}


def _has_typed_evidence(write: MemoryWriteProposal) -> bool:
    return any(
        isinstance(ref, dict)
        and ref.get("ref_type")
        and ref.get("ref_id")
        and ref.get("produced_by")
        for ref in write.evidence_refs
    )


def route_by_taxonomy(write: MemoryWriteProposal, blocks: list[MemoryBlock]) -> MemoryRouteDecision:
    tags = tuple(dict.fromkeys([*write.tags, write.kind]))
    by_id = {block.block_id: block for block in blocks}
    block = by_id.get(write.requested_block_id) if write.requested_block_id else None
    issues: list[str] = []
    verified = write.evidence_kind in _VERIFIED_EVIDENCE_KINDS and _has_typed_evidence(write)
    if write.kind in _DIRECTIVE_KINDS or write.metadata.get("surface") == "directive":
        return MemoryRouteDecision(
            False,
            "direct_directive_write_requires_promotion_gate",
            "active_bkb",
            "palace.lesson",
            "warm",
            "cross_run_durable",
            "directive",
            "candidate",
            "agent_claim",
            False,
            tags,
            ("promotion_gate_required",),
        )
    if block is not None:
        return MemoryRouteDecision(
            True,
            "requested_block_policy",
            block.block_id,
            write.requested_store or block.store,
            block.tier,
            block.scope,
            block.default_surface,
            "active" if verified else "candidate",
            "mutation_verified" if verified else "agent_claim",
            verified,
            tags,
        )
    if write.kind in {"hypothesis", "idea", "research_lead"}:
        return MemoryRouteDecision(
            True,
            "idea_jsonl_only_unverified",
            "project_ideas",
            "ideas.jsonl",
            "cold",
            "workspace",
            "reflection_candidate",
            "candidate",
            "agent_claim",
            False,
            tags,
        )
    if write.kind in {"github_project", "repo_capsule", "code_chunk"}:
        return MemoryRouteDecision(
            True,
            "external_project_passive_codeptr",
            "github_repo_capsules",
            "palace.codeptr",
            "warm",
            "cross_run_durable",
            "supplemental_evidence",
            "active",
            "public_verified" if verified else "agent_claim",
            verified,
            tags,
        )
    if write.kind == "research_finding":
        if not _has_typed_evidence(write):
            issues.append("research_finding_requires_typed_evidence")
        return MemoryRouteDecision(
            not issues,
            "research_finding_hot_with_evidence" if not issues else "blocked_missing_evidence",
            "current_run_state",
            "palace.run",
            "hot",
            "run_scoped",
            "current_run_state",
            "active" if not issues else "candidate",
            "mutation_verified" if not issues else "agent_claim",
            not issues,
            tags,
            tuple(issues),
        )
    if write.kind in {"lesson", "verified_finding"} and verified:
        return MemoryRouteDecision(
            True,
            "verified_lesson_passive",
            "verified_lessons",
            "palace.lesson",
            "warm",
            "cross_run_durable",
            "supplemental_evidence",
            "active",
            "mutation_verified",
            True,
            tags,
        )
    if write.kind in {"tool_log", "terminal_scrollback", "watcher_incident"}:
        return MemoryRouteDecision(
            True,
            "log_archive_hint",
            "transient_logs",
            "palace.transient",
            "transient",
            "transient",
            "archive_hint",
            "active",
            "agent_claim",
            False,
            tags,
        )
    return MemoryRouteDecision(
        True,
        "default_passive_candidate",
        "project_ideas",
        "ideas.jsonl",
        "cold",
        "workspace",
        "reflection_candidate",
        "candidate",
        "agent_claim",
        False,
        tags,
    )
