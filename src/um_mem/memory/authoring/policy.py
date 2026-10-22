"""Policy helpers for agent-authored memory routing."""

from um_mem.contracts.memory import reserved_memory_metadata_keys
from um_mem.memory.authoring.models import MemoryRouteDecision, MemoryWriteProposal
from um_mem.memory.blocks.models import MemoryBlock


def _decision_for_block(
    *,
    block: MemoryBlock,
    accepted: bool,
    reason: str,
    tags: tuple[str, ...],
    policy_issues: tuple[str, ...] = (),
    suggested_block_id: str = "",
    suggested_action: str = "",
) -> MemoryRouteDecision:
    return MemoryRouteDecision(
        accepted=bool(accepted and not policy_issues),
        reason=reason,
        block_id=block.block_id,
        retention=block.retention,
        delivery_policy=block.delivery_policy,
        temperature=block.temperature,
        authority="advisory",
        scope=block.scope,
        trust_level="agent_reported",
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
        status="candidate",
        tags=tags,
        policy_issues=policy_issues,
        suggested_block_id=suggested_block_id,
        suggested_action=suggested_action,
    )


def _unbound_decision(
    *,
    block_id: str,
    reason: str,
    tags: tuple[str, ...],
    policy_issues: tuple[str, ...],
) -> MemoryRouteDecision:
    return MemoryRouteDecision(
        accepted=False,
        reason=reason,
        block_id=block_id,
        retention="durable",
        delivery_policy="on_demand",
        temperature="warm",
        authority="advisory",
        scope="workspace",
        trust_level="agent_reported",
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
        status="candidate",
        tags=tags,
        policy_issues=policy_issues,
    )


def route_by_taxonomy(write: MemoryWriteProposal, blocks: list[MemoryBlock]) -> MemoryRouteDecision:
    """Validate one explicit structural block choice without semantic rerouting."""

    tags = tuple(dict.fromkeys([*write.tags, write.kind]))
    requested_block_id = str(write.requested_block_id or "").strip()
    if not requested_block_id:
        return _unbound_decision(
            block_id="",
            reason="requested_block_id_required",
            tags=tags,
            policy_issues=("requested_block_id_required",),
        )
    by_id = {block.block_id: block for block in blocks}
    requested = by_id.get(requested_block_id)
    if requested is None:
        return _unbound_decision(
            block_id=requested_block_id,
            reason="unknown_requested_block",
            tags=tags,
            policy_issues=("unknown_logical_block_id",),
        )
    parent_ids = {block.parent_id for block in blocks if block.parent_id}
    writable_leaf_ids = sorted(
        block.block_id
        for block in blocks
        if block.agent_writable
        and not block.index_only
        and block.authority == "advisory"
        and not block.child_ids
        and block.block_id not in parent_ids
    )
    suggested_block_id = (
        "decisions_and_contracts"
        if "decisions_and_contracts" in writable_leaf_ids
        else (writable_leaf_ids[0] if writable_leaf_ids else "")
    )
    suggested_action = "retry_remember" if suggested_block_id else ""

    reserved = reserved_memory_metadata_keys(write.metadata)
    if reserved:
        return _decision_for_block(
            block=requested,
            accepted=False,
            reason="reserved_metadata_governance_field",
            tags=tags,
            policy_issues=tuple(f"reserved_metadata:{key}" for key in reserved),
        )
    if requested.block_id == "verified_lessons":
        reflection = by_id.get("reflection_candidates")
        suggested_reflection = (
            reflection.block_id
            if reflection is not None and reflection.agent_writable and not reflection.index_only
            else ""
        )
        return _decision_for_block(
            block=requested,
            accepted=False,
            reason="verified_lesson_requires_typed_promotion",
            tags=tags,
            policy_issues=("typed_verification_and_tribunal_required",),
            suggested_block_id=suggested_reflection,
            suggested_action="retry_remember" if suggested_reflection else "",
        )
    if requested.index_only:
        return _decision_for_block(
            block=requested,
            accepted=False,
            reason="index_only_block_cannot_store_records",
            tags=tags,
            policy_issues=("index_only_not_record_storage",),
            suggested_block_id=suggested_block_id,
            suggested_action=suggested_action,
        )
    if not requested.agent_writable:
        return _decision_for_block(
            block=requested,
            accepted=False,
            reason="host_managed_block_cannot_accept_agent_writes",
            tags=tags,
            policy_issues=("host_managed_block",),
            suggested_block_id=suggested_block_id,
            suggested_action=suggested_action,
        )
    if requested.child_ids or any(block.parent_id == requested.block_id for block in blocks):
        return _decision_for_block(
            block=requested,
            accepted=False,
            reason="non_leaf_block_cannot_store_records",
            tags=tags,
            policy_issues=("non_leaf_not_record_storage",),
            suggested_block_id=suggested_block_id,
            suggested_action=suggested_action,
        )
    if requested.authority != "advisory":
        return _decision_for_block(
            block=requested,
            accepted=False,
            reason="agent_write_requires_typed_promotion_route",
            tags=tags,
            policy_issues=("typed_promotion_route_required",),
            suggested_block_id=suggested_block_id,
            suggested_action=suggested_action,
        )
    return _decision_for_block(
        block=requested,
        accepted=True,
        reason="requested_block_policy",
        tags=tags,
    )
