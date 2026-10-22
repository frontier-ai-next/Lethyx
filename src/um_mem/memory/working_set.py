"""Exact, owner-scoped access to unpublished working records."""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from um_mem.contracts import MemoryBlockRecord
from um_mem.contracts.memory import MemoryScope
from um_mem.contracts.models import EvidenceRef
from um_mem.llm_budget import estimate_tokens
from um_mem.memory.context_payload import PackedEvidencePayload, pack_evidence_payload
from um_mem.memory.record_refs import PinnedRecordRef, pin_memory_record
from um_mem.memory.selection import MemorySelectionPolicy, SelectionContext


@dataclass(frozen=True, slots=True)
class WorkingRecordRef:
    record_id: str
    revision_id: str
    content_hash: str
    logical_block_id: str


@dataclass(frozen=True, slots=True)
class WorkingSetRead:
    """The bounded context is the only content-bearing field; refs remain exact."""

    records: tuple[WorkingRecordRef, ...]
    rendered_context: str
    token_budget: int
    estimated_tokens: int
    byte_budget: int
    rendered_bytes: int
    truncated: bool
    omitted_record_ids: tuple[str, ...]
    unavailable_record_ids: tuple[str, ...]
    token_accounting: str = "um_mem_estimate"
    visibility: str = "working_unverified"


@dataclass(frozen=True, slots=True)
class WorkingRecordSnapshot:
    """Complete candidate content at one exact revision, never a recall excerpt."""

    reference: PinnedRecordRef
    title: str
    content: str
    kind: str
    scope: MemoryScope
    evidence_refs: tuple[EvidenceRef, ...]


class WorkingSnapshotBudgetError(ValueError):
    """Complete candidate content cannot fit the caller's explicit snapshot budget."""


def snapshot_working_record(
    record: MemoryBlockRecord | None,
    *,
    record_id: str,
    workspace_id: str,
    profile_id: str,
    scope_chain: tuple[str, ...],
    owner_id: str,
    generation_id: str,
    expected_revision_id: str,
    expected_content_hash: str,
    max_tokens: int,
    max_bytes: int,
) -> WorkingRecordSnapshot:
    """Read a pinned candidate without granting normal recall or exposing metadata."""

    if not owner_id.strip() or not generation_id.strip() or not expected_revision_id.strip():
        raise ValueError("working snapshot requires an exact owner, generation and revision")
    if len(expected_content_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_content_hash
    ):
        raise ValueError("working snapshot requires an exact SHA-256 content hash")
    if type(max_tokens) is not int or type(max_bytes) is not int or min(max_tokens, max_bytes) < 1:
        raise ValueError("working snapshot budgets must be positive integers")
    reference = pin_memory_record(
        record,
        record_id=record_id,
        workspace_id=workspace_id,
        profile_id=profile_id,
        scope_chain=scope_chain,
        owner_id=owner_id,
        generation_id=generation_id,
    )
    if (
        record is None
        or reference.status != "candidate"
        or reference.revision_id != expected_revision_id
        or reference.content_hash != expected_content_hash
        or record.content_mode != "full"
    ):
        raise ValueError("working record snapshot is unavailable at the exact requested revision")
    snapshot = WorkingRecordSnapshot(
        reference=reference,
        title=record.title,
        content=record.content,
        kind=record.kind,
        scope=record.scope,
        evidence_refs=record.evidence_refs,
    )
    if len(record.content) + len(record.title) > max_bytes:
        raise WorkingSnapshotBudgetError(
            "complete working record snapshot exceeds its explicit content budget"
        )
    encoded = json.dumps(asdict(snapshot), ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > max_bytes or estimate_tokens(encoded) > max_tokens:
        raise WorkingSnapshotBudgetError(
            "complete working record snapshot exceeds its explicit content budget"
        )
    return snapshot


def read_working_records(
    get_record: Callable[[str], MemoryBlockRecord | None],
    *,
    workspace_id: str,
    profile_id: str,
    scope_chain: tuple[str, ...],
    owner_id: str,
    generation_id: str,
    record_ids: Sequence[str],
    max_tokens: int,
    max_bytes: int,
) -> WorkingSetRead:
    if not owner_id.strip() or not generation_id.strip():
        raise ValueError("working set requires an exact owner and generation")
    if isinstance(record_ids, str | bytes) or not 1 <= len(record_ids) <= 32:
        raise ValueError("working set requires between 1 and 32 exact record IDs")
    if any(not isinstance(item, str) or not item.strip() for item in record_ids):
        raise ValueError("working set record IDs must be nonempty strings")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("working set record IDs must be unique")
    if type(max_tokens) is not int or type(max_bytes) is not int or min(max_tokens, max_bytes) < 1:
        raise ValueError("working set budgets must be positive integers")
    expected_owner = {"owner_id": owner_id, "generation_id": generation_id}
    policy = MemorySelectionPolicy()
    records: list[MemoryBlockRecord] = []
    unavailable: list[str] = []
    for record_id in record_ids:
        record = get_record(record_id)
        if (
            record is None
            or record.block_id != record_id
            or record.workspace_id != workspace_id
            or record.profile_id != profile_id
            or record.status != "candidate"
            or record.metadata.get("working_set_owner") != expected_owner
            or not policy.eligible_for_recall(
                record,
                SelectionContext(
                    scope_chain=scope_chain,
                    run_id=record.run_id,
                    include_candidates=True,
                    source_horizon="durable",
                ),
            )
        ):
            unavailable.append(record_id)
            continue
        records.append(record)
    refs = tuple(
        WorkingRecordRef(
            record_id=record.block_id,
            revision_id=record.revision_id,
            content_hash=hashlib.sha256(record.content.encode("utf-8")).hexdigest(),
            logical_block_id=record.block_membership.primary_block_id
            if record.block_membership
            else "",
        )
        for record in records
    )
    cards = [
        {
            "text": record.content,
            "source": record.block_id,
            "title": record.title,
            "status": "candidate",
            "authority": "advisory",
            "sensitivity": record.sensitivity,
            "content_mode": record.content_mode,
            "operational_risk": record.operational_risk,
            "metadata": {
                "block_id": record.block_id,
                "revision_id": record.revision_id,
                "status": "candidate",
            },
        }
        for record in records
    ]

    def pack(budget: int) -> PackedEvidencePayload:
        return pack_evidence_payload(cards, max_tokens=budget)

    packed = pack(max_tokens)
    if len(packed.rendered_context.encode("utf-8")) > max_bytes:
        lower, upper = 0, max_tokens
        packed = pack(0)
        while lower <= upper:
            middle = (lower + upper) // 2
            candidate = pack(middle)
            if len(candidate.rendered_context.encode("utf-8")) <= max_bytes:
                packed = candidate
                lower = middle + 1
            else:
                upper = middle - 1
    rendered = packed.rendered_context if packed.items else ""
    visible = {entry.record_id for entry in packed.visibility if entry.state != "not_visible"}
    omitted = tuple(record.block_id for record in records if record.block_id not in visible)
    return WorkingSetRead(
        records=refs,
        rendered_context=rendered,
        token_budget=max_tokens,
        estimated_tokens=packed.estimated_tokens if rendered else 0,
        byte_budget=max_bytes,
        rendered_bytes=len(rendered.encode("utf-8")),
        truncated=bool(omitted or packed.truncated_item_count),
        omitted_record_ids=omitted,
        unavailable_record_ids=tuple(unavailable),
    )
