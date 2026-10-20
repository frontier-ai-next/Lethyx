"""Content-free references to exact, currently authorized memory revisions."""

from dataclasses import dataclass
from hashlib import sha256

from lethyx.contracts import MemoryBlockRecord
from lethyx.contracts.memory import MemoryStatus
from lethyx.memory.selection import MemorySelectionPolicy, SelectionContext


@dataclass(frozen=True, slots=True)
class PinnedRecordRef:
    record_id: str
    revision_id: str
    content_hash: str
    logical_block_id: str
    status: MemoryStatus


def pin_memory_record(
    record: MemoryBlockRecord | None,
    *,
    record_id: str,
    workspace_id: str,
    profile_id: str,
    scope_chain: tuple[str, ...],
    owner_id: str,
    generation_id: str,
) -> PinnedRecordRef:
    candidate_owned = (
        record is not None
        and record.status == "candidate"
        and bool(owner_id.strip() and generation_id.strip())
        and record.metadata.get("working_set_owner")
        == {"owner_id": owner_id, "generation_id": generation_id}
    )
    if (
        record is None
        or record.block_id != record_id
        or record.workspace_id != workspace_id
        or record.profile_id != profile_id
        or record.status not in {"candidate", "accepted"}
        or (record.status == "candidate" and not candidate_owned)
        or not MemorySelectionPolicy().eligible_for_recall(
            record,
            SelectionContext(
                scope_chain=scope_chain,
                run_id=record.run_id,
                include_candidates=candidate_owned,
                source_horizon="durable",
            ),
        )
    ):
        raise ValueError("memory record is unavailable in the current scope and generation")
    return PinnedRecordRef(
        record_id=record.block_id,
        revision_id=record.revision_id,
        content_hash=sha256(record.content.encode("utf-8")).hexdigest(),
        logical_block_id=record.block_membership.primary_block_id
        if record.block_membership
        else "",
        status=record.status,
    )
