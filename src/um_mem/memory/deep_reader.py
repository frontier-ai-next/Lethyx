"""Explicit bounded block-local reader escalation over one selected backend."""

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from um_mem.contracts import MemoryBlockRecord
from um_mem.memory.activation_lease import ActivationLease
from um_mem.memory.recall_plan import RecallPlan


@dataclass(frozen=True)
class ReaderGuidanceResolution:
    outcome: str
    card: dict[str, Any] | None
    silent_miss: bool = False


@dataclass(frozen=True)
class DeepReadTrace:
    mode: str
    backend_id: str
    logical_block_id: str
    limit: int
    max_tokens: int
    reconstruction_steps: int
    guidance_outcome: str
    silent_guidance_miss: bool
    host_actions_allowed: bool = False
    reverse_provenance_depth: Literal[1] = 1
    final_output: Literal["evidence"] = "evidence"
    write_permission: Literal[False] = False
    promotion_permission: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            self.host_actions_allowed is not False
            or self.reverse_provenance_depth != 1
            or self.final_output != "evidence"
            or self.write_permission is not False
            or self.promotion_permission is not False
        ):
            raise ValueError("deep reader must remain block-local, read-only evidence")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeepReadResult:
    lease: ActivationLease
    trace: DeepReadTrace

    def to_mapping(self) -> dict[str, Any]:
        return {
            "items": self.lease.to_context_items(),
            "rendered_context": self.lease.rendered_context,
            "activation_trace": self.lease.trace.to_mapping(),
            "deep_read_trace": self.trace.to_mapping(),
        }


@dataclass(frozen=True)
class BlockLocalEvidenceRegion:
    """Read-only evidence already selected from one backend and logical block."""

    backend_id: str
    workspace_id: str
    logical_block_id: str
    primary_record_ids: tuple[str, ...]
    records: tuple[MemoryBlockRecord, ...]

    def __post_init__(self) -> None:
        for name in ("backend_id", "workspace_id", "logical_block_id"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"{name} must be a non-empty identifier")
        if not self.primary_record_ids or any(
            not isinstance(value, str) or not value.strip() for value in self.primary_record_ids
        ):
            raise ValueError("primary_record_ids must contain evidence identifiers")
        if any(not isinstance(record, MemoryBlockRecord) for record in self.records):
            raise TypeError("records must contain MemoryBlockRecord values")
        by_id = {record.block_id: record for record in self.records}
        if len(by_id) != len(self.records):
            raise ValueError("evidence region record IDs must be unique")
        missing = sorted(set(self.primary_record_ids).difference(by_id))
        if missing:
            raise ValueError("evidence region is missing a primary record")
        for record in self.records:
            if record.workspace_id and record.workspace_id != self.workspace_id:
                raise ValueError("evidence region contains a foreign workspace record")
            if record.source_backend and record.source_backend != self.backend_id:
                raise ValueError("evidence region contains a foreign backend record")
        for record_id in self.primary_record_ids:
            membership = by_id[record_id].block_membership
            if membership is None or self.logical_block_id not in {
                membership.primary_block_id,
                *membership.secondary_block_ids,
            }:
                raise ValueError("primary evidence is outside the selected logical block")


@dataclass(frozen=True)
class DeepEvidenceItem:
    record_id: str
    content: str
    evidence_kind: Literal["selected_evidence", "canonical_raw_span"]
    source_record_id: str
    canonical_content_sha256: str
    surface: Literal["supplemental_evidence"] = "supplemental_evidence"
    authority: Literal["advisory"] = "advisory"
    write_permission: Literal[False] = False
    promotion_permission: Literal[False] = False

    def __post_init__(self) -> None:
        if not self.record_id or not self.source_record_id:
            raise ValueError("deep evidence identifiers must be non-empty")
        if self.evidence_kind not in {"selected_evidence", "canonical_raw_span"}:
            raise ValueError("deep evidence kind is invalid")
        expected = "sha256:" + hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.canonical_content_sha256 != expected:
            raise ValueError("deep evidence content hash does not match content")
        if (
            self.surface != "supplemental_evidence"
            or self.authority != "advisory"
            or self.write_permission is not False
            or self.promotion_permission is not False
        ):
            raise ValueError("deep reader output must remain read-only advisory evidence")


@dataclass(frozen=True)
class BlockLocalDeepEvidenceResult:
    backend_id: str
    workspace_id: str
    logical_block_id: str
    items: tuple[DeepEvidenceItem, ...]
    reverse_provenance_depth: Literal[1] = 1
    backend_path: Literal["selected_backend"] = "selected_backend"
    final_output: Literal["evidence"] = "evidence"
    write_permission: Literal[False] = False
    promotion_permission: Literal[False] = False

    def __post_init__(self) -> None:
        if any(not isinstance(item, DeepEvidenceItem) for item in self.items):
            raise TypeError("deep reader result must contain typed evidence")
        if (
            self.reverse_provenance_depth != 1
            or self.backend_path != "selected_backend"
            or self.final_output != "evidence"
            or self.write_permission is not False
            or self.promotion_permission is not False
        ):
            raise ValueError("deep reader cannot widen backend or gain mutation authority")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def read_block_local_evidence_region(
    region: BlockLocalEvidenceRegion,
    *,
    reverse_provenance_sources: Callable[[str], tuple[str, ...]],
    max_primary_records: int = 4,
    max_source_spans: int = 4,
) -> BlockLocalDeepEvidenceResult:
    """Read selected evidence and depth-one canonical spans without a write path."""

    if not isinstance(region, BlockLocalEvidenceRegion):
        raise TypeError("region must be a BlockLocalEvidenceRegion")
    if not callable(reverse_provenance_sources):
        raise TypeError("reverse_provenance_sources must be callable")
    for name, value in (
        ("max_primary_records", max_primary_records),
        ("max_source_spans", max_source_spans),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    records = {record.block_id: record for record in region.records}
    items: list[DeepEvidenceItem] = []
    seen: set[str] = set()
    primary_ids = region.primary_record_ids[:max_primary_records]
    for record_id in primary_ids:
        record = records[record_id]
        items.append(_deep_evidence_item(record, evidence_kind="selected_evidence"))
        seen.add(record_id)
    source_count = 0
    for record_id in primary_ids:
        for source_id in tuple(reverse_provenance_sources(record_id) or ()):
            canonical_id = str(source_id or "").strip()
            if not canonical_id or canonical_id in seen:
                continue
            source = records.get(canonical_id)
            if source is None:
                continue
            items.append(
                _deep_evidence_item(
                    source,
                    evidence_kind="canonical_raw_span",
                    source_record_id=record_id,
                )
            )
            seen.add(canonical_id)
            source_count += 1
            if source_count >= max_source_spans:
                break
        if source_count >= max_source_spans:
            break
    return BlockLocalDeepEvidenceResult(
        backend_id=region.backend_id,
        workspace_id=region.workspace_id,
        logical_block_id=region.logical_block_id,
        items=tuple(items),
    )


def _deep_evidence_item(
    record: MemoryBlockRecord,
    *,
    evidence_kind: Literal["selected_evidence", "canonical_raw_span"],
    source_record_id: str = "",
) -> DeepEvidenceItem:
    return DeepEvidenceItem(
        record_id=record.block_id,
        content=record.content,
        evidence_kind=evidence_kind,
        source_record_id=source_record_id or record.block_id,
        canonical_content_sha256="sha256:"
        + hashlib.sha256(record.content.encode("utf-8")).hexdigest(),
    )


def _valid_block_descriptor(value: Any, *, logical_block_id: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if str(value.get("block_id") or "") != logical_block_id:
        return None
    guidance = value.get("guidance")
    if not isinstance(guidance, dict):
        return None
    if str(guidance.get("block_id") or "") != logical_block_id:
        return None
    synopsis = guidance.get("synopsis")
    use_when = guidance.get("use_when")
    avoid_when = guidance.get("avoid_when", ())
    if not isinstance(synopsis, str) or not synopsis.strip():
        return None
    if not isinstance(use_when, (list, tuple)) or not use_when:
        return None
    if any(not isinstance(item, str) or not item.strip() for item in use_when):
        return None
    if not isinstance(avoid_when, (list, tuple)) or any(
        not isinstance(item, str) for item in avoid_when
    ):
        return None
    return {
        "block_id": logical_block_id,
        "synopsis": synopsis.strip(),
        "use_when": [str(item) for item in use_when],
        "avoid_when": [str(item) for item in avoid_when],
    }


def resolve_reader_guidance(
    plan: RecallPlan,
    *,
    logical_block_id: str,
) -> ReaderGuidanceResolution:
    """Resolve current or last-known-good guidance; fail visibly to block evidence."""

    current = plan.metadata.get("block_descriptors")
    current_cards = list(current) if isinstance(current, (list, tuple)) else []
    matching_current = [
        value
        for value in current_cards
        if isinstance(value, dict) and str(value.get("block_id") or "") == logical_block_id
    ]
    for value in matching_current:
        valid = _valid_block_descriptor(value, logical_block_id=logical_block_id)
        if valid is not None:
            return ReaderGuidanceResolution("current_guidance", valid)

    last_known_good = plan.metadata.get("last_known_good_block_descriptors")
    lkg_cards = list(last_known_good) if isinstance(last_known_good, (list, tuple)) else []
    for value in lkg_cards:
        valid = _valid_block_descriptor(value, logical_block_id=logical_block_id)
        if valid is not None:
            return ReaderGuidanceResolution("last_known_good_guidance", valid)

    if matching_current:
        return ReaderGuidanceResolution("incompatible_widened_to_block_evidence", None)
    return ReaderGuidanceResolution("missing_widened_to_block_evidence", None)


def deep_read_result(
    *,
    lease: ActivationLease,
    plan: RecallPlan,
    backend_id: str,
    logical_block_id: str,
    limit: int,
) -> DeepReadResult:
    guidance = resolve_reader_guidance(plan, logical_block_id=logical_block_id)
    corrective = any(
        isinstance(item.metadata.get("corrective_recall"), dict) for item in lease.items
    )
    provenance = any(item.metadata.get("reverse_provenance_depth") == 1 for item in lease.items)
    return DeepReadResult(
        lease=lease,
        trace=DeepReadTrace(
            mode="deep_block_local",
            backend_id=backend_id,
            logical_block_id=logical_block_id,
            limit=limit,
            max_tokens=lease.token_budget,
            reconstruction_steps=1 + int(corrective or provenance),
            guidance_outcome=guidance.outcome,
            silent_guidance_miss=guidance.silent_miss,
        ),
    )
