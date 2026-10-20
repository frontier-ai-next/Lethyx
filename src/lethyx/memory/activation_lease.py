"""Traced, immutable activation of one already-packed advisory context payload."""

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from lethyx.memory.context_payload import PackedEvidencePayload
from lethyx.memory.recall_plan import RecallPlan


@dataclass(frozen=True)
class ActivationItem:
    """A prompt projection that cannot promote or mutate its source record."""

    block_id: str
    text: str
    source: str = ""
    authority: str = "non_directive"
    surface: str = "supplemental_evidence"
    lifecycle: str = "active"
    score: float | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", str(self.block_id or ""))
        object.__setattr__(self, "text", str(self.text or ""))
        object.__setattr__(self, "source", str(self.source or ""))
        object.__setattr__(self, "authority", "non_directive")
        object.__setattr__(self, "surface", "supplemental_evidence")
        object.__setattr__(self, "lifecycle", str(self.lifecycle or "active"))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags if str(tag)))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_context_item(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "authority": self.authority,
            "surface": self.surface,
            "lifecycle": self.lifecycle,
        }


@dataclass(frozen=True)
class ActivationTrace:
    """Out-of-prompt audit trace for an activation decision."""

    mode: str
    backend_id: str
    plan_reason: str
    query_type: str
    requested_tokens: int
    allocated_tokens: int
    estimated_tokens: int
    metadata_tokens: int
    rendered_item_count: int
    truncated_item_count: int
    logical_block_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    visibility: tuple[dict[str, Any], ...]
    exact_literal_capsule: tuple[str, ...]
    exact_literal_losses: tuple[dict[str, Any], ...]
    corrective_recall: dict[str, Any] = field(default_factory=dict)
    reverse_provenance: dict[str, Any] = field(default_factory=dict)
    event_type: str = "recall_activation"
    mask_id: str = ""
    mask_revision: int = 0
    catalog_revision: int = 0
    backend_revision: str = ""
    refresh_reason: str = ""
    excluded_lifecycle_statuses: tuple[str, ...] = ()
    scope_validation_result: str = "matched"
    resolution_status: str = "resolved"
    semantic_recall_performed: bool = False
    out_of_prompt: bool = True

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationLease:
    """One bounded working-set lease over a final packer payload.

    The lease owns no authority state and performs no rendering or truncation.
    ``render`` therefore returns the final packer's bytes unchanged.
    """

    query: str
    workspace_id: str
    run_id: str
    logical_block_ids: tuple[str, ...]
    items: tuple[ActivationItem, ...]
    rendered_context: str
    token_budget: int
    estimated_tokens: int
    metadata_tokens: int
    trace: ActivationTrace
    lease_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", str(self.query or ""))
        object.__setattr__(self, "workspace_id", str(self.workspace_id or ""))
        object.__setattr__(self, "run_id", str(self.run_id or ""))
        object.__setattr__(self, "logical_block_ids", tuple(self.logical_block_ids or ()))
        object.__setattr__(self, "items", tuple(self.items or ()))
        if not self.lease_id:
            object.__setattr__(self, "lease_id", f"lease_{uuid.uuid4().hex[:12]}")

    def render(self) -> str:
        return self.rendered_context

    def to_context_items(self) -> list[dict[str, Any]]:
        return [item.to_context_item() for item in self.items]


def activation_lease_from_payload(
    *,
    plan: RecallPlan,
    payload: PackedEvidencePayload,
    backend_id: str,
    mode: str = "fast_one_hop",
) -> ActivationLease:
    """Project one final payload into a non-directive lease without repacking it."""

    items: list[ActivationItem] = []
    corrective_recall: dict[str, Any] = {}
    reverse_sources: list[dict[str, Any]] = []
    for item in payload.items:
        metadata = dict(item.get("metadata") or {})
        source_authority = str(
            metadata.get("source_authority")
            or metadata.get("authority")
            or item.get("authority")
            or "advisory"
        )
        metadata["source_authority"] = source_authority
        metadata["authority"] = "non_directive"
        if isinstance(metadata.get("corrective_recall"), dict):
            corrective_recall = dict(metadata["corrective_recall"])
        if metadata.get("reverse_provenance_depth") == 1:
            reverse_sources.append(
                {
                    "record_id": str(metadata.get("block_id") or item.get("source") or ""),
                    "derived_id": str(metadata.get("reverse_provenance_derived_id") or ""),
                    "root_id": str(metadata.get("reverse_provenance_root_id") or ""),
                    "reference_only": metadata.get("reverse_provenance_reference_only") is True,
                }
            )
        block_id = str(metadata.get("block_id") or item.get("source") or "")
        items.append(
            ActivationItem(
                block_id=block_id,
                text=str(item.get("text") or ""),
                source=str(item.get("source") or block_id),
                lifecycle=str(item.get("lifecycle") or metadata.get("lifecycle") or "active"),
                score=item.get("score") if isinstance(item.get("score"), (int, float)) else None,
                tags=tuple(str(tag) for tag in (item.get("tags") or ())),
                metadata=metadata,
            )
        )
    trace = ActivationTrace(
        mode=mode,
        backend_id=backend_id,
        plan_reason=plan.reason,
        query_type=str(plan.query_type),
        requested_tokens=plan.max_tokens,
        allocated_tokens=payload.token_budget,
        estimated_tokens=payload.estimated_tokens,
        metadata_tokens=payload.metadata_tokens,
        rendered_item_count=len(payload.items),
        truncated_item_count=payload.truncated_item_count,
        logical_block_ids=plan.logical_block_ids,
        record_ids=plan.record_ids,
        visibility=tuple(asdict(row) for row in payload.visibility),
        exact_literal_capsule=payload.exact_literal_capsule,
        exact_literal_losses=tuple(asdict(loss) for loss in payload.exact_literal_losses),
        corrective_recall=corrective_recall,
        reverse_provenance={"canonical_source_refs": reverse_sources} if reverse_sources else {},
    )
    return ActivationLease(
        query=plan.query,
        workspace_id=plan.workspace_id,
        run_id=plan.run_id,
        logical_block_ids=plan.logical_block_ids,
        items=tuple(items),
        rendered_context=payload.rendered_context,
        token_budget=payload.token_budget,
        estimated_tokens=payload.estimated_tokens,
        metadata_tokens=payload.metadata_tokens,
        trace=trace,
    )
