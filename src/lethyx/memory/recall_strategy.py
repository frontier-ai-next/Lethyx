"""Bounded context allocation and explicit deep-read helpers."""

from dataclasses import dataclass
from typing import Any, cast

from lethyx.contracts.memory import (
    _closed_value,
    MemoryStatus,
    MemoryRetention,
    DeliveryPolicy,
    MemoryTemperature,
    MemoryScope,
    VerificationStatus,
    PromotionDecision,
)
from lethyx.contracts.models import TrustLevel

from lethyx.contracts import MemoryBlockRecord
from lethyx.memory.backends.base import MemoryHit
from lethyx.memory.backends.contracts import MemoryEdgeRecord, MemoryIndexNode
from lethyx.memory.recall_plan import RecallPlan
from lethyx.memory.reconstructive_recall import (
    PinnedRecallFrame,
    ReconstructiveLitePlan,
    compile_reconstructive_lite_plan,
)


@dataclass(frozen=True)
class RecallBudgetAllocation:
    """The caller-owned hard context budget, without semantic down-allocation."""

    requested_tokens: int
    allocated_tokens: int
    query_type: str
    query_term_count: int
    evidence_role_count: int
    high_risk: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "requested_tokens": self.requested_tokens,
            "allocated_tokens": self.allocated_tokens,
            "query_type": self.query_type,
            "query_term_count": self.query_term_count,
            "evidence_role_count": self.evidence_role_count,
            "high_risk": self.high_risk,
        }


def allocate_recall_budget(plan: RecallPlan) -> RecallBudgetAllocation:
    """Preserve the exact caller budget; the backend owns candidate ranking."""

    requested = max(0, int(plan.max_tokens))
    return RecallBudgetAllocation(
        requested_tokens=requested,
        allocated_tokens=requested,
        query_type=str(plan.query_type),
        query_term_count=0,
        evidence_role_count=len(plan.evidence_roles),
        high_risk=bool(
            plan.metadata.get("high_risk") is True
            or plan.metadata.get("transition_reason") == "high_risk_boundary"
            or plan.metadata.get("operational_risk") in {"high", "critical"}
        ),
    )


def provenance_reference_record(hit: MemoryHit) -> MemoryBlockRecord | None:
    """Project a non-persisted canonical source pointer onto MCP's record transport."""

    metadata = dict(hit.metadata or {})
    if metadata.get("reverse_provenance_reference_only") is not True:
        return None
    return MemoryBlockRecord(
        block_id=str(metadata.get("block_id") or ""),
        content=hit.text,
        title="Canonical source reference",
        kind="observation",
        status=cast(
            MemoryStatus,
            _closed_value(
                str(metadata.get("status") or "candidate"), field_name="status", shape=MemoryStatus
            ),
        ),
        retention=cast(
            MemoryRetention,
            _closed_value(
                str(metadata.get("retention") or "durable"),
                field_name="retention",
                shape=MemoryRetention,
            ),
        ),
        delivery_policy=cast(
            DeliveryPolicy,
            _closed_value(
                str(metadata.get("delivery_policy") or "on_demand"),
                field_name="delivery_policy",
                shape=DeliveryPolicy,
            ),
        ),
        temperature=cast(
            MemoryTemperature,
            _closed_value(
                str(metadata.get("temperature") or "warm"),
                field_name="temperature",
                shape=MemoryTemperature,
            ),
        ),
        authority="advisory",
        trust_level=cast(
            TrustLevel,
            _closed_value(
                str(metadata.get("trust_level") or "agent_reported"),
                field_name="trust_level",
                shape=TrustLevel,
            ),
        ),
        scope=cast(
            MemoryScope,
            _closed_value(
                str(metadata.get("scope") or "workspace"), field_name="scope", shape=MemoryScope
            ),
        ),
        verification_status=cast(
            VerificationStatus,
            _closed_value(
                str(metadata.get("verification_status") or "unverified"),
                field_name="verification_status",
                shape=VerificationStatus,
            ),
        ),
        verification_methods=tuple(metadata.get("verification_methods") or ()),
        promotion_decision=cast(
            PromotionDecision,
            _closed_value(
                str(metadata.get("promotion_decision") or "not_requested"),
                field_name="promotion_decision",
                shape=PromotionDecision,
            ),
        ),
        workspace_id=str(metadata.get("workspace_id") or ""),
        run_id=str(metadata.get("run_id") or ""),
        source_backend=str(metadata.get("source_backend") or ""),
        tags=tuple(hit.tags),
        sensitivity=metadata.get("sensitivity") or "internal",
        content_mode=metadata.get("content_mode") or "full",
        operational_risk=metadata.get("operational_risk") or "low",
        metadata={
            key: value
            for key, value in metadata.items()
            if key
            in {
                "canonical_source_ref",
                "evidence_roles",
                "reverse_provenance_depth",
                "reverse_provenance_derived_id",
                "reverse_provenance_reference_only",
                "reverse_provenance_root_id",
                "recall_budget",
                "corrective_recall",
            }
        },
    )


def compile_reconstructive_strategy(
    plan: RecallPlan,
    *,
    frame: PinnedRecallFrame,
    index_nodes: tuple[MemoryIndexNode, ...] = (),
    edges: tuple[MemoryEdgeRecord, ...] = (),
) -> ReconstructiveLitePlan:
    """Expose explicit reconstructive deep-read planning at the shared boundary."""

    return compile_reconstructive_lite_plan(
        plan,
        frame=frame,
        index_nodes=index_nodes,
        edges=edges,
    )
