"""Typed contract for agent-authored memory writes."""

from dataclasses import dataclass, field
from typing import Any

from um_mem.contracts import TurnChildClaim
from um_mem.contracts.memory import (
    DeliveryPolicy,
    MemoryAuthority,
    MemoryRetention,
    MemoryScope,
    MemoryStatus,
    MemoryTemperature,
    PromotionDecision,
    VerificationMethod,
    VerificationStatus,
)
from um_mem.contracts.models import TrustLevel


@dataclass(frozen=True)
class MemoryWriteProposal:
    title: str
    content: str
    kind: str
    tags: tuple[str, ...]
    workspace_id: str = ""
    run_id: str = ""
    stage_id: str = ""
    subtask_id: str = ""
    source_id: str = ""
    evidence_kind: str = ""
    evidence_refs: tuple[dict[str, Any], ...] = ()
    child_claims: tuple[TurnChildClaim, ...] = ()
    write_mode: str = "standard"
    requested_block_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryRouteDecision:
    accepted: bool
    reason: str
    block_id: str
    retention: MemoryRetention
    delivery_policy: DeliveryPolicy
    temperature: MemoryTemperature
    authority: MemoryAuthority
    scope: MemoryScope
    trust_level: TrustLevel
    verification_status: VerificationStatus
    verification_methods: tuple[VerificationMethod, ...]
    promotion_decision: PromotionDecision
    status: MemoryStatus
    tags: tuple[str, ...]
    policy_issues: tuple[str, ...] = ()
    catalog_revision_id: str = ""
    suggested_block_id: str = ""
    suggested_action: str = ""
