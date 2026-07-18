"""Typed contract for agent-authored memory writes."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryWriteProposal:
    title: str
    content: str
    kind: str
    tags: tuple[str, ...]
    workspace_id: str = ""
    run_id: str = ""
    phase_id: str = ""
    subtask_id: str = ""
    source_id: str = ""
    evidence_kind: str = ""
    evidence_refs: tuple[dict[str, Any], ...] = ()
    requested_block_id: str = ""
    requested_store: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryRouteDecision:
    accepted: bool
    reason: str
    block_id: str
    store: str
    tier: str
    scope: str
    surface: str
    lifecycle: str
    trust_level: str
    verified: bool
    tags: tuple[str, ...]
    policy_issues: tuple[str, ...] = ()
