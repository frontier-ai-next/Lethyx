"""Typed node/edge models for memory graph and tree."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    block_id: str = ""
    store: str = ""
    tier: str = ""
    scope: str = ""
    surface: str = "supplemental_evidence"
    lifecycle: str = "active"
    trust_level: str = "agent_claim"
    workspace_id: str = ""
    run_id: str = ""
    phase_id: str = ""
    parent_id: str = ""
    path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    src_id: str
    dst_id: str
    edge_type: str
    reason: str = ""
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProofScore:
    candidate_id: str
    score: float
    verified_support_paths: int = 0
    contradiction_paths: int = 0
    hard_contradiction: bool = False
    details: dict[str, Any] = field(default_factory=dict)
