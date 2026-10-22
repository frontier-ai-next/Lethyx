"""Typed graph evaluation over the selected backend's edge contract."""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from um_mem.memory.backends.contracts import EdgeType, MemoryEdgeRecord


@dataclass(frozen=True)
class MemoryProofScore:
    candidate_id: str
    score: float
    verified_support_paths: int = 0
    contradiction_paths: int = 0
    hard_contradiction: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def score_memory_edges(
    candidate_id: str,
    edges: Iterable[MemoryEdgeRecord],
    *,
    trust_weight: float = 0.5,
    recency_weight: float = 0.0,
    outcome_correlation: float = 0.0,
) -> MemoryProofScore:
    """Score direct proof/contradiction paths without opening another store."""

    target = str(candidate_id or "").strip()
    if not target:
        raise ValueError("candidate_id is required")
    typed_edges = tuple(edges)
    if any(not isinstance(edge, MemoryEdgeRecord) for edge in typed_edges):
        raise TypeError("edge graph requires MemoryEdgeRecord values")
    support_paths = [
        [edge.src_id, edge.dst_id]
        for edge in typed_edges
        if edge.dst_id == target and edge.edge_type in {EdgeType.SUPPORTS, EdgeType.VERIFIED_BY}
    ]
    contradiction_paths = [
        [edge.src_id, edge.dst_id]
        for edge in typed_edges
        if target in {edge.src_id, edge.dst_id} and edge.edge_type == EdgeType.CONTRADICTS
    ]
    score = (
        math.log1p(len(support_paths))
        - 1.5 * math.log1p(len(contradiction_paths))
        + 0.3 * recency_weight
        + 0.7 * outcome_correlation
        + 0.8 * trust_weight
    )
    return MemoryProofScore(
        candidate_id=target,
        score=score,
        verified_support_paths=len(support_paths),
        contradiction_paths=len(contradiction_paths),
        hard_contradiction=bool(contradiction_paths),
        details={
            "support_paths": support_paths,
            "contradiction_paths": contradiction_paths,
        },
    )
