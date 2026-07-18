"""Proof and contradiction scoring."""

import math
from pathlib import Path

from lethyx.memory.graph.models import ProofScore
from lethyx.memory.graph.query import find_contradiction_paths, find_supporting_paths


def proof_score(
    candidate_id: str,
    drive_root: Path | None,
    trust_weight: float = 0.5,
    recency_weight: float = 0.0,
    outcome_correlation: float = 0.0,
) -> ProofScore:
    support = find_supporting_paths(candidate_id, drive_root)
    contradictions = find_contradiction_paths(candidate_id, drive_root)
    score = (
        1.0 * math.log(1 + len(support))
        - 1.5 * math.log(1 + len(contradictions))
        + 0.3 * recency_weight
        + 0.7 * outcome_correlation
        + 0.8 * trust_weight
    )
    return ProofScore(
        candidate_id,
        score,
        len(support),
        len(contradictions),
        bool(contradictions),
        {"support_paths": support, "contradiction_paths": contradictions},
    )
