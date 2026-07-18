"""Evaluation metrics for the memory_bench framework.

All functions operate on pre-normalised strings (call text.normalize() first)
or accept raw strings and normalise internally.  This module has no imports
from lethyx or any memory backend.
"""

import math
import random
from collections import Counter
from typing import Sequence

from memory_bench.core.text import normalize, tokenize

# ---------------------------------------------------------------------------
# Token-level F1 (LoCoMo / SQuAD style)
# ---------------------------------------------------------------------------


def _token_overlap(pred: str, gold: str) -> tuple[float, float, float]:
    """Return (precision, recall, f1) over token bags."""
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        exact = normalize(pred) == normalize(gold)
        score = 1.0 if exact else 0.0
        return score, score, score
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0, 0.0, 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and one gold answer."""
    _, _, score = _token_overlap(prediction, gold)
    return score


def f1_max(prediction: str, golds: Sequence[str]) -> float:
    """Max token-level F1 against a set of acceptable gold answers."""
    if not golds:
        return 0.0
    return max(f1(prediction, g) for g in golds)


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalised prediction == normalised gold, else 0.0."""
    return 1.0 if normalize(prediction) == normalize(gold) else 0.0


def exact_match_max(prediction: str, golds: Sequence[str]) -> float:
    if not golds:
        return 0.0
    return max(exact_match(prediction, g) for g in golds)


# ---------------------------------------------------------------------------
# Retrieval metrics (recall@k, MRR, nDCG)
# ---------------------------------------------------------------------------


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of relevant items found in the top-k retrieved items.

    ``retrieved_ids`` are the MemoryRecord.evidence_id values returned by
    MemoryStore.retrieve() in ranked order.
    """
    if not relevant_ids:
        return 1.0  # nothing to retrieve → perfect by convention
    top_k = retrieved_ids[:k]
    hits = len({rid for rid in top_k if rid in relevant_ids})
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1/(rank of first hit), or 0 if no hit."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at k (binary relevance)."""
    top_k = retrieved_ids[:k]
    seen: set[str] = set()
    dcg = 0.0
    for i, rid in enumerate(top_k):
        if rid in relevant_ids and rid not in seen:
            dcg += 1.0 / math.log2(i + 2)
            seen.add(rid)
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> dict[str, float]:
    """Bundle of recall@k, mrr, ndcg@k for a single probe."""
    return {
        f"recall_at_{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr": mrr(retrieved_ids, relevant_ids),
        f"ndcg_at_{k}": ndcg_at_k(retrieved_ids, relevant_ids, k),
        "n_retrieved": len(retrieved_ids),
        "n_relevant": len(relevant_ids),
    }


# ---------------------------------------------------------------------------
# Abstention accuracy
# ---------------------------------------------------------------------------


def abstention_accuracy(abstained: bool, probe_expects_abstain: bool) -> float:
    """1.0 when the model's abstention matches what the probe expects."""
    return 1.0 if abstained == probe_expects_abstain else 0.0


# ---------------------------------------------------------------------------
# Nugget coverage (BEAM style)
# ---------------------------------------------------------------------------


def nugget_coverage(prediction: str, nuggets: Sequence[str]) -> float:
    """Fraction of atomic nuggets covered by the prediction.

    A nugget is considered covered if:
    - token-level F1 between prediction and nugget >= 0.5, OR
    - all tokens in the nugget are present in the prediction tokens (handles
      short / single-word nuggets where F1 is diluted by prediction length).

    This mirrors the BEAM paper's nugget scoring approach.
    """
    if not nuggets:
        return 1.0
    pred_norm = normalize(prediction)
    pred_token_set = set(tokenize(pred_norm))

    def _covered(n: str) -> bool:
        n_norm = normalize(n)
        n_tokens = set(tokenize(n_norm))
        if not n_tokens:
            return True
        # Token containment: all nugget tokens present in prediction
        if n_tokens.issubset(pred_token_set):
            return True
        # F1 threshold for multi-token nuggets
        return f1(pred_norm, n_norm) >= 0.5

    hits = sum(1 for n in nuggets if _covered(n))
    return hits / len(nuggets)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(num: int, denom: int, *, ndigits: int = 2) -> float:
    return round(100.0 * num / denom, ndigits) if denom else 0.0


def bootstrap_ci(
    values: list[float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval (lower, upper) for the mean.

    Returns (mean - half_width, mean + half_width) at ``confidence`` level.
    """
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    boot_means = sorted(mean(rng.choices(values, k=n)) for _ in range(n_bootstrap))
    lo_idx = int(n_bootstrap * (1 - confidence) / 2)
    hi_idx = n_bootstrap - lo_idx - 1
    return boot_means[lo_idx], boot_means[hi_idx]


def aggregate_scores(
    scores: list[float],
    *,
    category: str = "",
    ci: bool = False,
    seed: int = 0,
) -> dict[str, float]:
    """Aggregate per-probe scores into summary statistics."""
    result: dict[str, float] = {
        "n": len(scores),
        "mean": mean(scores),
        "sum": sum(scores),
    }
    if ci and len(scores) >= 10:
        lo, hi = bootstrap_ci(scores, seed=seed)
        result["ci_lower"] = lo
        result["ci_upper"] = hi
    return result
