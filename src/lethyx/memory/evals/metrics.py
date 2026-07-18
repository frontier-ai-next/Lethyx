"""Deterministic static memory metrics."""

from typing import Any


def first_action_alignment_at_1(expected: list[str], observed: list[str]) -> float:
    return 1.0 if expected and observed and observed[0] in expected else 0.0


def directive_precision(items: list[dict[str, Any]]) -> float:
    directives = [i for i in items if i.get("directive")]
    if not directives:
        return 1.0
    good = [
        i
        for i in directives
        if i.get("surface") == "directive"
        and i.get("trust") in {"verified", "curated", "mutation_verified", "public_verified"}
    ]
    return len(good) / len(directives)


def directive_recall(expected_ids: list[str], items: list[dict[str, Any]]) -> float:
    if not expected_ids:
        return 1.0
    ids = {str(i.get("id")) for i in items if i.get("directive")}
    return len([i for i in expected_ids if i in ids]) / len(expected_ids)


def contamination_rate(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    bad = [i for i in items if i.get("directive") and i.get("surface") != "directive"]
    return len(bad) / len(items)


def retrieval_leakage_rate(items: list[dict[str, Any]]) -> float:
    retrieval = {
        "hindsight",
        "mempalace_search",
        "graph",
        "semantic_search",
        "warm_recall",
        "palace.recall",
        "mnemosyne",
    }
    if not items:
        return 0.0
    bad = [i for i in items if i.get("directive") and i.get("source_backend") in retrieval]
    return len(bad) / len(items)


def evidence_coverage(items: list[dict[str, Any]]) -> float:
    if not items:
        return 1.0
    covered = [
        i for i in items if i.get("evidence_refs") or i.get("trust") in {"curated", "verified"}
    ]
    return len(covered) / len(items)


def token_roi(success_score: float, token_cost: int) -> float:
    return success_score / max(1, token_cost)


def false_promotion_rate(candidates: list[dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    false = [c for c in candidates if c.get("accepted") and not c.get("typed_evidence")]
    return len(false) / len(candidates)


def hindsight_candidate_yield(candidates: list[dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    useful = [c for c in candidates if c.get("typed_evidence")]
    return len(useful) / len(candidates)
