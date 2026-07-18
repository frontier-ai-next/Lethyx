"""
Relevance scoring and lesson deduplication.

Finds the most relevant lessons for a given context
and removes duplicate/similar lessons.
"""

import logging
from typing import Any

from lethyx.memory.models import (
    MemoryQuery,
    LessonRecord,
)
from lethyx.memory.store import MemoryStore

log = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================


def query_relevant_lessons(
    store: MemoryStore,
    query: MemoryQuery,
) -> list[LessonRecord]:
    """Query lessons and sort by relevance score.

    Args:
        store: Memory store
        query: Query parameters

    Returns:
        Lessons sorted by relevance
    """
    # Get base results from store
    lessons = store.query_lessons(query)

    # Score and sort by relevance
    scored = [(lesson, score_relevance(lesson, query, store.config)) for lesson in lessons]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [lesson for lesson, _score in scored]


def score_relevance(lesson: LessonRecord, query: MemoryQuery, config: Any) -> float:
    """Calculate relevance score for a lesson.

    Higher score = more relevant.

    Factors:
    - Tag match (weighted)
    - Priority (weighted)
    - Recency (weighted)
    - Decay score (weighted)
    """
    score = 0.0

    # Tag matching
    if query.tags:
        overlap = len(query.tags.intersection(lesson.tags))
        if overlap > 0:
            score += overlap * config.tag_match_weight

    # Priority
    score += lesson.priority * config.priority_weight

    # Recency (newer is slightly better)

    age_days = lesson.age_seconds / 86400
    recency_bonus = max(0, 1 - age_days / 365)  # Decays over a year
    score += recency_bonus * config.recency_weight

    # Decay score (fresher is better)
    score += lesson.decay_score * config.decay_weight

    return score


def deduplicate_lessons(
    lessons: list[LessonRecord],
    similarity_threshold: float = 0.8,
) -> list[LessonRecord]:
    """Remove duplicate or very similar lessons.

    Uses simple text similarity on conclusions and change summaries.

    Args:
        lessons: Lessons to deduplicate
        similarity_threshold: Similarity threshold (0-1)

    Returns:
        Deduplicated lesson list
    """
    if not lessons:
        return []

    # Sort by priority (keep higher priority versions)
    sorted_lessons = sorted(lessons, key=lambda lesson: lesson.priority, reverse=True)

    unique_lessons = []
    seen_signatures = set()

    for lesson in sorted_lessons:
        # Create a simple signature
        signature = _lesson_signature(lesson)

        # Check for similar signatures
        is_duplicate = False
        for seen in seen_signatures:
            if _signature_similarity(signature, seen) >= similarity_threshold:
                is_duplicate = True
                # Still mark the original as accessed (boost its score)
                lesson.touch()
                break

        if not is_duplicate:
            unique_lessons.append(lesson)
            seen_signatures.add(signature)

    return unique_lessons


def cluster_lessons_by_pattern(
    lessons: list[LessonRecord],
) -> dict[str, list[LessonRecord]]:
    """Cluster lessons by repeat/avoid patterns.

    Useful for identifying what works vs what doesn't.

    Returns:
        Dict mapping pattern -> list of lessons with that pattern
    """
    clusters: dict[str, list[LessonRecord]] = {}

    for lesson in lessons:
        for pattern in lesson.repeat_tags + lesson.avoid_tags:
            if pattern not in clusters:
                clusters[pattern] = []
            clusters[pattern].append(lesson)

    return clusters


# =============================================================================
# Internal Helpers
# =============================================================================


def _lesson_signature(lesson: LessonRecord) -> tuple[str, str, str]:
    """Create a simple signature for similarity comparison."""
    # Use key fields for signature
    conclusion_normalized = _normalize_text(lesson.conclusion)
    change_normalized = _normalize_text(lesson.change_summary)
    type_key = lesson.lesson_type

    return (type_key, conclusion_normalized, change_normalized)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re

    # Lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove common stopwords (very basic)
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
    words = text.split()
    words = [w for w in words if w not in stopwords and len(w) > 2]

    return " ".join(words)


def _signature_similarity(sig1: tuple[str, str, str], sig2: tuple[str, str, str]) -> float:
    """Calculate similarity between two lesson signatures.

    Uses Jaccard similarity on word sets.
    """
    # Must be same type
    if sig1[0] != sig2[0]:
        return 0.0

    # Compare conclusions
    words1 = set(sig1[1].split())
    words2 = set(sig2[1].split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    jaccard = intersection / union if union > 0 else 0.0

    # Also check change summary
    change_words1 = set(sig1[2].split())
    change_words2 = set(sig2[2].split())

    if change_words1 and change_words2:
        c_intersection = len(change_words1.intersection(change_words2))
        c_union = len(change_words1.union(change_words2))
        change_jaccard = c_intersection / c_union if c_union > 0 else 0.0
    else:
        change_jaccard = 0.0

    # Weight conclusion more heavily
    return 0.7 * jaccard + 0.3 * change_jaccard
