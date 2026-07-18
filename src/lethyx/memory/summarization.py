"""
Memory summarization - compact representations for context injection.

Converts raw memory into compact bundles suitable for LLM prompts.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from lethyx.memory.models import (
    MemoryQuery,
    MemorySummaryBundle,
    LessonType,
)
from lethyx.memory.store import MemoryStore
from lethyx.memory.relevance import query_relevant_lessons

log = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================


def summarize_workspace_run(
    store: MemoryStore,
    run_path: Path,
    task_id: str,
    workspace_id: str,
) -> dict[str, Any]:
    """Summarize a workspace run into structured evidence.

    Returns a dict with:
    - status: success/failure/partial/unknown
    - evidence_summary: Brief description
    - success_patterns: List of patterns to repeat
    - failure_patterns: List of patterns to avoid
    - metrics: Basic metrics if available
    """
    status = _detect_run_status(run_path)
    success_patterns = _extract_success_patterns(run_path)
    failure_patterns = _extract_failure_patterns(run_path)
    metrics = _extract_run_metrics(run_path)

    return {
        "status": status,
        "evidence_summary": _build_evidence_summary(run_path, status),
        "success_patterns": success_patterns,
        "failure_patterns": failure_patterns,
        "metrics": metrics,
        "task_id": task_id,
        "workspace_id": workspace_id,
    }


def summarize_manager_state(
    store: MemoryStore,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Summarize the current manager state.

    Returns:
    - Total lessons by type
    - Active competency gaps
    - Recent signals
    - Capability areas with concerns
    """
    stats = store.get_stats()
    active_gaps = store.get_active_gaps()

    # Group gaps by capability area
    gaps_by_area: dict[str, list[dict]] = {}
    for gap in active_gaps:
        area = gap.capability_area
        if area not in gaps_by_area:
            gaps_by_area[area] = []
        gaps_by_area[area].append(
            {
                "id": gap.id,
                "severity": gap.severity,
                "description": gap.description[:100],
                "occurrences": gap.occurrences,
            }
        )

    # Recent concerns
    recent_signals = store.get_recent_negative_signals(seconds=3600, limit=20)
    concern_areas = list({s.capability_area for s in recent_signals if s.is_negative})

    return {
        "stats": {
            "total_lessons": stats.total_lessons,
            "workspace_lessons": stats.workspace_lessons,
            "manager_lessons": stats.manager_lessons,
            "active_gaps": stats.active_gaps,
        },
        "gaps_by_area": gaps_by_area,
        "concern_areas": concern_areas,
        "recent_negative_signals": len(recent_signals),
    }


def build_memory_summary_bundle(
    store: MemoryStore,
    task_id: str,
    task_class: str | None = None,
    workspace_id: str | None = None,
    max_lessons: int = 10,
    max_gaps: int = 5,
) -> MemorySummaryBundle:
    """Build a compact memory summary for prompt injection.

    This is the main entry point for getting manager memory into context.

    Args:
        store: Memory store
        task_id: Current task
        task_class: Optional task class for broader matching
        workspace_id: Optional workspace for workspace-specific lessons
        max_lessons: Maximum lessons to include per type
        max_gaps: Maximum gaps to include

    Returns:
        Compact bundle ready for prompt injection
    """
    # Query relevant lessons
    query = MemoryQuery(
        task_id=task_id,
        task_class=task_class,
        workspace_id=workspace_id,
        min_decay_score=0.3,
        limit=max_lessons * 2,  # Get more, will filter by relevance
    )

    all_relevant = query_relevant_lessons(store, query)

    # Separate by type
    workspace_lessons = [
        lesson for lesson in all_relevant if lesson.lesson_type == LessonType.WORKSPACE
    ][:max_lessons]
    manager_lessons = [
        lesson for lesson in all_relevant if lesson.lesson_type == LessonType.MANAGER
    ][:max_lessons]

    # Get active gaps
    active_gaps = store.get_active_gaps()[:max_gaps]

    # Look for repeated patterns
    repeated_failures = _find_repeated_patterns(store, "failure")
    repeated_successes = _find_repeated_patterns(store, "success")

    # Build capability warnings from gaps
    capability_warnings = []
    for gap in active_gaps:
        if gap.severity in ("high", "critical") and not gap.is_workspace_level:
            capability_warnings.append(f"{gap.capability_area}: {gap.description[:80]}")

    # Compact lesson representations
    compact_workspace = [_compact_lesson(lesson) for lesson in workspace_lessons]
    compact_manager = [_compact_lesson(lesson) for lesson in manager_lessons]
    compact_gaps = [_compact_gap(g) for g in active_gaps]

    return MemorySummaryBundle(
        task_id=task_id,
        relevant_workspace_lessons=compact_workspace,
        relevant_manager_lessons=compact_manager,
        active_gaps=compact_gaps,
        capability_warnings=capability_warnings,
        repeated_failures=[
            {"pattern_description": p[0], "occurrence_count": p[1]} for p in repeated_failures
        ],
        repeated_successes=[
            {"pattern_description": p[0], "occurrence_count": p[1]} for p in repeated_successes
        ],
        stats=store.get_stats(),
    )


# =============================================================================
# Internal Helpers
# =============================================================================


def _detect_run_status(
    run_path: Path,
) -> Literal["success", "failure", "partial", "unknown"]:
    """Detect overall run status."""
    # Check reports for explicit status
    reports_path = run_path / "reports"
    if reports_path.exists():
        for report_file in reports_path.glob("*.md"):
            content = report_file.read_text(encoding="utf-8").lower()
            if "success" in content or "complete" in content or "delivered" in content:
                return "success"
            if "fail" in content or "error" in content:
                return "failure"
            if "partial" in content:
                return "partial"

    # Check memory for error signals
    memory_path = run_path / "memory"
    if memory_path.exists():
        for agent_dir in memory_path.iterdir():
            if not agent_dir.is_dir():
                continue
            for md_file in agent_dir.glob("*.md"):
                content = md_file.read_text(encoding="utf-8").lower()
                if "fatal" in content or "critical error" in content:
                    return "failure"
                if "error" in content or "exception" in content:
                    return "partial"

    return "unknown"


def _extract_success_patterns(run_path: Path) -> list[str]:
    """Extract patterns that contributed to success."""
    patterns = []

    reports_path = run_path / "reports"
    if reports_path.exists():
        for report_file in reports_path.glob("*.md"):
            content = report_file.read_text(encoding="utf-8")
            for line in content.split("\n"):
                line_lower = line.lower()
                if any(w in line_lower for w in ["successful", "worked", "effective"]):
                    stripped = line.strip()
                    if len(stripped) < 150:
                        patterns.append(stripped)

    return list(set(patterns))[:5]


def _extract_failure_patterns(run_path: Path) -> list[str]:
    """Extract patterns that caused failure."""
    patterns = []

    memory_path = run_path / "memory"
    if memory_path.exists():
        for agent_dir in memory_path.iterdir():
            if not agent_dir.is_dir():
                continue
            for md_file in agent_dir.glob("*.md"):
                content = md_file.read_text(encoding="utf-8")
                for line in content.split("\n"):
                    line_lower = line.lower()
                    if any(w in line_lower for w in ["error:", "failed to", "unable"]):
                        stripped = line.strip()
                        if len(stripped) < 150:
                            patterns.append(stripped)

    return list(set(patterns))[:5]


def _extract_run_metrics(run_path: Path) -> dict[str, Any]:
    """Extract basic metrics from run."""
    metrics = {}

    # Count outputs
    memory_path = run_path / "memory"
    if memory_path.exists():
        agent_count = len([d for d in memory_path.iterdir() if d.is_dir()])
        metrics["agents_active"] = agent_count

    reports_path = run_path / "reports"
    if reports_path.exists():
        metrics["reports_generated"] = len(list(reports_path.glob("*.md")))

    artifacts_path = run_path / "artifacts"
    if artifacts_path.exists():
        metrics["artifacts"] = len(list(artifacts_path.iterdir()))

    return metrics


def _build_evidence_summary(run_path: Path, status: str) -> str:
    """Build a brief evidence summary."""
    parts = [f"Status: {status}"]

    metrics = _extract_run_metrics(run_path)
    if metrics:
        parts.append(f"Agents: {metrics.get('agents_active', 0)}")
        parts.append(f"Reports: {metrics.get('reports_generated', 0)}")

    return "; ".join(parts)


def _find_repeated_patterns(
    store: MemoryStore, pattern_type: Literal["failure", "success"]
) -> list[tuple[str, int]]:
    """Find repeated patterns across lessons.

    Returns:
        List of (pattern_description, occurrence_count) tuples
    """
    query = MemoryQuery(limit=200)
    all_lessons = store.query_lessons(query)

    pattern_counts: Counter = Counter()

    for lesson in all_lessons:
        if pattern_type == "success":
            patterns = lesson.repeat_tags
        else:
            patterns = lesson.avoid_tags

        for pattern in patterns:
            pattern_counts[pattern] += 1

    # Get patterns that appear multiple times
    repeated = [(p, c) for p, c in pattern_counts.items() if c >= 2]
    repeated.sort(key=lambda x: x[1], reverse=True)

    return repeated[:10]


def _compact_lesson(lesson: Any) -> dict[str, Any]:
    """Create a compact representation of a lesson."""
    return {
        "id": lesson.id,
        "workspace_id": lesson.workspace_id,
        "conclusion": lesson.conclusion[:200],
        "change_summary": lesson.change_summary[:150],
        "priority": lesson.priority,
        "repeat_tags": lesson.repeat_tags[:3],
        "avoid_tags": lesson.avoid_tags[:3],
        "age_days": lesson.age_seconds / 86400,
    }


def _compact_gap(gap: Any) -> dict[str, Any]:
    """Create a compact representation of a gap."""
    return {
        "id": gap.id,
        "capability_area": gap.capability_area,
        "severity": gap.severity,
        "description": gap.description[:150],
        "occurrences": gap.occurrences,
        "is_workspace_level": gap.is_workspace_level,
    }
