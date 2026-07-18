"""
Context builder - assembles memory into prompt-ready bundles.

This is the main interface for other Lethyx modules to get memory
into LLM context efficiently.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from lethyx.memory.models import (
    MemoryQuery,
    MemorySummaryBundle,
    WorkingMemoryRecord,
)
from lethyx.memory.store import MemoryStore
from lethyx.memory.summarization import (
    build_memory_summary_bundle,
    summarize_workspace_run,
)

log = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================


def build_manager_context_bundle(
    store: MemoryStore,
    task_id: str,
    task_class: str | None = None,
    workspace_id: str | None = None,
    max_lessons: int = 10,
    max_gaps: int = 5,
    include_stats: bool = True,
) -> MemorySummaryBundle:
    """Build a context bundle for manager decision-making.

    This is the primary interface for getting manager memory into prompts.

    Args:
        store: Memory store
        task_id: Current task ID
        task_class: Optional task class for broader matching
        workspace_id: Optional workspace ID for workspace-specific context
        max_lessons: Maximum lessons per type
        max_gaps: Maximum competency gaps to include
        include_stats: Whether to include memory statistics

    Returns:
        Compact bundle ready for prompt injection
    """
    bundle = build_memory_summary_bundle(
        store=store,
        task_id=task_id,
        task_class=task_class,
        workspace_id=workspace_id,
        max_lessons=max_lessons,
        max_gaps=max_gaps,
    )

    log.info(
        f"Built manager context: {len(bundle.relevant_workspace_lessons)} workspace lessons, "
        f"{len(bundle.relevant_manager_lessons)} manager lessons, "
        f"{len(bundle.active_gaps)} active gaps"
    )

    return bundle


def build_workspace_context_bundle(
    store: MemoryStore,
    task_id: str,
    workspace_id: str,
    include_workspace_memory: bool = True,
    include_manager_lessons: bool = True,
    include_contrastive: bool = True,
    max_lessons: int = 5,
) -> dict[str, Any]:
    """Build context bundle for workspace-level operations.

    Includes:
    - Workspace-specific memory if available
    - Relevant manager lessons that apply to this workspace type
    - Competency warnings relevant to this workspace

    Args:
        store: Memory store
        task_id: Current task ID
        workspace_id: Workspace to get context for
        include_workspace_memory: Include workspace memory record
        include_manager_lessons: Include applicable manager lessons
        max_lessons: Maximum lessons to include

    Returns:
        Context dict with structured memory info
    """
    context: dict[str, Any] = {
        "task_id": task_id,
        "workspace_id": workspace_id,
    }

    # Get workspace-specific lessons
    ws_query = MemoryQuery(
        task_id=task_id,
        workspace_id=workspace_id,
        limit=max_lessons,
    )
    workspace_lessons = store.query_lessons(ws_query)

    context["workspace_lessons"] = [
        _compact_lesson_for_context(lesson) for lesson in workspace_lessons
    ]

    # Get relevant manager lessons
    if include_manager_lessons:
        # Query for lessons that might apply to this workspace type
        mgr_query = MemoryQuery(
            lesson_type="manager",
            limit=max_lessons,
        )
        manager_lessons = store.query_lessons(mgr_query)
        context["manager_lessons"] = [
            _compact_lesson_for_context(lesson) for lesson in manager_lessons
        ]

    # Check for relevant capability gaps
    active_gaps = store.get_active_gaps()
    relevant_gaps = [
        g
        for g in active_gaps
        if not g.is_workspace_level or g.metadata.get("workspace_id") == workspace_id
    ]
    context["capability_gaps"] = [
        {
            "capability_area": g.capability_area,
            "severity": g.severity,
            "description": g.description[:100],
        }
        for g in relevant_gaps[:3]
    ]

    # Contrastive lessons
    if include_contrastive:
        try:
            from lethyx.memory.contrastive import retrieve_contrastive_lessons

            context["contrastive_lessons"] = retrieve_contrastive_lessons(
                store,
                workspace_id=workspace_id,
                limit_successes=3,
                limit_failures=3,
            )
        except Exception:
            log.debug("Contrastive memory retrieval failed (non-fatal)", exc_info=True)
            context["contrastive_lessons"] = {}

    return context


def update_working_memory(
    store: MemoryStore,
    task_id: str,
    workspace_id: str,
    brief: str | None = None,
    hypothesis: str | None = None,
    last_run_id: str | None = None,
    last_run_status: Literal["success", "failure", "partial", "unknown"] = "unknown",
    patch_plan: str | None = None,
) -> WorkingMemoryRecord:
    """Update working memory for current task iteration.

    Working memory is short-lived and reset between tasks.
    This function creates or updates the record.

    Args:
        store: Memory store
        task_id: Current task ID
        workspace_id: Current workspace
        brief: Task brief
        hypothesis: Current hypothesis
        last_run_id: Last run identifier
        last_run_status: Status of last run
        patch_plan: Current patch plan

    Returns:
        The working memory record
    """
    # Note: Working memory is typically kept in-memory, not persisted
    # This is a simple representation - in practice you'd use a cache

    record = WorkingMemoryRecord(
        task_id=task_id,
        workspace_id=workspace_id,
        brief=brief or "",
        hypothesis=hypothesis or "",
        last_run_id=last_run_id,
        last_run_status=last_run_status,
        patch_plan=patch_plan,
    )

    return record


def ingest_workspace_run(
    store: MemoryStore,
    run_path: Path,
    task_id: str,
    workspace_id: str,
    auto_extract_lessons: bool = True,
) -> dict[str, Any]:
    """Ingest a completed workspace run into memory.

    This is the main entry point for converting raw run outputs
    into structured memory records.

    Args:
        store: Memory store
        run_path: Path to run directory
        task_id: Associated task
        workspace_id: Workspace that was run
        auto_extract_lessons: Whether to auto-extract lessons

    Returns:
        Summary of what was ingested
    """
    summary = summarize_workspace_run(
        store=store,
        run_path=run_path,
        task_id=task_id,
        workspace_id=workspace_id,
    )

    ingested: dict[str, Any] = {
        "run_summary": summary,
        "lessons_created": [],
        "signals_recorded": [],
    }

    # Auto-extract lessons if enabled
    if auto_extract_lessons:
        from lethyx.memory.lessons import promote_log_evidence_to_lesson

        lesson = promote_log_evidence_to_lesson(
            store=store,
            task_id=task_id,
            workspace_id=workspace_id,
            run_logs_path=run_path,
            run_manifest=summary.get("metrics"),
        )

        if lesson:
            ingested["lessons_created"].append(lesson.id)

    # Record competency signals based on run outcome
    if summary["status"] == "failure":
        from lethyx.memory.competency import record_competency_signal, SignalCategory

        signal = record_competency_signal(
            store=store,
            category=SignalCategory.REPEATED_FAILURE_MODE,
            capability_area="workspace_execution",
            strength=-0.5,
            evidence_summary=summary["evidence_summary"],
            task_id=task_id,
            workspace_id=workspace_id,
        )
        ingested["signals_recorded"].append(signal.id)

    log.info(f"Ingested workspace run from {run_path}: {ingested}")
    return ingested


# =============================================================================
# Internal Helpers
# =============================================================================


def _compact_lesson_for_context(lesson: Any) -> dict[str, Any]:
    """Create a compact lesson representation for context."""
    return {
        "conclusion": lesson.conclusion[:150],
        "repeat_patterns": lesson.repeat_tags[:2],
        "avoid_patterns": lesson.avoid_tags[:2],
        "priority": lesson.priority,
    }
