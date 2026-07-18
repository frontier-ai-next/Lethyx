"""
Lesson recording and extraction.

Converts workspace run outputs into structured lessons.
Handles both explicit lessons and implicit pattern extraction.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Literal

from lethyx.memory.models import (
    WorkspaceLessonRecord,
    ManagerLessonRecord,
    LessonRecord,
    generate_lesson_id,
)
from lethyx.memory.store import MemoryStore

log = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================


def record_workspace_lesson(
    store: MemoryStore,
    task_id: str,
    workspace_id: str,
    change_summary: str,
    expected_effect: str,
    observed_effect: str,
    conclusion: str,
    evidence_summary: str = "",
    repeat_tags: list[str] | None = None,
    avoid_tags: list[str] | None = None,
    priority: int = 5,
    tags: set[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> WorkspaceLessonRecord:
    """Record a workspace-level lesson.

    Args:
        store: Memory store
        task_id: Task identifier
        workspace_id: Workspace identifier
        change_summary: What was changed
        expected_effect: What we expected to happen
        observed_effect: What actually happened
        conclusion: What we learned
        evidence_summary: Key evidence
        repeat_tags: Patterns to repeat
        avoid_tags: Patterns to avoid
        priority: Lesson priority (higher = more important)
        tags: Search tags
        metadata: Additional metadata

    Returns:
        The created lesson record
    """
    lesson = WorkspaceLessonRecord(
        id=generate_lesson_id(),
        task_id=task_id,
        workspace_id=workspace_id,
        change_summary=change_summary,
        expected_effect=expected_effect,
        observed_effect=observed_effect,
        conclusion=conclusion,
        evidence_summary=evidence_summary,
        repeat_tags=repeat_tags or [],
        avoid_tags=avoid_tags or [],
        priority=priority,
        tags=tags or set(),
        metadata=metadata or {},
    )

    store.add_lesson(lesson)
    log.info(f"Recorded workspace lesson {lesson.id} for workspace {workspace_id}")
    return lesson


def record_manager_lesson(
    store: MemoryStore,
    task_id: str,
    change_summary: str,
    expected_effect: str,
    observed_effect: str,
    conclusion: str,
    evidence_summary: str = "",
    capability_area: str | None = None,
    was_self_improvement: bool = False,
    self_patch_outcome: Literal["success", "failure", "mixed", "unknown"] = "unknown",
    repeat_tags: list[str] | None = None,
    avoid_tags: list[str] | None = None,
    priority: int = 5,
    tags: set[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ManagerLessonRecord:
    """Record a manager-level lesson.

    Args:
        store: Memory store
        task_id: Task identifier
        change_summary: What was changed
        expected_effect: What we expected to happen
        observed_effect: What actually happened
        conclusion: What we learned
        evidence_summary: Key evidence
        capability_area: Affected capability area (e.g., "gmas_knowledge")
        was_self_improvement: Was this a self-improvement?
        self_patch_outcome: Outcome of self-patch
        repeat_tags: Patterns to repeat
        avoid_tags: Patterns to avoid
        priority: Lesson priority (higher = more important)
        tags: Search tags
        metadata: Additional metadata

    Returns:
        The created lesson record
    """
    lesson = ManagerLessonRecord(
        id=generate_lesson_id(),
        task_id=task_id,
        workspace_id=None,  # Manager lessons are workspace-agnostic
        change_summary=change_summary,
        expected_effect=expected_effect,
        observed_effect=observed_effect,
        conclusion=conclusion,
        evidence_summary=evidence_summary,
        affected_capability_area=capability_area,
        was_self_improvement=was_self_improvement,
        self_patch_outcome=self_patch_outcome,
        repeat_tags=repeat_tags or [],
        avoid_tags=avoid_tags or [],
        priority=priority,
        tags=tags or set(),
        metadata=metadata or {},
    )

    store.add_lesson(lesson)
    log.info(f"Recorded manager lesson {lesson.id}")
    return lesson


def promote_log_evidence_to_lesson(
    store: MemoryStore,
    task_id: str,
    workspace_id: str,
    run_logs_path: Path,
    run_manifest: dict[str, Any] | None = None,
) -> LessonRecord | None:
    """Extract a lesson from workspace run logs.

    Analyzes run outputs to identify:
    - Success patterns to repeat
    - Failure patterns to avoid
    - Unexpected outcomes worth learning

    Returns None if no clear lesson can be extracted.

    Args:
        store: Memory store
        task_id: Task identifier
        workspace_id: Workspace identifier
        run_logs_path: Path to run directory
        run_manifest: Optional run manifest metadata

    Returns:
        Extracted lesson or None
    """
    # Check for common success/failure indicators
    status = _detect_run_status(run_logs_path, run_manifest)

    if status == "success":
        return _extract_success_lesson(store, task_id, workspace_id, run_logs_path, run_manifest)
    elif status == "failure":
        return _extract_failure_lesson(store, task_id, workspace_id, run_logs_path, run_manifest)
    elif status == "partial":
        return _extract_partial_lesson(store, task_id, workspace_id, run_logs_path, run_manifest)
    else:
        # Unclear outcome - don't force a lesson
        return None


# =============================================================================
# Extraction Helpers
# =============================================================================


def _detect_run_status(
    run_logs_path: Path,
    manifest: dict[str, Any] | None,
) -> Literal["success", "failure", "partial", "unknown"]:
    """Detect the overall status of a workspace run."""
    # Check manifest first
    if manifest:
        manifest_status = manifest.get("status", manifest.get("outcome"))
        if manifest_status:
            manifest_status_lower = manifest_status.lower()
            if "success" in manifest_status_lower or "complete" in manifest_status_lower:
                return "success"
            elif "fail" in manifest_status_lower or "error" in manifest_status_lower:
                return "failure"
            elif "partial" in manifest_status_lower:
                return "partial"

    # Check for error patterns in logs
    memory_path = run_logs_path / "memory"
    if memory_path.exists():
        # Look for error signals in agent memory
        for agent_dir in memory_path.iterdir():
            if not agent_dir.is_dir():
                continue
            for md_file in agent_dir.glob("*.md"):
                content = md_file.read_text(encoding="utf-8").lower()
                if "error" in content or "exception" in content or "failed" in content:
                    # Check if it was fatal or recoverable
                    if "fatal" in content or "critical" in content:
                        return "failure"
                    return "partial"

    # Check for success signals
    reports_path = run_logs_path / "reports"
    if reports_path.exists():
        for report_file in reports_path.glob("*.md"):
            content = report_file.read_text(encoding="utf-8").lower()
            if "success" in content or "complete" in content or "delivered" in content:
                return "success"

    return "unknown"


def _extract_success_lesson(
    store: MemoryStore,
    task_id: str,
    workspace_id: str,
    run_logs_path: Path,
    manifest: dict[str, Any] | None,
) -> WorkspaceLessonRecord:
    """Extract a lesson from a successful run."""
    # Identify what worked
    success_patterns = _identify_success_patterns(run_logs_path)

    return record_workspace_lesson(
        store=store,
        task_id=task_id,
        workspace_id=workspace_id,
        change_summary=_infer_changes_from_manifest(manifest),
        expected_effect="Task completion",
        observed_effect="Run succeeded",
        conclusion=f"Successful run. Patterns to repeat: {', '.join(success_patterns) if success_patterns else 'base configuration'}",
        evidence_summary=_summarize_run_evidence(run_logs_path),
        repeat_tags=success_patterns,
        avoid_tags=[],
        priority=7,  # Success patterns are high value
        tags={"success", workspace_id},
    )


def _extract_failure_lesson(
    store: MemoryStore,
    task_id: str,
    workspace_id: str,
    run_logs_path: Path,
    manifest: dict[str, Any] | None,
) -> WorkspaceLessonRecord:
    """Extract a lesson from a failed run."""
    # Identify what went wrong
    failure_patterns = _identify_failure_patterns(run_logs_path)
    error_summary = _summarize_errors(run_logs_path)

    return record_workspace_lesson(
        store=store,
        task_id=task_id,
        workspace_id=workspace_id,
        change_summary=_infer_changes_from_manifest(manifest),
        expected_effect="Task completion",
        observed_effect="Run failed",
        conclusion=f"Run failed. Issue: {error_summary}",
        evidence_summary=_summarize_run_evidence(run_logs_path),
        repeat_tags=[],
        avoid_tags=failure_patterns,
        priority=8,  # Failure lessons are very important
        tags={"failure", workspace_id},
    )


def _extract_partial_lesson(
    store: MemoryStore,
    task_id: str,
    workspace_id: str,
    run_logs_path: Path,
    manifest: dict[str, Any] | None,
) -> WorkspaceLessonRecord:
    """Extract a lesson from a partially successful run."""
    success_patterns = _identify_success_patterns(run_logs_path)
    failure_patterns = _identify_failure_patterns(run_logs_path)

    return record_workspace_lesson(
        store=store,
        task_id=task_id,
        workspace_id=workspace_id,
        change_summary=_infer_changes_from_manifest(manifest),
        expected_effect="Task completion",
        observed_effect="Partial success",
        conclusion=f"Partial run. Worked: {', '.join(success_patterns) if success_patterns else 'some aspects'}. Issues: {', '.join(failure_patterns) if failure_patterns else 'some aspects failed'}.",
        evidence_summary=_summarize_run_evidence(run_logs_path),
        repeat_tags=success_patterns,
        avoid_tags=failure_patterns,
        priority=6,
        tags={"partial", workspace_id},
    )


# =============================================================================
# Analysis Helpers
# =============================================================================


def _infer_changes_from_manifest(manifest: dict[str, Any] | None) -> str:
    """Infer what changed from run manifest."""
    if not manifest:
        return "Standard workspace run"

    changes = []

    if manifest.get("graph_modified"):
        changes.append("graph topology")
    if manifest.get("agents_modified"):
        changes.append("agent configurations")
    if manifest.get("prompts_modified"):
        changes.append("prompts")
    if manifest.get("tools_modified"):
        changes.append("tools")

    return f"Modified: {', '.join(changes)}" if changes else "Standard run"


def _identify_success_patterns(run_logs_path: Path) -> list[str]:
    """Identify patterns that contributed to success."""
    patterns = []

    # Check reports for success indicators
    reports_path = run_logs_path / "reports"
    if reports_path.exists():
        for report_file in reports_path.glob("*.md"):
            content = report_file.read_text(encoding="utf-8")
            # Simple keyword extraction
            for line in content.split("\n"):
                line_lower = line.lower()
                if any(
                    word in line_lower
                    for word in [
                        "successful",
                        "worked well",
                        "effective",
                        "good result",
                    ]
                ):
                    # Extract the pattern description
                    stripped = line.strip()
                    if len(stripped) < 200:  # Keep it concise
                        patterns.append(stripped[:100])

    return patterns[:5]  # Top 5 patterns


def _identify_failure_patterns(run_logs_path: Path) -> list[str]:
    """Identify patterns that caused failure."""
    patterns = []

    memory_path = run_logs_path / "memory"
    if memory_path.exists():
        for agent_dir in memory_path.iterdir():
            if not agent_dir.is_dir():
                continue
            for md_file in agent_dir.glob("*.md"):
                content = md_file.read_text(encoding="utf-8")
                # Look for error/failure patterns
                for line in content.split("\n"):
                    line_lower = line.lower()
                    if any(
                        word in line_lower
                        for word in ["error:", "exception:", "failed to", "unable to"]
                    ):
                        stripped = line.strip()
                        if len(stripped) < 200:
                            patterns.append(stripped[:100])

    return list(set(patterns))[:5]  # Dedupe and limit


def _summarize_errors(run_logs_path: Path) -> str:
    """Summarize errors from the run."""
    error_messages = []

    memory_path = run_logs_path / "memory"
    if memory_path.exists():
        for agent_dir in memory_path.iterdir():
            if not agent_dir.is_dir():
                continue
            for md_file in agent_dir.glob("*.md"):
                content = md_file.read_text(encoding="utf-8")
                # Extract error messages
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if "error" in line_lower or "exception" in line_lower:
                        # Grab context
                        context = "\n".join(lines[max(0, i - 1) : min(len(lines), i + 2)])
                        if len(context) < 300:
                            error_messages.append(context.strip())

    if error_messages:
        # Dedupe by hash
        unique_errors = list(
            {hashlib.md5(e.encode()).hexdigest(): e for e in error_messages}.values()
        )
        return "; ".join(unique_errors[:3])  # Top 3 unique errors

    return "Unknown error"


def _summarize_run_evidence(run_logs_path: Path) -> str:
    """Create a brief summary of run evidence."""
    evidence_parts = []

    # Check reports
    reports_path = run_logs_path / "reports"
    if reports_path.exists():
        report_count = len(list(reports_path.glob("*.md")))
        evidence_parts.append(f"{report_count} reports generated")

    # Check memory outputs
    memory_path = run_logs_path / "memory"
    if memory_path.exists():
        agent_count = len([d for d in memory_path.iterdir() if d.is_dir()])
        evidence_parts.append(f"{agent_count} agents active")

    # Check for artifacts
    artifacts_path = run_logs_path / "artifacts"
    if artifacts_path.exists():
        artifact_count = len(list(artifacts_path.iterdir()))
        evidence_parts.append(f"{artifact_count} artifacts")

    return "; ".join(evidence_parts) if evidence_parts else "Run completed"
