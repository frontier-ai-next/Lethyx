"""
Memory store - persistence layer for structured memory.

Simple file-backed storage using JSONL for:
- Lessons (workspace and manager)
- Competency gaps
- Capability signals

Designed for local-first operation with easy inspection.
"""

import json
import logging

from lethyx.memory.models import (
    MemoryConfig,
    MemoryStats,
    LessonRecord,
    WorkspaceLessonRecord,
    ManagerLessonRecord,
    CompetencyGapRecord,
    CapabilitySignal,
    FailureSignature,
    MemoryQuery,
)

log = logging.getLogger(__name__)


class MemoryStore:
    """File-backed store for structured memory records.

    Uses JSONL for human-readability and easy appending.
    Maintains in-memory indices for fast queries.

    # MIGRATION: MemoryStore JSONL is being replaced by lethyx.memory.palace.MemPalace.
    # This class remains for backward compatibility. New code should use MemPalace.add() directly.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self._ensure_directories()

        # In-memory storage
        self._workspace_lessons: dict[str, WorkspaceLessonRecord] = {}
        self._manager_lessons: dict[str, ManagerLessonRecord] = {}
        self._gaps: dict[str, CompetencyGapRecord] = {}
        self._signals: dict[str, CapabilitySignal] = {}
        self._failure_signatures: dict[str, FailureSignature] = {}

        # Load existing data
        self._load_all()

    # =========================================================================
    # Setup
    # =========================================================================

    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        self.config.memory_root.mkdir(parents=True, exist_ok=True)

    def _load_all(self) -> None:
        """Load all records from disk."""
        self._load_lessons()
        self._load_gaps()
        self._load_signals()

    def _load_lessons(self) -> None:
        """Load lessons from JSONL file."""
        path = self.config.lessons_path
        if not path.exists():
            return

        try:
            for line in path.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    lesson_type = data.get("lesson_type")

                    if lesson_type == "workspace":
                        lesson = WorkspaceLessonRecord(**data)
                        self._workspace_lessons[lesson.id] = lesson
                    elif lesson_type == "manager":
                        lesson = ManagerLessonRecord(**data)
                        self._manager_lessons[lesson.id] = lesson
                except Exception as e:
                    log.warning(f"Failed to load lesson: {e}", exc_info=True)

            log.info(
                f"Loaded {len(self._workspace_lessons)} workspace lessons, "
                f"{len(self._manager_lessons)} manager lessons"
            )
        except Exception as e:
            log.error(f"Failed to load lessons file: {e}", exc_info=True)

    def _load_gaps(self) -> None:
        """Load competency gaps from JSONL file."""
        path = self.config.gaps_path
        if not path.exists():
            return

        try:
            for line in path.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    gap = CompetencyGapRecord(**data)
                    self._gaps[gap.id] = gap
                except Exception as e:
                    log.warning(f"Failed to load gap: {e}", exc_info=True)

            log.info(f"Loaded {len(self._gaps)} competency gaps")
        except Exception as e:
            log.error(f"Failed to load gaps file: {e}", exc_info=True)

    def _load_signals(self) -> None:
        """Load capability signals from JSONL file."""
        path = self.config.signals_path
        if not path.exists():
            return

        try:
            for line in path.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    signal = CapabilitySignal(**data)
                    self._signals[signal.id] = signal
                except Exception as e:
                    log.warning(f"Failed to load signal: {e}", exc_info=True)

            log.info(f"Loaded {len(self._signals)} capability signals")
        except Exception as e:
            log.error(f"Failed to load signals file: {e}", exc_info=True)

    # =========================================================================
    # Lessons CRUD
    # =========================================================================

    def add_lesson(self, lesson: LessonRecord) -> None:
        """Add a lesson to storage."""
        if isinstance(lesson, WorkspaceLessonRecord):
            self._workspace_lessons[lesson.id] = lesson
        elif isinstance(lesson, ManagerLessonRecord):
            self._manager_lessons[lesson.id] = lesson
        else:
            raise ValueError(f"Unknown lesson type: {type(lesson)}")

        self._append_lesson(lesson)

    def _append_lesson(self, lesson: LessonRecord) -> None:
        """Append lesson to JSONL file."""
        self.config.lessons_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config.lessons_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(lesson.model_dump(mode="json"), ensure_ascii=False))
            f.write("\n")

    def get_lesson(self, lesson_id: str) -> LessonRecord | None:
        """Get a lesson by ID."""
        return self._workspace_lessons.get(lesson_id) or self._manager_lessons.get(lesson_id)

    def query_lessons(self, query: MemoryQuery) -> list[LessonRecord]:
        """Query lessons by criteria."""
        results: list[LessonRecord] = []

        # Select source based on type
        if query.lesson_type is None:
            # Return lessons from both types
            source = list(self._workspace_lessons.values()) + list(self._manager_lessons.values())
        elif query.lesson_type == "workspace":
            source = self._workspace_lessons.values()
        else:  # lesson_type == "manager"
            source = self._manager_lessons.values()

        for lesson in source:
            # Apply filters
            if query.task_id is not None and lesson.task_id != query.task_id:
                continue
            if query.workspace_id is not None and lesson.workspace_id != query.workspace_id:
                continue
            if query.tags and not query.tags.intersection(lesson.tags):
                continue
            if lesson.priority < query.min_priority:
                continue
            if not query.include_stale and lesson.is_stale:
                continue
            if lesson.decay_score < query.min_decay_score:
                continue
            if query.max_age_seconds is not None and lesson.age_seconds > query.max_age_seconds:
                continue

            results.append(lesson)

        # Sort by priority and recency
        results.sort(key=lambda lesson: (lesson.priority, lesson.created_at), reverse=True)

        return results[: query.limit]

    def update_lesson(self, lesson_id: str, **updates) -> bool:
        """Update lesson fields."""
        lesson = self.get_lesson(lesson_id)
        if lesson is None:
            return False

        for key, value in updates.items():
            if hasattr(lesson, key):
                setattr(lesson, key, value)

        lesson.touch()
        return True

    def delete_lesson(self, lesson_id: str) -> bool:
        """Delete a lesson (marks as deleted, doesn't rewrite file)."""
        if lesson_id in self._workspace_lessons:
            del self._workspace_lessons[lesson_id]
            return True
        if lesson_id in self._manager_lessons:
            del self._manager_lessons[lesson_id]
            return True
        return False

    # =========================================================================
    # Competency Gaps CRUD
    # =========================================================================

    def add_gap(self, gap: CompetencyGapRecord) -> None:
        """Add a competency gap."""
        self._gaps[gap.id] = gap
        self._append_gap(gap)

    def _append_gap(self, gap: CompetencyGapRecord) -> None:
        """Append gap to JSONL file."""
        self.config.gaps_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config.gaps_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(gap.model_dump(mode="json"), ensure_ascii=False))
            f.write("\n")

    def get_gap(self, gap_id: str) -> CompetencyGapRecord | None:
        """Get a gap by ID."""
        return self._gaps.get(gap_id)

    def get_active_gaps(self, capability_area: str | None = None) -> list[CompetencyGapRecord]:
        """Get all active (open/investigating) gaps."""
        gaps = [g for g in self._gaps.values() if g.status in ("open", "investigating")]

        if capability_area:
            gaps = [g for g in gaps if g.capability_area == capability_area]

        # Sort by severity and recency
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        gaps.sort(
            key=lambda g: (severity_order.get(g.severity, 0), g.last_seen_at),
            reverse=True,
        )

        return gaps

    def update_gap(self, gap_id: str, **updates) -> bool:
        """Update gap fields."""
        gap = self.get_gap(gap_id)
        if gap is None:
            return False

        for key, value in updates.items():
            if hasattr(gap, key):
                setattr(gap, key, value)

        gap.touch()
        return True

    def close_gap(self, gap_id: str, resolution: str) -> bool:
        """Mark a gap as resolved."""
        gap = self.get_gap(gap_id)
        if gap is None:
            return False

        gap.close(resolution)
        return True

    # =========================================================================
    # Capability Signals
    # =========================================================================

    def add_signal(self, signal: CapabilitySignal) -> None:
        """Add a capability signal."""
        self._signals[signal.id] = signal
        self._append_signal(signal)

    def _append_signal(self, signal: CapabilitySignal) -> None:
        """Append signal to JSONL file."""
        self.config.signals_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config.signals_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(signal.model_dump(mode="json"), ensure_ascii=False))
            f.write("\n")

    def get_signals(
        self,
        capability_area: str | None = None,
        category: str | None = None,
        min_strength: float | None = None,
        limit: int = 100,
    ) -> list[CapabilitySignal]:
        """Query capability signals."""
        results = list(self._signals.values())

        if capability_area:
            results = [s for s in results if s.capability_area == capability_area]
        if category:
            results = [s for s in results if s.category == category]
        if min_strength is not None:
            results = [s for s in results if s.strength >= min_strength]

        results.sort(key=lambda s: s.timestamp, reverse=True)
        return results[:limit]

    def get_recent_negative_signals(
        self, seconds: float = 3600, limit: int = 50
    ) -> list[CapabilitySignal]:
        """Get recent negative (problem) signals."""
        import time

        cutoff = time.time() - seconds
        results = [s for s in self._signals.values() if s.is_negative and s.timestamp > cutoff]
        results.sort(key=lambda s: s.timestamp, reverse=True)
        return results[:limit]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> MemoryStats:
        """Get statistics about memory contents."""

        total_size = 0
        for path in [
            self.config.lessons_path,
            self.config.gaps_path,
            self.config.signals_path,
        ]:
            if path.exists():
                total_size += path.stat().st_size

        active_gaps = len([g for g in self._gaps.values() if g.status in ("open", "investigating")])
        closed_gaps = len(self._gaps) - active_gaps

        return MemoryStats(
            total_lessons=len(self._workspace_lessons) + len(self._manager_lessons),
            workspace_lessons=len(self._workspace_lessons),
            manager_lessons=len(self._manager_lessons),
            active_gaps=active_gaps,
            closed_gaps=closed_gaps,
            total_signals=len(self._signals),
            memory_size_bytes=total_size,
        )

    # =========================================================================
    # Maintenance
    # =========================================================================

    def reprioritize_memory(self) -> None:
        """Apply decay and priority adjustments to all lessons."""
        import time

        half_life_seconds = self.config.decay_half_life_days * 24 * 3600
        decay_factor = 0.5 ** (1 / half_life_seconds)  # Per-second decay

        for lesson in list(self._workspace_lessons.values()) + list(self._manager_lessons.values()):
            # Apply time decay
            age = lesson.age_seconds
            lesson.decay_score *= decay_factor**age

            # Boost if recently accessed
            if lesson.last_accessed_at:
                time_since_access = time.time() - lesson.last_accessed_at
                if time_since_access < 86400:  # Accessed in last 24h
                    lesson.decay_score = min(1.0, lesson.decay_score + self.config.access_boost)

            # Remove stale lessons below threshold
            if lesson.decay_score < self.config.stale_decay_threshold and lesson.access_count == 0:
                self.delete_lesson(lesson.id)

    def cleanup_stale_gaps(self) -> int:
        """Close or remove stale competency gaps."""

        removed = 0
        for gap_id, gap in list(self._gaps.items()):
            if gap.is_stale and gap.status in ("open", "investigating"):
                # Mark as deferred rather than deleting
                gap.status = "deferred"
                removed += 1

        return removed

    def compact_storage(self) -> None:
        """Rewrite JSONL files removing deleted/stale records."""
        # This is expensive - only run explicitly
        self._rewrite_lessons()
        self._rewrite_gaps()
        self._rewrite_signals()

    def _rewrite_lessons(self) -> None:
        """Rewrite lessons file with current in-memory contents."""
        with open(self.config.lessons_path, "w", encoding="utf-8") as f:
            for lesson in list(self._workspace_lessons.values()) + list(
                self._manager_lessons.values()
            ):
                f.write(json.dumps(lesson.model_dump(mode="json"), ensure_ascii=False))
                f.write("\n")

    def _rewrite_gaps(self) -> None:
        """Rewrite gaps file with current in-memory contents."""
        with open(self.config.gaps_path, "w", encoding="utf-8") as f:
            for gap in self._gaps.values():
                f.write(json.dumps(gap.model_dump(mode="json"), ensure_ascii=False))
                f.write("\n")

    def _rewrite_signals(self) -> None:
        """Rewrite signals file with current in-memory contents."""
        with open(self.config.signals_path, "w", encoding="utf-8") as f:
            for signal in self._signals.values():
                f.write(json.dumps(signal.model_dump(mode="json"), ensure_ascii=False))
                f.write("\n")


# =============================================================================
# Convenience API functions
# =============================================================================


def reprioritize_memory(store: MemoryStore) -> None:
    """Apply decay and priority adjustments to all lessons in a store.

    This is a convenience wrapper around MemoryStore.reprioritize_memory()
    for easier importing and use.

    Args:
        store: The memory store to reprioritize
    """
    store.reprioritize_memory()
