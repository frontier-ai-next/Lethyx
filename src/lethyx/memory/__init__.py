"""
Lethyx Memory System.

Runtime memory writes and reads go through exactly one selected durable backend
(`mempalace`, `hindsight`, or `mnemosyne`) plus Lethyx typed governance.
Legacy JSONL/wing/hall/room readers are migration-only.

Model layer: MemoryType, LessonRecord, CompetencyGapRecord, MemoryQuery, MemoryConfig, etc.
"""

from lethyx.memory.models import (
    # Memory types
    MemoryType,
    WorkingMemoryRecord,
    WorkspaceMemoryRecord,
    ManagerMemoryRecord,
    CompetencyMemoryRecord,
    # Lesson records
    LessonRecord,
    WorkspaceLessonRecord,
    ManagerLessonRecord,
    # Competency tracking
    CompetencyGapRecord,
    CapabilitySignal,
    FailureSignature,
    # Query and summary
    MemoryQuery,
    MemorySummaryBundle,
    MemoryStats,
    # Config
    MemoryConfig,
)

from lethyx.memory.store import MemoryStore
from lethyx.memory.store import reprioritize_memory as _reprioritize_memory_impl
from lethyx.memory.paths import (
    get_workspace_store,
    manager_memory_root,
    palace_path_for,
    workspace_memory_root,
)
from lethyx.memory.lessons import (
    record_workspace_lesson,
    record_manager_lesson,
    promote_log_evidence_to_lesson,
)
from lethyx.memory.competency import (
    record_competency_signal,
    open_competency_gap,
    update_competency_gap,
    get_active_gaps,
    check_capability_deficit,
)
from lethyx.memory.summarization import (
    summarize_workspace_run,
    summarize_manager_state,
    build_memory_summary_bundle,
)
from lethyx.memory.relevance import (
    query_relevant_lessons,
    score_relevance,
    deduplicate_lessons,
)
from lethyx.memory.context_builder import (
    build_manager_context_bundle,
    build_workspace_context_bundle,
)

# Canonical recall bundle is the tiered MemPalace one; the flat prompt-line
# bundle is exported under its own name to avoid ambiguity.
from lethyx.memory.palace.recall import RecallBundle
from lethyx.memory.recall import PromptRecallBundle, summarized_palace_for_prompt
from lethyx.memory.reflection import ReflectionResult, run_reflection_phase
from lethyx.memory.harness_stats import (
    ingest_run_harness_selections,
    update_harness_stats,
)
from lethyx.memory.user_profile import (
    load_user_profile_overlay,
    record_human_feedback,
    record_user_insight,
)


def reprioritize_memory(store: MemoryStore) -> None:
    """Apply decay and priority adjustments to all lessons.

    Convenience wrapper for MemoryStore.reprioritize_memory().
    """
    _reprioritize_memory_impl(store)


__all__ = [
    # Models
    "MemoryType",
    "WorkingMemoryRecord",
    "WorkspaceMemoryRecord",
    "ManagerMemoryRecord",
    "CompetencyMemoryRecord",
    "LessonRecord",
    "WorkspaceLessonRecord",
    "ManagerLessonRecord",
    "CompetencyGapRecord",
    "CapabilitySignal",
    "FailureSignature",
    "MemoryQuery",
    "MemorySummaryBundle",
    "MemoryStats",
    "MemoryConfig",
    # Store
    "MemoryStore",
    "reprioritize_memory",
    "get_workspace_store",
    "manager_memory_root",
    "palace_path_for",
    "workspace_memory_root",
    # Lessons
    "record_workspace_lesson",
    "record_manager_lesson",
    "promote_log_evidence_to_lesson",
    # Competency
    "record_competency_signal",
    "open_competency_gap",
    "update_competency_gap",
    "get_active_gaps",
    "check_capability_deficit",
    # Summarization
    "summarize_workspace_run",
    "summarize_manager_state",
    "build_memory_summary_bundle",
    # Relevance
    "query_relevant_lessons",
    "score_relevance",
    "deduplicate_lessons",
    # Context
    "build_manager_context_bundle",
    "build_workspace_context_bundle",
    "RecallBundle",
    "PromptRecallBundle",
    "summarized_palace_for_prompt",
    "ReflectionResult",
    "run_reflection_phase",
    # Self-improvement loops
    "ingest_run_harness_selections",
    "update_harness_stats",
    "load_user_profile_overlay",
    "record_human_feedback",
    "record_user_insight",
]
