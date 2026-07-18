"""
Memory system data models.

Defines schemas for all memory types and records.
Structured, not free-form - enables querying and decision policies.
"""

import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """Types of memory in the system."""

    WORKING = "working"
    WORKSPACE = "workspace"
    MANAGER = "manager"
    COMPETENCY = "competency"


class LessonType(StrEnum):
    """Types of lessons."""

    WORKSPACE = "workspace"
    MANAGER = "manager"


class GapSeverity(StrEnum):
    """Severity levels for competency gaps."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GapStatus(StrEnum):
    """Status of a competency gap."""

    OPEN = "open"
    INVESTIGATING = "investigating"
    ADDRESSED = "addressed"
    FALSE_POSITIVE = "false_positive"
    DEFERRED = "deferred"


class SignalCategory(StrEnum):
    """Categories of capability signals."""

    NO_PROGRESS_ITERATIONS = "no_progress_iterations"
    RETRIEVAL_MISSES = "retrieval_misses"
    REPEATED_FAILURE_MODE = "repeated_failure_mode"
    HUMAN_FEEDBACK = "human_feedback"
    HIGH_COST_NO_GAIN = "high_cost_no_gain"
    MISSING_CAPABILITY = "missing_capability"


# =============================================================================
# Base Memory Records
# =============================================================================


@dataclass
class MemoryStats:
    """Statistics about memory contents."""

    total_lessons: int = 0
    workspace_lessons: int = 0
    manager_lessons: int = 0
    active_gaps: int = 0
    closed_gaps: int = 0
    total_signals: int = 0
    memory_size_bytes: int = 0


class WorkingMemoryRecord(BaseModel):
    """Short-lived memory for the current task iteration.

    Contains:
    - Current task brief
    - Current hypothesis
    - Selected workspace
    - Last run results
    - Current patch plan
    """

    task_id: str
    workspace_id: str
    brief: str = ""
    hypothesis: str = ""
    last_run_id: str | None = None
    last_run_status: Literal["success", "failure", "partial", "unknown"] = "unknown"
    patch_plan: str | None = None
    iteration_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = time.time()


class WorkspaceMemoryRecord(BaseModel):
    """Workspace-specific memory.

    Stores:
    - Past successful/failed patches
    - Local invariants
    - Known limitations
    - Startup patterns
    - Domain recipes
    """

    workspace_id: str
    task_class: str
    lessons: list[str] = Field(default_factory=list)  # Lesson IDs
    invariants: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    successful_patterns: list[str] = Field(default_factory=list)
    failure_patterns: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = time.time()


class ManagerMemoryRecord(BaseModel):
    """Manager-level memory across all workspaces.

    Stores:
    - Which workspaces work for which task classes
    - Strategy patterns
    - Signs of workspace vs manager problems
    - Retrieval patterns that worked
    """

    task_class: str
    preferred_workspace_id: str | None = None
    successful_strategies: list[str] = Field(default_factory=list)
    failed_strategies: list[str] = Field(default_factory=list)
    workspace_vs_manager_clues: list[str] = Field(default_factory=list)
    retrieval_patterns: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = time.time()


class CompetencyMemoryRecord(BaseModel):
    """Memory about manager capabilities and gaps.

    Stores:
    - What the manager couldn't do
    - Which self-patches closed real blockers
    - Which self-patches were useless
    - Skills to develop next
    """

    capability_area: str  # e.g., "gmas_knowledge", "retrieval", "prompt_design"
    current_level: float = 0.0  # 0.0 to 1.0
    demonstrated_gaps: list[str] = Field(default_factory=list)  # Gap IDs
    successful_improvements: list[str] = Field(default_factory=list)
    failed_improvements: list[str] = Field(default_factory=list)
    next_skills: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = time.time()


# =============================================================================
# Lesson Records
# =============================================================================


class LessonRecord(BaseModel):
    """Base lesson record with common fields."""

    id: str
    lesson_type: LessonType
    task_id: str
    workspace_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    priority: int = 0  # Higher = more important
    access_count: int = 0
    last_accessed_at: float | None = None
    tags: set[str] = Field(default_factory=set)
    decay_score: float = 1.0  # 1.0 = fresh, 0.0 = stale

    # Core lesson content
    change_summary: str  # What was changed
    expected_effect: str  # What we expected to happen
    observed_effect: str  # What actually happened
    conclusion: str  # What we learned
    evidence_summary: str  # Key evidence

    # Actionable tags
    repeat_tags: list[str] = Field(default_factory=list)  # Patterns to repeat
    avoid_tags: list[str] = Field(default_factory=list)  # Patterns to avoid

    # Raw evidence pointers (Meta-Harness traceability)
    raw_evidence_paths: list[str] = Field(default_factory=list)
    candidate_id: str | None = None
    experiment_id: str | None = None
    source_run_id: str | None = None
    source_task_result_path: str | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Record an access and update timestamp."""
        self.access_count += 1
        self.last_accessed_at = time.time()

    @property
    def age_seconds(self) -> float:
        """Age of the lesson in seconds."""
        return time.time() - self.created_at

    @property
    def is_stale(self) -> bool:
        """Check if lesson is considered stale (decay_score < 0.3)."""
        return self.decay_score < 0.3


class WorkspaceLessonRecord(LessonRecord):
    """A lesson learned from workspace iteration.

    Workspace-specific: apply to this workspace or similar ones.
    """

    lesson_type: Literal[LessonType.WORKSPACE] = LessonType.WORKSPACE
    workspace_version: str | None = None  # Git sha or version identifier
    files_changed: list[str] = Field(default_factory=list)
    was_promoted: bool = False  # Was this promoted to seed?


class ManagerLessonRecord(LessonRecord):
    """A lesson learned about manager-level behavior.

    Manager-wide: applies across all workspaces.
    """

    lesson_type: Literal[LessonType.MANAGER] = LessonType.MANAGER
    affected_capability_area: str | None = None  # e.g., "gmas_knowledge"
    was_self_improvement: bool = False
    self_patch_outcome: Literal["success", "failure", "mixed", "unknown"] = "unknown"


# =============================================================================
# Competency Tracking
# =============================================================================


class FailureSignature(BaseModel):
    """A signature of a failure mode for pattern matching."""

    signature_hash: str
    pattern_description: str
    error_types: list[str] = Field(default_factory=list)
    context_clues: list[str] = Field(default_factory=list)
    occurrence_count: int = 1
    first_seen_at: float = Field(default_factory=time.time)
    last_seen_at: float = Field(default_factory=time.time)


class CapabilitySignal(BaseModel):
    """A signal indicating potential capability gap or strength."""

    id: str
    category: SignalCategory
    capability_area: str  # e.g., "gmas_knowledge", "retrieval", "planning"
    strength: float  # -1.0 (weakness) to 1.0 (strength)
    evidence_summary: str
    task_id: str
    workspace_id: str | None = None
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_negative(self) -> bool:
        """True if this signal indicates a problem/weakness."""
        return self.strength < 0


class CompetencyGapRecord(BaseModel):
    """A record of a capability gap that may need self-improvement."""

    id: str
    capability_area: str
    severity: GapSeverity
    status: GapStatus = GapStatus.OPEN
    description: str
    evidence_signals: list[str] = Field(default_factory=list)  # Signal IDs
    occurrences: int = 1
    first_seen_at: float = Field(default_factory=time.time)
    last_seen_at: float = Field(default_factory=time.time)
    resolved_at: float | None = None
    resolution_summary: str | None = None
    suspected_root_cause: str | None = None
    suggested_actions: list[str] = Field(default_factory=list)
    is_workspace_level: bool = False  # True if problem is in workspace, not manager
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Update last_seen_at and increment occurrences."""
        self.last_seen_at = time.time()
        self.occurrences += 1

    @property
    def age_seconds(self) -> float:
        """Age of the gap in seconds."""
        return time.time() - self.first_seen_at

    @property
    def is_stale(self) -> bool:
        """Check if gap is stale (no activity for 7 days)."""
        return (time.time() - self.last_seen_at) > (7 * 24 * 3600)

    def close(self, resolution: str) -> None:
        """Mark the gap as resolved."""
        self.status = GapStatus.ADDRESSED
        self.resolved_at = time.time()
        self.resolution_summary = resolution


# =============================================================================
# Query and Summary
# =============================================================================


class MemoryQuery(BaseModel):
    """Query for retrieving relevant memory."""

    task_id: str | None = None
    task_class: str | None = None
    workspace_id: str | None = None
    lesson_type: LessonType | None = None
    tags: set[str] = Field(default_factory=set)
    min_priority: int = 0
    min_decay_score: float = 0.0
    max_age_seconds: float | None = None
    limit: int = 50
    include_stale: bool = False


class MemorySummaryBundle(BaseModel):
    """A compact summary of memory for prompt injection.

    Designed to be context-efficient:
    - Top relevant lessons only
    - Open competency gaps
    - Recent repeated failures
    - Stats overview
    """

    task_id: str
    generated_at: float = Field(default_factory=time.time)

    # Compact lessons (not full records)
    relevant_workspace_lessons: list[dict[str, Any]] = Field(default_factory=list)
    relevant_manager_lessons: list[dict[str, Any]] = Field(default_factory=list)

    # Competency info
    active_gaps: list[dict[str, Any]] = Field(default_factory=list)
    capability_warnings: list[str] = Field(default_factory=list)

    # Patterns
    repeated_failures: list[dict[str, Any]] = Field(default_factory=list)
    repeated_successes: list[dict[str, Any]] = Field(default_factory=list)

    # Stats
    stats: MemoryStats = Field(default_factory=MemoryStats)

    def to_prompt_section(self) -> str:
        """Convert to a compact prompt section."""
        sections = []

        # Stats
        sections.append("## Memory Stats")
        sections.append(
            f"- Lessons: {self.stats.workspace_lessons} workspace, {self.stats.manager_lessons} manager\n"
            f"- Active gaps: {self.stats.active_gaps}\n"
            f"- Total signals: {self.stats.total_signals}"
        )

        # Active gaps
        if self.active_gaps:
            sections.append("\n## Active Competency Gaps")
            for gap in self.active_gaps:
                sections.append(
                    f"- [{gap['severity']}] {gap['capability_area']}: {gap['description'][:100]}"
                )

        # Relevant lessons
        if self.relevant_workspace_lessons:
            sections.append("\n## Relevant Workspace Lessons")
            for lesson in self.relevant_workspace_lessons[:5]:
                sections.append(
                    f"- {lesson.get('conclusion', lesson.get('change_summary', ''))[:150]}"
                )

        if self.relevant_manager_lessons:
            sections.append("\n## Relevant Manager Lessons")
            for lesson in self.relevant_manager_lessons[:5]:
                sections.append(
                    f"- {lesson.get('conclusion', lesson.get('change_summary', ''))[:150]}"
                )

        # Repeated failures
        if self.repeated_failures:
            sections.append("\n## Repeated Failure Patterns")
            for failure in self.repeated_failures[:3]:
                sections.append(
                    f"- {failure.get('pattern_description', 'Unknown pattern')}: {failure.get('occurrence_count', 0)} occurrences"
                )

        return "\n".join(sections)


# =============================================================================
# Configuration
# =============================================================================


class MemoryConfig(BaseModel):
    """Configuration for the memory system."""

    # Storage paths
    memory_root: Path = Field(default_factory=lambda: Path(".Lethyx/memory"))
    lessons_path: Path = Field(default_factory=lambda: Path(".Lethyx/memory/lessons.jsonl"))
    gaps_path: Path = Field(default_factory=lambda: Path(".Lethyx/memory/gaps.jsonl"))
    signals_path: Path = Field(default_factory=lambda: Path(".Lethyx/memory/signals.jsonl"))

    # Lesson management
    max_working_lessons: int = 100
    max_workspace_lessons: int = 500
    max_manager_lessons: int = 200
    default_lesson_priority: int = 5

    # Decay settings
    decay_half_life_days: float = 30.0  # Days for priority to halve
    stale_decay_threshold: float = 0.3
    access_boost: float = 0.1  # Boost decay_score on each access

    # Competency settings
    gap_threshold_signals: int = 3  # Signals needed to open a gap
    gap_cooldown_seconds: float = 3600.0  # Min time between same-gap signals
    max_active_gaps: int = 20

    # Relevance scoring
    tag_match_weight: float = 2.0
    recency_weight: float = 1.0
    priority_weight: float = 1.5
    decay_weight: float = 1.0

    # Deduplication
    dedup_similarity_threshold: float = 0.8  # For lesson clustering


# =============================================================================
# Factory helpers
# =============================================================================


def generate_lesson_id() -> str:
    """Generate a unique lesson ID."""
    import uuid

    return f"lesson_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def generate_gap_id() -> str:
    """Generate a unique competency gap ID."""
    import uuid

    return f"gap_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def generate_signal_id() -> str:
    """Generate a unique capability signal ID."""
    import uuid

    return f"signal_{int(time.time())}_{uuid.uuid4().hex[:8]}"
