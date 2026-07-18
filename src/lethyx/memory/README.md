# Lethyx Memory System

Structured memory system for the manager layer (Task 05: Memory and Competency Ledger).

## Overview

The memory system provides four types of structured memory:

1. **Working Memory** - Short-lived memory for current task iteration
2. **Workspace Memory** - Workspace-specific lessons and patterns
3. **Manager Memory** - Cross-workspace patterns and strategies
4. **Competency Memory** - Capability gaps and self-improvement tracking

## Architecture

```
Lethyx/memory/
├── __init__.py           # Public API exports
├── models.py             # Pydantic schemas for all memory types
├── store.py              # File-backed JSONL persistence layer
├── lessons.py            # Lesson recording and extraction
├── competency.py         # Competency ledger and gap tracking
├── summarization.py      # Compact summaries for context injection
├── relevance.py          # Relevance scoring and deduplication
├── context_builder.py    # Main interface for building memory bundles
└── README.md             # This file
```

## Key Features

### Structured Lessons

Lessons are structured records with:
- `change_summary`: What was changed
- `expected_effect`: What we expected to happen
- `observed_effect`: What actually happened
- `conclusion`: What we learned
- `repeat_tags`: Patterns to repeat
- `avoid_tags`: Patterns to avoid

### Competency Ledger

Tracks manager-level capability gaps:
- Detects repeated failure patterns
- Distinguishes workspace vs manager issues
- Triggers self-improvement when appropriate
- Maintains gap severity and status

### Memory Decay

Lessons automatically decay over time:
- Fresh lessons have `decay_score = 1.0`
- Decay follows half-life schedule (default 30 days)
- Accessing a lesson boosts its score
- Stale lessons (`decay_score < 0.3`) are filtered out

## Usage

### Basic Lesson Recording

```python
from lethyx.memory import record_workspace_lesson, MemoryStore

store = MemoryStore()

# Record a lesson from a workspace iteration
lesson = record_workspace_lesson(
    store=store,
    task_id="task_123",
    workspace_id="agent_research",
    change_summary="Added parallel execution to graph",
    expected_effect="Faster completion",
    observed_effect="2x speed improvement",
    conclusion="Parallel execution is effective for this task type",
    evidence_summary="Wall time reduced from 10m to 5m",
    repeat_tags=["parallel_execution"],
    avoid_tags=[],
    priority=8,
)
```

### Competency Tracking

```python
from lethyx.memory import record_competency_signal, get_active_gaps

# Record a negative signal
signal = record_competency_signal(
    store=store,
    category="retrieval_misses",
    capability_area="gmas_knowledge",
    strength=-0.7,
    evidence_summary="Failed to find GMAS routing API 3 times",
    task_id="task_123",
)

# Check for capability gaps
gaps = get_active_gaps(store, capability_area="gmas_knowledge")
for gap in gaps:
    print(f"{gap.severity}: {gap.description}")
```

### Building Context for LLM

```python
from lethyx.memory import build_manager_context_bundle

# Build compact memory bundle for prompt injection
bundle = build_manager_context_bundle(
    store=store,
    task_id="task_123",
    workspace_id="agent_research",
    max_lessons=10,
    max_gaps=5,
)

# Convert to prompt section
prompt_section = bundle.to_prompt_section()
```

### Ingesting Workspace Runs

```python
from lethyx.memory import ingest_workspace_run
from pathlib import Path

# Auto-extract lessons from a completed run
ingested = ingest_workspace_run(
    store=store,
    run_path=Path("workspaces/agent_research/runs/run_123"),
    task_id="task_123",
    workspace_id="agent_research",
    auto_extract_lessons=True,
)

print(f"Created {len(ingested['lessons_created'])} lessons")
print(f"Recorded {len(ingested['signals_recorded'])} signals")
```

## Memory Types

### WorkingMemoryRecord

Short-lived state for current iteration:
- Current task brief
- Current hypothesis
- Selected workspace
- Last run results
- Current patch plan

### WorkspaceLessonRecord

Workspace-specific lessons:
- What changes worked/didn't work
- Patterns to repeat/avoid
- Workspace-specific invariants
- Files changed

### ManagerLessonRecord

Manager-wide lessons:
- Strategy patterns
- Retrieval improvements
- Self-patch outcomes
- Cross-workspace insights

### CompetencyGapRecord

Capability gap tracking:
- Capability area (e.g., "gmas_knowledge")
- Severity level
- Status (open/investigating/addressed)
- Evidence signals
- Suggested actions

## Storage

Memory is stored in JSONL files under `.Lethyx/memory/`:

```
.Lethyx/memory/
├── lessons.jsonl    # All lessons (workspace + manager)
├── gaps.jsonl        # Competency gaps
└── signals.jsonl     # Capability signals
```

JSONL format is:
- Human-readable
- Easy to append
- Simple to inspect/debug
- Compatible with Unix tools

## Durable memory backends

Environment variables: **[docs/memory-durable-backends.md](../../docs/memory-durable-backends.md)**.

Workspace vs manager isolation (what is shared across projects): **[docs/workspaces.md#memory-layout-per-workspace-vs-manager](../../docs/workspaces.md#memory-layout-per-workspace-vs-manager)**.

## Configuration

```python
from lethyx.memory import MemoryConfig

config = MemoryConfig(
    memory_root=Path(".Lethyx/memory"),
    max_workspace_lessons=500,
    max_manager_lessons=200,
    decay_half_life_days=30.0,
    gap_threshold_signals=3,  # Signals needed to open a gap
)
```

## Testing

Run memory system tests:

```bash
pytest Lethyx/tests/test_memory.py -v
```

All Lethyx tests:

```bash
pytest Lethyx/tests/ -v
```

## Integration Points

The memory system integrates with:

- **Workspace Runtime** ([`Lethyx/workspace_runtime/`](../workspace_runtime/))
  - Records lessons after each run
  - Summarizes run outcomes

- **Control Plane** ([`Lethyx/control_plane/`](../control_plane/)) (future)
  - Queries memory for decision-making
  - Updates competency tracking

- **Retrieval** ([`Lethyx/retrieval/`](../retrieval/))
  - Provides retrieval usefulness signals
  - Stores retrieval patterns that worked

## Design Principles

1. **Structured over free-form** - Lessons have stable fields for querying
2. **Workspace-first** - Most learning is workspace-specific
3. **Manager as exception** - Self-improvement only for clear capability gaps
4. **Decay and deduplication** - Memory doesn't grow forever
5. **Hot vs cold storage** - Structured lessons in memory, raw logs in artifacts

## Acceptance Criteria (from Task 05)

- ✅ Separate memory into working/workspace/manager/competency types
- ✅ Store structured lessons, not just summaries
- ✅ Track repeated failure modes and success patterns
- ✅ Track when problems are workspace vs manager level
- ✅ Support memory decay and reprioritization
- ✅ Keep raw logs out of hot memory
- ✅ Convert runtime evidence into structured lessons
- ✅ Answer "what have we tried on this task/class"
- ✅ Detect repeated manager-level failure signals
- ✅ Store reusable lessons from iterations
- ✅ Memory is structured for decision policies

## Future Enhancements

- LLM-based lesson extraction from run logs
- Cross-workspace pattern clustering
- Automatic lesson promotion to seed workspaces
- Integration with eval systems for lesson validation
- Retrieval over lessons (not just metadata filtering)
