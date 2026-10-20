from dataclasses import replace

import pytest

from lethyx.contracts import RecallFrame
from lethyx.memory.backends.contracts import EdgeType, MemoryEdgeRecord, MemoryIndexNode
from lethyx.memory.recall_plan import RecallPlan
from lethyx.memory.recall_strategy import compile_reconstructive_strategy
from lethyx.memory.reconstructive_recall import pin_recall_frame


_QUERY = "How did ReleasePlanner decision procedure change src/api.py before release?"


def _frame(**overrides) -> RecallFrame:
    values = {
        "query": _QUERY,
        "workspace_id": "alpha",
        "run_id": "run-1",
        "task_id": "task-1",
        "subgoal_id": "subgoal-2",
        "subtask_id": "subtask-3",
        "episode_id": "episode-4",
        "scope_chain": ("workspace:alpha",),
        "repository_id": "repo-1",
        "worktree": "worktree-1",
        "revision_id": "revision-7",
        "dirty_state": "dirty",
        "actor_id": "agent-1",
        "speaker_id": "user-1",
        "belief_holder_id": "agent-1",
        "relevant_entities": ("ReleasePlanner",),
        "relevant_paths": ("src/api.py",),
        "relevant_symbols": ("ReleasePlanner.compile",),
        "event_time": "2026-07-14T10:00:00Z",
        "observation_time": "2026-07-14T10:01:00Z",
        "query_time": "2026-07-15T10:00:00Z",
        "tool_name": "pytest",
        "tool_state": "passed",
        "environment_state": ("python:3.12", "branch:release"),
        "desired_temporal_mode": "historical",
        "required_evidence_roles": ("decision", "source"),
        "expected_source_types": ("procedure",),
    }
    values.update(overrides)
    return RecallFrame(**values)


def _pin(frame: RecallFrame | None = None):
    return pin_recall_frame(
        frame or _frame(),
        canonical_snapshot_revision="sha256:" + "a" * 64,
        catalog_revision="catalog:revision-1",
        policy_revision="policy:revision-1",
    )


def _plan() -> RecallPlan:
    return RecallPlan(
        query=_QUERY,
        workspace_id="alpha",
        run_id="run-1",
        scope_chain=("workspace:alpha",),
        max_tokens=2_048,
        limit=12,
    )


def _nodes() -> tuple[MemoryIndexNode, ...]:
    return (
        MemoryIndexNode(
            node_id="cue-node",
            title="ReleasePlanner release change",
            node_type="descriptor",
            workspace_id="alpha",
            tags=("release",),
            metadata={"record_id": "record-cue", "kind": "observation"},
        ),
        MemoryIndexNode(
            node_id="entity-node",
            title="ReleasePlanner entity",
            node_type="entity",
            workspace_id="alpha",
            metadata={"record_id": "record-entity", "kind": "entity"},
        ),
        MemoryIndexNode(
            node_id="path-node",
            title="src/api.py ReleasePlanner.compile",
            node_type="path",
            workspace_id="alpha",
            metadata={"record_id": "record-path", "kind": "path"},
        ),
        MemoryIndexNode(
            node_id="time-node",
            title="historical release state",
            node_type="temporal",
            workspace_id="alpha",
            metadata={"record_id": "record-time", "kind": "temporal"},
        ),
        MemoryIndexNode(
            node_id="decision-node",
            title="release decision",
            node_type="decision",
            workspace_id="alpha",
            metadata={"record_id": "record-decision", "kind": "decision"},
        ),
        MemoryIndexNode(
            node_id="procedure-node",
            title="release procedure steps",
            node_type="procedure",
            workspace_id="alpha",
            metadata={"record_id": "record-procedure", "kind": "procedure"},
        ),
        MemoryIndexNode(
            node_id="foreign-node",
            title="ReleasePlanner foreign",
            node_type="decision",
            workspace_id="beta",
            metadata={"record_id": "record-foreign", "kind": "decision"},
        ),
    )


def _edges() -> tuple[MemoryEdgeRecord, ...]:
    return (
        MemoryEdgeRecord(
            src_id="record-decision",
            dst_id="record-path",
            edge_type=EdgeType.IMPLEMENTS,
            workspace_id="alpha",
        ),
        MemoryEdgeRecord(
            src_id="record-procedure",
            dst_id="record-decision",
            edge_type=EdgeType.APPLIED_REFLECTION,
            workspace_id="alpha",
        ),
    )


def test_pinned_frame_revision_is_deterministic_and_snapshot_bound() -> None:
    first = _pin()
    second = _pin()
    changed = pin_recall_frame(
        _frame(),
        canonical_snapshot_revision="sha256:" + "b" * 64,
        catalog_revision="catalog:revision-1",
        policy_revision="policy:revision-1",
    )

    assert first == second
    assert first.frame_revision.startswith("sha256:")
    assert changed.frame_revision != first.frame_revision


def test_pinned_frame_reinstates_complete_deep_agent_context() -> None:
    context = _pin().reinstated_context()

    assert context["episode_id"] == "episode-4"
    assert context["task_id"] == "task-1"
    assert context["subgoal_id"] == "subgoal-2"
    assert context["repository"] == {
        "repository_id": "repo-1",
        "worktree": "worktree-1",
        "revision_id": "revision-7",
        "dirty_state": "dirty",
    }
    assert context["actor"]["actor_id"] == "agent-1"
    assert context["tool_state"]["tool_state"] == "passed"
    assert context["temporal"]["mode"] == "historical"


def test_shared_planner_compiles_all_six_operators_deterministically() -> None:
    first = compile_reconstructive_strategy(
        _plan(),
        frame=_pin(),
        index_nodes=_nodes(),
        edges=_edges(),
    )
    reordered = compile_reconstructive_strategy(
        _plan(),
        frame=_pin(),
        index_nodes=tuple(reversed(_nodes())),
        edges=tuple(reversed(_edges())),
    )

    assert first == reordered
    assert first.operators == ("cue", "entity", "path", "time", "decision", "procedure")
    assert first.strategy == "reconstructive_lite"
    assert first.backend_path == "selected_backend"
    assert all(step.plan.workspace_id == "alpha" for step in first.steps)
    assert all(step.plan.scope_chain == ("workspace:alpha",) for step in first.steps)
    assert all("record-foreign" not in step.candidate_ids for step in first.steps)
    assert first.steps[3].plan.temporal_mode == "historical"
    assert first.steps[3].plan.as_of == "2026-07-15T10:00:00Z"
    assert "decision" in first.steps[4].plan.evidence_roles
    assert "procedure" in first.steps[5].plan.tags


def test_operator_candidates_come_only_from_canonical_index_records_and_edges() -> None:
    plan = compile_reconstructive_strategy(
        _plan(),
        frame=_pin(),
        index_nodes=_nodes(),
        edges=_edges(),
    )
    known = {
        "record-cue",
        "record-entity",
        "record-path",
        "record-time",
        "record-decision",
        "record-procedure",
    }

    assert set(candidate for step in plan.steps for candidate in step.candidate_ids) <= known
    assert all(
        not candidate.endswith("-node") for step in plan.steps for candidate in step.candidate_ids
    )


def test_frame_plan_identity_and_scope_mismatches_fail_closed() -> None:
    with pytest.raises(ValueError, match="workspace"):
        compile_reconstructive_strategy(
            _plan(),
            frame=_pin(_frame(workspace_id="beta")),
            index_nodes=_nodes(),
        )
    with pytest.raises(ValueError, match="query"):
        compile_reconstructive_strategy(
            replace(_plan(), query="different query"),
            frame=_pin(),
            index_nodes=_nodes(),
        )
