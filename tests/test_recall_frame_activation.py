from dataclasses import asdict
from typing import Any

import pytest

from lethyx import MemoryClient
from lethyx.contracts import RecallFrame
from lethyx.memory.recall_activation import (
    compile_recall_frame,
    recall_activation_decision,
)


def test_recall_frame_compiles_live_state_as_widenable_soft_priors() -> None:
    frame = compile_recall_frame(
        query="Why did the parser change?",
        workspace_id="alpha",
        run_id="run-1",
        task_id="task-1",
        subtask_id="subtask-2",
        episode_id="episode-3",
        scope_chain=("workspace:alpha", "task:task-1"),
        transition_reason="source_change",
        repository_id="lethyx",
        worktree="C:/work/lethyx",
        revision_id="abc123",
        dirty_state="dirty",
        actor_id="agent:sol",
        speaker_id="user:operator",
        belief_holder_id="agent:sol",
        relevant_entities=("RecallPlan",),
        relevant_paths=("src/lethyx/memory/recall_plan.py",),
        relevant_symbols=("RecallPlan.compile",),
        relevant_configs=("pyproject.toml",),
        event_time="2026-07-14T20:00:00Z",
        mention_time="2026-07-14T20:01:00Z",
        observation_time="2026-07-14T20:02:00Z",
        query_time="2026-07-14T20:03:00Z",
        tool_name="pytest",
        tool_state="failed",
        environment_state=("python:3.12", "platform:windows"),
        desired_temporal_mode="why_changed",
        required_evidence_roles=("source", "verification"),
        expected_source_types=("file_snapshot", "test_run"),
        uncertainty=0.7,
        coverage=0.3,
    )

    assert isinstance(frame, RecallFrame)
    assert frame.desired_temporal_mode == "why_changed"
    assert frame.widening_policy == "backend_only"
    assert frame.widening_allowed is True
    assert set(frame.auto_derived_fields) >= {
        "task_id",
        "subtask_id",
        "repository_id",
        "tool_state",
        "desired_temporal_mode",
        "required_evidence_roles",
        "uncertainty",
        "coverage",
    }
    assert asdict(frame)["relevant_symbols"] == ("RecallPlan.compile",)

    with pytest.raises(ValueError, match="desired_temporal_mode"):
        compile_recall_frame(query="parser", desired_temporal_mode="future")
    with pytest.raises(ValueError, match="uncertainty"):
        compile_recall_frame(query="parser", uncertainty=1.1)


@pytest.mark.parametrize(
    "transition_reason",
    [
        "subgoal_transition",
        "tool_result",
        "tool_failure",
        "test_failure",
        "verification_success",
        "plan_revision",
        "component_switch",
        "repository_switch",
        "source_change",
        "uncertainty_spike",
        "context_fold",
        "page_back",
        "prospective_trigger",
        "high_risk_boundary",
    ],
)
def test_mid_task_transitions_refresh_recall_strategy(transition_reason: str) -> None:
    previous = compile_recall_frame(query="parser", task_id="task-1")
    current = compile_recall_frame(
        query="parser",
        task_id="task-1",
        transition_reason=transition_reason,
    )

    decision = recall_activation_decision(previous, current)

    assert decision.should_refresh is True
    assert decision.strategy_refresh is True
    assert decision.reason == transition_reason
    assert decision.widening_allowed is True


def test_implicit_frame_delta_refreshes_but_unchanged_frame_does_not() -> None:
    previous = compile_recall_frame(
        query="parser",
        repository_id="lethyx",
        revision_id="abc",
        uncertainty=0.2,
    )

    assert recall_activation_decision(previous, previous).should_refresh is False
    repository_change = compile_recall_frame(
        query="parser",
        repository_id="other-repo",
        revision_id="def",
        uncertainty=0.2,
    )
    assert recall_activation_decision(previous, repository_change).reason == "repository_switch"
    uncertainty_spike = compile_recall_frame(
        query="parser",
        repository_id="lethyx",
        revision_id="abc",
        uncertainty=0.7,
    )
    assert recall_activation_decision(previous, uncertainty_spike).reason == "uncertainty_spike"


def test_caller_supplied_logical_ids_remain_strict(tmp_path: Any) -> None:
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=object())

    plan = client.compile_recall_plan(
        "celestial cobalt orchard",
        logical_block_ids=["strict-leaf"],
    )

    assert plan is not None
    assert plan.logical_block_ids == ("strict-leaf",)
    assert plan.reason == "explicit_block_first_recall"
    assert "routing_trace" not in plan.metadata
