"""Recall-frame compilation and cheap mid-task activation decisions."""

from dataclasses import dataclass

from lethyx.contracts import RecallFrame, RecallTemporalMode, RecallWideningPolicy


_STRATEGY_TRANSITIONS = frozenset(
    {
        "compaction",
        "component_switch",
        "context_fold",
        "high_risk_boundary",
        "page_back",
        "plan_revision",
        "prospective_trigger",
        "repository_switch",
        "session_boundary",
        "source_change",
        "subgoal_transition",
        "turn_completion",
        "test_failure",
        "tool_failure",
        "tool_result",
        "uncertainty_spike",
        "verification_result",
        "verification_success",
    }
)


@dataclass(frozen=True)
class RecallActivationDecision:
    """Content-free decision to refresh advisory recall for a live frame."""

    should_refresh: bool
    reason: str
    strategy_refresh: bool
    widening_allowed: bool


def compile_recall_frame(
    query: str,
    *,
    workspace_id: str = "",
    run_id: str = "",
    task_id: str = "",
    subtask_id: str = "",
    episode_id: str = "",
    scope_chain: tuple[str, ...] = (),
    transition_reason: str = "",
    repository_id: str = "",
    worktree: str = "",
    revision_id: str = "",
    dirty_state: str = "unknown",
    actor_id: str = "",
    speaker_id: str = "",
    belief_holder_id: str = "",
    relevant_entities: tuple[str, ...] = (),
    relevant_paths: tuple[str, ...] = (),
    relevant_symbols: tuple[str, ...] = (),
    relevant_configs: tuple[str, ...] = (),
    event_time: str = "",
    mention_time: str = "",
    observation_time: str = "",
    query_time: str = "",
    tool_name: str = "",
    tool_state: str = "",
    environment_state: tuple[str, ...] = (),
    desired_temporal_mode: RecallTemporalMode = "current",
    required_evidence_roles: tuple[str, ...] = (),
    expected_source_types: tuple[str, ...] = (),
    uncertainty: float | None = None,
    coverage: float | None = None,
    widening_policy: RecallWideningPolicy = "backend_only",
) -> RecallFrame:
    """Compile whitelisted live state into a soft, always-widenable recall hypothesis."""

    soft_values: dict[str, object] = {
        "task_id": task_id,
        "subtask_id": subtask_id,
        "episode_id": episode_id,
        "scope_chain": scope_chain,
        "transition_reason": transition_reason,
        "repository_id": repository_id,
        "worktree": worktree,
        "revision_id": revision_id,
        "dirty_state": dirty_state if dirty_state != "unknown" else "",
        "actor_id": actor_id,
        "speaker_id": speaker_id,
        "belief_holder_id": belief_holder_id,
        "relevant_entities": relevant_entities,
        "relevant_paths": relevant_paths,
        "relevant_symbols": relevant_symbols,
        "relevant_configs": relevant_configs,
        "event_time": event_time,
        "mention_time": mention_time,
        "observation_time": observation_time,
        "query_time": query_time,
        "tool_name": tool_name,
        "tool_state": tool_state,
        "environment_state": environment_state,
        "desired_temporal_mode": desired_temporal_mode,
        "required_evidence_roles": required_evidence_roles,
        "expected_source_types": expected_source_types,
        "uncertainty": uncertainty,
        "coverage": coverage,
        "widening_policy": widening_policy,
        "widening_allowed": True,
    }
    auto_derived_fields = tuple(
        field_name
        for field_name, value in soft_values.items()
        if value is not None and value != "" and value != () and value is not False
    )
    return RecallFrame(
        query=query,
        workspace_id=workspace_id,
        run_id=run_id,
        task_id=task_id,
        subtask_id=subtask_id,
        episode_id=episode_id,
        scope_chain=scope_chain,
        transition_reason=transition_reason,
        repository_id=repository_id,
        worktree=worktree,
        revision_id=revision_id,
        dirty_state=dirty_state,
        actor_id=actor_id,
        speaker_id=speaker_id,
        belief_holder_id=belief_holder_id,
        relevant_entities=relevant_entities,
        relevant_paths=relevant_paths,
        relevant_symbols=relevant_symbols,
        relevant_configs=relevant_configs,
        event_time=event_time,
        mention_time=mention_time,
        observation_time=observation_time,
        query_time=query_time,
        tool_name=tool_name,
        tool_state=tool_state,
        environment_state=environment_state,
        desired_temporal_mode=desired_temporal_mode,
        required_evidence_roles=required_evidence_roles,
        expected_source_types=expected_source_types,
        uncertainty=1.0 if uncertainty is None else uncertainty,
        coverage=0.0 if coverage is None else coverage,
        widening_policy=widening_policy,
        widening_allowed=True,
        auto_derived_fields=auto_derived_fields,
    )


def recall_activation_decision(
    previous: RecallFrame | None,
    current: RecallFrame,
) -> RecallActivationDecision:
    """Apply a deterministic state-delta gate before retrieval or strategy refresh."""

    if previous is None:
        return RecallActivationDecision(True, "initial_frame", True, current.widening_allowed)
    if previous.query != current.query:
        return RecallActivationDecision(True, "query_change", True, current.widening_allowed)
    if (
        current.transition_reason
        and current.transition_reason != previous.transition_reason
        and current.transition_reason in _STRATEGY_TRANSITIONS
    ):
        return RecallActivationDecision(
            True,
            current.transition_reason,
            True,
            current.widening_allowed,
        )
    if (previous.task_id, previous.subtask_id, previous.episode_id) != (
        current.task_id,
        current.subtask_id,
        current.episode_id,
    ):
        return RecallActivationDecision(True, "subgoal_transition", True, current.widening_allowed)
    if (previous.repository_id, previous.worktree) != (
        current.repository_id,
        current.worktree,
    ):
        return RecallActivationDecision(True, "repository_switch", True, current.widening_allowed)
    if (
        previous.revision_id,
        previous.dirty_state,
        previous.relevant_paths,
        previous.relevant_symbols,
        previous.relevant_configs,
    ) != (
        current.revision_id,
        current.dirty_state,
        current.relevant_paths,
        current.relevant_symbols,
        current.relevant_configs,
    ):
        return RecallActivationDecision(True, "source_change", True, current.widening_allowed)
    if (previous.tool_name, previous.tool_state, previous.environment_state) != (
        current.tool_name,
        current.tool_state,
        current.environment_state,
    ):
        reason = "tool_failure" if "fail" in current.tool_state.casefold() else "tool_result"
        return RecallActivationDecision(True, reason, True, current.widening_allowed)
    if (
        current.uncertainty - previous.uncertainty >= 0.15
        or previous.coverage - current.coverage >= 0.15
    ):
        return RecallActivationDecision(True, "uncertainty_spike", True, current.widening_allowed)
    if (
        previous.desired_temporal_mode,
        previous.required_evidence_roles,
        previous.expected_source_types,
    ) != (
        current.desired_temporal_mode,
        current.required_evidence_roles,
        current.expected_source_types,
    ):
        return RecallActivationDecision(True, "strategy_change", True, current.widening_allowed)
    if previous != current:
        return RecallActivationDecision(True, "frame_state_change", True, current.widening_allowed)
    return RecallActivationDecision(False, "no_material_change", False, current.widening_allowed)
