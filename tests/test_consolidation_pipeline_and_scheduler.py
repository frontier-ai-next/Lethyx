from dataclasses import FrozenInstanceError, replace

import pytest

from lethyx.contracts import (
    ConsolidationDecision,
    EvidenceRef,
    MemoryBlockRecord,
    MemoryTransitionAssessment,
)
from lethyx.memory.consolidation import (
    CONSOLIDATION_TRIGGERS,
    CanonicalConsolidationPipeline,
    ConsolidationScheduler,
    ConsolidationSignal,
)


def _evidence(ref_id: str = "phase-7") -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="test_run",
            ref_id=ref_id,
            produced_by="verifier",
            producer_id="verifier:phase-7",
        ),
    )


def _record(block_id: str, *, workspace_id: str = "ws") -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        canonical_key=f"episode:{block_id}",
        revision_id=f"{block_id}@1",
        content=f"canonical episode {block_id}",
        workspace_id=workspace_id,
        run_id="run-1",
        task_id="task-1",
        kind="episode",
        evidence_refs=_evidence(block_id),
    )


def _decision(**changes: object) -> ConsolidationDecision:
    values: dict[str, object] = {
        "decision_id": "decision-1",
        "workspace_id": "ws",
        "action": "consolidate",
        "source_episode_ids": ("episode-a", "episode-b"),
        "grouping_key": "release-parser",
        "task_family": "release",
        "trigger": "verified_reuse_harm",
        "expected_applicability": "same parser release workflow",
        "risk": "medium",
        "cost_budget_tokens": 512,
        "rationale": "Two verified outcomes justify a bounded candidate view.",
        "evidence_refs": _evidence(),
    }
    values.update(changes)
    return ConsolidationDecision(**values)  # type: ignore[arg-type]


def test_consolidation_decision_is_complete_proposed_and_never_forced_per_turn() -> None:
    decision = _decision()

    assert decision.source_episode_ids == ("episode-a", "episode-b")
    assert decision.grouping_key == "release-parser"
    assert decision.task_family == "release"
    assert decision.trigger == "verified_reuse_harm"
    assert decision.expected_applicability == "same parser release workflow"
    assert decision.risk == "medium"
    assert decision.cost_budget_tokens == 512
    assert decision.status == "proposed"
    assert decision.forced_rewrite is False
    assert decision.host_action_authorized is False


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"source_episode_ids": ()}, "source_episode_ids"),
        ({"trigger": "turn_completed"}, "consolidation trigger"),
        ({"cost_budget_tokens": 0}, "cost_budget_tokens"),
        ({"rationale": ""}, "rationale"),
        ({"status": "committed"}, "must start proposed"),
    ),
)
def test_consolidation_decision_rejects_incomplete_or_implicit_rewrite_inputs(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _decision(**changes)


def test_canonical_pipeline_builds_immutable_reproducible_transition_and_assessment() -> None:
    pipeline = CanonicalConsolidationPipeline()
    records = (_record("episode-b"), _record("episode-a"))
    kwargs = {
        "new_evidence_refs": _evidence("new-outcome"),
        "proposed_operations": ("create_materialized_view",),
        "touched_view_ids": ("view-release-parser",),
        "post_state_candidate": {
            "view_id": "view-release-parser",
            "source_episode_ids": ["episode-a", "episode-b"],
            "authority": "advisory",
        },
        "source_coverage_after": ("episode-a", "episode-b"),
        "policy_version": "consolidation-policy@1",
        "model_version": "deterministic@1",
        "prompt_version": "none@1",
    }

    first = pipeline.propose(_decision(), source_records=records, **kwargs)
    second = pipeline.propose(_decision(), source_records=tuple(reversed(records)), **kwargs)
    assessment = pipeline.assess(
        first,
        assessment_id="assessment-1",
        verdict="defer",
        reasons=("tribunal is owned by a later slice",),
        evidence_refs=_evidence("assessment"),
    )

    assert first == second
    assert first.touched_record_ids == ("episode-a", "episode-b")
    assert first.touched_view_ids == ("view-release-parser",)
    assert first.pre_state_snapshot_sha256.startswith("sha256:")
    assert first.post_state_candidate_sha256.startswith("sha256:")
    assert first.source_coverage_diff.lost_source_ids == ()
    assert first.policy_version == "consolidation-policy@1"
    assert isinstance(assessment, MemoryTransitionAssessment)
    assert assessment.transition_id == first.transition_id
    assert assessment.verdict == "defer"
    assert assessment.canonical_mutation_applied is False
    assert assessment.host_action_authorized is False
    with pytest.raises(FrozenInstanceError):
        first.policy_version = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="does not match"):
        replace(first, post_state_candidate_sha256=f"sha256:{'0' * 64}")


@pytest.mark.parametrize(
    ("records", "message"),
    (
        ((object(),), "canonical MemoryBlockRecord"),
        ((_record("episode-a"),), "source episode set"),
        (
            (_record("episode-a"), _record("episode-b", workspace_id="other")),
            "workspace",
        ),
    ),
)
def test_canonical_pipeline_rejects_noncanonical_mixed_or_incomplete_sources(
    records: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CanonicalConsolidationPipeline().propose(
            _decision(),
            source_records=records,
            new_evidence_refs=_evidence("new"),
            proposed_operations=("retain_sources",),
            touched_view_ids=(),
            post_state_candidate={"action": "retain"},
            source_coverage_after=("episode-a", "episode-b"),
            policy_version="policy@1",
            model_version="deterministic@1",
            prompt_version="none@1",
        )


def test_scheduler_exposes_every_independent_trigger_without_recurrence_gate() -> None:
    signals = tuple(
        ConsolidationSignal(
            workspace_id="ws",
            source_episode_ids=(f"episode-{index}",),
            grouping_key=f"group-{index}",
            task_family="release",
            trigger=trigger,
            proposed_action="defer",
            expected_applicability="same release family",
            risk="medium",
            cost_budget_tokens=128,
            rationale=f"independent {trigger} signal",
            evidence_refs=_evidence(trigger),
        )
        for index, trigger in enumerate(sorted(CONSOLIDATION_TRIGGERS), start=1)
    )

    decisions = ConsolidationScheduler().schedule(signals)

    assert {decision.trigger for decision in decisions} == CONSOLIDATION_TRIGGERS
    assert len(decisions) == len(CONSOLIDATION_TRIGGERS)
    assert all(decision.status == "proposed" for decision in decisions)
    assert all(decision.forced_rewrite is False for decision in decisions)
    assert all(decision.host_action_authorized is False for decision in decisions)
    assert "turn_completed" not in CONSOLIDATION_TRIGGERS


def test_scheduler_is_deterministic_deduplicated_and_does_not_execute() -> None:
    signal = ConsolidationSignal(
        workspace_id="ws",
        source_episode_ids=("episode-a",),
        grouping_key="release",
        task_family="release",
        trigger="correction_conflict",
        proposed_action="split",
        expected_applicability="current workspace conflict",
        risk="high",
        cost_budget_tokens=256,
        rationale="Preserve both sides until conflict adjudication.",
        evidence_refs=_evidence("conflict"),
    )
    scheduler = ConsolidationScheduler()

    first = scheduler.schedule((signal, signal))
    second = scheduler.schedule((signal,))

    assert first == second
    assert len(first) == 1
    assert first[0].action == "split"
    assert first[0].host_action_authorized is False
    assert scheduler.schedule(()) == ()
