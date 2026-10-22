from itertools import permutations

from um_mem.contracts import EvidenceRef
from um_mem.memory.consolidation import (
    ConsolidationExitMetrics,
    ConsolidationInputGate,
    ConsolidationScheduler,
    ConsolidationSignal,
)


def _evidence(name: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="verification_report",
            ref_id=name,
            produced_by="verifier",
            producer_id="verifier:phase-7-exit",
        ),
    )


def _signal(name: str) -> ConsolidationSignal:
    return ConsolidationSignal(
        workspace_id="ws",
        source_episode_ids=(f"episode-{name}",),
        grouping_key="release",
        task_family="release",
        trigger="capacity_pressure",
        proposed_action="consolidate",
        expected_applicability="release",
        risk="medium",
        cost_budget_tokens=128,
        rationale=f"consolidate {name}",
        evidence_refs=_evidence(name),
        cluster_key="release-cluster",
        expected_value=0.5,
        age_seconds=100,
        estimated_cost_tokens=32,
    )


def test_episodic_fallback_is_safe_and_heterogeneous_streamed_default_off() -> None:
    gate = ConsolidationInputGate()

    episodic = gate.evaluate(input_mode="episodic_only", preservation_qualified=False)
    heterogeneous = gate.evaluate(
        input_mode="heterogeneous",
        preservation_qualified=False,
    )
    streamed = gate.evaluate(input_mode="streamed", preservation_qualified=False)
    qualified = gate.evaluate(input_mode="streamed", preservation_qualified=True)

    assert episodic.consolidation_allowed is True
    assert heterogeneous.consolidation_allowed is False
    assert streamed.consolidation_allowed is False
    assert heterogeneous.fallback_action == "retain"
    assert streamed.shadow_only is True
    assert qualified.consolidation_allowed is True


def test_schedule_is_invariant_under_ingestion_reordering() -> None:
    signals = (_signal("a"), _signal("b"), _signal("c"))
    scheduler = ConsolidationScheduler()
    schedules = [
        scheduler.build_schedule(tuple(ordering), window="release-window")
        for ordering in permutations(signals)
    ]

    assert len({schedule.input_set_fingerprint_sha256 for schedule in schedules}) == 1
    assert (
        len(
            {
                tuple(entry.decision.decision_id for entry in schedule.entries)
                for schedule in schedules
            }
        )
        == 1
    )


def test_consolidated_bank_must_be_smaller_with_full_raw_coverage() -> None:
    passed = ConsolidationExitMetrics(
        raw_record_ids=("raw-1", "raw-2", "raw-3"),
        active_view_ids=("view-1", "view-2"),
        covered_raw_record_ids=("raw-1", "raw-2", "raw-3"),
        lost_valid_fact_ids=(),
        unsupported_addition_ids=(),
    ).evaluate()
    assert passed.passed is True

    no_reduction = ConsolidationExitMetrics(
        raw_record_ids=("raw-1", "raw-2"),
        active_view_ids=("view-1", "view-2"),
        covered_raw_record_ids=("raw-1", "raw-2"),
        lost_valid_fact_ids=(),
        unsupported_addition_ids=(),
    ).evaluate()
    lost_fact = ConsolidationExitMetrics(
        raw_record_ids=("raw-1", "raw-2", "raw-3"),
        active_view_ids=("view-1",),
        covered_raw_record_ids=("raw-1", "raw-2"),
        lost_valid_fact_ids=("fact-3",),
        unsupported_addition_ids=(),
    ).evaluate()
    assert no_reduction.passed is False
    assert "active_bank_not_smaller" in no_reduction.reasons
    assert lost_fact.passed is False
    assert "raw_coverage_incomplete" in lost_fact.reasons
    assert "lost_valid_fact" in lost_fact.reasons


def test_exit_metrics_reject_unsupported_additions() -> None:
    result = ConsolidationExitMetrics(
        raw_record_ids=("raw-1", "raw-2"),
        active_view_ids=("view-1",),
        covered_raw_record_ids=("raw-1", "raw-2"),
        lost_valid_fact_ids=(),
        unsupported_addition_ids=("hallucination-1",),
    ).evaluate()
    assert result.passed is False
    assert "unsupported_addition" in result.reasons
