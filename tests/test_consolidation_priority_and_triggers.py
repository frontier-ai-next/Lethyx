from itertools import permutations

import pytest

from lethyx.contracts import EvidenceRef, MemoryBlockRecord
from lethyx.memory.consolidation import (
    CONSOLIDATION_TRIGGERS,
    AtomicMemoryClaim,
    CanonicalConsolidationPipeline,
    ConsolidationScheduler,
    ConsolidationSignal,
    MaterializedMemoryView,
)


def _evidence(ref_id: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="test_run",
            ref_id=ref_id,
            produced_by="verifier",
            producer_id="verifier:phase-7-priority",
        ),
    )


def _signal(
    name: str,
    *,
    trigger: str = "capacity_pressure",
    risk: str = "low",
    rare_critical: bool = False,
    expected_value: float = 0.2,
    age_seconds: int = 1,
    estimated_cost_tokens: int = 32,
    dependency_ids: tuple[str, ...] = (),
    dependencies_closed: bool = True,
    recurrence_count: int = 2,
) -> ConsolidationSignal:
    return ConsolidationSignal(
        workspace_id="ws",
        source_episode_ids=(f"episode-{name}",),
        grouping_key=f"group-{name}",
        cluster_key=f"cluster-{name}",
        task_family="release",
        trigger=trigger,  # type: ignore[arg-type]
        proposed_action="defer",
        expected_applicability="same release family",
        risk=risk,  # type: ignore[arg-type]
        cost_budget_tokens=256,
        rationale=f"schedule {name}",
        evidence_refs=_evidence(name),
        rare_critical=rare_critical,
        expected_value=expected_value,
        age_seconds=age_seconds,
        estimated_cost_tokens=estimated_cost_tokens,
        dependency_ids=dependency_ids,
        dependencies_closed=dependencies_closed,
        recurrence_count=recurrence_count,
    )


def _record(block_id: str) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        canonical_key=f"episode:{block_id}",
        revision_id=f"{block_id}@1",
        content=f"raw episode {block_id}",
        workspace_id="ws",
        kind="episode",
        authority="advisory",
        evidence_refs=_evidence(block_id),
    )


def test_priority_considers_rare_risk_value_age_cost_and_not_fifo() -> None:
    fifo_first = _signal(
        "fifo-first",
        risk="low",
        expected_value=0.1,
        age_seconds=10,
        estimated_cost_tokens=128,
    )
    important_later = _signal(
        "important-later",
        risk="critical",
        rare_critical=True,
        expected_value=0.9,
        age_seconds=100,
        estimated_cost_tokens=16,
    )

    schedule = ConsolidationScheduler().build_schedule(
        (fifo_first, important_later),
        window="2026-07-15T02:00:00Z/2026-07-15T03:00:00Z",
    )

    assert tuple(entry.signal.source_episode_ids[0] for entry in schedule.entries) == (
        "episode-important-later",
        "episode-fifo-first",
    )
    assert schedule.entries[0].order == 1
    assert schedule.entries[0].rare_critical is True
    assert schedule.entries[0].priority_score > schedule.entries[1].priority_score


def test_dependency_closure_defers_and_rare_critical_bypasses_recurrence_threshold() -> None:
    blocked = _signal(
        "blocked",
        risk="critical",
        rare_critical=True,
        dependency_ids=("decision-parent",),
        dependencies_closed=False,
    )
    ordinary_single = _signal(
        "ordinary-single",
        trigger="recurrence",
        recurrence_count=1,
    )
    rare_single = _signal(
        "rare-single",
        trigger="recurrence",
        rare_critical=True,
        recurrence_count=1,
    )

    schedule = ConsolidationScheduler().build_schedule(
        (blocked, ordinary_single, rare_single),
        window="window-1",
    )
    by_episode = {entry.signal.source_episode_ids[0]: entry for entry in schedule.entries}

    assert by_episode["episode-blocked"].disposition == "deferred"
    assert by_episode["episode-blocked"].reason == "dependency_closure_incomplete"
    assert by_episode["episode-ordinary-single"].disposition == "deferred"
    assert by_episode["episode-ordinary-single"].reason == "recurrence_threshold_not_met"
    assert by_episode["episode-rare-single"].disposition == "scheduled"
    assert by_episode["episode-rare-single"].reason == "rare_critical_bypass"


def test_schedule_records_window_order_grouping_cluster_and_stable_input_fingerprint() -> None:
    signals = (
        _signal("a", risk="high"),
        _signal("b", expected_value=0.8),
        _signal("c", age_seconds=500),
    )
    scheduler = ConsolidationScheduler()
    baseline = scheduler.build_schedule(signals, window="bounded-window")

    for reordered in permutations(signals):
        assert scheduler.build_schedule(reordered, window="bounded-window") == baseline

    assert baseline.window == "bounded-window"
    assert baseline.input_set_fingerprint_sha256.startswith("sha256:")
    assert baseline.grouping_keys == ("group-a", "group-b", "group-c")
    assert baseline.cluster_keys == ("cluster-a", "cluster-b", "cluster-c")
    assert tuple(entry.order for entry in baseline.entries) == (1, 2, 3)
    assert scheduler.build_schedule((*signals, signals[0]), window="bounded-window") == baseline


def test_exact_phase_seven_event_triggers_are_typed_and_independently_schedulable() -> None:
    triggers = {
        "session_end",
        "task_end",
        "token_pressure",
        "duplicate_density",
        "source_change",
        "unresolved_conflict",
        "repeated_verified_reuse",
    }
    assert triggers <= CONSOLIDATION_TRIGGERS

    decisions = ConsolidationScheduler().schedule(
        tuple(_signal(trigger, trigger=trigger) for trigger in sorted(triggers))
    )

    assert {decision.trigger for decision in decisions} == triggers
    assert all(decision.host_action_authorized is False for decision in decisions)


def test_raw_to_atomic_claim_to_materialized_view_has_transitive_provenance_dag() -> None:
    records = (_record("raw-b"), _record("raw-a"))
    claims = (
        AtomicMemoryClaim(
            claim_id="claim-parser",
            workspace_id="ws",
            content="The release parser is verified.",
            source_record_ids=("raw-a",),
            evidence_refs=_evidence("claim-parser"),
        ),
        AtomicMemoryClaim(
            claim_id="claim-rollback",
            workspace_id="ws",
            content="Rollback remains available.",
            source_record_ids=("raw-b",),
            evidence_refs=_evidence("claim-rollback"),
        ),
    )
    view = MaterializedMemoryView(
        view_id="view-release",
        workspace_id="ws",
        content="Verified release parser with rollback.",
        claim_ids=("claim-parser", "claim-rollback"),
        transitive_raw_record_ids=("raw-a", "raw-b"),
    )

    dag = CanonicalConsolidationPipeline().build_provenance_dag(
        source_records=records,
        atomic_claims=claims,
        materialized_views=(view,),
    )
    reordered = CanonicalConsolidationPipeline().build_provenance_dag(
        source_records=tuple(reversed(records)),
        atomic_claims=tuple(reversed(claims)),
        materialized_views=(view,),
    )

    assert dag == reordered
    assert dag.raw_record_ids == ("raw-a", "raw-b")
    assert dag.claim_ids == ("claim-parser", "claim-rollback")
    assert dag.view_ids == ("view-release",)
    assert set(dag.edges) == {
        ("raw-a", "claim-parser", "supports_claim"),
        ("raw-b", "claim-rollback", "supports_claim"),
        ("claim-parser", "view-release", "materializes_view"),
        ("claim-rollback", "view-release", "materializes_view"),
    }
    assert dag.canonical_mutation_applied is False
    assert dag.host_action_authorized is False
    assert all(claim.authority == "advisory" for claim in dag.atomic_claims)
    assert all(candidate.status == "candidate" for candidate in dag.materialized_views)


def test_provenance_dag_rejects_missing_raw_or_claim_coverage() -> None:
    record = _record("raw-a")
    foreign_claim = AtomicMemoryClaim(
        claim_id="claim-foreign",
        workspace_id="ws",
        content="Unsupported claim.",
        source_record_ids=("raw-missing",),
        evidence_refs=_evidence("claim-foreign"),
    )
    with pytest.raises(ValueError, match="unknown raw source"):
        CanonicalConsolidationPipeline().build_provenance_dag(
            source_records=(record,),
            atomic_claims=(foreign_claim,),
            materialized_views=(),
        )

    claim = AtomicMemoryClaim(
        claim_id="claim-a",
        workspace_id="ws",
        content="Supported claim.",
        source_record_ids=("raw-a",),
        evidence_refs=_evidence("claim-a"),
    )
    incomplete_view = MaterializedMemoryView(
        view_id="view-incomplete",
        workspace_id="ws",
        content="Broken provenance.",
        claim_ids=("claim-a",),
        transitive_raw_record_ids=("raw-missing",),
    )
    with pytest.raises(ValueError, match="transitive raw coverage"):
        CanonicalConsolidationPipeline().build_provenance_dag(
            source_records=(record,),
            atomic_claims=(claim,),
            materialized_views=(incomplete_view,),
        )
