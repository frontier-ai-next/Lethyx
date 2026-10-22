import pytest

from um_mem.contracts import EvidenceRef, MemoryBlockRecord, RecallFrame, RecordBlockMembership
from um_mem.memory.backends.contracts import EdgeType, MemoryEdgeRecord, MemoryIndexNode
from um_mem.memory.deep_reader import (
    BlockLocalEvidenceRegion,
    read_block_local_evidence_region,
)
from um_mem.memory.operational_slo import RecallBudget
from um_mem.memory.recall_plan import RecallPlan
from um_mem.memory.recall_strategy import compile_reconstructive_strategy
from um_mem.memory.reconstruction_qualification import (
    ReconstructionQualificationCase,
    qualify_reconstructive_product,
)
from um_mem.memory.reconstructive_recall import (
    ReconstructionObservation,
    decide_reconstruction_escalation,
    pin_recall_frame,
    run_reconstructive_lite,
)


def _source(ref_id: str) -> EvidenceRef:
    return EvidenceRef(
        ref_type="source_span",
        ref_id=ref_id,
        produced_by="agent",
        producer_id="capture",
    )


def _qualification():
    exact = ReconstructionQualificationCase(
        case_id="exact-one-hop",
        declared_multi_hop=False,
        exact_lookup=True,
        one_hop_coverage=1.0,
        reconstruction_coverage=1.0,
        one_hop_latency_ms=3,
        reconstruction_latency_ms=0,
        latency_budget_ms=50,
        one_hop_tokens=12,
        reconstruction_tokens=0,
        token_budget=100,
        reconstruction_steps=0,
    )
    multi_hop = ReconstructionQualificationCase(
        case_id="declared-multi-hop",
        declared_multi_hop=True,
        one_hop_coverage=0.5,
        reconstruction_coverage=0.9,
        one_hop_latency_ms=5,
        reconstruction_latency_ms=20,
        latency_budget_ms=50,
        one_hop_tokens=20,
        reconstruction_tokens=80,
        token_budget=100,
        reconstruction_steps=2,
    )
    return qualify_reconstructive_product(
        (exact, multi_hop),
        evidence_revision="sha256:" + "c" * 64,
    )


def _plan_and_frame(
    query: str,
    *,
    query_type: str,
    episode_id: str = "episode-current",
    coverage: float = 0.2,
):
    frame = RecallFrame(
        query=query,
        workspace_id="alpha",
        run_id="run-1",
        episode_id=episode_id,
        scope_chain=("workspace:alpha",),
        coverage=coverage,
    )
    pinned = pin_recall_frame(
        frame,
        canonical_snapshot_revision="sha256:" + "a" * 64,
        catalog_revision="catalog:1",
        policy_revision="policy:1",
    )
    plan = RecallPlan(
        query=query,
        query_type=query_type,
        workspace_id="alpha",
        run_id="run-1",
        scope_chain=("workspace:alpha",),
        limit=8,
    )
    return plan, pinned


def _nodes() -> tuple[MemoryIndexNode, ...]:
    return (
        MemoryIndexNode(
            node_id="cue",
            title="Planner release procedure",
            node_type="observation",
            workspace_id="alpha",
            metadata={"record_id": "record-cue", "episode_id": "episode-current"},
        ),
        MemoryIndexNode(
            node_id="decision",
            title="Planner release decision",
            node_type="decision",
            workspace_id="alpha",
            metadata={
                "record_id": "record-decision",
                "kind": "decision",
                "episode_id": "episode-current",
            },
        ),
    )


def test_exit_exact_lookup_has_zero_reconstruction_and_policy_cost() -> None:
    plan, pinned = _plan_and_frame("exact record-cue", query_type="exact_lookup")
    compiled = compile_reconstructive_strategy(plan, frame=pinned, index_nodes=_nodes())
    decision = decide_reconstruction_escalation(
        plan,
        frame=pinned,
        product_qualification=_qualification(),
    )

    def forbidden(*_args):
        raise AssertionError("exact lookup invoked reconstruction")

    result = run_reconstructive_lite(
        compiled,
        escalation=decision,
        executor=forbidden,
        operator_policy=forbidden,
    )

    assert result.evidence_ids == ()
    assert result.trace.steps == ()
    assert result.trace.total_tokens == 0
    assert result.trace.total_latency_ms == 0


def test_exit_declared_multi_hop_qualification_has_no_budget_or_safety_regression() -> None:
    result = _qualification()

    assert result.qualified is True
    assert result.safe_one_hop_fallback is True
    assert result.declared_multi_hop_cases == 1
    assert result.reasons == ("qualified_against_safe_one_hop_fallback",)


def test_exit_deep_output_has_zero_directive_contamination_or_scope_leakage() -> None:
    primary = MemoryBlockRecord(
        block_id="primary",
        content="current workspace evidence",
        workspace_id="alpha",
        source_backend="mempalace",
        block_membership=RecordBlockMembership(
            primary_block_id="decisions",
            catalog_revision_id="catalog:test",
        ),
    )
    source = MemoryBlockRecord(
        block_id="source",
        content="canonical source span",
        workspace_id="alpha",
        source_backend="mempalace",
        block_membership=RecordBlockMembership(
            primary_block_id="raw_events",
            catalog_revision_id="catalog:test",
        ),
    )
    region = BlockLocalEvidenceRegion(
        backend_id="mempalace",
        workspace_id="alpha",
        logical_block_id="decisions",
        primary_record_ids=("primary",),
        records=(primary, source),
    )

    result = read_block_local_evidence_region(
        region,
        reverse_provenance_sources=lambda _record_id: ("source",),
    )

    assert result.final_output == "evidence"
    assert all(item.authority == "advisory" for item in result.items)
    assert all(item.surface == "supplemental_evidence" for item in result.items)
    assert "directive" not in str(result.to_mapping()).casefold()
    foreign = MemoryBlockRecord(
        block_id="foreign",
        content="foreign secret",
        workspace_id="beta",
        source_backend="mempalace",
        block_membership=RecordBlockMembership(
            primary_block_id="decisions",
            catalog_revision_id="catalog:test",
        ),
    )
    with pytest.raises(ValueError, match="foreign workspace"):
        BlockLocalEvidenceRegion(
            backend_id="mempalace",
            workspace_id="alpha",
            logical_block_id="decisions",
            primary_record_ids=("primary",),
            records=(primary, foreign),
        )


def test_exit_trace_exposes_measured_benefit_for_every_executed_step() -> None:
    plan, pinned = _plan_and_frame("compare Planner release decisions", query_type="synthesis")
    compiled = compile_reconstructive_strategy(plan, frame=pinned, index_nodes=_nodes())
    decision = decide_reconstruction_escalation(
        plan,
        frame=pinned,
        product_qualification=_qualification(),
    )

    def executor(step, _budget):
        return ReconstructionObservation(
            (f"evidence-{step.operator}",),
            5,
            5,
            min(0.95, 0.2 + 0.4 * step.ordinal),
        )

    result = run_reconstructive_lite(
        compiled,
        escalation=decision,
        executor=executor,
        budget=RecallBudget(100, 2, 100, 16),
    )

    assert result.trace.steps
    assert all(step.useful for step in result.trace.steps)
    assert all(
        step.coverage_after > step.coverage_before or step.evidence_ids
        for step in result.trace.steps
    )


def test_cue_step_delegates_semantic_query_to_backend_without_index_selection() -> None:
    query = "Planner release procedure"
    plan, pinned = _plan_and_frame(query, query_type="synthesis")
    current = MemoryIndexNode(
        node_id="current",
        title=query,
        node_type="observation",
        workspace_id="alpha",
        metadata={"record_id": "current", "episode_id": "episode-current"},
    )
    distractor = MemoryIndexNode(
        node_id="distractor",
        title=query,
        node_type="observation",
        workspace_id="alpha",
        metadata={"record_id": "distractor", "episode_id": "episode-old"},
    )

    unsupported = compile_reconstructive_strategy(
        plan,
        frame=pinned,
        index_nodes=(current, distractor),
    )
    supported = compile_reconstructive_strategy(
        plan,
        frame=pinned,
        index_nodes=(current, distractor),
        edges=(
            MemoryEdgeRecord(
                src_id="distractor",
                dst_id="current",
                edge_type=EdgeType.SUPPORTS,
                workspace_id="alpha",
                evidence_refs=(_source("context-link"),),
            ),
        ),
    )

    assert unsupported.steps[0].candidate_ids == ()
    assert supported.steps[0].candidate_ids == ()
    assert unsupported.steps[0].plan.record_ids == ()
    assert supported.steps[0].plan.record_ids == ()


def test_exit_timeout_returns_safe_abstention_or_prior_partial_evidence() -> None:
    plan, pinned = _plan_and_frame("compare Planner release decisions", query_type="synthesis")
    compiled = compile_reconstructive_strategy(plan, frame=pinned, index_nodes=_nodes())
    decision = decide_reconstruction_escalation(
        plan,
        frame=pinned,
        product_qualification=_qualification(),
    )

    def timeout(_step, _budget):
        raise TimeoutError

    abstained = run_reconstructive_lite(
        compiled,
        escalation=decision,
        executor=timeout,
        budget=RecallBudget(20, 2, 100, 16),
    )
    calls = 0

    def partial_then_timeout(step, _budget):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError
        return ReconstructionObservation((f"evidence-{step.operator}",), 5, 5, 0.5)

    partial = run_reconstructive_lite(
        compiled,
        escalation=decision,
        executor=partial_then_timeout,
        budget=RecallBudget(20, 2, 100, 16),
    )

    assert abstained.trace.disposition == "abstained"
    assert abstained.trace.reason == "operator_timeout"
    assert abstained.evidence_ids == ()
    assert partial.trace.disposition == "partial"
    assert partial.trace.reason == "operator_timeout"
    assert partial.evidence_ids == ("evidence-cue",)
    assert partial.trace.steps[-1].evidence_ids == ()
