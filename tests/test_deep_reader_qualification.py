import pytest

from um_mem.contracts import EvidenceRef, MemoryBlockRecord, RecallFrame, RecordBlockMembership
from um_mem.memory.backends.contracts import MemoryIndexNode
from um_mem.memory.deep_reader import (
    BlockLocalEvidenceRegion,
    read_block_local_evidence_region,
)
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
    reconstruction_relation,
    run_reconstructive_lite,
    select_reconstruction_operator,
)


def _compiled_plan():
    query = "compare Planner procedure decisions"
    frame = RecallFrame(
        query=query,
        workspace_id="alpha",
        run_id="run-1",
        scope_chain=("workspace:alpha",),
        relevant_entities=("Planner",),
        expected_source_types=("procedure",),
        coverage=0.2,
    )
    pinned = pin_recall_frame(
        frame,
        canonical_snapshot_revision="sha256:" + "a" * 64,
        catalog_revision="catalog:1",
        policy_revision="policy:1",
    )
    base = RecallPlan(
        query=query,
        query_type="synthesis",
        workspace_id="alpha",
        run_id="run-1",
        scope_chain=("workspace:alpha",),
        limit=8,
    )
    nodes = tuple(
        MemoryIndexNode(
            node_id=f"node-{index}",
            title=f"Planner procedure decision {index}",
            node_type=kind,
            workspace_id="alpha",
            metadata={"record_id": f"record-{index}", "kind": kind},
        )
        for index, kind in enumerate(("observation", "decision", "procedure"), start=1)
    )
    return base, pinned, compile_reconstructive_strategy(base, frame=pinned, index_nodes=nodes)


def _case(
    case_id: str = "multi-hop",
    *,
    coverage: float = 0.9,
    safety_violations: int = 0,
    latency_ms: float = 20,
    tokens: int = 80,
) -> ReconstructionQualificationCase:
    return ReconstructionQualificationCase(
        case_id=case_id,
        declared_multi_hop=True,
        one_hop_coverage=0.5,
        reconstruction_coverage=coverage,
        one_hop_latency_ms=5,
        reconstruction_latency_ms=latency_ms,
        latency_budget_ms=50,
        one_hop_tokens=20,
        reconstruction_tokens=tokens,
        token_budget=100,
        reconstruction_safety_violations=safety_violations,
        reconstruction_steps=2,
    )


def _qualification(*cases: ReconstructionQualificationCase):
    return qualify_reconstructive_product(
        tuple(cases or (_case(),)),
        evidence_revision="sha256:" + "b" * 64,
    )


def _evidence(ref_id: str, *, verifier: bool = False) -> EvidenceRef:
    return EvidenceRef(
        ref_type="verification_report" if verifier else "source_span",
        ref_id=ref_id,
        produced_by="verifier" if verifier else "agent",
        producer_id="qa" if verifier else "capture",
    )


def test_optional_policy_can_only_select_an_allowed_compiled_operator() -> None:
    _, _, compiled = _compiled_plan()
    seen_requests = []

    def malicious_policy(request):
        seen_requests.append(request)
        return "promote:procedure"

    rejected = select_reconstruction_operator(
        compiled,
        trigger_reasons=("synthesis",),
        policy=malicious_policy,
    )
    selected = select_reconstruction_operator(
        compiled,
        trigger_reasons=("synthesis",),
        policy=lambda _request: "procedure",
    )

    assert rejected.source == "deterministic"
    assert rejected.step == compiled.steps[0]
    assert rejected.rejected_policy_value == "promote:procedure"
    assert selected.source == "llm_policy"
    assert selected.step is not None and selected.step.operator == "procedure"
    assert set(vars(seen_requests[0])) == {
        "frame_revision",
        "allowed_operators",
        "attempted_operators",
        "trigger_reasons",
    }
    assert rejected.authority_change_allowed is False
    assert rejected.scope_change_allowed is False
    assert rejected.promotion_allowed is False


def test_runner_uses_policy_only_for_operator_order() -> None:
    base, pinned, compiled = _compiled_plan()
    decision = decide_reconstruction_escalation(
        base,
        frame=pinned,
        product_qualification=_qualification(),
    )

    result = run_reconstructive_lite(
        compiled,
        escalation=decision,
        operator_policy=lambda _request: "procedure",
        executor=lambda step, _budget: ReconstructionObservation(
            (f"evidence-{step.operator}",),
            5,
            5,
            0.95,
        ),
    )

    assert result.trace.steps[0].operator == "procedure"
    assert result.trace.host_actions_allowed is False


def test_block_local_reader_expands_depth_one_raw_spans_as_read_only_evidence() -> None:
    primary = MemoryBlockRecord(
        block_id="derived",
        content="materialized procedure",
        workspace_id="alpha",
        source_backend="mempalace",
        block_membership=RecordBlockMembership(
            primary_block_id="procedures",
            catalog_revision_id="catalog:test",
        ),
    )
    raw = MemoryBlockRecord(
        block_id="raw-event",
        content="canonical raw tool span",
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
        logical_block_id="procedures",
        primary_record_ids=("derived",),
        records=(primary, raw),
    )

    result = read_block_local_evidence_region(
        region,
        reverse_provenance_sources=lambda record_id: (
            ("raw-event",) if record_id == "derived" else ()
        ),
    )

    assert [item.evidence_kind for item in result.items] == [
        "selected_evidence",
        "canonical_raw_span",
    ]
    assert result.items[1].content == raw.content
    assert result.items[1].source_record_id == "derived"
    assert result.backend_path == "selected_backend"
    assert result.final_output == "evidence"
    assert result.write_permission is False
    assert result.promotion_permission is False
    assert all(item.surface == "supplemental_evidence" for item in result.items)


def test_dependency_semantics_require_source_and_verifier_refs() -> None:
    source = (_evidence("source:1"),)
    verifier = (_evidence("verify:1", verifier=True),)
    kinds = (
        reconstruction_relation(
            src_id="a",
            dst_id="b",
            requested_relation=kind,
            source_refs=source,
            verifier_refs=verifier if kind == "verified_dependency" else (),
        ).relation
        for kind in (
            "reported_dependency",
            "temporal_precondition",
            "semantic_association",
            "verified_dependency",
        )
    )

    assert tuple(kinds) == (
        "reported_dependency",
        "temporal_precondition",
        "semantic_association",
        "verified_dependency",
    )
    llm_edge = reconstruction_relation(
        src_id="a",
        dst_id="b",
        requested_relation="causal_effect",
        source_refs=source,
        origin="llm_extracted",
    )
    assert llm_edge.relation == "semantic_association"
    with pytest.raises(ValueError, match="verifier refs"):
        reconstruction_relation(
            src_id="a",
            dst_id="b",
            requested_relation="verified_dependency",
            source_refs=source,
        )


def test_escalation_fails_closed_until_product_qualification_passes() -> None:
    base, pinned, _ = _compiled_plan()
    missing = decide_reconstruction_escalation(base, frame=pinned)
    unsafe = decide_reconstruction_escalation(
        base,
        frame=pinned,
        product_qualification=_qualification(_case(safety_violations=1)),
    )
    over_budget = decide_reconstruction_escalation(
        base,
        frame=pinned,
        product_qualification=_qualification(_case(latency_ms=60, tokens=101)),
    )
    qualified = decide_reconstruction_escalation(
        base,
        frame=pinned,
        product_qualification=_qualification(),
    )

    assert missing.reasons == ("product_not_qualified",)
    assert unsafe.escalate is False
    assert "reconstruction_safety_regression" in _qualification(_case(safety_violations=1)).reasons
    assert over_budget.escalate is False
    assert qualified.escalate is True
    assert qualified.one_hop_fallback_required is True
