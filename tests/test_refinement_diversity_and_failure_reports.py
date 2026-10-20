from dataclasses import replace

import pytest

from lethyx import MemoryClient
from lethyx.contracts import (
    EvidenceRef,
    MemoryDecisionEvent,
    MemoryDecisionTrace,
    MemoryOutcomeReport,
    MemoryUsageEvent,
    OutcomeCreditEvent,
    TransferEnvelope,
    UtilitySelectionCandidate,
)
from lethyx.memory.blocks import RefinementCase
from lethyx.memory.refinement import (
    build_memory_outcome_report,
    evaluate_advisory_association,
    generate_refinement_case,
    procedure_promotion_gate,
    qualify_procedure_reuse,
    select_diverse_utility_candidates,
)


def _evidence(name: str, *, ref_type: str = "test_run") -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type=ref_type,
            ref_id=name,
            produced_by="verifier",
            producer_id=f"verifier:{name}",
        ),
    )


def _decision() -> MemoryDecisionEvent:
    return MemoryDecisionEvent(
        decision_id="decision-report",
        workspace_id="ws",
        run_id="run-report",
        task_signature="repair:python:unit-test",
        action_trace_id="action-report",
        action="recall",
        canonical_snapshot_sha256=f"sha256:{'a' * 64}",
        context_snapshot_sha256=f"sha256:{'b' * 64}",
        live_state_snapshot_sha256=f"sha256:{'c' * 64}",
        live_trigger="verification:failure",
        candidate_ids=("memory-helped", "memory-extra", "memory-hurt"),
        selected_candidate_ids=("memory-helped", "memory-extra", "memory-hurt"),
        policy_revision="policy-1",
        catalog_revision="catalog-1",
        cost_tokens=120,
        cost_latency_ms=9,
        downstream_outcome="verified",
        evidence_refs=_evidence("decision-report"),
    )


def _usage(
    memory_id: str,
    usage_type: str,
    sequence: int,
) -> MemoryUsageEvent:
    return MemoryUsageEvent(
        event_id=f"usage-{memory_id}-{usage_type}",
        decision_id="decision-report",
        memory_id=memory_id,
        workspace_id="ws",
        task_signature="repair:python:unit-test",
        action_trace_id="action-report",
        usage_type=usage_type,
        sequence=sequence,
        evidence_refs=_evidence(f"usage-{sequence}"),
    )


def _credit(
    memory_id: str,
    outcome: str,
    *,
    domain: str,
    decision_id: str = "decision-report",
) -> OutcomeCreditEvent:
    helped = outcome == "helped"
    return OutcomeCreditEvent(
        credit_id=f"credit-{memory_id}-{outcome}-{domain}",
        decision_id=decision_id,
        workspace_id="ws",
        task_signature="repair:python:unit-test",
        action_trace_id="action-report",
        memory_ids=(memory_id,),
        applied_usage_event_ids=(f"usage-{memory_id}-applied",),
        outcome=outcome,
        attribution_basis="deterministic_verifier" if helped else "proven_regression",
        verifier_result="passed" if helped else "failed",
        approval_principal_id="",
        task_family="repair",
        stage="verification",
        project_id="lethyx",
        host_id="codex",
        profile_id="deep-harness",
        independence_domain=domain,
        evidence_refs=_evidence(
            f"credit-{memory_id}-{domain}",
            ref_type="test_run" if helped else "verification_report",
        ),
        reason=f"{outcome} was causally verified",
    )


def test_procedure_reuse_requires_same_signature_or_qualified_transfer() -> None:
    same = qualify_procedure_reuse(
        memory_id="procedure-1",
        source_task_signature="repair:python",
        requested_task_signature="repair:python",
    )
    assert same.allowed is True
    assert same.reason == "same_task_signature"

    denied = qualify_procedure_reuse(
        memory_id="procedure-1",
        source_task_signature="repair:python",
        requested_task_signature="repair:rust",
    )
    assert denied.allowed is False
    assert denied.promotion_authorized is False

    envelope = TransferEnvelope(
        envelope_id="transfer-procedure-1",
        status="accepted",
        source_scope="workspace:lethyx",
        target_scope="workspace:rust-port",
        source_task_signature="repair:python",
        target_task_signature="repair:rust",
        qualified_memory_ids=("procedure-1",),
        verified_outcome_ids=("credit-a", "credit-b"),
        independence_domains=("project:lethyx", "project:rust-port"),
        source_workflow="repair-python",
        target_workflow="repair-rust",
        source_project_id="lethyx",
        target_project_id="rust-port",
        source_host="hermes",
        target_host="codex",
        source_profile="hermes:deep",
        target_profile="codex:deep",
        source_stages=("diagnose", "repair", "verify"),
        target_stages=("diagnose", "repair", "verify"),
        prerequisites=("typed-tests",),
        contraindications=("untrusted-origin",),
        exclusions=("production-erase",),
        support_count=2,
        source_revision_id="lethyx@abc123",
        valid_from="2026-01-01T00:00:00Z",
        expires_at="2027-01-01T00:00:00Z",
        drift_fingerprint="sha256:transfer-procedure-1",
        evidence_refs=_evidence("transfer-procedure-1", ref_type="verification_report"),
    )
    transferred = qualify_procedure_reuse(
        memory_id="procedure-1",
        source_task_signature="repair:python",
        requested_task_signature="repair:rust",
        transfer_envelope=envelope,
    )
    assert transferred.allowed is True
    assert transferred.reason == "qualified_transfer_envelope"
    assert transferred.host_action_authorized is False
    with pytest.raises(ValueError, match="multiple independently verified outcomes"):
        replace(
            envelope,
            verified_outcome_ids=("credit-a",),
            independence_domains=("project:lethyx",),
        )


def test_one_success_cannot_promote_procedure() -> None:
    one = procedure_promotion_gate(
        verified_outcome_ids=("credit-a",),
        independence_domains=("run-a",),
    )
    assert one.allowed is False
    assert one.reason == "insufficient_independent_verified_outcomes"
    two = procedure_promotion_gate(
        verified_outcome_ids=("credit-a", "credit-b"),
        independence_domains=("run-a", "run-b"),
    )
    assert two.allowed is True


def test_coactivation_is_candidate_only_until_verified_outcome_diversity() -> None:
    coactivation = _evidence("coactivation", ref_type="session_event")
    candidate = evaluate_advisory_association(
        association_id="association-1",
        memory_ids=("memory-a", "memory-b"),
        coactivation_evidence_refs=coactivation,
        outcome_credits=(),
    )
    assert candidate.status == "candidate"
    assert candidate.advisory_weight == 0.0
    assert candidate.affects_authority is False
    assert candidate.durable_procedure is False

    credits = (
        replace(
            _credit("memory-a", "helped", domain="run-a", decision_id="association-a"),
            memory_ids=("memory-a", "memory-b"),
            applied_usage_event_ids=("applied-a", "applied-b"),
        ),
        replace(
            _credit("memory-a", "helped", domain="run-b", decision_id="association-b"),
            credit_id="association-credit-b",
            memory_ids=("memory-a", "memory-b"),
            applied_usage_event_ids=("applied-c", "applied-d"),
        ),
    )
    strengthened = evaluate_advisory_association(
        association_id="association-1",
        memory_ids=("memory-a", "memory-b"),
        coactivation_evidence_refs=coactivation,
        outcome_credits=credits,
    )
    assert strengthened.status == "strengthened"
    assert strengthened.outcome_diversity == 2
    assert strengthened.evidence_retained is True

    harmed = evaluate_advisory_association(
        association_id="association-1",
        memory_ids=("memory-a", "memory-b"),
        coactivation_evidence_refs=coactivation,
        outcome_credits=(
            replace(credits[0], outcome="hurt", attribution_basis="proven_regression"),
        ),
    )
    assert harmed.status == "weakened"
    assert harmed.evidence_retained is True
    quarantined = evaluate_advisory_association(
        association_id="association-1",
        memory_ids=("memory-a", "memory-b"),
        coactivation_evidence_refs=coactivation,
        outcome_credits=(
            replace(
                credits[0],
                outcome="hurt",
                attribution_basis="proven_regression",
            ),
            replace(
                credits[1],
                outcome="hurt",
                attribution_basis="proven_contamination",
            ),
        ),
    )
    assert quarantined.status == "quarantined"
    assert quarantined.evidence_refs


def test_deterministic_diversity_recency_cap_preserves_older_task_family() -> None:
    candidates = (
        UtilitySelectionCandidate(
            memory_id="recent-best",
            task_family="recent-repair",
            shadow_utility=1.0,
            support_count=8,
            newest_credit_at="2026-07-15T00:00:00Z",
        ),
        UtilitySelectionCandidate(
            memory_id="recent-second",
            task_family="recent-repair",
            shadow_utility=0.9,
            support_count=7,
            newest_credit_at="2026-07-14T23:00:00Z",
        ),
        UtilitySelectionCandidate(
            memory_id="older-family",
            task_family="release-audit",
            shadow_utility=0.4,
            support_count=3,
            newest_credit_at="2026-05-01T00:00:00Z",
        ),
    )
    selected = select_diverse_utility_candidates(
        candidates,
        limit=2,
        max_per_task_family=1,
        as_of="2026-07-15T01:00:00Z",
        recent_window_days=7,
    )
    assert tuple(item.memory_id for item in selected) == (
        "older-family",
        "recent-best",
    )


@pytest.mark.parametrize(
    ("case_type", "failure_stage", "target"),
    [
        ("capture_miss", "storage", "capture"),
        ("selection_miss", "selection", "taxonomy"),
        ("overretrieval", "retrieval", "retrieval"),
        ("reader_misuse", "rendering", "renderer"),
        ("reader_misuse", "comprehension", "reader"),
        ("novel_concept", "selection", "taxonomy"),
        ("label_disagreement", "selection", "taxonomy"),
        ("stale_guidance", "action", "host_action"),
        ("weak_model_failure", "verifier", "verifier"),
    ],
)
def test_refinement_case_has_closed_type_and_separate_failure_owner(
    case_type: str,
    failure_stage: str,
    target: str,
) -> None:
    case = generate_refinement_case(
        case_id=f"case-{case_type}",
        case_type=case_type,
        failure_stage=failure_stage,
        query="why did memory fail?",
        decision_id="decision-report",
        task_signature="repair:python:unit-test",
        action_trace_id="action-report",
        evidence_refs=_evidence(f"case-{case_type}"),
        reason="typed failure attribution",
    )
    assert isinstance(case, RefinementCase)
    assert case.refinement_target == target
    assert case.taxonomy_change_recommended is (target == "taxonomy")
    with pytest.raises(ValueError, match="unsupported refinement case type"):
        replace(case, case_type="weak_model_misuse")


def test_operator_report_classifies_helped_hurt_and_unnecessary_records() -> None:
    decision = _decision()
    usage = (
        _usage("memory-helped", "applied", 0),
        _usage("memory-extra", "rendered", 1),
        _usage("memory-hurt", "applied", 2),
    )
    report = build_memory_outcome_report(
        report_id="report-1",
        trace=MemoryDecisionTrace(decision=decision, usage_events=usage),
        outcome_credits=(
            _credit("memory-helped", "helped", domain="run-a"),
            _credit("memory-hurt", "hurt", domain="run-b"),
        ),
    )
    assert isinstance(report, MemoryOutcomeReport)
    assert {entry.memory_id: entry.disposition for entry in report.entries} == {
        "memory-helped": "helped",
        "memory-extra": "unknown",
        "memory-hurt": "hurt",
    }
    assert report.ordinary_recall is False
    assert report.host_action_authorized is False
    assert report.affects_authority is False


@pytest.mark.parametrize("backend_id", ["mempalace", "hindsight", "mnemosyne"])
def test_operator_report_reconstructs_from_selected_backend_after_restart(
    tmp_path,
    backend_id: str,
) -> None:
    root = tmp_path / backend_id
    decision = _decision()
    usage = (
        _usage("memory-helped", "applied", 0),
        _usage("memory-extra", "rendered", 1),
        _usage("memory-hurt", "applied", 2),
    )
    credits = (
        _credit("memory-helped", "helped", domain="run-a"),
        _credit("memory-hurt", "hurt", domain="run-b"),
    )
    client = MemoryClient(root, workspace_id="ws", backend=backend_id)
    try:
        client.record_memory_decision(decision)
        for event in usage:
            client.record_memory_usage_event(event)
        for credit in credits:
            client.record_outcome_credit(credit)
    finally:
        client.backend.close()

    reopened = MemoryClient(root, workspace_id="ws", backend=backend_id)
    try:
        report = reopened.memory_operator_report(decision.decision_id)
        assert {entry.disposition for entry in report.entries} == {
            "helped",
            "hurt",
            "unknown",
        }
        assert tuple(
            credit.credit_id for credit in reopened.outcome_credits(decision.decision_id)
        ) == (
            credits[0].credit_id,
            credits[1].credit_id,
        )
    finally:
        reopened.backend.close()


def test_unnecessary_requires_an_explicit_completed_nonuse_observation() -> None:
    decision = _decision()
    report = build_memory_outcome_report(
        report_id="complete-nonuse",
        trace=MemoryDecisionTrace(
            decision=decision,
            usage_events=(_usage("memory-extra", "not_applied", 0),),
        ),
        outcome_credits=(),
    )
    values = {entry.memory_id: entry.disposition for entry in report.entries}
    assert values["memory-extra"] == "unnecessary"
    assert values["memory-helped"] == "unknown"
    assert values["memory-hurt"] == "unknown"


def test_action_nonuse_cannot_contradict_applied_evidence(tmp_path) -> None:
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        client.record_memory_decision(_decision())
        client.record_memory_usage_event(_usage("memory-helped", "applied", 0))
        with pytest.raises(ValueError, match="contradictory"):
            client.record_memory_usage_event(_usage("memory-helped", "not_applied", 1))
        assert len(client.memory_decision_trace(_decision().decision_id).usage_events) == 1
    finally:
        client.backend.close()
