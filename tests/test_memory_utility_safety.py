from dataclasses import replace

import pytest

from lethyx import MemoryClient
from lethyx.contracts import EvidenceRef, MemoryDecisionEvent, MemoryUsageEvent, OutcomeCreditEvent
from lethyx.memory.backends.contracts import MemoryUsageRecord
from lethyx.memory.refinement import (
    evaluate_advisory_association,
    procedure_promotion_gate,
    qualify_procedure_reuse,
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
        decision_id="phase-6-exit-decision",
        workspace_id="ws",
        run_id="run-exit",
        task_signature="repair:python",
        action_trace_id="action-exit",
        action="recall",
        canonical_snapshot_sha256=f"sha256:{'a' * 64}",
        context_snapshot_sha256=f"sha256:{'b' * 64}",
        live_state_snapshot_sha256=f"sha256:{'c' * 64}",
        live_trigger="exit_gate",
        candidate_ids=("memory-help", "memory-hurt"),
        selected_candidate_ids=("memory-help", "memory-hurt"),
        policy_revision="policy-exit",
        catalog_revision="catalog-exit",
        cost_tokens=20,
        cost_latency_ms=4,
        downstream_outcome="verified",
        evidence_refs=_evidence("exit-decision"),
    )


def _usage(memory_id: str, sequence: int) -> MemoryUsageEvent:
    return MemoryUsageEvent(
        event_id=f"usage-{memory_id}",
        decision_id="phase-6-exit-decision",
        memory_id=memory_id,
        workspace_id="ws",
        task_signature="repair:python",
        action_trace_id="action-exit",
        usage_type="applied",
        sequence=sequence,
        evidence_refs=_evidence(f"usage-{memory_id}"),
    )


def _credit(memory_id: str, outcome: str, domain: str) -> OutcomeCreditEvent:
    helped = outcome == "helped"
    return OutcomeCreditEvent(
        credit_id=f"credit-{memory_id}",
        decision_id="phase-6-exit-decision",
        workspace_id="ws",
        task_signature="repair:python",
        action_trace_id="action-exit",
        memory_ids=(memory_id,),
        applied_usage_event_ids=(f"usage-{memory_id}",),
        outcome=outcome,
        attribution_basis="deterministic_verifier" if helped else "proven_regression",
        verifier_result="passed" if helped else "failed",
        approval_principal_id="",
        task_family="repair",
        stage="verification",
        project_id="lethyx",
        host_id="codex",
        profile_id="release",
        independence_domain=domain,
        evidence_refs=_evidence(
            f"credit-{memory_id}",
            ref_type="test_run" if helped else "verification_report",
        ),
        reason=f"verified {outcome}",
    )


def test_exit_retrieval_popularity_does_not_change_trust(tmp_path) -> None:
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        saved = client.remember(
            "Candidate observation",
            requested_block_id="workspace_episodes",
            evidence_refs=list(_evidence("candidate-observation")),
        )
        before = client.backend.get_block(saved.memory_id)
        for index in range(20):
            client.backend.record_usage(
                MemoryUsageRecord(
                    block_id=saved.memory_id,
                    workspace_id="ws",
                    run_id=f"retrieval-{index}",
                    included=True,
                    outcome="neutral",
                    evidence_refs=_evidence(f"retrieval-{index}"),
                )
            )
        after = client.backend.get_block(saved.memory_id)
        assert before is not None and after is not None
        assert after.trust_level == before.trust_level == "agent_reported"
        assert after.verification_status == before.verification_status == "unverified"
    finally:
        client.backend.close()


def test_exit_one_unrelated_success_does_not_globalize_procedure() -> None:
    gate = procedure_promotion_gate(
        verified_outcome_ids=("credit-one",),
        independence_domains=("run-one",),
    )
    reuse = qualify_procedure_reuse(
        memory_id="procedure-1",
        source_task_signature="repair:python",
        requested_task_signature="deploy:rust",
    )
    assert gate.allowed is False
    assert reuse.allowed is False
    assert reuse.promotion_authorized is False


def test_exit_verified_helped_hurt_trace_is_reproducible_after_restart(tmp_path) -> None:
    decision = _decision()
    usage = (_usage("memory-help", 0), _usage("memory-hurt", 1))
    credits = (
        _credit("memory-help", "helped", "run-help"),
        _credit("memory-hurt", "hurt", "run-hurt"),
    )
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        client.record_memory_decision(decision)
        for event in usage:
            client.record_memory_usage_event(event)
        for credit in credits:
            client.record_outcome_credit(credit)
    finally:
        client.backend.close()
    reopened = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        assert reopened.memory_decision_trace(decision.decision_id).usage_events == usage
        assert reopened.outcome_credits(decision.decision_id) == credits
        assert tuple(
            entry.disposition
            for entry in reopened.memory_operator_report(decision.decision_id).entries
        ) == ("helped", "hurt")
    finally:
        reopened.backend.close()


def test_exit_uncertain_attribution_stays_neutral() -> None:
    neutral = OutcomeCreditEvent(
        credit_id="credit-neutral",
        decision_id="decision-neutral",
        workspace_id="ws",
        task_signature="repair:python",
        action_trace_id="action-neutral",
        memory_ids=("memory-neutral",),
        applied_usage_event_ids=(),
        outcome="neutral",
        attribution_basis="insufficient_linkage",
        verifier_result="inconclusive",
        approval_principal_id="",
        task_family="repair",
        stage="verification",
        project_id="lethyx",
        host_id="codex",
        profile_id="release",
        independence_domain="run-neutral",
        evidence_refs=_evidence("neutral-attribution"),
        reason="causal link is insufficient",
    )
    assert neutral.outcome == "neutral"
    with pytest.raises(ValueError, match="requires applied memory linkage"):
        replace(
            neutral,
            outcome="helped",
            attribution_basis="deterministic_verifier",
            verifier_result="passed",
        )


def test_exit_context_local_utility_does_not_transfer_without_envelope(tmp_path) -> None:
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        decision = _decision()
        usage = _usage("memory-help", 0)
        credit = _credit("memory-help", "helped", "run-help")
        client.record_memory_decision(decision)
        client.record_memory_usage_event(usage)
        client.record_outcome_credit(credit)
        local = client.contextual_utility_summary(
            memory_id="memory-help",
            task_family="repair",
            stage="verification",
            project_id="lethyx",
            host_id="codex",
            profile_id="release",
            minimum_support=1,
            minimum_independent_outcomes=1,
        )
        unrelated = client.contextual_utility_summary(
            memory_id="memory-help",
            task_family="deploy",
            stage="verification",
            project_id="other-project",
            host_id="codex",
            profile_id="release",
            minimum_support=1,
            minimum_independent_outcomes=1,
        )
        assert local.support_count == 1
        assert unrelated.support_count == 0
        assert (
            client.qualify_procedure_reuse(
                memory_id="memory-help",
                source_task_signature="repair:python",
                requested_task_signature="deploy:rust",
            ).allowed
            is False
        )
    finally:
        client.backend.close()


def test_exit_popularity_and_coactivation_do_not_increase_authority() -> None:
    association = evaluate_advisory_association(
        association_id="association-exit",
        memory_ids=("memory-a", "memory-b"),
        coactivation_evidence_refs=_evidence("coactivation", ref_type="session_event"),
        outcome_credits=(),
    )
    assert association.status == "candidate"
    assert association.advisory_weight == 0.0
    assert association.affects_authority is False
    assert association.durable_procedure is False
