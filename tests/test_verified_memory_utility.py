from dataclasses import replace

import pytest

from lethyx import MemoryClient
from lethyx.contracts import (
    EvidenceRef,
    MemoryDecisionEvent,
    MemoryUsageEvent,
    OutcomeCreditEvent,
)
from lethyx.memory.backends.contracts import CanonicalHeadConflict, MemoryUsageRecord


def _evidence(
    name: str,
    *,
    ref_type: str = "test_run",
    produced_by: str = "verifier",
) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type=ref_type,
            ref_id=name,
            produced_by=produced_by,
            producer_id=f"{produced_by}:{name}",
        ),
    )


def _decision(index: int) -> MemoryDecisionEvent:
    return MemoryDecisionEvent(
        decision_id=f"decision-{index}",
        workspace_id="ws",
        run_id=f"run-{index}",
        task_signature="repair:python:unit-test",
        action_trace_id=f"action-{index}",
        action="recall",
        canonical_snapshot_sha256=f"sha256:{'a' * 64}",
        context_snapshot_sha256=f"sha256:{'b' * 64}",
        live_state_snapshot_sha256=f"sha256:{'c' * 64}",
        live_trigger="test_state:required_suite_failed",
        candidate_ids=("memory-1",),
        selected_candidate_ids=("memory-1",),
        policy_revision="policy-7",
        catalog_revision="catalog-11",
        cost_tokens=100,
        cost_latency_ms=5,
        downstream_outcome="observed",
        evidence_refs=_evidence(f"decision-{index}"),
    )


def _usage(index: int, usage_type: str, sequence: int) -> MemoryUsageEvent:
    return MemoryUsageEvent(
        event_id=f"usage-{index}-{usage_type}",
        decision_id=f"decision-{index}",
        memory_id="memory-1",
        workspace_id="ws",
        task_signature="repair:python:unit-test",
        action_trace_id=f"action-{index}",
        usage_type=usage_type,
        sequence=sequence,
        evidence_refs=_evidence(f"usage-{index}-{usage_type}"),
    )


def _credit(
    index: int,
    outcome: str,
    basis: str,
    *,
    usage_event_id: str = "",
    verifier_result: str = "",
    approval_principal_id: str = "",
    evidence_refs: tuple[EvidenceRef, ...] | None = None,
) -> OutcomeCreditEvent:
    return OutcomeCreditEvent(
        credit_id=f"credit-{index}-{outcome}",
        decision_id=f"decision-{index}",
        workspace_id="ws",
        task_signature="repair:python:unit-test",
        action_trace_id=f"action-{index}",
        memory_ids=("memory-1",),
        applied_usage_event_ids=((usage_event_id,) if usage_event_id else ()),
        outcome=outcome,
        attribution_basis=basis,
        verifier_result=verifier_result,
        approval_principal_id=approval_principal_id,
        task_family="repair",
        stage="verification",
        project_id="lethyx",
        host_id="codex",
        profile_id="deep-harness",
        independence_domain=f"independent-run-{index}",
        evidence_refs=evidence_refs or _evidence(f"credit-{index}"),
        reason=f"{basis} observation",
    )


def test_helped_hurt_and_neutral_credit_rules_are_closed() -> None:
    helped = _credit(
        0,
        "helped",
        "deterministic_verifier",
        usage_event_id="usage-0-applied",
        verifier_result="passed",
    )
    assert helped.outcome == "helped"
    with pytest.raises(ValueError, match="passed deterministic verifier"):
        replace(helped, verifier_result="failed")

    approval = _credit(
        0,
        "helped",
        "trusted_approval",
        usage_event_id="usage-0-applied",
        approval_principal_id="human:release-owner",
        evidence_refs=_evidence(
            "approval",
            ref_type="human_approval",
            produced_by="human",
        ),
    )
    assert approval.approval_principal_id == "human:release-owner"
    with pytest.raises(ValueError, match="trusted approval evidence"):
        replace(approval, evidence_refs=_evidence("not-approval"))

    hurt = _credit(
        1,
        "hurt",
        "proven_regression",
        usage_event_id="usage-1-applied",
    )
    assert hurt.outcome == "hurt"
    with pytest.raises(ValueError, match="proven harm basis"):
        replace(hurt, attribution_basis="insufficient_linkage")

    neutral = _credit(2, "neutral", "insufficient_linkage")
    assert neutral.applied_usage_event_ids == ()
    with pytest.raises(ValueError, match="insufficient_linkage"):
        replace(neutral, attribution_basis="deterministic_verifier")


@pytest.mark.parametrize("backend_id", ["mempalace", "hindsight", "mnemosyne"])
def test_verified_credit_builds_contextual_shadow_summary_after_restart(
    tmp_path,
    backend_id: str,
) -> None:
    root = tmp_path / backend_id
    client = MemoryClient(root, workspace_id="ws", backend=backend_id)
    credits: list[OutcomeCreditEvent] = []
    try:
        for index in range(3):
            client.record_memory_decision(_decision(index))
            client.record_memory_usage_event(_usage(index, "candidate", 0))
            client.record_memory_usage_event(_usage(index, "applied", 1))
        credits = [
            _credit(
                0,
                "helped",
                "deterministic_verifier",
                usage_event_id="usage-0-applied",
                verifier_result="passed",
            ),
            _credit(
                1,
                "hurt",
                "proven_regression",
                usage_event_id="usage-1-applied",
                evidence_refs=_evidence("regression", ref_type="verification_report"),
            ),
            _credit(
                2,
                "neutral",
                "insufficient_linkage",
                usage_event_id="usage-2-candidate",
            ),
        ]
        first = client.record_outcome_credit(credits[0])
        assert first["replayed"] is False
        assert client.record_outcome_credit(credits[0])["replayed"] is True
        with pytest.raises(CanonicalHeadConflict, match="outcome_credit_idempotency_conflict"):
            client.record_outcome_credit(replace(credits[0], reason="changed"))
        with pytest.raises(
            CanonicalHeadConflict,
            match="outcome_credit_applied_usage_conflict",
        ):
            client.record_outcome_credit(replace(credits[0], credit_id="duplicate-applied-credit"))
        for credit in credits[1:]:
            client.record_outcome_credit(credit)
        with pytest.raises(ValueError, match="applied usage linkage"):
            client.record_outcome_credit(
                replace(
                    credits[0],
                    credit_id="bad-candidate-credit",
                    applied_usage_event_ids=("usage-0-candidate",),
                )
            )
        summary = client.contextual_utility_summary(
            memory_id="memory-1",
            task_family="repair",
            stage="verification",
            project_id="lethyx",
            host_id="codex",
            profile_id="deep-harness",
            minimum_support=2,
            minimum_independent_outcomes=2,
        )
        assert (summary.helped_count, summary.hurt_count, summary.neutral_count) == (1, 1, 1)
        assert summary.support_count == 2
        assert summary.outcome_diversity == 2
        assert summary.shadow_utility == 0.0
        assert summary.shadow_eligible is True
        assert summary.ranking_mode == "shadow_only"
        assert summary.affects_trust is False
        assert summary.affects_truth is False
        assert summary.affects_deletion is False
        assert summary.affects_authority is False
        unrelated = client.contextual_utility_summary(
            memory_id="memory-1",
            task_family="unrelated",
            stage="verification",
            project_id="lethyx",
            host_id="codex",
            profile_id="deep-harness",
            minimum_support=1,
            minimum_independent_outcomes=1,
        )
        assert unrelated.support_count == 0
        assert unrelated.shadow_eligible is False
    finally:
        client.backend.close()

    reopened = MemoryClient(root, workspace_id="ws", backend=backend_id)
    try:
        restarted = reopened.contextual_utility_summary(
            memory_id="memory-1",
            task_family="repair",
            stage="verification",
            project_id="lethyx",
            host_id="codex",
            profile_id="deep-harness",
            minimum_support=2,
            minimum_independent_outcomes=2,
        )
        assert restarted.support_count == 2
        assert restarted.newest_credit_at
    finally:
        reopened.backend.close()


def test_retrieval_or_ignore_cannot_create_unverified_credit(tmp_path) -> None:
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        with pytest.raises(ValueError, match="verified OutcomeCreditEvent"):
            client.backend.record_usage(
                MemoryUsageRecord(
                    block_id="memory-1",
                    workspace_id="ws",
                    included=True,
                    outcome="helped",
                    helped=True,
                )
            )
    finally:
        client.backend.close()
