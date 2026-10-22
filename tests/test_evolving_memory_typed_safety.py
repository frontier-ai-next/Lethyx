from dataclasses import replace

import pytest

from um_mem.contracts import EvidenceRef, OutcomeCreditEvent


def _neutral(reason: str) -> OutcomeCreditEvent:
    return OutcomeCreditEvent(
        credit_id="credit-1",
        decision_id="decision-1",
        workspace_id="alpha",
        task_signature="typed-task",
        action_trace_id="action-1",
        memory_ids=("memory-1",),
        applied_usage_event_ids=(),
        outcome="neutral",
        attribution_basis="insufficient_linkage",
        verifier_result="inconclusive",
        approval_principal_id="",
        task_family="repair",
        stage="verification",
        project_id="project-1",
        host_id="host-1",
        profile_id="profile-1",
        independence_domain="run-1",
        evidence_refs=(
            EvidenceRef(
                ref_type="tool_result",
                ref_id="tool-1",
                produced_by="watcher",
                producer_id="watcher-1",
            ),
        ),
        reason=reason,
    )


def test_free_text_cannot_turn_unlinked_tool_failure_into_hurt() -> None:
    ordinary = _neutral("ordinary typed tool result")
    alarming = _neutral("catastrophic regression error hurt failure")

    assert ordinary.outcome == alarming.outcome == "neutral"
    assert ordinary.attribution_basis == alarming.attribution_basis == "insufficient_linkage"
    with pytest.raises(ValueError, match="requires applied memory linkage"):
        replace(
            alarming,
            outcome="hurt",
            attribution_basis="proven_regression",
        )
