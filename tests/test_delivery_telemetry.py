from dataclasses import replace

import pytest

from lethyx import MemoryClient
from lethyx.contracts import (
    EvidenceRef,
    OriginRef,
    ProspectiveLifecycleEvent,
    ProspectiveObligation,
    ProspectiveTelemetry,
    ProspectiveTriggerPredicate,
)
from lethyx.memory.prospective import ProspectiveTriggerContext


def _evidence(name: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="test_run", ref_id=name, produced_by="verifier", producer_id=f"verifier:{name}"
        ),
    )


def _fired() -> ProspectiveObligation:
    pending = ProspectiveObligation(
        obligation_id="delivery",
        scope="workspace",
        evidence_refs=_evidence("source"),
        required_intervention="Tell the host that release proof is ready.",
        trigger_predicates=(
            ProspectiveTriggerPredicate(
                predicate_type="registered_condition",
                field="release_proof_ready",
                operator="eq",
                value=True,
            ),
        ),
        satisfaction_verifier="artifact_sha256_matches",
    )
    armed = pending.transition_to("armed", evidence_refs=_evidence("arm"))
    return armed.transition_to("fired", evidence_refs=_evidence("fire"))


def test_fired_obligation_projects_only_labelled_non_executing_working_state(tmp_path) -> None:
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        client.create_prospective_obligation(
            replace(_fired(), status="pending", transition_from=None, transition_evidence_refs=()),
            event_id="create",
        )
        with pytest.raises(ValueError, match="status_fired_required"):
            client.prospective_intervention("delivery")
    finally:
        client.backend.close()

    projection = MemoryClient.prospective_intervention_from_obligation(_fired())
    assert projection.surface == "current_working_state"
    assert projection.label == "prospective_advisory_intervention"
    assert projection.text == "Tell the host that release proof is ready."
    assert projection.host_action_authorized is False
    assert projection.executes_action is False
    assert projection.consequential_action_requires_host_authorization is True


def test_satisfaction_preserves_passed_verifier_result_and_complete_telemetry() -> None:
    fired = _fired().with_telemetry(
        false_positive_delta=1,
        missed_trigger_delta=2,
        overloaded_trigger_delta=3,
        attention_interruption_delta=4,
    )
    with pytest.raises(ValueError, match="passed verifier result"):
        fired.transition_to(
            "satisfied",
            evidence_refs=_evidence("satisfied"),
            satisfaction_evidence_refs=_evidence("proof"),
            satisfaction_verifier="artifact_sha256_matches",
            satisfaction_verifier_result="failed",
        )
    satisfied = fired.transition_to(
        "satisfied",
        evidence_refs=_evidence("satisfied"),
        satisfaction_evidence_refs=_evidence("proof"),
        satisfaction_verifier="artifact_sha256_matches",
        satisfaction_verifier_result="passed",
    )
    assert satisfied.satisfaction_verifier_result == "passed"
    assert satisfied.telemetry == ProspectiveTelemetry(
        false_positive_count=1,
        missed_trigger_count=2,
        overloaded_trigger_count=3,
        attention_interruption_count=4,
    )


def test_selected_backend_round_trips_delivery_verifier_and_attention_telemetry(
    tmp_path,
) -> None:
    pending = ProspectiveObligation(
        obligation_id="durable-delivery",
        scope="workspace",
        evidence_refs=_evidence("source"),
        required_intervention="Present the release proof to the host.",
        trigger_predicates=(
            ProspectiveTriggerPredicate(
                predicate_type="registered_condition",
                field="proof_ready",
                operator="eq",
                value=True,
            ),
        ),
        satisfaction_verifier="artifact_sha256_matches",
        telemetry=ProspectiveTelemetry(
            false_positive_count=1,
            missed_trigger_count=2,
            overloaded_trigger_count=3,
            attention_interruption_count=4,
        ),
    )
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        client.create_prospective_obligation(pending, event_id="create")
        client.apply_prospective_lifecycle_event(
            pending.obligation_id,
            ProspectiveLifecycleEvent(
                event_id="arm",
                event_type="arm",
                expected_version=0,
                expected_status="pending",
                evidence_refs=_evidence("arm"),
            ),
        )
        client.apply_prospective_lifecycle_event(
            pending.obligation_id,
            ProspectiveLifecycleEvent(
                event_id="fire",
                event_type="fire",
                expected_version=1,
                expected_status="armed",
                evidence_refs=_evidence("fire"),
            ),
            trigger_context=ProspectiveTriggerContext(
                registered_condition={"proof_ready": True},
                origin=OriginRef(
                    channel="host_event",
                    principal_id="host:codex",
                    integrity_class="observed",
                ),
                observed_at="2026-07-15T01:00:00Z",
            ),
        )
        assert client.prospective_intervention(pending.obligation_id).text == (
            "Present the release proof to the host."
        )
        client.apply_prospective_lifecycle_event(
            pending.obligation_id,
            ProspectiveLifecycleEvent(
                event_id="satisfy",
                event_type="satisfy",
                expected_version=2,
                expected_status="fired",
                evidence_refs=_evidence("satisfy"),
                satisfaction_evidence_refs=_evidence("proof"),
                satisfaction_verifier="artifact_sha256_matches",
                satisfaction_verifier_result="passed",
            ),
        )
    finally:
        client.backend.close()

    reopened = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        current = reopened.get_prospective_obligation(pending.obligation_id)
        assert current is not None
        assert current.status == "satisfied"
        assert current.satisfaction_verifier_result == "passed"
        assert current.telemetry == pending.telemetry
        with pytest.raises(ValueError, match="status_fired_required"):
            reopened.prospective_intervention(pending.obligation_id)
    finally:
        reopened.backend.close()
