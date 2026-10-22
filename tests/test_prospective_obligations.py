from dataclasses import replace
from datetime import UTC, datetime

import pytest

from um_mem import MemoryClient
from um_mem.contracts import (
    AuthenticatedHostEvent,
    EvidenceRef,
    OriginRef,
    ProspectiveLifecycleEvent,
    ProspectiveObligation,
    ProspectiveTelemetry,
    ProspectiveTriggerPredicate,
)
from um_mem.memory.backends.hindsight import HindsightBackend
from um_mem.memory.backends.mempalace import MemPalaceBackend
from um_mem.memory.backends.mnemosyne import MnemosyneBackend
from um_mem.memory.prospective import (
    ProspectiveTriggerContext,
    evaluate_prospective_trigger,
)


def _evidence(name: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="test_run",
            ref_id=name,
            produced_by="verifier",
            producer_id=f"verifier:prospective:{name}",
        ),
    )


def _host_event(event_type: str) -> AuthenticatedHostEvent:
    return AuthenticatedHostEvent(
        event_id=f"host-{event_type}",
        event_type=event_type,
        principal_id="host:test",
        channel_id="local-service:ws",
        authentication_evidence_ref=EvidenceRef(
            ref_type="session_event",
            ref_id=f"auth-{event_type}",
            hash="a" * 64,
            produced_by="supervisor",
            producer_id="local-service-listener",
        ),
    )


def _obligation(obligation_id: str = "release-proof") -> ProspectiveObligation:
    return ProspectiveObligation(
        obligation_id=obligation_id,
        scope="workspace",
        evidence_refs=_evidence("source"),
        required_intervention="Ask the host to publish the verified release proof.",
        trigger_predicates=(
            ProspectiveTriggerPredicate(
                predicate_type="host_event",
                field="event_type",
                operator="eq",
                value="verification_complete",
            ),
        ),
        semantic_trigger_cues=("release proof ready", "verification complete"),
        earliest_at="2026-07-14T19:00:00Z",
        deadline_at="2026-07-15T06:00:00Z",
        expires_at="2026-07-15T07:00:00Z",
        recurrence_policy="once",
        suppress_predicates=(
            ProspectiveTriggerPredicate(
                predicate_type="registered_condition",
                field="proof_already_published",
                operator="eq",
                value=True,
            ),
        ),
        cancel_predicates=(
            ProspectiveTriggerPredicate(
                predicate_type="registered_condition",
                field="release_cancelled",
                operator="eq",
                value=True,
            ),
        ),
        satisfaction_verifier="artifact_sha256_matches",
    )


def _armed(obligation: ProspectiveObligation | None = None) -> ProspectiveObligation:
    return (obligation or _obligation()).transition_to("armed", evidence_refs=_evidence("armed"))


def test_first_class_obligation_validates_time_lifecycle_proof_and_telemetry() -> None:
    obligation = _obligation()
    assert obligation.telemetry == ProspectiveTelemetry()
    assert obligation.semantic_trigger_cues == (
        "release proof ready",
        "verification complete",
    )

    armed = _armed(obligation)
    fired = armed.transition_to("fired", evidence_refs=_evidence("fired"))
    with pytest.raises(ValueError, match="proof-of-satisfaction"):
        fired.transition_to("satisfied", evidence_refs=_evidence("satisfied"))
    satisfied = fired.transition_to(
        "satisfied",
        evidence_refs=_evidence("satisfied"),
        satisfaction_evidence_refs=_evidence("proof"),
        satisfaction_verifier_result="passed",
    )
    assert satisfied.satisfaction_evidence_refs == _evidence("proof")
    assert satisfied.satisfaction_verifier == "artifact_sha256_matches"

    measured = obligation.with_telemetry(
        false_positive_delta=1,
        missed_trigger_delta=2,
        overloaded_trigger_delta=3,
    )
    assert measured.telemetry == ProspectiveTelemetry(
        false_positive_count=1,
        missed_trigger_count=2,
        overloaded_trigger_count=3,
    )

    with pytest.raises(ValueError, match="deadline_at must not precede earliest_at"):
        ProspectiveObligation(
            obligation_id="bad-time",
            scope="workspace",
            evidence_refs=_evidence("bad-time"),
            required_intervention="Notify the host.",
            trigger_predicates=obligation.trigger_predicates,
            earliest_at="2026-07-15T06:00:00Z",
            deadline_at="2026-07-15T05:00:00Z",
        )


def test_typed_and_semantic_trigger_evaluation_are_separate_and_advisory() -> None:
    obligation = _armed()
    typed = evaluate_prospective_trigger(
        obligation,
        ProspectiveTriggerContext(
            host_event=_host_event("verification_complete"),
            observed_at="2026-07-14T20:00:00Z",
        ),
    )
    assert typed.matched is True
    assert typed.match_kind == "typed"
    assert typed.delivery_eligible is True
    assert typed.host_action_authorized is False

    semantic = evaluate_prospective_trigger(
        obligation,
        ProspectiveTriggerContext(
            host_event=_host_event("unrelated"),
            semantic_text="The verification complete report says the release proof ready.",
            observed_at="2026-07-14T20:00:00Z",
        ),
    )
    assert semantic.matched is True
    assert semantic.match_kind == "semantic"
    assert semantic.delivery_eligible is False
    assert semantic.host_action_authorized is False
    assert semantic.reason == "semantic_candidate_requires_typed_gate"


def test_suppression_cancellation_and_expiry_are_lifecycle_gates_not_actions() -> None:
    obligation = _armed()
    suppressed = evaluate_prospective_trigger(
        obligation,
        ProspectiveTriggerContext(
            host_event=_host_event("verification_complete"),
            registered_condition={"proof_already_published": True},
            observed_at="2026-07-14T20:00:00Z",
        ),
    )
    assert suppressed.matched is False
    assert suppressed.reason == "suppressed_by_current_state"

    cancelled = evaluate_prospective_trigger(
        obligation,
        ProspectiveTriggerContext(
            registered_condition={"release_cancelled": True},
            observed_at="2026-07-14T20:00:00Z",
        ),
    )
    assert cancelled.match_kind == "lifecycle"
    assert cancelled.recommended_transition == "cancelled"
    assert cancelled.host_action_authorized is False

    expired = evaluate_prospective_trigger(
        obligation,
        ProspectiveTriggerContext(observed_at="2026-07-15T07:00:01Z"),
    )
    assert expired.match_kind == "lifecycle"
    assert expired.recommended_transition == "expired"
    assert expired.delivery_eligible is False


def test_selected_backend_persists_and_updates_trigger_index_without_action_payload(
    tmp_path,
) -> None:
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    obligation = _obligation()
    created = client.create_prospective_obligation(obligation, event_id="create-1")
    assert created["version"] == 0

    nodes = client.backend.select_index_nodes(
        workspace_id="ws", node_type="prospective_trigger", limit=10
    )
    assert len(nodes) == 1
    node = nodes[0]
    assert node.metadata["obligation_id"] == obligation.obligation_id
    assert node.metadata["status"] == "pending"
    assert node.metadata["version"] == 0
    assert node.metadata["host_action_authorized"] is False
    assert "required_intervention" not in node.metadata

    client.transition_prospective_obligation(
        obligation.obligation_id,
        expected_version=0,
        expected_status="pending",
        status="armed",
        evidence_refs=_evidence("armed"),
        event_id="arm-1",
    )
    updated = client.backend.select_index_nodes(
        workspace_id="ws", node_type="prospective_trigger", limit=10
    )
    assert len(updated) == 1
    assert updated[0].node_id == node.node_id
    assert updated[0].metadata["status"] == "armed"
    assert updated[0].metadata["version"] == 1

    candidates = client.prospective_trigger_candidates(
        ProspectiveTriggerContext(
            host_event=_host_event("verification_complete"),
            observed_at="2026-07-14T20:00:00Z",
        )
    )
    assert [candidate.obligation_id for candidate in candidates] == [obligation.obligation_id]
    assert candidates[0].match_kind == "typed"
    assert candidates[0].host_action_authorized is False
    client.backend.close()

    reopened = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        semantic = reopened.prospective_trigger_candidates(
            ProspectiveTriggerContext(
                host_event=_host_event("unrelated"),
                semantic_text="release proof ready",
                observed_at="2026-07-14T20:00:00Z",
            )
        )
        assert len(semantic) == 1
        assert semantic[0].match_kind == "semantic"
        assert semantic[0].delivery_eligible is False
    finally:
        reopened.backend.close()


def test_normal_semantic_write_cannot_model_a_prospective_obligation(tmp_path) -> None:
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        result = client.remember(
            "Remind me later",
            kind="prospective_trigger",
            requested_block_id="workspace_episodes",
        )
        assert result.saved is False
        assert result.raw["reason"] == "prospective_obligation_requires_typed_api"
        assert result.raw["policy_issues"] == ["prospective_obligation_not_semantic_memory"]
    finally:
        client.backend.close()


def test_selected_backend_satisfaction_cas_requires_and_round_trips_proof(tmp_path) -> None:
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    obligation = _obligation("satisfaction-proof").with_telemetry(missed_trigger_delta=1)
    try:
        client.create_prospective_obligation(obligation, event_id="create-satisfaction")
        client.transition_prospective_obligation(
            obligation.obligation_id,
            expected_version=0,
            expected_status="pending",
            status="armed",
            evidence_refs=_evidence("armed"),
            event_id="arm-satisfaction",
        )
        client.apply_prospective_lifecycle_event(
            obligation.obligation_id,
            ProspectiveLifecycleEvent(
                event_id="fire-satisfaction",
                event_type="fire",
                expected_version=1,
                expected_status="armed",
                evidence_refs=_evidence("fired"),
            ),
            trigger_context=ProspectiveTriggerContext(
                host_event=_host_event("verification_complete"),
                origin=OriginRef(
                    channel="host_event",
                    principal_id="host:test",
                    integrity_class="observed",
                    origin_id="local-service:ws:verification-complete",
                ),
                observed_at="2026-07-14T20:00:00Z",
            ),
        )
        with pytest.raises(ValueError, match="proof-of-satisfaction"):
            client.transition_prospective_obligation(
                obligation.obligation_id,
                expected_version=2,
                expected_status="fired",
                status="satisfied",
                evidence_refs=_evidence("satisfied"),
                event_id="satisfy-without-proof",
            )
        satisfied = client.transition_prospective_obligation(
            obligation.obligation_id,
            expected_version=2,
            expected_status="fired",
            status="satisfied",
            evidence_refs=_evidence("satisfied"),
            satisfaction_evidence_refs=_evidence("proof"),
            satisfaction_verifier="artifact_sha256_matches",
            satisfaction_verifier_result="passed",
            event_id="satisfy-with-proof",
        )
        assert satisfied["version"] == 3
        current = client.get_prospective_obligation(obligation.obligation_id)
        assert current is not None
        assert current.status == "satisfied"
        assert current.satisfaction_evidence_refs == _evidence("proof")
        assert current.telemetry.missed_trigger_count == 1
        node = client.backend.select_index_nodes(
            workspace_id="ws", node_type="prospective_trigger", limit=10
        )[0]
        assert node.metadata["status"] == "satisfied"
        assert node.metadata["version"] == 3
    finally:
        client.backend.close()


@pytest.mark.parametrize("backend_id", ["mempalace", "hindsight", "mnemosyne"])
def test_each_selected_backend_adapter_owns_the_same_prospective_index(
    tmp_path,
    backend_id: str,
) -> None:
    if backend_id == "mempalace":
        backend = MemPalaceBackend(tmp_path, workspace_id="ws", reconcile_pending_writes=False)
    elif backend_id == "hindsight":
        backend = HindsightBackend(
            repo_root=tmp_path,
            workspace_id="ws",
            reconcile_pending_writes=False,
        )
    else:
        backend = MnemosyneBackend(
            tmp_path,
            workspace_id="ws",
            enabled=False,
            reconcile_pending_writes=False,
        )
    obligation = replace(
        _obligation(f"adapter-{backend_id}"),
        scope="task",
        scope_id=f"task-{backend_id}",
        recurrence_policy="repeat",
        recurrence_interval_seconds=300,
    )
    try:
        backend.create_prospective_obligation(obligation, event_id="create")
        assert backend.get_prospective_obligation(obligation.obligation_id) == obligation
        nodes = backend.select_index_nodes(
            workspace_id="ws", node_type="prospective_trigger", limit=10
        )
        assert len(nodes) == 1
        assert nodes[0].metadata["obligation_id"] == obligation.obligation_id
        assert nodes[0].metadata["scope"] == "task"
        assert nodes[0].metadata["scope_id"] == f"task-{backend_id}"
        assert nodes[0].metadata["recurrence_policy"] == "repeat"
        assert nodes[0].metadata["recurrence_interval_seconds"] == 300
        assert nodes[0].metadata["host_action_authorized"] is False
    finally:
        backend.close()


def test_trigger_context_requires_timezone_aware_observed_at() -> None:
    with pytest.raises(ValueError, match="observed_at must include a timezone"):
        ProspectiveTriggerContext(observed_at="2026-07-14T20:00:00")
    now = datetime.now(UTC).isoformat()
    assert ProspectiveTriggerContext(observed_at=now).observed_at == now
