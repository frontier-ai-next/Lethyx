from dataclasses import replace

import pytest

from lethyx import MemoryClient
from lethyx.contracts import (
    AuthenticatedHostEvent,
    EvidenceRef,
    OriginRef,
    ProspectiveLifecycleEvent,
    ProspectiveObligation,
    ProspectiveTriggerPredicate,
)
from lethyx.memory.backends.contracts import CanonicalHeadConflict
from lethyx.memory.prospective import (
    ProspectiveTriggerContext,
    evaluate_prospective_trigger,
)


def _evidence(name: str, *, produced_by: str = "verifier") -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="test_run",
            ref_id=name,
            produced_by=produced_by,
            producer_id=f"{produced_by}:{name}",
        ),
    )


def _obligation(
    obligation_id: str,
    predicate_type: str,
    field: str,
    value: str | int | bool,
    *,
    scope: str = "workspace",
    scope_id: str = "",
) -> ProspectiveObligation:
    return ProspectiveObligation(
        obligation_id=obligation_id,
        scope=scope,
        scope_id=scope_id,
        evidence_refs=_evidence(f"source-{obligation_id}"),
        required_intervention=f"Present advisory intervention for {obligation_id}.",
        trigger_predicates=(
            ProspectiveTriggerPredicate(
                predicate_type=predicate_type,
                field=field,
                operator="eq",
                value=value,
            ),
        ),
    ).transition_to("armed", evidence_refs=_evidence(f"arm-{obligation_id}"))


def _authenticated_event() -> AuthenticatedHostEvent:
    return AuthenticatedHostEvent(
        event_id="host-event-41",
        event_type="verification_complete",
        principal_id="host:codex",
        channel_id="local-service:ws",
        authentication_evidence_ref=EvidenceRef(
            ref_type="session_event",
            ref_id="auth-receipt-41",
            hash="a" * 64,
            produced_by="supervisor",
            producer_id="local-service-listener",
        ),
        payload={"suite": "release", "passed": True},
    )


def _fire_context() -> ProspectiveTriggerContext:
    return ProspectiveTriggerContext(
        registered_condition={"proof_ready": True},
        origin=OriginRef(
            channel="host_event",
            principal_id="host:codex",
            integrity_class="observed",
            origin_id="local-service:ws:proof-ready",
        ),
        observed_at="2026-07-15T01:00:00Z",
    )


@pytest.mark.parametrize(
    ("predicate_type", "field", "value", "context"),
    [
        (
            "time",
            "window",
            "release",
            ProspectiveTriggerContext(
                time={"window": "release"}, observed_at="2026-07-15T01:00:00Z"
            ),
        ),
        (
            "deadline",
            "gate",
            "reached",
            ProspectiveTriggerContext(
                deadline={"gate": "reached"}, observed_at="2026-07-15T01:00:00Z"
            ),
        ),
        (
            "host_event",
            "event_type",
            "verification_complete",
            ProspectiveTriggerContext(
                host_event=_authenticated_event(), observed_at="2026-07-15T01:00:00Z"
            ),
        ),
        (
            "tool_state",
            "pytest",
            "passed",
            ProspectiveTriggerContext(
                tool_state={"pytest": "passed"}, observed_at="2026-07-15T01:00:00Z"
            ),
        ),
        (
            "test_state",
            "release_suite",
            "passed",
            ProspectiveTriggerContext(
                test_state={"release_suite": "passed"},
                observed_at="2026-07-15T01:00:00Z",
            ),
        ),
        (
            "file_state",
            "dist_manifest",
            "present",
            ProspectiveTriggerContext(
                file_state={"dist_manifest": "present"},
                observed_at="2026-07-15T01:00:00Z",
            ),
        ),
        (
            "registered_condition",
            "release_proof_ready",
            True,
            ProspectiveTriggerContext(
                registered_condition={"release_proof_ready": True},
                observed_at="2026-07-15T01:00:00Z",
            ),
        ),
    ],
)
def test_first_release_typed_trigger_channels_are_distinct_and_advisory(
    predicate_type: str,
    field: str,
    value: str | int | bool,
    context: ProspectiveTriggerContext,
) -> None:
    decision = evaluate_prospective_trigger(
        _obligation(predicate_type, predicate_type, field, value),
        context,
    )

    assert decision.matched is True
    assert decision.match_kind == "typed"
    assert decision.trigger_type == predicate_type
    assert decision.delivery_eligible is True
    assert decision.host_action_authorized is False


def test_host_event_requires_authenticated_typed_envelope() -> None:
    with pytest.raises(ValueError, match="AuthenticatedHostEvent"):
        ProspectiveTriggerContext(
            host_event={"event_type": "verification_complete"},
            observed_at="2026-07-15T01:00:00Z",
        )
    with pytest.raises(ValueError, match="authentication evidence"):
        AuthenticatedHostEvent(
            event_id="host-event-unverified",
            event_type="verification_complete",
            principal_id="host:codex",
            channel_id="local-service:ws",
            authentication_evidence_ref=EvidenceRef(
                ref_type="message",
                ref_id="self-claim",
                produced_by="agent",
                producer_id="host:codex",
            ),
        )
    with pytest.raises(ValueError, match="prospective predicate_type"):
        ProspectiveTriggerPredicate(
            predicate_type="event",
            field="event_type",
            operator="eq",
            value="verification_complete",
        )


def test_scope_timing_recurrence_and_conditions_are_closed_contracts() -> None:
    scoped = replace(
        _obligation(
            "task-proof",
            "registered_condition",
            "proof_ready",
            True,
            scope="task",
            scope_id="task-7",
        ),
        earliest_at="2026-07-15T01:00:00Z",
        deadline_at="2026-07-15T02:00:00Z",
        expires_at="2026-07-15T03:00:00Z",
        recurrence_policy="repeat",
        recurrence_interval_seconds=900,
        suppress_predicates=(
            ProspectiveTriggerPredicate(
                predicate_type="file_state",
                field="proof_published",
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
    )
    wrong_scope = evaluate_prospective_trigger(
        scoped,
        ProspectiveTriggerContext(
            registered_condition={"proof_ready": True},
            scope="task",
            scope_id="task-8",
            observed_at="2026-07-15T01:30:00Z",
        ),
    )
    assert wrong_scope.reason == "scope_mismatch"
    assert wrong_scope.delivery_eligible is False

    suppressed = evaluate_prospective_trigger(
        scoped,
        ProspectiveTriggerContext(
            registered_condition={"proof_ready": True},
            file_state={"proof_published": True},
            scope="task",
            scope_id="task-7",
            observed_at="2026-07-15T01:30:00Z",
        ),
    )
    assert suppressed.reason == "suppressed_by_current_state"

    with pytest.raises(ValueError, match="scope_id"):
        replace(scoped, scope="run", scope_id="")
    with pytest.raises(ValueError, match="recurrence_interval_seconds"):
        replace(scoped, recurrence_policy="repeat", recurrence_interval_seconds=0)


def test_typed_fire_and_satisfaction_events_are_restart_idempotent(tmp_path) -> None:
    obligation = replace(
        _obligation(
            "release-proof",
            "registered_condition",
            "proof_ready",
            True,
        ),
        status="pending",
        transition_from=None,
        transition_evidence_refs=(),
        satisfaction_verifier="artifact_sha256_matches",
    )
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        client.create_prospective_obligation(obligation, event_id="create-release-proof")
        armed = client.apply_prospective_lifecycle_event(
            obligation.obligation_id,
            ProspectiveLifecycleEvent(
                event_id="arm-release-proof",
                event_type="arm",
                expected_version=0,
                expected_status="pending",
                evidence_refs=_evidence("arm-release-proof"),
            ),
        )
        assert armed["status"] == "armed"
        fired_event = ProspectiveLifecycleEvent(
            event_id="fire-release-proof",
            idempotency_key="trigger:registered:proof_ready:1",
            event_type="fire",
            expected_version=1,
            expected_status="armed",
            evidence_refs=_evidence("fire-release-proof"),
        )
        fired = client.apply_prospective_lifecycle_event(
            obligation.obligation_id,
            fired_event,
            trigger_context=_fire_context(),
        )
        replayed = client.apply_prospective_lifecycle_event(
            obligation.obligation_id,
            fired_event,
        )
        assert fired["status"] == "fired"
        assert fired["replayed"] is False
        assert replayed["receipt_id"] == fired["receipt_id"]
        assert replayed["replayed"] is True
    finally:
        client.backend.close()

    reopened = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        replayed_after_restart = reopened.apply_prospective_lifecycle_event(
            obligation.obligation_id,
            fired_event,
        )
        assert replayed_after_restart["receipt_id"] == fired["receipt_id"]
        assert replayed_after_restart["replayed"] is True

        satisfied_event = ProspectiveLifecycleEvent(
            event_id="satisfy-release-proof",
            idempotency_key="satisfaction:artifact:release-proof:1",
            event_type="satisfy",
            expected_version=2,
            expected_status="fired",
            evidence_refs=_evidence("satisfy-release-proof"),
            satisfaction_evidence_refs=_evidence("release-proof-artifact"),
            satisfaction_verifier="artifact_sha256_matches",
            satisfaction_verifier_result="passed",
        )
        satisfied = reopened.apply_prospective_lifecycle_event(
            obligation.obligation_id,
            satisfied_event,
        )
        satisfied_replay = reopened.apply_prospective_lifecycle_event(
            obligation.obligation_id,
            satisfied_event,
        )
        assert satisfied["status"] == "satisfied"
        assert satisfied_replay["receipt_id"] == satisfied["receipt_id"]
        assert satisfied_replay["replayed"] is True

        late_fire_replay = reopened.apply_prospective_lifecycle_event(
            obligation.obligation_id,
            fired_event,
        )
        assert late_fire_replay["receipt_id"] == fired["receipt_id"]
        assert late_fire_replay["replayed"] is True
        current = reopened.get_prospective_obligation(obligation.obligation_id)
        assert current is not None and current.status == "satisfied"
        node = reopened.backend.select_index_nodes(
            workspace_id="ws", node_type="prospective_trigger", limit=10
        )[0]
        assert node.metadata["status"] == "satisfied"
        assert node.metadata["version"] == 3

        with pytest.raises(CanonicalHeadConflict, match="idempotency"):
            reopened.apply_prospective_lifecycle_event(
                obligation.obligation_id,
                replace(satisfied_event, event_id="different-event-id"),
            )
    finally:
        reopened.backend.close()


def test_lifecycle_event_type_fixes_expected_transition_and_satisfaction_proof() -> None:
    with pytest.raises(ValueError, match="fire event requires armed"):
        ProspectiveLifecycleEvent(
            event_id="bad-fire",
            event_type="fire",
            expected_version=0,
            expected_status="pending",
            evidence_refs=_evidence("bad-fire"),
        )
    with pytest.raises(ValueError, match="proof-of-satisfaction"):
        ProspectiveLifecycleEvent(
            event_id="bad-satisfaction",
            event_type="satisfy",
            expected_version=2,
            expected_status="fired",
            evidence_refs=_evidence("bad-satisfaction"),
        )
