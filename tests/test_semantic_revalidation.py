from dataclasses import replace

import pytest

from um_mem import MemoryClient
from um_mem.contracts import (
    EvidenceRef,
    OriginRef,
    ProspectiveLifecycleEvent,
    ProspectiveObligation,
    ProspectiveTriggerPredicate,
)
from um_mem.memory.backends.contracts import CanonicalHeadConflict
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
            producer_id=f"verifier:{name}",
        ),
    )


def _obligation(*, scope: str = "workspace", scope_id: str = "") -> ProspectiveObligation:
    return ProspectiveObligation(
        obligation_id="semantic-revalidation",
        scope=scope,
        scope_id=scope_id,
        evidence_refs=_evidence("source"),
        required_intervention="Present the verified advisory reminder.",
        trigger_predicates=(
            ProspectiveTriggerPredicate(
                predicate_type="registered_condition",
                field="proof_ready",
                operator="eq",
                value=True,
            ),
        ),
        semantic_trigger_cues=("proof looks ready",),
        cancel_predicates=(
            ProspectiveTriggerPredicate(
                predicate_type="registered_condition",
                field="release_cancelled",
                operator="eq",
                value=True,
            ),
        ),
    )


def _origin(integrity: str = "observed") -> OriginRef:
    return OriginRef(
        channel="host_event",
        principal_id="host:codex",
        integrity_class=integrity,
        origin_id="local-service:ws:event-41",
    )


def _context(**overrides: object) -> ProspectiveTriggerContext:
    values: dict[str, object] = {
        "registered_condition": {"proof_ready": True},
        "origin": _origin(),
        "authority_ceiling": "advisory",
        "observed_at": "2026-07-15T01:00:00Z",
    }
    values.update(overrides)
    return ProspectiveTriggerContext(**values)


def _fire_event(event_id: str = "fire-1") -> ProspectiveLifecycleEvent:
    return ProspectiveLifecycleEvent(
        event_id=event_id,
        idempotency_key="registered:proof_ready:1",
        event_type="fire",
        expected_version=1,
        expected_status="armed",
        evidence_refs=_evidence("fire"),
    )


def _armed_client(tmp_path, obligation: ProspectiveObligation | None = None) -> MemoryClient:
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    current = obligation or _obligation()
    client.create_prospective_obligation(current, event_id="create")
    client.apply_prospective_lifecycle_event(
        current.obligation_id,
        ProspectiveLifecycleEvent(
            event_id="arm",
            event_type="arm",
            expected_version=0,
            expected_status="pending",
            evidence_refs=_evidence("arm"),
        ),
    )
    return client


def test_semantic_trigger_is_explicitly_experimental_unadvertised_and_cannot_fire(
    tmp_path,
) -> None:
    client = _armed_client(tmp_path)
    semantic_context = ProspectiveTriggerContext(
        semantic_text="The proof looks ready now.",
        origin=_origin(),
        authority_ceiling="advisory",
        observed_at="2026-07-15T01:00:00Z",
    )
    try:
        obligation = client.get_prospective_obligation("semantic-revalidation")
        assert obligation is not None
        decision = evaluate_prospective_trigger(obligation, semantic_context)
        assert decision.match_kind == "semantic"
        assert decision.candidate_class == "experimental"
        assert decision.advertised is False
        assert decision.delivery_eligible is False
        assert decision.host_action_authorized is False

        with pytest.raises(ValueError, match="delivery-eligible typed trigger"):
            client.apply_prospective_lifecycle_event(
                obligation.obligation_id,
                _fire_event(),
                trigger_context=semantic_context,
            )
        current = client.get_prospective_obligation(obligation.obligation_id)
        assert current is not None and current.status == "armed"
    finally:
        client.backend.close()


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (
            _context(registered_condition={"proof_ready": True, "release_cancelled": True}),
            "cancel_condition_matched",
        ),
        (_context(origin=_origin("reported")), "observed origin integrity"),
        (_context(authority_ceiling="directive"), "advisory authority ceiling"),
    ],
)
def test_fire_revalidation_fails_closed_before_mutation(
    tmp_path,
    context: ProspectiveTriggerContext,
    message: str,
) -> None:
    client = _armed_client(tmp_path)
    try:
        with pytest.raises(ValueError, match=message):
            client.apply_prospective_lifecycle_event(
                "semantic-revalidation",
                _fire_event(),
                trigger_context=context,
            )
        current = client.get_prospective_obligation("semantic-revalidation")
        assert current is not None and current.status == "armed"
    finally:
        client.backend.close()


def test_fire_revalidates_scope_current_state_and_duplicate_before_cas(tmp_path) -> None:
    scoped = replace(_obligation(), scope="task", scope_id="task-7")
    client = _armed_client(tmp_path, scoped)
    try:
        wrong_scope = _context(scope="task", scope_id="task-8")
        with pytest.raises(ValueError, match="scope_mismatch"):
            client.apply_prospective_lifecycle_event(
                scoped.obligation_id,
                _fire_event(),
                trigger_context=wrong_scope,
            )

        valid_scope = _context(scope="task", scope_id="task-7")
        fired = client.apply_prospective_lifecycle_event(
            scoped.obligation_id,
            _fire_event(),
            trigger_context=valid_scope,
        )
        assert fired["status"] == "fired"

        replay = client.apply_prospective_lifecycle_event(
            scoped.obligation_id,
            _fire_event(),
        )
        assert replay["replayed"] is True
        assert replay["receipt_id"] == fired["receipt_id"]

        with pytest.raises(CanonicalHeadConflict, match="stale"):
            client.apply_prospective_lifecycle_event(
                scoped.obligation_id,
                replace(
                    _fire_event("duplicate-fire-2"),
                    idempotency_key="registered:proof_ready:2",
                ),
                trigger_context=valid_scope,
            )
    finally:
        client.backend.close()


def test_low_level_public_transition_cannot_bypass_fire_revalidation(tmp_path) -> None:
    client = _armed_client(tmp_path)
    try:
        with pytest.raises(ValueError, match="apply_prospective_lifecycle_event"):
            client.transition_prospective_obligation(
                "semantic-revalidation",
                expected_version=1,
                expected_status="armed",
                status="fired",
                evidence_refs=_evidence("bypass"),
                event_id="bypass-fire",
            )
    finally:
        client.backend.close()
