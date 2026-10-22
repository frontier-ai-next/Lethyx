from dataclasses import replace

import pytest

from um_mem import MemoryClient
from um_mem.contracts import (
    EvidenceRef,
    OriginRef,
    ProspectiveLifecycleEvent,
    ProspectiveObligation,
    ProspectiveTelemetry,
    ProspectiveTriggerPredicate,
)
from um_mem.memory.backends.contracts import CanonicalHeadConflict
from um_mem.memory.prospective import ProspectiveTriggerContext, evaluate_prospective_trigger


def _evidence(name: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="test_run",
            ref_id=name,
            produced_by="verifier",
            producer_id=f"verifier:{name}",
        ),
    )


def _predicate(kind: str, field: str, value: str | bool) -> ProspectiveTriggerPredicate:
    return ProspectiveTriggerPredicate(
        predicate_type=kind,
        field=field,
        operator="eq",
        value=value,
    )


def _armed(
    obligation_id: str = "guard-restore", *, scope_id: str = "task-a"
) -> ProspectiveObligation:
    return ProspectiveObligation(
        obligation_id=obligation_id,
        scope="task",
        scope_id=scope_id,
        evidence_refs=_evidence(f"source-{obligation_id}"),
        required_intervention=f"Handle safe-coding obligation {obligation_id}.",
        trigger_predicates=(_predicate("registered_condition", "ready", True),),
        cancel_predicates=(_predicate("registered_condition", "cancelled", True),),
        deadline_at="2026-07-15T02:00:00Z",
        expires_at="2026-07-15T03:00:00Z",
    ).transition_to("armed", evidence_refs=_evidence(f"arm-{obligation_id}"))


def _origin() -> OriginRef:
    return OriginRef(
        channel="host_event",
        principal_id="host:codex",
        integrity_class="observed",
        origin_id="local-service:ws:exit",
    )


@pytest.mark.parametrize(
    ("obligation_id", "kind", "field", "value"),
    [
        ("restore-guard", "file_state", "temporary_guard_disabled", True),
        ("remove-debug-bypass", "file_state", "debug_bypass_present", True),
        ("update-schema-fixtures-docs", "registered_condition", "schema_changed", True),
        ("required-suite-before-release", "test_state", "required_suite", "failed"),
    ],
)
def test_safe_coding_obligation_fixtures_use_typed_conditions(
    obligation_id: str,
    kind: str,
    field: str,
    value: str | bool,
) -> None:
    obligation = ProspectiveObligation(
        obligation_id=obligation_id,
        scope="workspace",
        evidence_refs=_evidence(obligation_id),
        required_intervention=f"Resolve {obligation_id} before release.",
        trigger_predicates=(_predicate(kind, field, value),),
    ).transition_to("armed", evidence_refs=_evidence(f"arm-{obligation_id}"))
    context_values = {kind: {field: value}, "observed_at": "2026-07-15T01:00:00Z"}
    positive = evaluate_prospective_trigger(obligation, ProspectiveTriggerContext(**context_values))
    negative = evaluate_prospective_trigger(
        obligation,
        ProspectiveTriggerContext(observed_at="2026-07-15T01:00:00Z"),
    )
    assert positive.delivery_eligible is True
    assert positive.trigger_type == kind
    assert negative.matched is False


def test_unconditioned_always_remind_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unconditioned always-remind"):
        ProspectiveObligation(
            obligation_id="always-remind",
            scope="workspace",
            evidence_refs=_evidence("always-remind"),
            required_intervention="Repeat this on every turn.",
        )


def test_cancel_and_expire_never_create_delivery_alarm() -> None:
    obligation = _armed()
    cancel = evaluate_prospective_trigger(
        obligation,
        ProspectiveTriggerContext(
            registered_condition={"cancelled": True},
            scope="task",
            scope_id="task-a",
            observed_at="2026-07-15T01:00:00Z",
        ),
    )
    expire = evaluate_prospective_trigger(
        obligation,
        ProspectiveTriggerContext(
            scope="task",
            scope_id="task-a",
            observed_at="2026-07-15T03:00:01Z",
        ),
    )
    assert (cancel.recommended_transition, cancel.delivery_eligible) == ("cancelled", False)
    assert (expire.recommended_transition, expire.delivery_eligible) == ("expired", False)
    for terminal in ("cancelled", "expired"):
        ended = obligation.transition_to(terminal, evidence_refs=_evidence(terminal))
        assert (
            evaluate_prospective_trigger(
                ended,
                ProspectiveTriggerContext(
                    registered_condition={"ready": True},
                    scope="task",
                    scope_id="task-a",
                    observed_at="2026-07-15T01:00:00Z",
                ),
            ).matched
            is False
        )


def test_missed_deadline_and_overload_have_dedicated_telemetry() -> None:
    obligation = _armed().with_telemetry(
        missed_trigger_delta=1,
        overloaded_trigger_delta=2,
        late_trigger_delta=3,
    )
    late_deadline = evaluate_prospective_trigger(
        obligation,
        ProspectiveTriggerContext(
            scope="task",
            scope_id="task-a",
            observed_at="2026-07-15T02:00:01Z",
        ),
    )
    assert late_deadline.delivery_eligible is True
    assert late_deadline.trigger_type == "deadline"
    assert obligation.telemetry.missed_trigger_count == 1
    assert obligation.telemetry.overloaded_trigger_count == 2
    assert obligation.telemetry.late_trigger_count == 3


def test_restart_competing_task_and_exact_once_fire_exit_gate(tmp_path) -> None:
    armed = _armed()
    pending = replace(armed, status="pending", transition_from=None, transition_evidence_refs=())
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
        assert (
            client.prospective_trigger_candidates(
                ProspectiveTriggerContext(
                    registered_condition={"ready": True},
                    scope="task",
                    scope_id="task-b",
                    observed_at="2026-07-15T01:00:00Z",
                )
            )
            == []
        )
    finally:
        client.backend.close()

    reopened = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    fire = ProspectiveLifecycleEvent(
        event_id="fire",
        idempotency_key="task-a:ready:1",
        event_type="fire",
        expected_version=1,
        expected_status="armed",
        evidence_refs=_evidence("fire"),
    )
    context = ProspectiveTriggerContext(
        registered_condition={"ready": True},
        scope="task",
        scope_id="task-a",
        origin=_origin(),
        observed_at="2026-07-15T01:00:00Z",
    )
    try:
        first = reopened.apply_prospective_lifecycle_event(
            pending.obligation_id, fire, trigger_context=context
        )
        replay = reopened.apply_prospective_lifecycle_event(pending.obligation_id, fire)
        assert replay["replayed"] is True
        assert replay["receipt_id"] == first["receipt_id"]
        with pytest.raises(CanonicalHeadConflict, match="stale"):
            reopened.apply_prospective_lifecycle_event(
                pending.obligation_id,
                replace(fire, event_id="fire-duplicate", idempotency_key="task-a:ready:2"),
                trigger_context=context,
            )
    finally:
        reopened.backend.close()


def test_prospective_telemetry_is_published_outside_ordinary_recall(tmp_path) -> None:
    obligation = replace(
        _armed("telemetry"),
        status="pending",
        transition_from=None,
        transition_evidence_refs=(),
        telemetry=ProspectiveTelemetry(
            false_positive_count=1,
            missed_trigger_count=2,
            overloaded_trigger_count=3,
            attention_interruption_count=4,
            duplicate_fire_count=5,
            late_trigger_count=6,
        ),
    )
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        client.create_prospective_obligation(obligation, event_id="create")
        report = client.prospective_telemetry_report()
        assert report.channel == "prospective_obligation_telemetry"
        assert report.ordinary_recall is False
        assert report.host_action_authorized is False
        assert len(report.entries) == 1
        entry = report.entries[0]
        assert entry.false_activation_count == 1
        assert entry.missed_trigger_count == 2
        assert entry.overloaded_trigger_count == 3
        assert entry.attention_interruption_count == 4
        assert entry.duplicate_fire_count == 5
        assert entry.late_trigger_count == 6
    finally:
        client.backend.close()


def test_semantic_candidate_never_crosses_host_gate() -> None:
    obligation = replace(_armed(), semantic_trigger_cues=("looks ready",))
    decision = evaluate_prospective_trigger(
        obligation,
        ProspectiveTriggerContext(
            semantic_text="The change looks ready.",
            scope="task",
            scope_id="task-a",
            observed_at="2026-07-15T01:00:00Z",
        ),
    )
    assert decision.candidate_class == "experimental"
    assert decision.delivery_eligible is False
    assert decision.host_action_authorized is False
