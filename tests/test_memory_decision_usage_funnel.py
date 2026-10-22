from dataclasses import replace

import pytest

from um_mem import MemoryClient
from um_mem.contracts import EvidenceRef, MemoryDecisionEvent, MemoryUsageEvent
from um_mem.memory.backends.contracts import CanonicalHeadConflict, MemoryUsageRecord
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex


def _evidence(name: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="test_run",
            ref_id=name,
            produced_by="verifier",
            producer_id=f"verifier:{name}",
        ),
    )


def _decision(decision_id: str = "decision-1") -> MemoryDecisionEvent:
    return MemoryDecisionEvent(
        decision_id=decision_id,
        workspace_id="ws",
        run_id="run-1",
        task_signature="repair:python:unit-test",
        action_trace_id="action-trace-1",
        action="recall",
        canonical_snapshot_sha256=f"sha256:{'a' * 64}",
        context_snapshot_sha256=f"sha256:{'b' * 64}",
        live_state_snapshot_sha256=f"sha256:{'c' * 64}",
        live_trigger="test_state:required_suite_failed",
        candidate_ids=("memory-1", "memory-2"),
        selected_candidate_ids=("memory-1",),
        policy_revision="policy-7",
        catalog_revision="catalog-11",
        cost_tokens=321,
        cost_latency_ms=12,
        downstream_outcome="verification_pending",
        evidence_refs=_evidence(f"decision-{decision_id}"),
    )


def _usage(
    usage_type: str,
    sequence: int,
    *,
    memory_id: str = "memory-1",
    event_id: str = "",
) -> MemoryUsageEvent:
    return MemoryUsageEvent(
        event_id=event_id or f"usage-{sequence}-{usage_type}",
        decision_id="decision-1",
        memory_id=memory_id,
        workspace_id="ws",
        task_signature="repair:python:unit-test",
        action_trace_id="action-trace-1",
        usage_type=usage_type,
        sequence=sequence,
        evidence_refs=_evidence(f"usage-{sequence}"),
    )


def test_decision_contract_captures_full_context_action_cost_and_outcome() -> None:
    decision = _decision()
    assert decision.selected_candidate_ids == ("memory-1",)
    assert decision.cost_tokens == 321
    assert decision.downstream_outcome == "verification_pending"
    with pytest.raises(ValueError, match="selected_candidate_ids must be available candidates"):
        replace(decision, selected_candidate_ids=("not-available",))
    with pytest.raises(ValueError, match="cost_tokens must be a non-negative integer"):
        replace(decision, cost_tokens=True)
    with pytest.raises(ValueError, match="canonical_snapshot_sha256 must be a sha256 reference"):
        replace(decision, canonical_snapshot_sha256="not-a-snapshot")


@pytest.mark.parametrize(
    "usage_type",
    ["candidate", "selected", "rendered", "cited", "applied", "blocked"],
)
def test_usage_event_types_are_closed_and_linked(
    usage_type: str,
) -> None:
    event = _usage(usage_type, 0)
    assert event.usage_type == usage_type
    assert event.task_signature == _decision().task_signature
    assert event.action_trace_id == _decision().action_trace_id


@pytest.mark.parametrize("backend_id", ["mempalace", "hindsight", "mnemosyne"])
def test_selected_backend_round_trips_idempotent_decision_usage_trace(
    tmp_path,
    backend_id: str,
) -> None:
    root = tmp_path / backend_id
    decision = _decision()
    events = (
        _usage("candidate", 0),
        _usage("selected", 1),
        _usage("rendered", 2),
        _usage("cited", 3),
        _usage("applied", 4),
        _usage("blocked", 5, memory_id="memory-2"),
    )
    client = MemoryClient(root, workspace_id="ws", backend=backend_id)
    try:
        first = client.record_memory_decision(decision)
        replay = client.record_memory_decision(decision)
        assert first["replayed"] is False
        assert replay["replayed"] is True
        with pytest.raises(CanonicalHeadConflict, match="memory_decision_idempotency_conflict"):
            client.record_memory_decision(
                replace(decision, downstream_outcome="different-observation")
            )
        for event in events:
            assert client.record_memory_usage_event(event)["replayed"] is False
        assert client.record_memory_usage_event(events[0])["replayed"] is True
        assert client.backend.value_stats("memory-1")["uses_count"] == 0
        with pytest.raises(CanonicalHeadConflict, match="memory_usage_sequence_conflict"):
            client.record_memory_usage_event(
                _usage("candidate", 0, event_id="same-sequence-new-event")
            )
    finally:
        client.backend.close()

    reopened = MemoryClient(root, workspace_id="ws", backend=backend_id)
    try:
        trace = reopened.memory_decision_trace(decision.decision_id)
        assert trace.decision == decision
        assert trace.usage_events == events
        assert trace.ordinary_recall is False
        assert trace.host_action_authorized is False
        assert {event.usage_type for event in trace.usage_events} == {
            "candidate",
            "selected",
            "rendered",
            "cited",
            "applied",
            "blocked",
        }
    finally:
        reopened.backend.close()


def test_usage_recording_fails_closed_on_decision_linkage(tmp_path) -> None:
    client = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        with pytest.raises(CanonicalHeadConflict, match="memory_decision_missing"):
            client.record_memory_usage_event(_usage("candidate", 0))
        client.record_memory_decision(_decision())
        with pytest.raises(ValueError, match="decision journal namespace"):
            client.backend.record_usage(
                MemoryUsageRecord(
                    block_id="memory-decision-event:decision-1",
                    workspace_id="ws",
                    outcome="memory_decision",
                )
            )
        with pytest.raises(ValueError, match="task signature linkage"):
            client.record_memory_usage_event(
                replace(_usage("candidate", 0), task_signature="unrelated-task")
            )
        with pytest.raises(ValueError, match="available candidate linkage"):
            client.record_memory_usage_event(
                replace(_usage("candidate", 0), memory_id="not-available")
            )
        with pytest.raises(ValueError, match="selected candidate linkage"):
            client.record_memory_usage_event(_usage("applied", 0, memory_id="memory-2"))
        with pytest.raises(ValueError, match="workspace mismatch"):
            client.record_memory_decision(replace(_decision("wrong-ws"), workspace_id="other"))
    finally:
        client.backend.close()


def test_context_build_automatically_records_candidate_selected_and_rendered_neutral_trace(
    tmp_path,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "automatic.sqlite",
        backend_id="test",
        workspace_id="ws",
    )
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)
    try:
        saved = client.remember(
            "Parser source evidence is in src/parser.py.",
            kind="episode",
            run_id="run-automatic",
            requested_block_id="workspace_episodes",
        )
        items = client.build_context(
            "parser source evidence",
            run_id="run-automatic",
            include_candidates=True,
            logical_block_ids=["workspace_episodes"],
        )

        row = backend._conn.execute(
            "SELECT block_id FROM usage WHERE outcome='memory_decision'"
        ).fetchone()
        assert row is not None
        decision_id = str(row[0]).removeprefix("memory-decision-event:")
        trace = client.memory_decision_trace(decision_id)
        assert items
        assert trace.decision.downstream_outcome == "neutral_pending_verifier"
        assert trace.decision.candidate_ids == (saved.memory_id,)
        assert trace.decision.selected_candidate_ids == (saved.memory_id,)
        assert [event.usage_type for event in trace.usage_events] == [
            "candidate",
            "selected",
            "rendered",
        ]
    finally:
        backend.close()
