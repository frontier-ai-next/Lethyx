from typing import Any

import pytest

from um_mem.contracts import EvidenceRef, MemoryBlockRecord
from um_mem.memory.selection import MemorySelectionPolicy, SelectionContext


def _block(**overrides: Any) -> MemoryBlockRecord:
    base = {
        "block_id": "active",
        "content": "alpha project memory",
        "title": "Alpha",
        "kind": "observation",
        "status": "accepted",
        "retention": "durable",
        "delivery_policy": "on_demand",
        "temperature": "warm",
        "authority": "advisory",
        "trust_level": "workspace_verified",
        "scope": "workspace",
        "verification_status": "verified",
        "verification_methods": ("test_run",),
        "promotion_decision": "accepted",
        "workspace_id": "alpha",
        "run_id": "run-1",
        "producer": "test",
        "tags": ("alpha",),
        "evidence_refs": (
            EvidenceRef(
                ref_type="test_run",
                ref_id="policy",
                produced_by="verifier",
                producer_id="verifier:policy",
            ),
        ),
        "metadata": {},
    }
    base.update(overrides)
    return MemoryBlockRecord(**base)


class _Backend:
    def __init__(self, records: list[MemoryBlockRecord]) -> None:
        self.records = records

    def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
        del query
        return list(self.records)


def test_selection_policy_excludes_inactive_and_expired_records_by_default() -> None:
    policy = MemorySelectionPolicy()
    ctx = SelectionContext(query="alpha", scope_chain=("workspace:alpha",))

    assert policy.eligible_for_recall(_block(), ctx) is True
    assert (
        policy.eligible_for_recall(
            _block(),
            SelectionContext(
                query="alpha",
                scope_chain=("workspace:alpha",),
                tags=("backend-owned-filter",),
            ),
        )
        is True
    )
    assert policy.eligible_for_recall(_block(status="quarantined"), ctx) is False
    assert (
        policy.eligible_for_recall(
            _block(status="retracted", retention="archive", delivery_policy="hidden"),
            ctx,
        )
        is False
    )
    assert policy.eligible_for_recall(_block(expires_at="2000-01-01T00:00:00Z"), ctx) is False


def test_selection_policy_requires_explicit_horizon_and_candidate_gate() -> None:
    policy = MemorySelectionPolicy()
    active_ctx = SelectionContext(query="alpha", scope_chain=("workspace:alpha",))
    archive_ctx = SelectionContext(
        query="alpha", scope_chain=("workspace:alpha",), source_horizon="archive"
    )
    candidate_ctx = SelectionContext(
        query="alpha",
        scope_chain=("workspace:alpha",),
        source_horizon="archive",
        include_candidates=True,
    )

    cold = _block(block_id="cold", temperature="cold")
    archived = _block(block_id="archived", retention="archive")
    candidate = _block(block_id="candidate", status="candidate", run_id="")

    assert policy.eligible_for_recall(cold, active_ctx) is True
    assert policy.eligible_for_recall(archived, active_ctx) is False
    assert policy.eligible_for_recall(candidate, archive_ctx) is False
    assert policy.eligible_for_recall(cold, archive_ctx) is True
    assert policy.eligible_for_recall(archived, archive_ctx) is True
    assert policy.eligible_for_recall(candidate, candidate_ctx) is True


def test_selection_policy_limits_candidates_to_the_requested_run() -> None:
    policy = MemorySelectionPolicy()
    current_run = SelectionContext(
        query="alpha",
        scope_chain=("workspace:alpha",),
        include_candidates=True,
        run_id="run-1",
    )

    assert policy.eligible_for_recall(_block(status="candidate"), current_run) is True
    assert (
        policy.eligible_for_recall(
            _block(status="candidate"),
            SelectionContext(
                query="alpha",
                scope_chain=("workspace:alpha",),
                include_candidates=True,
            ),
        )
        is False
    )
    assert (
        policy.eligible_for_recall(
            _block(status="candidate", run_id="run-2"),
            current_run,
        )
        is False
    )
    assert policy.eligible_for_recall(_block(run_id="run-2"), current_run) is True


def test_selection_policy_keeps_workspace_boundaries_and_directive_gate() -> None:
    policy = MemorySelectionPolicy()
    ctx = SelectionContext(query="alpha", scope_chain=("workspace:alpha",))

    assert policy.eligible_for_recall(_block(workspace_id="beta"), ctx) is False
    assert policy.eligible_for_directive(_block(), ctx) is False
    assert (
        policy.eligible_for_directive(
            _block(
                authority="directive",
            ),
            ctx,
        )
        is True
    )


def test_historical_recall_uses_the_requested_time_and_never_grants_directive_authority() -> None:
    policy = MemorySelectionPolicy()
    record = _block(
        authority="directive",
        valid_from="2020-01-01T00:00:00Z",
        expires_at="2020-02-01T00:00:00Z",
    )
    historical = SelectionContext(
        scope_chain=("workspace:alpha",),
        historical=True,
        as_of="2020-01-15T00:00:00Z",
    )
    assert policy.eligible_for_recall(record, historical)
    assert not policy.eligible_for_directive(record, historical)
    assert not policy.eligible_for_recall(
        record, SelectionContext(scope_chain=("workspace:alpha",))
    )
    for date in ("2019-12-31T23:59:59Z", "2020-02-01T00:00:00Z"):
        assert not policy.eligible_for_recall(
            record,
            SelectionContext(
                scope_chain=("workspace:alpha",),
                historical=True,
                as_of=date,
            ),
        )
    assert policy.eligible_for_recall(
        record,
        SelectionContext(
            scope_chain=("workspace:alpha",),
            historical=True,
            source_horizon="archive",
        ),
    )


def test_selection_rejects_an_invalid_requested_time_before_comparing_records() -> None:
    with pytest.raises(ValueError, match="as_of"):
        SelectionContext(historical=True, as_of="not-a-time")


def test_unknown_event_date_is_not_replaced_by_recording_time() -> None:
    policy = MemorySelectionPolicy()
    record = _block(recorded_at="2026-10-17T12:00:00Z")
    assert policy.eligible_for_recall(
        record,
        SelectionContext(
            scope_chain=("workspace:alpha",), historical=True, as_of="2023-05-30T00:00:00Z"
        ),
    )
    assert not policy.eligible_for_recall(
        record,
        SelectionContext(
            scope_chain=("workspace:other",), historical=True, as_of="2023-05-30T00:00:00Z"
        ),
    )
