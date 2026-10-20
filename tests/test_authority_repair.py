from dataclasses import replace
import json
import time
from typing import Any
from urllib.parse import quote

import pytest
import lethyx.memory.backends.hindsight as hindsight_module

from lethyx.contracts import (
    EvidenceRef,
    MemoryBlockRecord,
    ProspectiveObligation,
    RecordBlockMembership,
)
from lethyx.memory.backends.contracts import BackendCapabilities, CanonicalHeadConflict
from lethyx.memory.backends.hindsight import HindsightBackend
from lethyx.memory.backends.query_policy import native_block_tag
from lethyx.memory.backends.mempalace import MemPalaceBackend, _record_partition_identity
from lethyx.memory.backends.mnemosyne import MnemosyneBackend
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.hindsight.config import HindsightConfig
from lethyx.memory.kernel.models import block_record_to_event
from lethyx.memory.kernel.writer import _write_memory_event
from lethyx.memory.mutation import MutationDecision
from lethyx import MemoryClient


def _evidence(name: str) -> EvidenceRef:
    return EvidenceRef(
        ref_type="test_run",
        ref_id=name,
        produced_by="verifier",
        producer_id=f"verifier:order16:{name}",
    )


def _record(block_id: str, **overrides: object) -> MemoryBlockRecord:
    data: dict[str, object] = {
        "block_id": block_id,
        "revision_id": f"revision:{block_id}",
        "content": f"content:{block_id}",
        "title": "Order16 record",
        "block_membership": RecordBlockMembership(
            primary_block_id="workspace_episodes",
            catalog_revision_id="test-catalog",
        ),
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
        "workspace_id": "ws",
        "run_id": "run",
        "stage_id": "stage",
        "evidence_refs": (_evidence(block_id),),
    }
    data.update(overrides)
    return MemoryBlockRecord(**data)


def _obligation(obligation_id: str) -> ProspectiveObligation:
    return ProspectiveObligation(
        obligation_id=obligation_id,
        scope="workspace",
        evidence_refs=(_evidence(obligation_id),),
    )


class _HindsightStatus404(Exception):
    status = 404


class _HindsightFreshDocumentClient:
    def __init__(self, *, error_status: int | None = None) -> None:
        self.error_status = error_status
        self.retained: list[dict[str, Any]] = []
        self.documents = self

    def get_document(self, bank_id: str, document_id: str) -> dict[str, Any]:
        for row in self.retained:
            if row["bank_id"] == bank_id and row["id"] == document_id:
                return dict(row)
        error = _HindsightStatus404("not found")
        if self.error_status is not None:
            error.status = self.error_status
        raise error

    def retain(self, **kwargs: Any) -> dict[str, Any]:
        document_id = str(kwargs["document_id"])
        self.retained.append(
            {
                "id": document_id,
                "bank_id": str(kwargs["bank_id"]),
                "original_text": str(kwargs["content"]),
                "document_metadata": dict(kwargs["metadata"]),
                "tags": list(kwargs.get("tags") or []),
            }
        )
        return {"ok": True, "document_id": document_id}


class _SyncHttpResponse:
    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_SyncHttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _HindsightCapabilityClient:
    class banks:
        @staticmethod
        def list() -> list[str]:
            return []


class _HindsightCompleteCapabilityClient(_HindsightCapabilityClient):
    def retain(self, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def recall(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []


class _MnemosyneShapeFake:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.remembered: list[str] = []

    def remember(
        self, content: str, *, source: str, metadata: dict[str, Any], **_kwargs: Any
    ) -> str:
        native_id = f"native-{len(self.rows) + 1}"
        self.remembered.append(content)
        self.rows[native_id] = {
            "id": native_id,
            "content": MnemosyneBackend._provider_temporal_tags(content),
            "source": source,
            "metadata": dict(metadata),
        }
        return native_id

    def get_all_memories(self) -> list[dict[str, Any]]:
        return [
            {
                "id": row["id"],
                "content": row["content"][:500],
                "source": row["source"],
                "timestamp": "now",
                "session_id": "ws",
                "importance": 0.5,
            }
            for row in self.rows.values()
        ]

    def get(self, native_id: str) -> dict[str, Any] | None:
        row = self.rows.get(native_id)
        return dict(row) if row is not None else None

    def recall(self, _query: str, *, top_k: int, source: str = "") -> list[dict[str, Any]]:
        return [
            {
                "id": row["id"],
                "content": row["content"][:500],
                "source": row["source"],
                "score": 1.0 - index / 10,
            }
            for index, row in enumerate(
                [row for row in self.rows.values() if not source or row["source"] == source][:top_k]
            )
        ]


def test_prospective_cas_retries_without_duplicate_events_and_reopens(tmp_path) -> None:
    path = tmp_path / "prospective.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    obligation = _obligation("obligation-1")
    try:
        created = index.create_prospective_obligation(obligation, event_id="create-1")
        replayed = index.create_prospective_obligation(obligation, event_id="create-1")
        assert replayed["replayed"] is True
        assert replayed["receipt_id"] == created["receipt_id"]

        armed = index.transition_prospective_obligation(
            obligation.obligation_id,
            expected_version=0,
            expected_status="pending",
            status="armed",
            evidence_refs=(_evidence("arm-1"),),
            event_id="arm-1",
        )
        armed_retry = index.transition_prospective_obligation(
            obligation.obligation_id,
            expected_version=0,
            expected_status="pending",
            status="armed",
            evidence_refs=(_evidence("arm-1"),),
            event_id="arm-1",
        )
        assert armed_retry["replayed"] is True
        assert armed_retry["receipt_id"] == armed["receipt_id"]
        before_stale = {
            "obligation": index._conn.execute(
                "SELECT * FROM prospective_obligations ORDER BY obligation_id"
            ).fetchall(),
            "events": index._conn.execute(
                "SELECT * FROM prospective_obligation_events ORDER BY obligation_id, event_id"
            ).fetchall(),
        }
        with pytest.raises(CanonicalHeadConflict, match="stale"):
            index.transition_prospective_obligation(
                obligation.obligation_id,
                expected_version=0,
                expected_status="pending",
                status="cancelled",
                evidence_refs=(_evidence("stale"),),
                event_id="stale-1",
            )
        after_stale = {
            "obligation": index._conn.execute(
                "SELECT * FROM prospective_obligations ORDER BY obligation_id"
            ).fetchall(),
            "events": index._conn.execute(
                "SELECT * FROM prospective_obligation_events ORDER BY obligation_id, event_id"
            ).fetchall(),
        }
        assert after_stale == before_stale
        with pytest.raises(CanonicalHeadConflict, match="idempotency"):
            index.transition_prospective_obligation(
                obligation.obligation_id,
                expected_version=0,
                expected_status="pending",
                status="cancelled",
                evidence_refs=(_evidence("conflict"),),
                event_id="other-event",
                idempotency_key="arm-1",
            )
        with pytest.raises(ValueError, match="invalid prospective transition"):
            index.transition_prospective_obligation(
                obligation.obligation_id,
                expected_version=1,
                expected_status="armed",
                status="satisfied",
                evidence_refs=(_evidence("illegal"),),
                event_id="illegal-1",
            )
        assert (
            index._conn.execute("SELECT COUNT(*) FROM prospective_obligation_events").fetchone()[0]
            == 2
        )
    finally:
        index.close()

    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    try:
        current = reopened.get_prospective_obligation(obligation.obligation_id)
        assert current is not None and current.status == "armed"
        assert (
            reopened._conn.execute("SELECT COUNT(*) FROM prospective_obligation_events").fetchone()[
                0
            ]
            == 2
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("corruption", ["missing", "hash", "kind"])
def test_canonical_payload_pointer_corruption_fails_closed_and_staged_row_is_reused(
    tmp_path, corruption: str
) -> None:
    index = BackendSQLiteIndex(tmp_path / "canonical.sqlite", backend_id="test", workspace_id="ws")
    record = _record("pointer-1", canonical_key="topic")
    mutation = MutationDecision(record=record)
    try:
        reservation = index.reserve_canonical_head(
            workspace_id="ws",
            canonical_key="topic",
            predecessor_id="",
            successor_id=record.block_id,
            successor_status=record.status,
            successor_authority=record.authority,
            mutation=mutation,
        )
        with pytest.raises(CanonicalHeadConflict, match="write_journal"):
            index.complete_canonical_head(reservation, mutation)
        entry = index.open_write_outbox(record=record, store="test")
        assert index._conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 0
        index.complete_canonical_head(reservation, mutation)
        index.complete_write_outbox(entry.commit_revision)
        assert index._conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 1
        assert index.canonical_history(canonical_key="topic", workspace_id="ws") == [record]
        payload_id, _payload_sha256 = index._conn.execute(
            "SELECT record_payload_id, record_payload_sha256 FROM canonical_mutations"
        ).fetchone()
        if corruption == "missing":
            index._conn.execute("DELETE FROM canonical_payloads WHERE payload_id=?", (payload_id,))
        elif corruption == "hash":
            index._conn.execute(
                "UPDATE canonical_mutations SET record_payload_sha256=?",
                ("0" * 64,),
            )
        else:
            foreign_id, foreign_sha256 = index._conn.execute(
                "SELECT payload_id, payload_sha256 FROM canonical_head_outbox"
            ).fetchone()
            index._conn.execute(
                "UPDATE canonical_mutations SET record_payload_id=?, record_payload_sha256=?",
                (foreign_id, foreign_sha256),
            )
        index._conn.commit()
        with pytest.raises(ValueError, match="canonical payload"):
            index.canonical_history(canonical_key="topic", workspace_id="ws")
    finally:
        index.close()
    reopened = BackendSQLiteIndex(
        tmp_path / "canonical.sqlite", backend_id="test", workspace_id="ws"
    )
    try:
        with pytest.raises(ValueError, match="canonical payload"):
            reopened.canonical_history(canonical_key="topic", workspace_id="ws")
    finally:
        reopened.close()


def test_prospective_two_connection_cas_has_one_transition(tmp_path) -> None:
    path = tmp_path / "prospective-race.sqlite"
    seed = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    obligation = _obligation("obligation-race")
    seed.create_prospective_obligation(obligation, event_id="create-race")
    seed.close()
    left = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    right = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    try:
        left_result = left.transition_prospective_obligation(
            obligation.obligation_id,
            expected_version=0,
            expected_status="pending",
            status="armed",
            evidence_refs=(_evidence("arm-race"),),
            event_id="arm-race",
        )
        with pytest.raises(CanonicalHeadConflict, match="stale"):
            right.transition_prospective_obligation(
                obligation.obligation_id,
                expected_version=0,
                expected_status="pending",
                status="cancelled",
                evidence_refs=(_evidence("cancel-race"),),
                event_id="cancel-race",
            )
        retry = right.transition_prospective_obligation(
            obligation.obligation_id,
            expected_version=0,
            expected_status="pending",
            status="armed",
            evidence_refs=(_evidence("arm-race"),),
            event_id="arm-race",
        )
        assert retry["replayed"] is True
        assert retry["receipt_id"] == left_result["receipt_id"]
        assert (
            left._conn.execute("SELECT COUNT(*) FROM prospective_obligation_events").fetchone()[0]
            == 2
        )
    finally:
        left.close()
        right.close()


def test_native_overlay_requires_committed_full_write_journal_and_time_window(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "overlay.sqlite", backend_id="test", workspace_id="ws")
    record = _record("overlay-1")
    query = {"scope_chain": ["workspace:ws"], "source_horizon": "durable"}
    try:
        index.retain_block(record)
        assert index.overlay_native_hit(record, workspace_id="ws", query=query) is None
        entry = index.open_write_outbox(record=record, store="test")
        assert index.get_block(record.block_id, workspace_id="ws") is None
        index.complete_write_outbox(entry.commit_revision)
        assert index.overlay_native_hit(record, workspace_id="ws", query=query) == record
        assert (
            index.overlay_native_hit(
                replace(record, content="tampered"), workspace_id="ws", query=query
            )
            is None
        )
        future = replace(record, block_id="overlay-future", valid_from="2999-01-01T00:00:00+00:00")
        future_entry = index.open_write_outbox(record=future, store="test")
        index.retain_block(future)
        index.complete_write_outbox(future_entry.commit_revision)
        assert index.overlay_native_hit(future, workspace_id="ws", query=query) is None
    finally:
        index.close()


def test_mempalace_stages_outbox_before_head_and_replays_missing_native_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    backend = MemPalaceBackend(tmp_path, workspace_id="ws")
    original_complete = backend.complete_canonical_head
    observed: list[str] = []

    def check_staged(reservation, mutation):
        pending = backend._index.pending_write_outbox()
        observed.extend(
            entry.status for entry in pending if entry.block_id == mutation.record.block_id
        )
        return original_complete(reservation, mutation)

    original_add = backend._native.add_record
    fail_once = True

    def fail_projection(record, *, commit_revision):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise OSError("native lag")
        return original_add(record, commit_revision=commit_revision)

    monkeypatch.setattr(backend, "complete_canonical_head", check_staged)
    monkeypatch.setattr(backend._native, "add_record", fail_projection)
    try:
        result = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend).remember(
            "recover native projection",
            title="Order16 replay",
            requested_block_id="workspace_episodes",
        )
        block_id = result.memory_id
        assert observed == ["pending"]
        assert result.saved is True
        assert result.raw["projection_pending"] is True
        assert backend._index.write_outbox_status(block_id) == "pending"
    finally:
        backend.close()

    original_project = MemPalaceBackend._project_write_outbox
    replay_head_status: list[str | None] = []

    def observe_replay_head(self, entry, record):
        replay_head_status.append(self._index.canonical_head_status(record.block_id))
        return original_project(self, entry, record)

    monkeypatch.setattr(MemPalaceBackend, "_project_write_outbox", observe_replay_head)
    reopened = MemPalaceBackend(tmp_path, workspace_id="ws")
    try:
        assert MemoryClient(
            tmp_path, workspace_id="ws", backend_instance=reopened
        ).projection_barrier(block_id, timeout_seconds=5)["ok"]
        assert replay_head_status == ["committed"]
        assert reopened._index.write_outbox_status(block_id) == "committed"
        stored = reopened.get_block(block_id)
        assert stored is not None
        assert reopened._index.canonical_head_status(block_id) == "committed"
        native = reopened._native.collection(
            "palace.idea",
            partition=_record_partition_identity(stored),
        ).get(ids=[block_id])
        assert native.ids == [block_id]
    finally:
        reopened.close()


def test_post_head_native_failure_is_visible_and_retry_projects_same_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    backend = MemPalaceBackend(tmp_path, workspace_id="ws")
    original_add = backend._native.add_record
    attempts: list[str] = []

    def fail_once(record, *, commit_revision: int):
        attempts.append(record.block_id)
        if len(attempts) == 1:
            raise OSError("native lag")
        return original_add(record, commit_revision=commit_revision)

    monkeypatch.setattr(backend._native, "add_record", fail_once)
    try:
        first = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend).remember(
            "retry the same canonical record",
            requested_block_id="workspace_episodes",
        )
        record = backend.get_block(first.memory_id)
        assert record is not None
        assert first.saved is True
        assert first.raw["projection_pending"] is True
        assert backend._index.canonical_head_status(record.block_id) == "committed"

        retry = _write_memory_event(
            tmp_path,
            block_record_to_event(record),
            workspace_id="ws",
            backend_instance=backend,
            mutation=MutationDecision(record=record, action="retry"),
        )
        assert retry.saved is True
        assert retry.raw["replayed"] is True
        assert MemoryClient(
            tmp_path, workspace_id="ws", backend_instance=backend
        ).projection_barrier(record.block_id, timeout_seconds=5)["ok"]
        assert backend._index.write_outbox_status(record.block_id) == "committed"
        assert attempts == [record.block_id, record.block_id]
    finally:
        backend.close()


class _HindsightWithoutDocumentLookup:
    def __init__(self) -> None:
        self.calls = 0

    def retain(self, **kwargs):
        self.calls += 1
        return {"ok": True, "document_id": kwargs["document_id"]}


def test_hindsight_replays_deterministic_document_when_lookup_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    native = _HindsightWithoutDocumentLookup()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
        reconcile_pending_writes=False,
    )
    monkeypatch.setattr(
        backend._index,
        "complete_write_outbox",
        lambda _revision: (_ for _ in ()).throw(OSError("crash after native retain")),
    )
    record = _record("hindsight-replay")
    try:
        with pytest.raises(OSError, match="crash after native retain"):
            backend.retain_block(record)
    finally:
        backend.close()

    reopened = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )
    try:
        deadline = time.monotonic() + 5
        while reopened._index.write_outbox_status(record.block_id) != "committed":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert native.calls == 2
        assert reopened._index.write_outbox_status(record.block_id) == "committed"
    finally:
        reopened.close()


def test_capability_defaults_and_unavailable_probes_are_false(tmp_path) -> None:
    defaults = BackendCapabilities("mempalace")
    assert not any(value for value in vars(defaults).values() if isinstance(value, bool))
    mempalace = MemPalaceBackend(tmp_path, workspace_id="mempalace-ws")
    hindsight = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="hindsight-ws",
        config=HindsightConfig(enabled=False),
    )
    mnemosyne = MnemosyneBackend(repo_root=tmp_path, workspace_id="mnemosyne-ws", enabled=False)
    try:
        probe = mempalace.health()
        effective = bool(probe.get("native_ok")) and probe.get("effective_mode") == "native"
        mempalace_caps = mempalace.capabilities()
        assert mempalace_caps.persistent is effective
        assert mempalace_caps.semantic_recall is effective
        assert mempalace_caps.fts_recall is effective
        assert not any(
            value for value in vars(hindsight.capabilities()).values() if isinstance(value, bool)
        )
        assert not any(
            value for value in vars(mnemosyne.capabilities()).values() if isinstance(value, bool)
        )
    finally:
        mempalace.close()
        hindsight.close()
        mnemosyne.close()


@pytest.mark.parametrize("backend_id", ["mempalace", "hindsight", "mnemosyne"])
def test_missing_canonical_write_journal_fails_reservation_and_continues(
    tmp_path, backend_id: str
) -> None:
    if backend_id == "mempalace":
        backend = MemPalaceBackend(tmp_path, workspace_id="ws", reconcile_pending_writes=False)
    elif backend_id == "hindsight":
        backend = HindsightBackend(
            repo_root=tmp_path,
            workspace_id="ws",
            config=HindsightConfig(enabled=False),
        )
    else:
        backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=False)
    record = replace(
        _record(f"missing-journal-{backend_id}"),
        canonical_key=f"missing-journal:{backend_id}",
    )
    mutation = MutationDecision(record=record)
    try:
        backend._index.reserve_canonical_head(
            workspace_id="ws",
            canonical_key=record.canonical_key,
            predecessor_id="",
            successor_id=record.block_id,
            successor_status=record.status,
            successor_authority=record.authority,
            mutation=mutation,
        )
    finally:
        backend.close()

    if backend_id == "mempalace":
        reopened = MemPalaceBackend(tmp_path, workspace_id="ws")
    elif backend_id == "hindsight":
        reopened = HindsightBackend(
            repo_root=tmp_path,
            workspace_id="ws",
            config=HindsightConfig(enabled=False),
        )
    else:
        reopened = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=False)
    try:
        assert reopened._index.canonical_head_status(record.block_id) == "failed"
        assert reopened._index.pending_canonical_head_reservations() == []
        assert reopened._index.get_block(record.block_id, workspace_id="ws") is None
    finally:
        reopened.close()


def test_hindsight_maps_only_real_404_as_missing_on_fresh_deterministic_write(tmp_path) -> None:
    native = _HindsightFreshDocumentClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )
    try:
        result = backend.retain_block(_record("hindsight-404"))
        assert result["native_written"] is True
        assert len(native.retained) == 1
    finally:
        backend.close()

    native_error = _HindsightFreshDocumentClient(error_status=500)
    backend = HindsightBackend(
        repo_root=tmp_path / "error",
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native_error,
    )
    try:
        with pytest.raises(_HindsightStatus404):
            backend._native_record_for_write_outbox(
                backend._index.open_write_outbox(record=_record("hindsight-500"), store="hindsight")
            )
    finally:
        backend.close()


def test_hindsight_external_projection_lookup_uses_sync_http_without_async_client_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(
            enabled=True,
            base_url="http://127.0.0.1:8888",
            embedded=False,
        ),
        reconcile_pending_writes=False,
    )
    record = _record("sync-http")
    entry = backend._index.open_write_outbox(record=record, store="hindsight")
    payload = {
        "id": f"Lethyx:block:{record.block_id}",
        "bank_id": backend._block_native_metadata(
            record,
            commit_revision=entry.commit_revision,
        )["bank_id"],
        "original_text": record.content,
        "document_metadata": backend._block_native_metadata(
            record,
            commit_revision=entry.commit_revision,
        ),
        "tags": [
            native_block_tag(record.block_membership.primary_block_id),
            *record.tags,
        ],
    }
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return _SyncHttpResponse(payload)

    monkeypatch.setattr(hindsight_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        backend,
        "_client",
        lambda: pytest.fail("external projection lookup opened the async client"),
    )
    try:
        restored, error = backend._native_record_for_write_outbox(entry)
    finally:
        backend.close()

    assert error == ""
    assert restored == record
    assert len(requests) == 1
    assert requests[0][1] == 30.0
    bank_id = str(payload["bank_id"])
    assert f"/v1/default/banks/{quote(bank_id, safe='')}/documents/" in requests[0][0].full_url


def test_mnemosyne_real_shape_hydrates_details_and_replays_compact_native_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    native = _MnemosyneShapeFake()
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    backend._native_memory = native
    content = "same content " + ("x" * 620) + " on 2026-07-14 for 3 days"
    first = replace(_record("mnemo-same-1"), content=content)
    second = replace(_record("mnemo-same-2"), content=content)
    try:
        first_result = backend.retain_block(first)
        second_result = backend.retain_block(second)
        assert first_result["memory_id"] != second_result["memory_id"]
        assert len(native.remembered) == 2
        assert native.remembered[0].startswith(content + "\n[UMM:")
        assert "[DATES:" not in native.remembered[0]
        assert "[DURATIONS:" not in native.remembered[0]
        stored_first = native.rows[first_result["memory_id"]]["content"]
        assert stored_first.count("[DATES:") == 1
        assert stored_first.count("[DURATIONS:") == 1

        recalled = backend.recall(
            {
                "query": "same content",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "source_horizon": "durable",
                "logical_block_ids": ["workspace_episodes"],
                "limit": 5,
            }
        )
        assert {row.block_id for row in recalled} == {first.block_id, second.block_id}
        assert all(row.content == content for row in recalled)

        entry = backend._index.write_outbox_entry(
            block_id=first.block_id,
            commit_revision=int(first_result["commit_revision"]),
        )
        assert entry is not None and entry.status == "committed"
        remembered = len(native.remembered)
        assert backend._project_write_outbox(entry, first) == first_result["memory_id"]
        assert len(native.remembered) == remembered

        third = replace(_record("mnemo-crash"), content=content)
        third_entry = backend._index.open_write_outbox(record=third, store="mnemosyne")
        original_set_ref = backend._index.set_write_outbox_native_ref
        failed = True

        def fail_set_ref(commit_revision: int, native_id: str) -> None:
            nonlocal failed
            if failed and commit_revision == third_entry.commit_revision:
                failed = False
                raise OSError("crash after Mnemosyne remember")
            original_set_ref(commit_revision, native_id)

        monkeypatch.setattr(backend._index, "set_write_outbox_native_ref", fail_set_ref)
        remembered = len(native.remembered)
        with pytest.raises(OSError, match="crash after Mnemosyne remember"):
            backend.retain_block(third)
        assert len(native.remembered) == remembered + 1
        third_native_id = next(
            native_id
            for native_id, row in native.rows.items()
            if row["metadata"]["source_identity"] == backend._native_source_identity(third)
        )
        monkeypatch.setattr(backend._index, "set_write_outbox_native_ref", original_set_ref)
        recovered = backend.retain_block(third)
        assert recovered["memory_id"] == third_native_id
        assert len(native.remembered) == remembered + 1
        assert backend._index.write_outbox_status(third.block_id) == "committed"

        native.rows[first_result["memory_id"]]["content"] = "tampered"
        with pytest.raises(RuntimeError, match="validation failed"):
            backend._project_write_outbox(entry, first)
    finally:
        backend.close()


def test_mnemosyne_readiness_requires_local_native_api_shape(tmp_path) -> None:
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    backend._native_memory = object()
    try:
        health = backend.health()
        assert health["native_ok"] is False
        assert health["effective_mode"] == "unavailable"
        assert backend.capabilities().semantic_recall is False
    finally:
        backend.close()


def test_hindsight_capabilities_require_native_retain_and_recall(tmp_path) -> None:
    incomplete = _HindsightCapabilityClient()
    incomplete_backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="incomplete",
        config=HindsightConfig(enabled=True),
        client=incomplete,
    )
    complete = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="on",
        config=HindsightConfig(enabled=True),
        client=_HindsightCompleteCapabilityClient(),
    )
    try:
        incomplete_caps = incomplete_backend.capabilities()
        assert incomplete_caps.persistent is False
        assert incomplete_caps.semantic_recall is False
        assert incomplete_caps.workspace_isolation is False
        complete_caps = complete.capabilities()
        assert complete_caps.persistent is True
        assert complete_caps.semantic_recall is True
        assert complete_caps.workspace_isolation is True
        assert complete_caps.run_subtask_isolation is True
        assert complete_caps.value_tracking is True
        assert complete_caps.export_import is True
    finally:
        incomplete_backend.close()
        complete.close()
