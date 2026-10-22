import json
import sqlite3
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from um_mem.api import MemoryClient
from um_mem.contracts import (
    EvidenceRef,
    MemoryBlockRecord,
    OriginRef,
    OriginSegment,
    RecordBlockMembership,
    TurnChildClaim,
)
from um_mem.memory.backends.factory import selected_memory_backend_mode
from um_mem.memory.backends.hindsight import HindsightBackend
from um_mem.memory.backends.mempalace import (
    MemPalaceBackend,
    _native_gate_for,
    _record_to_native_payload,
)
from um_mem.memory.backends.mnemosyne import MnemosyneBackend
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex, WriteOutboxEntry
from um_mem.memory.blocks.catalog import compile_semantic_catalog
from um_mem.memory.blocks.registry import load_memory_blocks
from um_mem.memory.catalog_store import BlockEvolutionTarget
from um_mem.memory.hindsight.config import HindsightConfig
from um_mem.memory.hindsight.errors import HindsightUnavailableError
from um_mem.memory.kernel.models import block_record_to_event
from um_mem.memory.kernel.writer import _write_memory_event
from um_mem.memory.mutation import plan_memory_mutation


def _wait_for_projection(backend: Any, record_id: str, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while backend._index.write_outbox_status(record_id) != "committed":
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


_CATALOG_REVISION = "test-catalog-revision"


def _block(block_id: str = "b1") -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        revision_id=f"revision:{block_id}",
        content="Native backend must store this record.",
        title="native-required",
        kind="observation",
        status="accepted",
        retention="durable",
        delivery_policy="on_demand",
        temperature="warm",
        authority="advisory",
        trust_level="workspace_verified",
        scope="workspace",
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
        workspace_id="ws",
        run_id="run",
        producer="test",
        tags=("native",),
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id=block_id,
                produced_by="verifier",
                producer_id=f"verifier:{block_id}",
            ),
        ),
        block_membership=RecordBlockMembership(
            primary_block_id="project_ideas",
            catalog_revision_id=_CATALOG_REVISION,
        ),
        metadata={"evidence_id": block_id},
    )


class _RecordingIndex:
    def projection_gate(self, record):
        return nullcontext()

    def __init__(self) -> None:
        self.retained: list[MemoryBlockRecord] = []
        self.outbox: dict[int, WriteOutboxEntry] = {}
        self.records: dict[str, MemoryBlockRecord] = {}
        self.native_refs: dict[int, str] = {}
        self._next_revision = 0

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        self.retained.append(record)
        self.records[record.block_id] = record
        return {"ok": True, "block_id": record.block_id, "backend": "index"}

    def open_write_outbox(
        self,
        *,
        record: MemoryBlockRecord,
        store: str,
    ) -> WriteOutboxEntry:
        self._next_revision += 1
        entry = WriteOutboxEntry(
            commit_revision=self._next_revision,
            block_id=record.block_id,
            workspace_id=record.workspace_id,
            store=store,
            payload_id="record",
            payload_sha256="record",
            status="pending",
            error="",
            created_at="test",
            updated_at="test",
        )
        self.outbox[entry.commit_revision] = entry
        self.records[record.block_id] = record
        return entry

    def pending_write_outbox(self) -> list[WriteOutboxEntry]:
        return [entry for entry in self.outbox.values() if entry.status == "pending"]

    def write_outbox_entry(self, *, block_id: str, commit_revision: int) -> WriteOutboxEntry | None:
        entry = self.outbox.get(commit_revision)
        return entry if entry is not None and entry.block_id == block_id else None

    def write_outbox_record(self, entry: WriteOutboxEntry) -> MemoryBlockRecord:
        return self.records[entry.block_id]

    def set_write_outbox_native_ref(self, commit_revision: int, native_id: str) -> None:
        self.native_refs[commit_revision] = native_id

    def complete_write_outbox(self, commit_revision: int) -> None:
        entry = self.outbox[commit_revision]
        self.outbox[commit_revision] = replace(entry, status="committed", error="")

    def fail_write_outbox(self, commit_revision: int, *, error: str) -> None:
        entry = self.outbox[commit_revision]
        self.outbox[commit_revision] = replace(entry, status="failed", error=error)


class _FakeNativeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.ids = [str(row["id"]) for row in rows]
        self.documents = [str(row["content"]) for row in rows]
        self.metadatas = [dict(row["metadata"]) for row in rows]


class _FakeNativeCollection:
    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self.rows = rows

    def get(self, *, ids: list[str]) -> _FakeNativeResult:
        values = [self.rows[block_id] for block_id in ids if block_id in self.rows]
        return _FakeNativeResult(values)


class _WorkingNative:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, dict[str, Any]]] = {}
        self.calls: list[MemoryBlockRecord] = []

    def collection(self, store: str, *, partition: object) -> _FakeNativeCollection:
        del partition
        return _FakeNativeCollection(self.rows.setdefault(store, {}))

    def add_record(self, record: MemoryBlockRecord, *, commit_revision: int) -> None:
        self.calls.append(record)
        store, metadata = _record_to_native_payload(
            record,
            commit_revision=commit_revision,
        )
        self.rows.setdefault(store, {})[record.block_id] = {
            "id": record.block_id,
            "content": record.content,
            "metadata": metadata,
        }

    def release(self) -> None:
        return None


class _RaisingNative(_WorkingNative):
    def add_record(self, record: MemoryBlockRecord, *, commit_revision: int) -> None:
        del record, commit_revision
        raise RuntimeError("native palace down")


class _FakeHindsightClient:
    def __init__(self) -> None:
        self.retained: list[dict[str, Any]] = []
        self.batches: list[dict[str, Any]] = []

    def retain(self, **kwargs: Any) -> dict[str, Any]:
        self.retained.append(kwargs)
        return {"ok": True, "document_id": kwargs.get("document_id", "native-id")}

    def retain_batch(self, **kwargs: Any) -> dict[str, Any]:
        self.batches.append(kwargs)
        return {"ok": True}

    def recall(self, **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return []


class _RecoveryHindsightDocuments:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    def get_document(self, bank_id: str, document_id: str) -> dict[str, Any] | None:
        row = self.rows.get((bank_id, document_id))
        return dict(row) if row is not None else None


class _RecoveryHindsightClient:
    def __init__(self, *, response: dict[str, Any] | None = None) -> None:
        self.response = response or {"ok": True}
        self.retained: list[dict[str, Any]] = []
        self.batches: list[dict[str, Any]] = []
        self.documents = _RecoveryHindsightDocuments()

    def _store_document(self, *, bank_id: str, item: dict[str, Any]) -> None:
        document_id = str(item["document_id"])
        self.documents.rows[(bank_id, document_id)] = {
            "id": document_id,
            "bank_id": bank_id,
            "original_text": str(item["content"]),
            "document_metadata": dict(item["metadata"]),
            "tags": list(item["tags"]),
        }

    def retain(self, **kwargs: Any) -> dict[str, Any]:
        self.retained.append(dict(kwargs))
        self._store_document(bank_id=str(kwargs["bank_id"]), item=kwargs)
        return dict(self.response)

    def retain_batch(self, **kwargs: Any) -> dict[str, Any]:
        self.batches.append(dict(kwargs))
        bank_id = str(kwargs["bank_id"])
        for item in list(kwargs["items"]):
            self._store_document(bank_id=bank_id, item=dict(item))
        return dict(self.response)

    def recall(self, **kwargs: Any) -> list[dict[str, Any]]:
        bank_id = str(kwargs["bank_id"])
        return [
            {
                "id": row["id"],
                "text": row["original_text"],
                "metadata": dict(row["document_metadata"]),
                "tags": list(row["tags"]),
            }
            for (row_bank_id, _), row in self.documents.rows.items()
            if row_bank_id == bank_id
        ]


def test_hindsight_split_projects_into_new_bank_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _RecoveryHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )
    monkeypatch.setattr(backend, "health", lambda: {"native_ok": True, "effective_mode": "native"})
    try:
        client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)
        root_id = client.block_map()[0]["block_id"]
        source = client.create_leaf_block(
            name="Source",
            synopsis="Agent-owned source facts.",
            parent_id=root_id,
            reason="test",
        )
        written = client.remember("Exact Hindsight fact.", requested_block_id=source["block_id"])
        assert client.projection_barrier(written.memory_id)["ok"]
        split = client.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=(
                BlockEvolutionTarget(
                    name="Target",
                    synopsis="Agent-selected target facts.",
                    record_ids=(written.memory_id,),
                ),
            ),
            reason="agent chose a more precise leaf",
        )
        target_id = split["targets"][0]["block_id"]
        hits = client.recall(
            "Exact Hindsight fact", logical_block_ids=(target_id,), include_candidates=True
        )
        assert any("Exact Hindsight fact." in hit.text for hit in hits)
        assert len(native.documents.rows) == 2
    finally:
        backend.close()


def _isolated_mempalace(native: _RaisingNative | _WorkingNative, root: Path) -> MemPalaceBackend:
    backend = MemPalaceBackend.__new__(MemPalaceBackend)
    backend._workspace_id = "ws"
    backend._index = _RecordingIndex()
    backend._native = native
    backend._native_operation_gate = _native_gate_for(root / "index.sqlite")
    backend._operation_lock_timeout_seconds = 30.0
    backend._operation_state = threading.local()
    return backend


def test_mempalace_retain_block_fails_closed_when_native_write_fails(tmp_path: Path) -> None:
    backend = _isolated_mempalace(_RaisingNative(), tmp_path)

    with pytest.raises(RuntimeError, match="native palace down"):
        backend.retain_block(_block())

    assert backend._index.retained == [_block()]
    assert backend._index.outbox[1].status == "pending"


def test_hindsight_config_default_mode_names_hindsight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UM_MEM_BACKEND", raising=False)
    monkeypatch.delenv("UM_MEM_MEMORY_BACKEND", raising=False)

    assert HindsightConfig.from_env().backend_mode == "hindsight"

    monkeypatch.setenv("UM_MEM_MEMORY_BACKEND", "hindsight")
    with pytest.raises(ValueError, match="UM_MEM_MEMORY_BACKEND"):
        HindsightConfig.from_env()
    with pytest.raises(ValueError, match="UM_MEM_MEMORY_BACKEND"):
        selected_memory_backend_mode()


def test_hindsight_retain_defaults_to_sync_for_immediate_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UM_MEM_HINDSIGHT_RETAIN_ASYNC", raising=False)
    assert HindsightConfig.from_env().retain_async is False

    monkeypatch.setenv("UM_MEM_HINDSIGHT_RETAIN_ASYNC", "true")
    assert HindsightConfig.from_env().retain_async is True


def test_hindsight_embedded_uses_its_explicit_configuration_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace

    captured: dict[str, Any] = {}

    class FakeEmbedded:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hindsight",
        SimpleNamespace(HindsightEmbedded=FakeEmbedded),
    )
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "ambient-hindsight-key")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv("LLM_API_KEY", "ambient-shared-key")
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(
            enabled=True,
            embedded=True,
            api_key="dedicated-key",
            llm_base_url="https://memory.example.test/v1",
        ),
        reconcile_pending_writes=False,
    )
    try:
        backend._create_client()
    finally:
        backend.close()

    assert captured["llm_api_key"] == "dedicated-key"
    assert captured["llm_base_url"] == "https://memory.example.test/v1"


def test_hindsight_selected_backend_requires_native_enabled(tmp_path: Path) -> None:
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=False),
    )

    with pytest.raises(HindsightUnavailableError, match="disabled"):
        backend.retain_block(_block())

    health = backend.health()
    assert health["ok"] is False
    assert health["native_ok"] is False
    assert health["effective_mode"] == "unavailable"


def test_hindsight_public_default_remember_indexes_candidate_in_native_recall(
    tmp_path: Path,
) -> None:
    fake = _RecoveryHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=fake,
    )
    client = MemoryClient(repo_root=tmp_path, workspace_id="ws", backend_instance=backend)
    try:
        result = client.remember(
            "Hindsight default observation is supplemental evidence.",
            requested_block_id="workspace_episodes",
        )
        assert client.projection_barrier(result.memory_id, timeout_seconds=5)["ok"]
        hits = client.recall(
            "default observation",
            logical_block_ids=["workspace_episodes"],
            include_candidates=True,
            limit=5,
        )
    finally:
        backend.close()

    assert result.saved is True
    assert result.raw["durable_status"] == "committed"
    assert len(fake.retained) == 1
    assert [hit.text for hit in hits] == ["Hindsight default observation is supplemental evidence."]


def test_hindsight_verified_remember_uses_sync_native_write_by_default(tmp_path: Path) -> None:
    fake = _FakeHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=fake,
    )
    try:
        result = backend.retain_block(_block("verified"))
    finally:
        backend.close()

    assert result["native_written"] is True
    assert fake.retained[0]["retain_async"] is False


def test_hindsight_verified_retain_passes_block_event_time_to_native_client(tmp_path: Path) -> None:
    fake = _FakeHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=fake,
    )
    record = _block("dated")
    record.metadata["ts"] = "2023-05-25T12:00:00Z"
    try:
        backend.retain_block(record)
    finally:
        backend.close()

    retained = fake.retained[0]
    assert retained["timestamp"] == "2023-05-25T12:00:00Z"
    assert "event_time: 2023-05-25T12:00:00Z" in retained["context"]


def test_hindsight_retain_blocks_passes_block_event_time_to_native_batch(tmp_path: Path) -> None:
    fake = _FakeHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=fake,
    )
    record = _block("batch-dated")
    record.metadata["event_time"] = "2023-06-01T09:30:00Z"
    try:
        backend.retain_blocks([record])
    finally:
        backend.close()

    item = fake.retained[0]
    assert item["timestamp"] == "2023-06-01T09:30:00Z"
    assert "event_time: 2023-06-01T09:30:00Z" in item["context"]


def test_hindsight_retain_block_sends_unset_for_non_iso_block_event_time(tmp_path: Path) -> None:
    fake = _FakeHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=fake,
    )
    record = _block("human-dated")
    record.metadata["ts"] = "1:56 pm on 8 May, 2023"
    try:
        backend.retain_block(record)
    finally:
        backend.close()

    retained = fake.retained[0]
    assert retained["timestamp"] == "unset"
    assert "event_time_raw: 1:56 pm on 8 May, 2023" in retained["context"]
    assert "event_time: unset" in retained["context"]


def test_hindsight_retain_blocks_sends_unset_for_non_iso_block_event_time(tmp_path: Path) -> None:
    fake = _FakeHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=fake,
    )
    record = _block("batch-human-dated")
    record.metadata["event_time"] = "March-15-2024"
    try:
        backend.retain_blocks([record])
    finally:
        backend.close()

    item = fake.retained[0]
    assert item["timestamp"] == "unset"
    assert "event_time_raw: March-15-2024" in item["context"]
    assert "event_time: unset" in item["context"]


def test_hindsight_reconciles_pending_native_write_after_sidecar_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native = _RecoveryHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )

    def crash_after_native_projection(_revision: int) -> None:
        raise OSError("crash after native projection")

    monkeypatch.setattr(backend._index, "complete_write_outbox", crash_after_native_projection)
    try:
        with pytest.raises(OSError, match="crash after native projection"):
            backend.retain_block(_block("hindsight-pending"))
        assert backend.get_block("hindsight-pending") is None
        assert (
            backend.recall(
                {
                    "query": "native backend",
                    "workspace_id": "ws",
                    "run_id": "run",
                    "scope_chain": ["workspace:ws"],
                    "search_all": True,
                    "limit": 5,
                }
            )
            == []
        )
    finally:
        backend.close()

    reopened = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )
    try:
        assert _wait_for_projection(reopened, "hindsight-pending")
        recovered = reopened.get_block("hindsight-pending")
        recalled = reopened.recall(
            {
                "query": "native backend",
                "workspace_id": "ws",
                "run_id": "run",
                "scope_chain": ["workspace:ws"],
                "search_all": True,
                "limit": 5,
            }
        )
        outbox_status = reopened._index.write_outbox_status("hindsight-pending")
    finally:
        reopened.close()

    assert recovered is not None
    assert [row.block_id for row in recalled] == ["hindsight-pending"]
    assert outbox_status == "committed"


def test_hindsight_pending_recovery_fails_closed_on_native_document_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native = _RecoveryHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )

    def crash_after_native_projection(_revision: int) -> None:
        raise OSError("crash after native projection")

    monkeypatch.setattr(backend._index, "complete_write_outbox", crash_after_native_projection)
    try:
        with pytest.raises(OSError, match="crash after native projection"):
            backend.retain_block(_block("hindsight-mismatch"))
    finally:
        backend.close()

    document = next(iter(native.documents.rows.values()))
    document["document_metadata"]["workspace_id"] = "other"
    reopened = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )
    try:
        assert reopened.reconcile(dry_run=False).failures > 0
        assert reopened._index.write_outbox_status("hindsight-mismatch") != "committed"
        assert reopened.get_block("hindsight-mismatch") is None
    finally:
        reopened.close()

    index = BackendSQLiteIndex(
        tmp_path / "workspaces" / "ws" / ".memory" / "hindsight" / "UM_MEM_index.sqlite",
        backend_id="hindsight",
        workspace_id="ws",
    )
    try:
        outbox_status = index.write_outbox_status("hindsight-mismatch")
    finally:
        index.close()

    assert native.documents.rows
    assert outbox_status == "pending"


@pytest.mark.parametrize("backend_id", ["hindsight", "mnemosyne"])
def test_identity_mismatch_fails_pending_canonical_head_across_restarts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_id: str,
) -> None:
    native = _RecoveryHindsightClient() if backend_id == "hindsight" else None
    if native is not None:
        backend: HindsightBackend | MnemosyneBackend = HindsightBackend(
            repo_root=tmp_path,
            workspace_id="ws",
            config=HindsightConfig(enabled=True),
            client=native,
        )
    else:
        backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)

    def fail_complete_write_outbox(_revision: int) -> None:
        raise OSError("test completion failure")

    monkeypatch.setattr(
        backend._index,
        "complete_write_outbox",
        fail_complete_write_outbox,
    )
    source = _block(f"{backend_id}-canonical-pending")
    catalog = compile_semantic_catalog(load_memory_blocks(tmp_path, workspace_id="ws", strict=True))
    source = replace(
        source,
        revision_id=f"revision:{source.block_id}",
        block_membership=RecordBlockMembership(
            primary_block_id="stable_project_facts",
            catalog_revision_id=catalog.revision_id,
        ),
    )
    mutation = plan_memory_mutation(source, existing=None)
    try:
        write = _write_memory_event(
            tmp_path,
            block_record_to_event(mutation.record),
            workspace_id="ws",
            backend_instance=backend,
            mutation=mutation,
        )
        assert write.saved is True
        assert write.raw["projection_pending"] is True
        assert backend._index.canonical_head_status(mutation.record.block_id) == "committed"
        backend._index.stop_native_projection()
        # Materialize the native row deterministically, then simulate a crash
        # before its completion receipt and corrupt that exact row below.
        with pytest.raises(OSError, match="test completion failure"):
            backend.retain_block(mutation.record)
        [entry] = backend._index.pending_write_outbox()
        if native is not None:
            document = next(iter(native.documents.rows.values()))
            document["document_metadata"]["workspace_id"] = "other"
        else:
            wrong_native_id = backend._native(
                backend._record_partition_identity(mutation.record)
            ).remember(
                "wrong native identity",
                source="wrong-native-identity",
                importance=0.1,
                metadata={"block_id": "wrong", "workspace_id": "ws"},
                scope="workspace",
                valid_until=None,
                trust_tier=None,
            )
            with backend._index._conn:
                backend._index._conn.execute(
                    "UPDATE write_outbox_native_refs SET native_id=? WHERE commit_revision=?",
                    (wrong_native_id, entry.commit_revision),
                )
    finally:
        backend.close()

    if native is not None:
        reopened = HindsightBackend(
            repo_root=tmp_path,
            workspace_id="ws",
            config=HindsightConfig(enabled=True),
            client=native,
        )
    else:
        reopened = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    try:
        assert reopened.reconcile(dry_run=False).failures > 0
        assert reopened.get_block(mutation.record.block_id) is not None
        assert not MemoryClient(
            tmp_path, workspace_id="ws", backend_instance=reopened
        ).projection_barrier(mutation.record.block_id, timeout_seconds=0)["ok"]
    finally:
        reopened.close()
    backend_name = "hindsight" if native is not None else "mnemosyne"
    index = BackendSQLiteIndex(
        tmp_path / "workspaces" / "ws" / ".memory" / backend_name / "UM_MEM_index.sqlite",
        backend_id=backend_name,
        workspace_id="ws",
    )
    try:
        assert index.canonical_head_status(mutation.record.block_id) == "committed"
        assert index.write_outbox_status(mutation.record.block_id) == "pending"
        assert (
            index._conn.execute(
                "SELECT COUNT(*) FROM canonical_mutations WHERE successor_id=?",
                (mutation.record.block_id,),
            ).fetchone()[0]
            == 1
        )
    finally:
        index.close()


def test_hindsight_negative_native_ack_does_not_commit_sidecar(tmp_path: Path) -> None:
    native = _RecoveryHindsightClient(response={"success": False})
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )
    retry: dict[str, Any]
    retry_status: str | None
    try:
        with pytest.raises(HindsightUnavailableError, match="negative acknowledgement"):
            backend.retain_block(_block("hindsight-negative-ack"))
        stored = backend.get_block("hindsight-negative-ack")
        recalled = backend.recall(
            {
                "query": "native backend",
                "workspace_id": "ws",
                "run_id": "run",
                "scope_chain": ["workspace:ws"],
                "search_all": True,
                "limit": 5,
            }
        )
        outbox_status = backend._index.write_outbox_status("hindsight-negative-ack")
        native.response = {"success": True}
        retry = backend.retain_block(_block("hindsight-negative-ack"))
        retry_status = backend._index.write_outbox_status("hindsight-negative-ack")
    finally:
        backend.close()

    assert stored is None
    assert recalled == []
    assert outbox_status == "pending"
    assert retry["native_written"] is True
    assert retry_status == "committed"


def test_hindsight_rejects_async_normal_write_without_durable_receipt(tmp_path: Path) -> None:
    native = _RecoveryHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True, retain_async=True),
        client=native,
    )
    try:
        with pytest.raises(HindsightUnavailableError, match="synchronous native acknowledgement"):
            backend.retain_block(_block("hindsight-async"))
    finally:
        backend.close()

    assert native.retained == []


def test_hindsight_batch_reconciles_pending_native_rows_after_sidecar_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native = _RecoveryHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )
    complete_write_outbox = backend._index.complete_write_outbox
    calls = 0

    def reject_second_completion(revision: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("test second completion failure")
        complete_write_outbox(revision)

    records = [
        _block("hindsight-batch-one"),
        _block("hindsight-batch-two"),
        _block("hindsight-batch-three"),
    ]
    monkeypatch.setattr(backend._index, "complete_write_outbox", reject_second_completion)
    try:
        with pytest.raises(OSError, match="test second completion failure"):
            backend.retain_blocks(records)
        assert backend.get_block("hindsight-batch-two") is None
        hidden = backend.recall(
            {
                "query": "native backend",
                "workspace_id": "ws",
                "run_id": "run",
                "scope_chain": ["workspace:ws"],
                "search_all": True,
                "limit": 5,
            }
        )
    finally:
        backend.close()

    reopened = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )
    try:
        assert _wait_for_projection(reopened, "hindsight-batch-two")
        recovered = reopened.recall(
            {
                "query": "native backend",
                "workspace_id": "ws",
                "run_id": "run",
                "scope_chain": ["workspace:ws"],
                "search_all": True,
                "limit": 5,
            }
        )
        outbox_statuses = [
            reopened._index.write_outbox_status(record.block_id) for record in records
        ]
    finally:
        reopened.close()

    assert [row.block_id for row in hidden] == ["hindsight-batch-one"]
    assert {row.block_id for row in recovered} == {
        "hindsight-batch-one",
        "hindsight-batch-two",
    }
    assert outbox_statuses == ["committed", "committed", None]
    assert len(native.retained) == 2
    assert native.batches == []


def test_hindsight_batch_negative_native_ack_does_not_commit_sidecar(tmp_path: Path) -> None:
    native = _RecoveryHindsightClient(response={"success": False})
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=native,
    )
    retry: dict[str, Any]
    retry_status: str | None
    outbox_status: str | None
    try:
        with pytest.raises(HindsightUnavailableError, match="negative acknowledgement"):
            backend.retain_blocks([_block("hindsight-batch-negative")])
        stored = backend.get_block("hindsight-batch-negative")
        recalled = backend.recall(
            {
                "query": "native backend",
                "workspace_id": "ws",
                "run_id": "run",
                "scope_chain": ["workspace:ws"],
                "search_all": True,
                "limit": 5,
            }
        )
        outbox_status = backend._index.write_outbox_status("hindsight-batch-negative")
        native.response = {"success": True}
        retry = backend.retain_blocks([_block("hindsight-batch-negative")])
        retry_status = backend._index.write_outbox_status("hindsight-batch-negative")
    finally:
        backend.close()

    assert stored is None
    assert recalled == []
    assert outbox_status == "pending"
    assert retry["n_written"] == 1
    assert retry_status == "committed"


def test_write_outbox_retry_reuses_first_payload_when_only_auto_timestamps_change(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "retry.sqlite",
        backend_id="hindsight",
        workspace_id="ws",
    )
    first = replace(
        _block("retry-auto-timestamps"),
        created_at="2026-07-15T02:00:00Z",
        updated_at="2026-07-15T02:00:00Z",
    )
    retry = replace(
        first,
        created_at="2026-07-15T02:00:01Z",
        updated_at="2026-07-15T02:00:01Z",
    )
    try:
        original_entry = index.open_write_outbox(record=first, store="hindsight")
        replayed_entry = index.open_write_outbox(record=retry, store="hindsight")
        stored = index.write_outbox_record(replayed_entry)
        with pytest.raises(ValueError, match="identity conflicts"):
            index.open_write_outbox(
                record=replace(retry, content="different logical payload"),
                store="hindsight",
            )
    finally:
        index.close()

    assert replayed_entry == original_entry
    assert stored == first


def test_hindsight_batch_rejects_async_without_durable_receipt(tmp_path: Path) -> None:
    native = _RecoveryHindsightClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True, retain_async=True),
        client=native,
    )
    try:
        with pytest.raises(HindsightUnavailableError, match="synchronous native acknowledgement"):
            backend.retain_blocks([_block("hindsight-batch-async")])
    finally:
        backend.close()

    assert native.batches == []


def test_mnemosyne_selected_backend_requires_native_enabled(tmp_path: Path) -> None:
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=False)

    with pytest.raises(RuntimeError, match="native Mnemosyne"):
        backend.retain_block(_block())

    health = backend.health()
    assert health["ok"] is False
    assert health["native_ok"] is False
    assert health["effective_mode"] == "unavailable"


def test_mnemosyne_empty_batch_does_not_start_workers_or_native(tmp_path: Path) -> None:
    backend = MnemosyneBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        enabled=True,
        reconcile_pending_writes=False,
    )
    try:
        assert backend.retain_blocks([]) == {
            "ok": True,
            "n_written": 0,
            "native_written": 0,
        }
        assert backend._native_partitions == {}
    finally:
        backend.close()


@pytest.mark.parametrize("backend_id", ["hindsight", "mnemosyne"])
def test_temporary_batch_workers_release_sqlite_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_id: str
) -> None:
    if backend_id == "hindsight":
        backend: HindsightBackend | MnemosyneBackend = HindsightBackend(
            repo_root=tmp_path,
            workspace_id="ws",
            config=HindsightConfig(enabled=False),
            reconcile_pending_writes=False,
        )
    else:
        backend = MnemosyneBackend(
            repo_root=tmp_path,
            workspace_id="ws",
            enabled=False,
            reconcile_pending_writes=False,
        )

    worker_connections: list[sqlite3.Connection] = []

    def retain_in_worker(_record: MemoryBlockRecord) -> dict[str, bool]:
        connection = backend._index._conn
        worker_connections.append(connection)
        connection.execute("SELECT 1")
        return {"ok": True}

    monkeypatch.setattr(backend, "retain_block", retain_in_worker)
    try:
        for iteration in range(12):
            records = [
                replace(
                    _block(f"batch-{iteration}-{number}"),
                    block_membership=RecordBlockMembership(
                        primary_block_id=f"leaf-{number}",
                        catalog_revision_id=_CATALOG_REVISION,
                    ),
                )
                for number in range(2)
            ]
            assert backend.retain_blocks(records)["n_written"] == 2
        assert len(backend._index._connections) <= 3
        for connection in worker_connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")

        def reject_in_worker(_record: MemoryBlockRecord) -> dict[str, bool]:
            backend._index._conn.execute("SELECT 1")
            raise RuntimeError("batch retain failed")

        monkeypatch.setattr(backend, "retain_block", reject_in_worker)
        with pytest.raises(RuntimeError, match="batch retain failed"):
            backend.retain_blocks(records)
        assert len(backend._index._connections) == 1
    finally:
        backend.close()


def test_mnemosyne_close_allows_same_process_native_reopen(tmp_path: Path) -> None:
    first = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    try:
        first.retain_block(_block("mnemosyne-reopen"))
    finally:
        first.close()

    second = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    try:
        rows = second.recall(
            {
                "query": "native backend",
                "workspace_id": "ws",
                "run_id": "run",
                "scope_chain": ["workspace:ws"],
                "search_all": True,
                "limit": 5,
            }
        )
    finally:
        second.close()

    assert [row.block_id for row in rows] == ["mnemosyne-reopen"]


def test_mnemosyne_real_native_child_claim_transport_survives_reopen(tmp_path: Path) -> None:
    claim = TurnChildClaim(
        claim_type="user_intent",
        source_role="user",
        content="Keep the native transport first class.",
        parent_session_id="session-alpha",
        origin_segments=(
            OriginSegment(
                claim_id="user-intent:session-alpha",
                origin=OriginRef(
                    channel="user_message",
                    principal_id="session-alpha",
                    integrity_class="reported",
                ),
                integrity_ceiling="reported",
                independence_domain="hermes-session:session-alpha:user",
                evidence_refs=(
                    EvidenceRef(
                        ref_type="artifact",
                        ref_id="hermes-session:session-alpha",
                        produced_by="agent",
                        producer_id="agent:session-alpha",
                    ),
                ),
            ),
        ),
        temperature="hot",
        tags=("intent",),
    )
    record = replace(_block("mnemosyne-child-claim"), child_claims=(claim,))
    first = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    try:
        result = first.retain_block(record)
        native_row = first._native(first._record_partition_identity(record)).get(
            result["memory_id"]
        )

        assert native_row is not None
        native_metadata = native_row.get("metadata_json") or native_row.get("metadata")
        if isinstance(native_metadata, str):
            native_metadata = json.loads(native_metadata)
        assert isinstance(native_metadata, dict)
        serialized = json.loads(native_metadata["canonical_record_json"])
        assert serialized["child_claims"][0]["claim_type"] == "user_intent"
    finally:
        first.close()

    second = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    try:
        [restored] = second.recall(
            {
                "query": "native backend",
                "workspace_id": "ws",
                "run_id": "run",
                "scope_chain": ["workspace:ws"],
                "search_all": True,
                "limit": 5,
            }
        )
    finally:
        second.close()

    assert restored.child_claims == (claim,)


def test_mnemosyne_first_write_does_not_scan_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    record = _block("mnemosyne-no-initial-scan")
    native = backend._native(backend._record_partition_identity(record))
    monkeypatch.setattr(
        native,
        "get_all_memories",
        lambda: pytest.fail("ordinary native write must not scan the partition"),
    )
    try:
        assert backend.retain_block(record)["saved"] is True
        assert backend.get_block(record.block_id) == record
    finally:
        backend.close()


def test_mnemosyne_reconciles_pending_native_write_after_sidecar_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    record = _block("mnemosyne-pending")
    revision = 0

    def fail_complete_write_outbox(_revision: int) -> None:
        raise OSError("test final completion failure")

    monkeypatch.setattr(
        backend._index,
        "complete_write_outbox",
        fail_complete_write_outbox,
    )
    try:
        with pytest.raises(OSError, match="test final completion failure"):
            backend.retain_block(record)
        [entry] = backend._index.pending_write_outbox()
        revision = entry.commit_revision
        assert backend.get_block("mnemosyne-pending") is None
        assert (
            backend.recall(
                {
                    "query": "native backend",
                    "workspace_id": "ws",
                    "run_id": "run",
                    "scope_chain": ["workspace:ws"],
                    "search_all": True,
                    "limit": 5,
                }
            )
            == []
        )
    finally:
        backend.close()

    reopened = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    try:
        assert _wait_for_projection(reopened, "mnemosyne-pending")
        recovered = reopened.get_block("mnemosyne-pending")
        recalled = reopened.recall(
            {
                "query": "native backend",
                "workspace_id": "ws",
                "run_id": "run",
                "scope_chain": ["workspace:ws"],
                "search_all": True,
                "limit": 5,
            }
        )
        outbox_status = reopened._index.write_outbox_status("mnemosyne-pending")
        native_id = str(reopened._index.write_outbox_native_id(revision) or "")
        native_row = reopened._native(reopened._record_partition_identity(record)).get(native_id)
    finally:
        reopened.close()

    assert recovered is not None
    assert recovered == record
    assert [row.block_id for row in recalled] == ["mnemosyne-pending"]
    assert outbox_status == "committed"
    assert native_id
    assert isinstance(native_row, dict)


def test_mnemosyne_missing_native_outbox_reference_fails_closed_after_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    original_set_native_ref = backend._index.set_write_outbox_native_ref
    failed = True

    def fail_native_reference_once(commit_revision: int, native_id: str) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise OSError("test native reference failure")
        original_set_native_ref(commit_revision, native_id)

    monkeypatch.setattr(
        backend._index,
        "set_write_outbox_native_ref",
        fail_native_reference_once,
    )
    record = _block("mnemosyne-missing-ref")
    revision = 0
    try:
        with pytest.raises(OSError, match="test native reference failure"):
            backend.retain_block(record)
        [entry] = backend._index.pending_write_outbox()
        revision = entry.commit_revision
        assert backend.get_block(record.block_id) is None
        assert (
            len(backend._native(backend._record_partition_identity(record)).get_all_memories()) == 1
        )
    finally:
        backend.close()

    reopened = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    try:
        assert _wait_for_projection(reopened, record.block_id)
        partition = reopened._native(reopened._record_partition_identity(record))
        [summary] = partition.get_all_memories()
        native_id = str(summary["id"])
        detail = partition.get(native_id)
        assert isinstance(detail, dict)
        assert detail["source"] == "um_mem:block:project_ideas"
        metadata = detail.get("metadata") or detail.get("metadata_json")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert isinstance(metadata, dict)
        assert metadata["block_id"] == record.block_id
        assert metadata["source_identity"] == "um-mem:ws:mnemosyne-missing-ref"
        assert reopened._index.write_outbox_native_id(revision) == native_id
        assert reopened._index.write_outbox_status(record.block_id) == "committed"
        assert reopened.get_block(record.block_id) == record
    finally:
        reopened.close()

    index = BackendSQLiteIndex(
        tmp_path / "workspaces" / "ws" / ".memory" / "mnemosyne" / "UM_MEM_index.sqlite",
        backend_id="mnemosyne",
        workspace_id="ws",
    )
    try:
        outbox_status = index.write_outbox_status("mnemosyne-missing-ref")
    finally:
        index.close()

    assert outbox_status == "committed"


def test_mnemosyne_pending_recovery_fails_closed_on_native_content_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    record = _block("mnemosyne-content-mismatch")

    def fail_complete_write_outbox(_revision: int) -> None:
        raise OSError("test final completion failure")

    monkeypatch.setattr(
        backend._index,
        "complete_write_outbox",
        fail_complete_write_outbox,
    )
    try:
        with pytest.raises(OSError, match="test final completion failure"):
            backend.retain_block(record)
        entry = backend._index.pending_write_outbox()[0]
        native_id = backend._index.write_outbox_native_id(entry.commit_revision)
        assert native_id is not None
        assert (
            backend._native(backend._record_partition_identity(record)).update(
                native_id,
                content="tampered native content",
            )
            is True
        )
    finally:
        backend.close()

    reopened = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    try:
        assert reopened.reconcile(dry_run=False).failures > 0
        assert reopened.get_block(record.block_id) is None
        assert reopened._index.write_outbox_status(record.block_id) != "committed"
    finally:
        reopened.close()

    index = BackendSQLiteIndex(
        tmp_path / "workspaces" / "ws" / ".memory" / "mnemosyne" / "UM_MEM_index.sqlite",
        backend_id="mnemosyne",
        workspace_id="ws",
    )
    try:
        outbox_status = index.write_outbox_status("mnemosyne-content-mismatch")
    finally:
        index.close()

    assert outbox_status == "pending"


def test_write_outbox_native_reference_is_immutable(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    try:
        record = _block("native-reference")
        entry = index.open_write_outbox(
            record=record,
            store="mnemosyne",
        )
        index.set_write_outbox_native_ref(entry.commit_revision, "native-one")
        index.set_write_outbox_native_ref(entry.commit_revision, "native-one")
        with pytest.raises(RuntimeError, match="native reference is immutable"):
            index.set_write_outbox_native_ref(entry.commit_revision, "native-two")
        native_id = index.write_outbox_native_id(entry.commit_revision)
    finally:
        index.close()

    assert native_id == "native-one"
