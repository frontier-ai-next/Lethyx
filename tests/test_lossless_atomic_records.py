import hashlib
import json
import threading
import time

import pytest

from lethyx import MemoryClient
from lethyx.integrations.hermes_provider import UmMemMemoryProvider
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.backends.hindsight import HindsightBackend
from lethyx.memory.backends.mempalace import MemPalaceBackend
from lethyx.memory.backends.mnemosyne import MnemosyneBackend
from lethyx.memory.service_receipts import claim_remember_receipt, complete_remember_receipt


def test_projection_barrier_ignores_another_pending_record(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    try:
        selected = client.remember(
            "Selected durable record",
            requested_block_id="stable_project_facts",
        )
        other = client.remember(
            "Unrelated durable record",
            requested_block_id="stable_project_facts",
        )
        assert selected.saved and other.saved
        with index._conn:
            index._conn.execute(
                "UPDATE write_outbox SET status='pending' WHERE block_id=?",
                (other.memory_id,),
            )
        assert index.write_outbox_status(other.memory_id) == "pending"
        entry = index.latest_write_outbox_entry(block_id=selected.memory_id, workspace_id="ws")
        assert entry is not None

        barrier = client.projection_barrier(
            selected.memory_id,
            commit_revision=entry.commit_revision,
            timeout_seconds=0,
        )

        assert barrier["ok"] is True
        assert barrier["projection_revision"] == entry.commit_revision
        assert index.write_outbox_status(other.memory_id) == "pending"
    finally:
        index.close()


def test_concurrent_same_operation_id_replays_one_creation(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    client.block_map()
    started = threading.Event()
    release = threading.Event()
    original_retain = index.retain_block
    results = []
    errors = []

    def delayed_retain(record):
        started.set()
        assert release.wait(5)
        return original_retain(record)

    index.retain_block = delayed_retain

    def write() -> None:
        try:
            results.append(
                client.remember(
                    "One durable fact",
                    requested_block_id="stable_project_facts",
                    operation_id="same-create",
                )
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            index.close_thread_connection()

    first = threading.Thread(target=write)
    second = threading.Thread(target=write)
    try:
        first.start()
        assert started.wait(5)
        second.start()
        time.sleep(0.05)
        release.set()
        first.join(5)
        second.join(5)
        assert not errors
        assert len(results) == 2
        assert all(result.saved for result in results)
        assert results[0].memory_id == results[1].memory_id
        assert len(index.export_blocks()) == 1
        assert client.operation_status("same-create")["durable_status"] == "committed"
    finally:
        release.set()
        first.join(5)
        second.join(5)
        index.close()


@pytest.mark.parametrize("backend_type", [MemPalaceBackend, HindsightBackend, MnemosyneBackend])
def test_native_adapter_exposes_revision_scoped_outbox_to_barrier(tmp_path, backend_type) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    adapter = object.__new__(backend_type)
    adapter._index = index
    if backend_type is MnemosyneBackend:
        adapter.workspace_id = "ws"
    else:
        adapter._workspace_id = "ws"
    try:
        canonical = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
        saved = canonical.remember("evidence", requested_block_id="stable_project_facts")
        expected = index.latest_write_outbox_entry(block_id=saved.memory_id, workspace_id="ws")
        assert expected is not None
        assert adapter.latest_write_outbox_entry(block_id=saved.memory_id) == expected
    finally:
        index.close()


def test_provider_exposes_and_forwards_lossless_segments(monkeypatch):
    provider = UmMemMemoryProvider()
    provider._config = {"workspace_id": "ws", "backend": "mempalace"}
    provider._session_id = "session"
    schema = next(s for s in provider.get_tool_schemas() if s["name"] == "lethyx_provider_remember")
    assert "lossless_atomic" in schema["parameters"]["properties"]["write_mode"]["enum"]
    segment_properties = schema["parameters"]["properties"]["source_segments"]["items"][
        "properties"
    ]
    assert {"source_sha256", "byte_start", "byte_end"} <= set(segment_properties)
    captured = []
    monkeypatch.setattr(
        provider, "_capture_provider_tool_witness", lambda *a, **k: ("w", [{"ref_id": "w"}], {})
    )
    monkeypatch.setattr(
        provider, "_remember_and_export", lambda args, **k: captured.append(args) or {"ok": True}
    )
    segments = [
        {"source_id": "turn", "text": "exact text", "speaker": "Ada", "occurred_at": "2026-10-15"}
    ]
    result = json.loads(
        provider.handle_tool_call(
            "lethyx_provider_remember",
            {
                "content": "exact text",
                "operation": "add",
                "requested_block_id": "leaf",
                "write_mode": "lossless_atomic",
                "source_segments": segments,
            },
        )
    )
    assert result["ok"]
    assert captured[0]["source_segments"] == segments
    assert captured[0]["write_mode"] == "lossless_atomic"


def test_provider_preserves_retryable_witness_failure_without_attempting_semantic_write(
    monkeypatch,
):
    provider = UmMemMemoryProvider()
    provider._config = {"workspace_id": "ws", "backend": "mempalace"}
    provider._session_id = "session"
    monkeypatch.setattr(
        provider,
        "_observe_many_and_export",
        lambda *a, **k: {
            "ok": False,
            "results": [
                {
                    "ok": False,
                    "reason_code": "native_backend_unavailable",
                    "retryable": True,
                    "error": "native offline",
                }
            ],
        },
    )
    writes = []
    monkeypatch.setattr(provider, "_remember_and_export", lambda *a, **k: writes.append(a))
    result = json.loads(
        provider.handle_tool_call(
            "lethyx_provider_remember",
            {
                "content": "exact",
                "operation": "add",
                "requested_block_id": "leaf",
            },
        )
    )
    assert result["saved"] is False
    assert result["reason_code"] == "native_backend_unavailable"
    assert result["retryable"] is True
    assert result["durable_status"] == "not_started"
    assert writes == []


@pytest.mark.parametrize("text", ["short summary", "  точная цитата\r\n"])
def test_lossless_rejects_payload_that_does_not_cover_exact_segments(tmp_path, text):
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    try:
        with pytest.raises(ValueError, match="exact concatenation"):
            client.remember(
                text,
                requested_block_id="stable_project_facts",
                write_mode="lossless_atomic",
                metadata={
                    "source_segments": [{"source_id": "turn-1", "text": "точная цитата\r\n"}]
                },
            )
        assert index.export_blocks() == []
    finally:
        index.close()


def test_lossless_preserves_whitespace_unicode_and_segment_provenance(tmp_path):
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    source = "  Ада: 42 🧠\r\n\t"
    try:
        result = client.remember(
            source,
            requested_block_id="stable_project_facts",
            write_mode="lossless_atomic",
            metadata={
                "source_segments": [
                    {
                        "source_id": "turn-1",
                        "text": source,
                        "speaker": "Ада",
                        "occurred_at": "2026-10-15",
                    }
                ]
            },
        )
        assert result.saved
        record = client.read_exact_record(record_id=result.memory_id)
        assert record.content.encode() == source.encode()
        assert record.metadata["source_bytes"] == len(source.encode())
        assert record.metadata["source_segments"][0]["speaker"] == "Ада"
        assert record.metadata["source_segments"][0]["occurred_at"] == "2026-10-15"
    finally:
        index.close()


def test_lossless_atomic_record_survives_reopen_and_reports_coverage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "index.sqlite"
    index = BackendSQLiteIndex(database, backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    source = "At 2026-10-14, Ada said: retain the exact quoted value 42."
    source_id = "conversation:turn-42"
    try:
        result = client.remember(
            source,
            title="Exact source turn",
            requested_block_id="stable_project_facts",
            write_mode="lossless_atomic",
            source_id=source_id,
            operation_id="lossless-turn-42",
        )
        assert result.saved is True
        assert result.raw["write_mode"] == "lossless_atomic"
        assert result.raw["summary_count"] == 0
        assert result.raw["exact_source_coverage"] == 0.0
        assert result.raw["source_coverage_status"] == "unregistered"
        assert result.raw["exact_segment_fidelity"] == 1.0
        assert result.raw["source_segments"][0]["source_id"] == source_id
        assert (
            result.raw["source_segments"][0]["sha256"]
            == hashlib.sha256(source.encode("utf-8")).hexdigest()
        )
        record = client.read_exact_record(record_id=result.memory_id)
        assert record.content == source
        assert record.metadata["memory_mode"] == "lossless_atomic"
        assert record.metadata["summary_count"] == 0
    finally:
        index.close()

    reopened_index = BackendSQLiteIndex(database, backend_id="test", workspace_id="ws")
    reopened = MemoryClient(tmp_path, workspace_id="ws", backend_instance=reopened_index)
    try:
        monkeypatch.setattr(
            reopened_index,
            "export_blocks",
            lambda: pytest.fail("exact and scoped reads must not export the store"),
        )
        exact = reopened.read_exact_record(evidence_ref=source_id)
        assert exact.content == source
        assert exact.metadata["source_segments"][0]["text"] == source
        page = reopened.list_records(scope="workspace", scope_id="ws")
        assert [item["record_id"] for item in page["records"]] == [result.memory_id]
    finally:
        reopened_index.close()


def test_exact_evidence_ref_rejects_ambiguous_records(tmp_path) -> None:
    from lethyx.api import AmbiguousEvidenceRefError

    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    try:
        first = client.remember(
            "first exact turn",
            requested_block_id="stable_project_facts",
            write_mode="lossless_atomic",
            source_id="turn-1",
            operation_id="first-turn",
        )
        second = client.remember(
            "second exact turn",
            requested_block_id="stable_project_facts",
            write_mode="lossless_atomic",
            source_id="turn-1",
            operation_id="second-turn",
        )
        assert first.saved and second.saved
        assert client.read_exact_record(record_id=first.memory_id).content == "first exact turn"
        with pytest.raises(AmbiguousEvidenceRefError, match="ambiguous_evidence_ref"):
            client.read_exact_record(evidence_ref="turn-1")
    finally:
        index.close()


def test_operation_status_reads_original_creation_receipt_after_reopen(tmp_path) -> None:
    database = tmp_path / "index.sqlite"
    index = BackendSQLiteIndex(database, backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    try:
        saved = client.remember(
            "durable fact",
            requested_block_id="stable_project_facts",
            operation_id="stable-create-1",
        )
        assert saved.saved
    finally:
        index.close()

    reopened_index = BackendSQLiteIndex(database, backend_id="test", workspace_id="ws")
    reopened = MemoryClient(tmp_path, workspace_id="ws", backend_instance=reopened_index)
    try:
        status = reopened.operation_status("stable-create-1")
        assert status["ok"] is True
        assert status["operation_id"] == "stable-create-1"
        assert status["record_id"] == saved.memory_id
        assert status["durable_status"] == "committed"
        assert status["original_receipt"]["memory_id"] == saved.memory_id
        assert status["projection_status"] == "committed"
        original_receipt = status["original_receipt"]
        reopened_index.write_outbox_status = lambda _record_id: "pending"  # type: ignore[method-assign]
        pending = reopened.operation_status("stable-create-1")
        assert pending["original_receipt"] == original_receipt
        assert pending["projection_status"] == "pending"
        assert pending["commit_revision"] == status["commit_revision"]
        assert pending["projection_revision"] == 0
        assert reopened.operation_status("unknown")["reason_code"] == "operation_not_found"
        foreign = MemoryClient(
            tmp_path, workspace_id="ws", profile_id="other", backend_instance=reopened_index
        )
        assert foreign.operation_status("stable-create-1")["reason_code"] == "operation_not_found"
    finally:
        reopened_index.close()


def test_operation_status_keeps_access_denial_terminal(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    identity = dict(
        workspace_id="ws",
        profile_id="default",
        backend=client._backend_id(),
        caller_role="memory_client",
        protocol_version=2,
        mutation_correlation_id="denied-1",
        arguments={"tool": "remember"},
    )
    try:
        assert claim_remember_receipt(tmp_path, **identity)[0] == "new"
        complete_remember_receipt(
            tmp_path,
            **identity,
            response={
                "ok": False,
                "error_type": "PermissionError",
                "reason_code": "access_denied",
                "error_sha256": "a" * 64,
            },
            outcome="failed",
        )
        status = client.operation_status("denied-1")
        assert status["ok"] is False
        assert status["reason_code"] == "access_denied"
        assert status["retryable"] is False
    finally:
        index.close()
