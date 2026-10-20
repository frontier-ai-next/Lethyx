import hashlib

import pytest

from lethyx import MemoryClient
from lethyx.contracts import HostObservation
from lethyx.memory.backends.contracts import EdgeType
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory import mcp_facade
from lethyx.memory.local_service import local_service_role_allows


def test_registered_source_preserves_bytes_provenance_and_event_identity(tmp_path) -> None:
    database = tmp_path / "index.sqlite"
    text = "  Ада 🧠\r\n\t42  "
    index = BackendSQLiteIndex(database, backend_id="test", workspace_id="ws")
    client = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-a", backend_instance=index
    )
    try:
        first = client.register_source(
            source_id="turn-1",
            text=text,
            producer_id="trusted-host",
            speaker=" Ада ",
            occurred_at=" 2026-10-15T10:00:00Z ",
        )
        replay = client.register_source(
            source_id="turn-1",
            text=text,
            producer_id="trusted-host",
            speaker=" Ада ",
            occurred_at=" 2026-10-15T10:00:00Z ",
        )
        second = client.register_source(
            source_id="turn-2",
            text=text,
            producer_id="trusted-host",
            speaker=" Ада ",
            occurred_at=" 2026-10-15T10:00:00Z ",
        )
        assert first == replay
        assert first["witness_id"] != second["witness_id"]
        assert first["source_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert first["source_bytes"] == len(text.encode("utf-8"))
        witness = client.read_exact_record(record_id=first["witness_id"])
        assert witness.content.encode("utf-8") == text.encode("utf-8")
        assert witness.metadata["source_speaker"] == " Ада "
        assert witness.metadata["source_occurred_at"] == " 2026-10-15T10:00:00Z "
        assert witness.session_id == "session-a"
        with pytest.raises(ValueError, match="registered_source_conflict"):
            client.register_source(
                source_id="turn-1",
                text="altered",
                producer_id="trusted-host",
                speaker=" Ада ",
                occurred_at=" 2026-10-15T10:00:00Z ",
            )
    finally:
        index.close()

    reopened_index = BackendSQLiteIndex(database, backend_id="test", workspace_id="ws")
    reopened = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-a", backend_instance=reopened_index
    )
    try:
        replay = reopened.register_source(
            source_id="turn-1",
            text=text,
            producer_id="trusted-host",
            speaker=" Ада ",
            occurred_at=" 2026-10-15T10:00:00Z ",
        )
        assert replay == first
    finally:
        reopened_index.close()


def test_registered_source_rejects_foreign_session_identity(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    first = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-a", backend_instance=index
    )
    second = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-b", backend_instance=index
    )
    try:
        first.register_source(source_id="turn-1", text="same", producer_id="trusted-host")
        with pytest.raises(ValueError, match="registered_source_conflict"):
            second.register_source(source_id="turn-1", text="same", producer_id="trusted-host")
    finally:
        index.close()


def test_conversation_witness_identity_rejects_changed_bytes_after_recheck(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-a", backend_instance=index
    )
    base = dict(
        event_id="turn-1",
        event_kind="conversation_turn",
        raw_evidence="exact",
        observed_at="2026-10-15T10:00:00Z",
        producer_id="trusted-host",
    )
    try:
        first = client.observe(HostObservation(**base))
        replay = client.observe(HostObservation(**{**base, "observed_at": "2026-10-15T11:00:00Z"}))
        assert first["witness_id"] == replay["witness_id"]
        with pytest.raises(ValueError, match="registered_source_conflict"):
            client.observe(HostObservation(**{**base, "raw_evidence": "altered"}))
    finally:
        index.close()


def test_host_observe_surface_preserves_registered_source_provenance(tmp_path, monkeypatch) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-a", backend_instance=index
    )
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)
    assert local_service_role_allows("host_hook", "observe")
    assert not local_service_role_allows("parent", "observe")
    try:
        result = mcp_facade.lethyx_observe(
            event_id="turn-1",
            event_kind="conversation_turn",
            raw_evidence="  Ада 🧠  ",
            observed_at="2026-10-15T10:00:00Z",
            producer_id="trusted-host",
            speaker=" Ада ",
            source_occurred_at=" 2026-10-15T09:00:00Z ",
        )
        assert result["ok"] is True
        assert result["source_id"] == "turn-1"
        assert result["source_sha256"] == hashlib.sha256("  Ада 🧠  ".encode()).hexdigest()
        witness = client.read_exact_record(record_id=result["witness_id"])
        assert witness.metadata["source_speaker"] == " Ада "
        assert witness.metadata["source_occurred_at"] == " 2026-10-15T09:00:00Z "
    finally:
        index.close()


def test_bound_segments_use_utf8_offsets_and_union_coverage(tmp_path, monkeypatch) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-a", backend_instance=index
    )
    text = "A🧠B"
    try:
        source = client.register_source(
            source_id="turn-1", text=text, producer_id="trusted-host", speaker="Ada"
        )
        first = client.remember(
            "A🧠",
            requested_block_id="stable_project_facts",
            write_mode="lossless_atomic",
            operation_id="segment-1",
            metadata={
                "source_segments": [
                    {
                        "source_id": "turn-1",
                        "source_sha256": source["source_sha256"],
                        "byte_start": 0,
                        "byte_end": 5,
                        "text": "A🧠",
                    }
                ]
            },
        )
        assert first.saved
        assert first.raw["exact_source_coverage"] == 5 / 6
        assert first.raw["source_segments"][0]["speaker"] == "Ada"
        assert any(
            edge.dst_id == source["witness_id"] and edge.edge_type == EdgeType.DERIVED_FROM
            for edge in index.edges_for(first.memory_id)
        )
        assert client.source_coverage(["turn-1"])["covered_bytes"] == 5
        second = client.remember(
            "🧠B",
            requested_block_id="stable_project_facts",
            write_mode="lossless_atomic",
            operation_id="segment-2",
            metadata={
                "source_segments": [
                    {
                        "source_id": "turn-1",
                        "source_sha256": source["source_sha256"],
                        "byte_start": 1,
                        "byte_end": 6,
                        "text": "🧠B",
                    }
                ]
            },
        )
        assert second.saved
        assert [
            record.content
            for record in client.read_exact_records([second.memory_id, first.memory_id])
        ] == ["🧠B", "A🧠"]

        def no_full_export():
            raise AssertionError("source coverage must use indexed source references")

        monkeypatch.setattr(index, "export_blocks", no_full_export)
        coverage = client.source_coverage(["turn-1"])
        assert coverage["covered_bytes"] == 6
        assert coverage["source_byte_fraction"] == 1.0
        assert coverage["source_id_fraction"] == 1.0
        assert len(coverage["record_ids"]) == 2
        unknown = client.source_coverage(["turn-1", "turn-unprovided"])
        assert unknown["source_byte_fraction"] is None
        assert unknown["unknown_source_ids"] == ["turn-unprovided"]
        assert unknown["source_id_fraction"] == 0.5
    finally:
        index.close()

    reopened_index = BackendSQLiteIndex(
        tmp_path / "index.sqlite", backend_id="test", workspace_id="ws"
    )
    reopened = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-a", backend_instance=reopened_index
    )
    try:
        assert reopened.source_coverage(["turn-1"])["source_byte_fraction"] == 1.0
    finally:
        reopened_index.close()


def test_bound_segment_rejects_foreign_session_and_speaker_spoof(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    owner = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-a", backend_instance=index
    )
    foreign = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-b", backend_instance=index
    )
    try:
        source = owner.register_source(
            source_id="turn-1", text="value", producer_id="trusted-host", speaker="Ada"
        )
        segment = {
            "source_id": "turn-1",
            "source_sha256": source["source_sha256"],
            "byte_start": 0,
            "byte_end": 5,
            "text": "value",
        }
        with pytest.raises(ValueError, match="source segment registration"):
            foreign.remember(
                "value",
                requested_block_id="stable_project_facts",
                write_mode="lossless_atomic",
                metadata={"source_segments": [segment]},
            )
        with pytest.raises(ValueError, match="source segment provenance"):
            owner.remember(
                "value",
                requested_block_id="stable_project_facts",
                write_mode="lossless_atomic",
                metadata={"source_segments": [{**segment, "speaker": "Bob"}]},
            )
    finally:
        index.close()


def test_registered_coverage_rejects_mixed_bound_and_unbound_segments(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-a", backend_instance=index
    )
    try:
        source = client.register_source(source_id="turn-1", text="AB", producer_id="host")
        with pytest.raises(ValueError, match="cannot mix"):
            client.remember(
                "AB",
                requested_block_id="stable_project_facts",
                write_mode="lossless_atomic",
                metadata={
                    "source_segments": [
                        {
                            "source_id": "turn-1",
                            "source_sha256": source["source_sha256"],
                            "byte_start": 0,
                            "byte_end": 1,
                            "text": "A",
                        },
                        {"source_id": "turn-1", "text": "B"},
                    ]
                },
            )
    finally:
        index.close()


@pytest.mark.parametrize(
    "segment",
    [
        {"byte_start": 2, "byte_end": 5, "text": "🧠"},
        {"byte_start": 0, "byte_end": 5, "text": "A🧠", "source_sha256": "0" * 64},
    ],
)
def test_bound_segment_rejects_invalid_byte_range_or_source_hash(tmp_path, segment) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session-a", backend_instance=index
    )
    try:
        source = client.register_source(source_id="turn-1", text="A🧠B", producer_id="trusted-host")
        payload = {"source_id": "turn-1", "source_sha256": source["source_sha256"], **segment}
        with pytest.raises(ValueError, match="source segment"):
            client.remember(
                segment["text"],
                requested_block_id="stable_project_facts",
                write_mode="lossless_atomic",
                metadata={"source_segments": [payload]},
            )
    finally:
        index.close()
