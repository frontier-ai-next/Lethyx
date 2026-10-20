import json
from typing import Any

import pytest

from lethyx.api import MemoryClient
from lethyx.contracts import EvidenceRef
from lethyx.memory import mcp_facade
from lethyx.contracts import MemoryBlockRecord
from lethyx.memory.backends.contracts import MemoryEdgeRecord
from lethyx.memory.kernel.telemetry import read_memory_events


class _EdgeBackend:
    def __init__(self, records: list[MemoryBlockRecord]) -> None:
        self.records = {record.block_id: record for record in records}
        self.edges: list[MemoryEdgeRecord] = []

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        return self.records.get(block_id)

    def add_edge(self, edge: MemoryEdgeRecord) -> dict[str, Any]:
        self.edges.append(edge)
        return {
            "ok": True,
            "src_id": edge.src_id,
            "dst_id": edge.dst_id,
            "backend": "capture",
        }


def _block(block_id: str, workspace_id: str) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        content="SENSITIVE_EDGE_CONTENT_MUST_NOT_REACH_TELEMETRY",
        scope="workspace",
        workspace_id=workspace_id,
    )


def test_mcp_add_edge_delegates_to_memory_client(monkeypatch) -> None:
    received: dict[str, Any] = {}

    class _Client:
        workspace_id = "alpha"

        def add_edge(self, edge: Any) -> dict[str, Any]:
            received["edge"] = edge
            return {
                "ok": True,
                "src_id": edge.src_id,
                "dst_id": edge.dst_id,
                "backend": "capture",
            }

    client = _Client()
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)

    payload = mcp_facade.lethyx_add_edge(
        src_id="source-1",
        dst_id="target-1",
        edge_type="supports",
        workspace_id="alpha",
        evidence_refs=[
            {
                "ref_type": "test_run",
                "ref_id": "verify-1",
                "produced_by": "verifier",
                "producer_id": "verifier:verify-1",
            }
        ],
    )

    assert payload["ok"] is True
    edge = received["edge"]
    assert edge.workspace_id == "alpha"
    assert edge.edge_type == "supports"
    assert edge.evidence_refs[0].ref_id == "verify-1"


def test_memory_client_writes_local_evidence_bound_edge_and_telemetry(tmp_path) -> None:
    backend = _EdgeBackend([_block("source-1", "alpha"), _block("target-1", "alpha")])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    edge = MemoryEdgeRecord(
        src_id="source-1",
        dst_id="target-1",
        edge_type="supports",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id="verify-1",
                produced_by="verifier",
                producer_id="verifier:verify-1",
            ),
        ),
    )

    result = client.add_edge(edge)

    [event] = [
        row for row in read_memory_events(tmp_path) if row["event_type"] == "memory_edge_transition"
    ]
    assert result["ok"] is True
    assert backend.edges[0].workspace_id == "alpha"
    assert event["selected_id_hashes"]
    assert "SENSITIVE_EDGE_CONTENT_MUST_NOT_REACH_TELEMETRY" not in json.dumps(event)


def test_memory_client_rejects_foreign_edge_endpoint(tmp_path) -> None:
    backend = _EdgeBackend([_block("source-1", "alpha"), _block("target-1", "beta")])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    edge = MemoryEdgeRecord(
        src_id="source-1",
        dst_id="target-1",
        edge_type="supports",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id="verify-1",
                produced_by="verifier",
                producer_id="verifier:verify-1",
            ),
        ),
    )

    with pytest.raises(ValueError, match="workspace"):
        client.add_edge(edge)

    assert backend.edges == []
