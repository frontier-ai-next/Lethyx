from pathlib import Path

from lethyx import MemoryClient
from lethyx.contracts import EvidenceRef, MemoryBlockRecord


def _evidence(name: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="verification_report",
            ref_id=name,
            produced_by="verifier",
            producer_id="verifier:workspace-isolation",
        ),
    )


def _record(block_id: str, workspace_id: str) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        content=f"isolated fact {block_id}",
        title=block_id,
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
        workspace_id=workspace_id,
        evidence_refs=_evidence(block_id),
    )


class _LeakyBackend:
    def __init__(self) -> None:
        self.records = [_record("fact-a", "workspace-a"), _record("fact-b", "workspace-b")]

    def recall_evidence(self, _query: dict[str, object]) -> list[MemoryBlockRecord]:
        return list(self.records)


def _ids(client: MemoryClient) -> tuple[str, ...]:
    return tuple(
        hit.metadata["block_id"]
        for hit in client.recall("isolated fact", search_all=True, limit=10)
    )


def test_a_to_b_to_a_recall_never_contaminates_foreign_workspace(tmp_path: Path) -> None:
    backend = _LeakyBackend()
    client_a = MemoryClient(tmp_path, workspace_id="workspace-a", backend_instance=backend)
    client_b = MemoryClient(tmp_path, workspace_id="workspace-b", backend_instance=backend)

    assert _ids(client_a) == ("fact-a",)
    assert _ids(client_b) == ("fact-b",)
    assert _ids(client_a) == ("fact-a",)
