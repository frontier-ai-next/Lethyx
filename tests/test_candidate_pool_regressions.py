from pathlib import Path
import threading

from lethyx.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from lethyx.memory.backends.mempalace import MemPalaceBackend, _native_gate_for
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex


def _record(
    block_id: str,
    content: str,
    *,
    status: str = "accepted",
    retention: str = "durable",
    delivery_policy: str = "on_demand",
    temperature: str = "warm",
    source_path: str = "",
) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        revision_id=f"revision:{block_id}",
        title=block_id,
        content=content,
        block_membership=RecordBlockMembership(
            primary_block_id="workspace_episodes",
            catalog_revision_id="test-catalog",
        ),
        status=status,
        retention=retention,
        delivery_policy=delivery_policy,
        temperature=temperature,
        authority="advisory",
        trust_level="workspace_verified" if status == "accepted" else "agent_reported",
        scope="workspace",
        verification_status="verified" if status == "accepted" else "unverified",
        verification_methods=("test_run",) if status == "accepted" else (),
        promotion_decision="accepted" if status == "accepted" else "not_requested",
        workspace_id="ws",
        producer="test",
        metadata={"source_path": source_path} if source_path else {},
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"candidate-pool:{block_id}",
                produced_by="verifier",
                producer_id="verifier:candidate-pool",
            ),
        )
        if status == "accepted"
        else (),
    )


def test_mempalace_exact_selected_ids_keep_passive_and_candidate_records(tmp_path: Path) -> None:
    """Selected IDs must not be routed through MemPalace's active-only merge."""

    class NativeMustNotRun:
        def rows(self, *_args: object, **_kwargs: object) -> list[object]:
            raise AssertionError("exact selected IDs must use the sidecar fetch path")

        def release(self) -> None:
            return None

        def close(self) -> None:
            return None

    backend = MemPalaceBackend.__new__(MemPalaceBackend)
    backend._workspace_id = "ws"
    backend._repo_root = tmp_path
    backend._native = NativeMustNotRun()
    backend._index = BackendSQLiteIndex(
        tmp_path / "mempalace.sqlite", backend_id="mempalace", workspace_id="ws"
    )
    backend._native_operation_gate = _native_gate_for(backend._index.path)
    backend._operation_lock_timeout_seconds = 30.0
    backend._operation_state = threading.local()
    try:
        backend._index.retain_block(_record("candidate", "candidate evidence", status="candidate"))
        backend._index.retain_block(
            _record(
                "passive",
                "passive evidence",
                retention="archive",
                temperature="cold",
            )
        )

        rows = backend.recall(
            {
                "query": "unrelated weak query",
                "workspace_id": "ws",
                "record_ids": ["candidate", "passive"],
                "include_candidates": True,
                "source_horizon": "archive",
                "scope_chain": ["workspace:ws"],
                "limit": 2,
            }
        )
    finally:
        backend.close()

    assert [row.block_id for row in rows] == ["candidate", "passive"]
