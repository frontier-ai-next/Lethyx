import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from lethyx.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from lethyx.memory.backends.contracts import block_to_json
from lethyx.memory.backends.hindsight import HindsightBackend
from lethyx.memory.backends.mnemosyne import (
    MNEMOSYNE_PARTITION_PROJECTION_MARKER,
    MnemosyneBackend,
)
from lethyx.memory.backends.query_policy import (
    NATIVE_PROJECTION_REBUILD_REQUIRED,
)
from lethyx.memory.hindsight.config import HindsightConfig


def _record(block_id: str, content: str) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        title=block_id,
        content=content,
        kind="observation",
        status="accepted",
        retention="durable",
        delivery_policy="on_demand",
        temperature="warm",
        authority="advisory",
        trust_level="workspace_verified",
        scope="workspace",
        workspace_id="ws",
        revision_id=f"revision:{block_id}",
        block_membership=RecordBlockMembership(
            primary_block_id="workspace_episodes",
            catalog_revision_id="catalog:test",
        ),
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"native-sidecar:{block_id}",
                produced_by="verifier",
                producer_id="native-sidecar-contract-test",
            ),
        ),
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
    )


def _native_metadata(
    record: MemoryBlockRecord,
    *,
    commit_revision: int | None = None,
    projection_marker: str = "block_partition_v1",
    session_partition_id: str = "",
) -> dict[str, Any]:
    metadata = {
        "block_id": record.block_id,
        "source_identity": f"lethyx:{record.workspace_id}:{record.block_id}",
        "primary_block_id": record.block_membership.primary_block_id,
        "projection_marker": projection_marker,
        "native_partition_id": HindsightBackend._record_partition_identity(record).partition_id,
        "catalog_revision_id": record.block_membership.catalog_revision_id,
        "canonical_record_json": json.dumps(
            block_to_json(record), ensure_ascii=False, sort_keys=True
        ),
    }
    if session_partition_id:
        metadata["session_partition_id"] = session_partition_id
    if commit_revision is not None:
        metadata.update({"commit_revision": commit_revision, "workspace_id": record.workspace_id})
    return metadata


class _HindsightNative:
    def __init__(self, record: MemoryBlockRecord) -> None:
        self.record = record

    def recall(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": self.record.block_id,
                "text": self.record.content,
                "score": 1.0,
                "tags": [f"lethyx:block:{self.record.block_membership.primary_block_id}"],
                "metadata": _native_metadata(self.record),
            }
        ]


class _MnemosyneNative:
    def __init__(self, record: MemoryBlockRecord) -> None:
        session_partition_id = MnemosyneBackend._native_session_partition_id(
            record.block_membership.primary_block_id,
            workspace_id=record.workspace_id,
            catalog_revision_id=record.block_membership.catalog_revision_id,
            backend_partition_id=record.block_membership.backend_partition_id,
        )
        self.row = {
            "id": record.block_id,
            "content": MnemosyneBackend._native_expected_content(record, 1),
            "source": f"lethyx:block:{record.block_membership.primary_block_id}",
            "session_id": session_partition_id,
            "metadata": _native_metadata(
                record,
                commit_revision=1,
                projection_marker=MNEMOSYNE_PARTITION_PROJECTION_MARKER,
                session_partition_id=session_partition_id,
            ),
            "score": 1.0,
        }

    def recall(
        self,
        _query: str,
        *,
        top_k: int,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        if source is not None and source != self.row["source"]:
            return []
        return [{key: self.row[key] for key in ("id", "content", "source", "score")}][:top_k]

    def remember(self, content: str, **_kwargs: Any) -> str:
        return content

    def get(self, native_id: str) -> dict[str, Any] | None:
        return dict(self.row) if native_id == self.row["id"] else None

    def get_all_memories(self) -> list[dict[str, Any]]:
        return [dict(self.row)]


NativeBackend = HindsightBackend | MnemosyneBackend


def _hindsight(tmp_path: Path, record: MemoryBlockRecord) -> HindsightBackend:
    return HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=_HindsightNative(record),
    )


def _mnemosyne(tmp_path: Path, record: MemoryBlockRecord) -> MnemosyneBackend:
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    backend.available = True
    backend._native_memory = _MnemosyneNative(record)
    return backend


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight, _mnemosyne],
    ids=["hindsight", "mnemosyne"],
)
def test_native_hit_without_committed_sidecar_fails_closed(
    tmp_path: Path,
    backend_factory: Callable[[Path, MemoryBlockRecord], NativeBackend],
) -> None:
    backend = backend_factory(tmp_path, _record("native-only", "native without authority"))
    try:
        query = {
            "query": "native without authority",
            "workspace_id": "ws",
            "scope_chain": ["workspace:ws"],
            "logical_block_ids": ["workspace_episodes"],
            "limit": 1,
        }
        recalled = backend.recall_evidence(query)
    finally:
        backend.close()

    assert recalled == []


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight, _mnemosyne],
    ids=["hindsight", "mnemosyne"],
)
def test_scoped_recall_rejects_missing_or_unsupported_native_projection(
    tmp_path: Path,
    backend_factory: Callable[[Path, MemoryBlockRecord], NativeBackend],
) -> None:
    native = _record("same-id", "stale native content")
    canonical = replace(native, content="current canonical content")
    backend = backend_factory(tmp_path, native)
    entry = backend._index.open_write_outbox(
        record=canonical,
        store="mnemosyne" if isinstance(backend, MnemosyneBackend) else "hindsight",
    )
    backend._index.retain_block(canonical)
    backend._index.complete_write_outbox(entry.commit_revision)
    try:
        query = {
            "query": "stale native content",
            "workspace_id": "ws",
            "scope_chain": ["workspace:ws"],
            "logical_block_ids": ["workspace_episodes"],
            "limit": 1,
        }
        if isinstance(backend, MnemosyneBackend):
            assert backend.recall_evidence(query) == []
        else:
            with pytest.raises(RuntimeError, match=NATIVE_PROJECTION_REBUILD_REQUIRED):
                backend.recall_evidence(query)
    finally:
        backend.close()
