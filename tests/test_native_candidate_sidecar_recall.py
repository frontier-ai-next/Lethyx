import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from um_mem import MemoryClient
from um_mem.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from um_mem.memory.backends.base import MemoryQuery
from um_mem.memory.backends.contracts import block_to_json
from um_mem.memory.backends.hindsight import HindsightBackend
from um_mem.memory.backends.mempalace import (
    _MEMPALACE_PARTITION_PROJECTION_MARKER,
    MemPalaceBackend,
    _native_gate_for,
    _NativeMemPalace,
    _record_partition_identity,
)
from um_mem.memory.backends.mnemosyne import (
    MNEMOSYNE_PARTITION_PROJECTION_MARKER,
    MnemosyneBackend,
)
from um_mem.memory.backends.query_policy import merge_native_block_results
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.hindsight.config import HindsightConfig


def test_external_hindsight_transport_survives_projection_thread_changes(tmp_path, monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            response = {"results": []} if self.path.endswith("/recall") else {"success": True}
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_PUT = do_POST

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(base_url=f"http://127.0.0.1:{server.server_port}"),
    )
    monkeypatch.setattr(backend, "_client", lambda: pytest.fail("shared SDK event loop used"))
    partition = backend._record_partition_identity(_record("one", "exact source"))
    timestamp = datetime(2026, 10, 18, tzinfo=UTC)

    def write(i):
        backend._ensure_partition_bank(partition)
        return backend._retain(
            bank_id=partition.partition_id,
            content=f"exact {i}",
            document_id=f"record-{i}",
            timestamp=timestamp,
            metadata={"source": f"source-{i}"},
            tags=["native-scope"],
            retain_async=False,
        )

    try:
        # Separate pools recreate the real worker lifecycle, then read from a third thread.
        for i in range(2):
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(write, i).result(timeout=3)["success"]
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert (
                executor.submit(
                    backend._native_recall_hits,
                    MemoryQuery(query="exact", max_tokens=256),
                    partition=partition,
                ).result(timeout=3)
                == []
            )
        retained = [body for path, body in requests if path.endswith("/memories")]
        assert len(retained) == 2
        assert [body["items"][0]["document_id"] for body in retained] == ["record-0", "record-1"]
        assert all(body["async"] is False for body in retained)
        assert all(body["items"][0]["timestamp"] == timestamp.isoformat() for body in retained)
        assert all(f"/banks/{partition.partition_id}" in path for path, _ in requests)
        assert requests[-1][1]["tags_match"] == "all_strict"
    finally:
        backend.close()
        server.shutdown()
        server.server_close()
        thread.join(3)


def _record(
    block_id: str,
    content: str,
    *,
    primary_block_id: str = "workspace_episodes",
) -> MemoryBlockRecord:
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
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
        workspace_id="ws",
        revision_id=f"revision:{block_id}",
        block_membership=RecordBlockMembership(
            primary_block_id=primary_block_id,
            catalog_revision_id="catalog:test",
        ),
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"native-order:{block_id}",
                produced_by="verifier",
                producer_id="native-order-test",
            ),
        ),
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
        "source_identity": f"um-mem:{record.workspace_id}:{record.block_id}",
        "primary_block_id": record.block_membership.primary_block_id,
        "projection_marker": projection_marker,
        "native_partition_id": HindsightBackend._record_partition_identity(record).partition_id,
        "catalog_revision_id": record.block_membership.catalog_revision_id,
        "bank_id": HindsightBackend._record_partition_identity(record).partition_id,
        "canonical_record_json": json.dumps(
            block_to_json(record), ensure_ascii=False, sort_keys=True
        ),
    }
    if session_partition_id:
        metadata["session_partition_id"] = session_partition_id
    if commit_revision is not None:
        metadata.update(
            {
                "commit_revision": commit_revision,
                "workspace_id": record.workspace_id,
            }
        )
    return metadata


class _FakeHindsightClient:
    def __init__(self, records: list[MemoryBlockRecord]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.documents = self
        self.banks = self
        self.rows = [
            {
                "id": record.block_id,
                "text": record.content,
                "score": 1.0 - index / 10,
                "bank_id": HindsightBackend._record_partition_identity(record).partition_id,
                "tags": [
                    f"um_mem:block:{record.block_membership.primary_block_id}",
                    *record.tags,
                ],
                "metadata": _native_metadata(record, commit_revision=index + 1),
            }
            for index, record in enumerate(records)
        ]

    def list_banks(self) -> list[dict[str, Any]]:
        return []

    def recall(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(dict(kwargs))
        if kwargs.get("query") == "no semantic match":
            return []
        tags = list(kwargs.get("tags") or ())
        bank_id = str(kwargs.get("bank_id") or "")
        rows = [
            row
            for row in self.rows
            if row["bank_id"] == bank_id and all(tag in row["tags"] for tag in tags)
        ]
        return rows[: int(kwargs.get("max_tokens") or len(rows))]

    def get_document(self, *, bank_id: str, document_id: str) -> dict[str, Any]:
        block_id = document_id.removeprefix("UM-MEM:block:")
        row = next(row for row in self.rows if row["id"] == block_id and row["bank_id"] == bank_id)
        return {
            "id": document_id,
            "bank_id": bank_id,
            "original_text": row["text"],
            "tags": row["tags"],
            "document_metadata": row["metadata"],
        }


class _FakeMnemosyneMemory:
    def __init__(
        self,
        records: list[MemoryBlockRecord],
        *,
        session_id: str = "",
        projection_marker: str = MNEMOSYNE_PARTITION_PROJECTION_MARKER,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.remember_calls: list[dict[str, Any]] = []
        self.session_id = session_id
        self.rows = {
            record.block_id: {
                "id": record.block_id,
                "content": MnemosyneBackend._native_expected_content(record, index + 1),
                "source": f"um_mem:block:{record.block_membership.primary_block_id}",
                "session_id": (
                    session_id
                    or MnemosyneBackend._native_session_partition_id(
                        record.block_membership.primary_block_id,
                        workspace_id=record.workspace_id,
                        catalog_revision_id=record.block_membership.catalog_revision_id,
                        backend_partition_id=record.block_membership.backend_partition_id,
                    )
                ),
                "metadata": _native_metadata(
                    record,
                    commit_revision=index + 1,
                    projection_marker=projection_marker,
                    session_partition_id=(
                        session_id
                        or MnemosyneBackend._native_session_partition_id(
                            record.block_membership.primary_block_id,
                            workspace_id=record.workspace_id,
                            catalog_revision_id=record.block_membership.catalog_revision_id,
                            backend_partition_id=record.block_membership.backend_partition_id,
                        )
                    ),
                ),
                "score": 1.0 - index / 10,
            }
            for index, record in enumerate(records)
        }

    def recall(
        self,
        query: str,
        *,
        top_k: int,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "top_k": top_k, "source": source})
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "source": row["source"],
                "score": row["score"],
            }
            for row in [
                value for value in self.rows.values() if source is None or value["source"] == source
            ][:top_k]
        ]

    def remember(self, content: str, **kwargs: Any) -> str:
        self.remember_calls.append({"content": content, **kwargs})
        metadata = dict(kwargs.get("metadata") or {})
        native_id = str(metadata.get("block_id") or content)
        self.rows[native_id] = {
            "id": native_id,
            "content": MnemosyneBackend._provider_temporal_tags(content),
            "source": str(kwargs.get("source") or ""),
            "session_id": self.session_id,
            "metadata": metadata,
            "score": 1.0,
        }
        return native_id

    def get_all_memories(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows.values()]

    def get(self, native_id: str) -> dict[str, Any] | None:
        row = self.rows.get(native_id)
        return dict(row) if row is not None else None

    def close(self) -> None:
        return None


class _PartitionedMnemosyneStore:
    def __init__(
        self,
        records: list[MemoryBlockRecord],
        *,
        projection_marker: str = MNEMOSYNE_PARTITION_PROJECTION_MARKER,
    ) -> None:
        self.instances: dict[str, _FakeMnemosyneMemory] = {}
        self.records = records
        self.projection_marker = projection_marker

    def construct(self, session_partition_id: str) -> _FakeMnemosyneMemory:
        if session_partition_id not in self.instances:
            rows = [
                record
                for record in self.records
                if MnemosyneBackend._native_session_partition_id(
                    record.block_membership.primary_block_id,
                    workspace_id=record.workspace_id,
                    catalog_revision_id=record.block_membership.catalog_revision_id,
                    backend_partition_id=record.block_membership.backend_partition_id,
                )
                == session_partition_id
            ]
            self.instances[session_partition_id] = _FakeMnemosyneMemory(
                rows,
                session_id=session_partition_id,
                projection_marker=self.projection_marker,
            )
        return self.instances[session_partition_id]


class _CapBeforeSourceMnemosyneMemory(_FakeMnemosyneMemory):
    def recall(
        self,
        query: str,
        *,
        top_k: int,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "top_k": top_k, "source": source})
        globally_capped = list(self.rows.values())[:top_k]
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "source": row["source"],
                "score": row["score"],
            }
            for row in globally_capped
            if source is None or row["source"] == source
        ]


class _FakeMemPalaceNative:
    def __init__(self, records: list[MemoryBlockRecord]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.release_count = 0
        self.release_counts_at_rows: list[int] = []
        self._rows = [
            {
                "id": record.block_id,
                "content": record.content,
                "canonical_record_json": json.dumps(
                    block_to_json(record), ensure_ascii=False, sort_keys=True
                ),
                "score": 1.0 - index / 10,
                "logical_block_id": record.block_membership.primary_block_id,
                "projection_marker": _MEMPALACE_PARTITION_PROJECTION_MARKER,
                "native_partition_id": _record_partition_identity(record).partition_id,
                "catalog_revision_id": record.block_membership.catalog_revision_id,
                "commit_revision": index + 1,
            }
            for index, record in enumerate(records)
        ]

    def rows(
        self,
        *,
        query: str,
        limit: int,
        partition: object,
        rebuild_records: tuple[tuple[MemoryBlockRecord, int], ...] = (),
    ) -> list[dict[str, Any]]:
        del rebuild_records
        partition_id = str(partition.partition_id)
        self.release_counts_at_rows.append(self.release_count)
        self.calls.append({"query": query, "limit": limit, "partition_id": partition_id})
        if query == "no semantic match":
            return []
        return [dict(row) for row in self._rows if row["native_partition_id"] == partition_id][
            :limit
        ]

    def collection(self, _store: str, *, partition: object) -> "_FakeMemPalaceNative":
        self._active_partition_id = str(partition.partition_id)
        return self

    def get(self, *, ids: list[str]) -> SimpleNamespace:
        rows = [
            row
            for row in self._rows
            if row["id"] in ids and row["native_partition_id"] == self._active_partition_id
        ]
        return SimpleNamespace(
            ids=[row["id"] for row in rows],
            documents=[row["content"] for row in rows],
            metadatas=[
                {key: value for key, value in row.items() if key not in {"id", "content", "score"}}
                for row in rows
            ],
        )

    def health(self, partitions: tuple[object, ...]) -> dict[str, Any]:
        return {
            "ok": True,
            "detail": "ready",
            "stores_ok": [f"{partition.partition_id}:palace.idea" for partition in partitions],
            "stores_fail": [],
        }

    def close(self) -> None:
        return None

    def release(self) -> None:
        self.release_count += 1


def _hindsight_backend(tmp_path: Path, records: list[MemoryBlockRecord]) -> HindsightBackend:
    return HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=_FakeHindsightClient(records),
        reconcile_pending_writes=False,
    )


def _mnemosyne_backend(tmp_path: Path, records: list[MemoryBlockRecord]) -> MnemosyneBackend:
    backend = MnemosyneBackend(
        repo_root=tmp_path, workspace_id="ws", enabled=True, reconcile_pending_writes=False
    )
    backend.available = True
    backend._native_memory = _FakeMnemosyneMemory(records)
    return backend


def _partitioned_mnemosyne_backend(
    tmp_path: Path,
    records: list[MemoryBlockRecord],
    *,
    projection_marker: str = MNEMOSYNE_PARTITION_PROJECTION_MARKER,
) -> tuple[MnemosyneBackend, _PartitionedMnemosyneStore]:
    backend = MnemosyneBackend(
        repo_root=tmp_path, workspace_id="ws", enabled=True, reconcile_pending_writes=False
    )
    backend.available = True
    backend._native_memory = None
    backend._native_partitions = {}
    store = _PartitionedMnemosyneStore(
        records,
        projection_marker=projection_marker,
    )
    backend._construct_native = store.construct
    return backend, store


def _mempalace_backend(tmp_path: Path, records: list[MemoryBlockRecord]) -> MemPalaceBackend:
    backend = MemPalaceBackend.__new__(MemPalaceBackend)
    backend._repo_root = tmp_path
    backend._workspace_id = "ws"
    backend._native = _FakeMemPalaceNative(records)
    backend._index = BackendSQLiteIndex(
        tmp_path / "mempalace.sqlite",
        backend_id="mempalace",
        workspace_id="ws",
    )
    backend._native_operation_gate = _native_gate_for(backend._index.path)
    backend._operation_lock_timeout_seconds = 30.0
    backend._operation_state = threading.local()
    return backend


@pytest.mark.parametrize("invalid_score", [None, float("nan"), float("inf")])
def test_cosine_merge_rejects_missing_or_nonfinite_scores(invalid_score: float | None) -> None:
    record = _record("invalid-score", "Native score must be explicit.")
    with pytest.raises(ValueError, match="finite similarity"):
        merge_native_block_results(
            [("workspace_episodes", [(record, invalid_score)])],
            limit=1,
            score_mode="cosine_similarity",
        )


def test_mempalace_operation_gate_releases_native_owner_after_outermost_operation(
    tmp_path: Path,
) -> None:
    backend = _mempalace_backend(tmp_path, [])
    try:
        with backend.operation_gate(), backend.operation_gate():
            assert backend._native.release_count == 0
        assert backend._native.release_count == 1
    finally:
        backend.close()


def test_mempalace_reopens_native_client_between_projection_audit_and_semantic_query(
    tmp_path: Path,
) -> None:
    record = _record("target", "target", primary_block_id="block-a")
    backend = _mempalace_backend(tmp_path, [record])
    _retain_projected_sidecar(backend, [record])
    try:
        recalled = backend.recall(
            {
                "query": "target",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["block-a"],
                "limit": 1,
            }
        )

        assert [item.block_id for item in recalled] == ["target"]
        assert backend._native.release_counts_at_rows == [1]
        assert backend._native.release_count == 2
    finally:
        backend.close()


def test_mempalace_scoped_query_does_not_backfill_from_unranked_native_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        def __init__(self) -> None:
            self.ids = [["ranked"]]
            self.documents = [["ranked content"]]
            self.metadatas = [[{"logical_block_id": "block-a"}]]
            self.distances = [[0.1]]

    class Collection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def count(self) -> int:
            return 1

        def query(self, **kwargs):
            self.calls.append(("query", kwargs))
            return Result()

        def get(self, **kwargs):
            self.calls.append(("get", kwargs))
            raise AssertionError("semantic recall must not use unranked collection.get backfill")

    collection = Collection()
    native = object.__new__(_NativeMemPalace)
    monkeypatch.setattr(native, "collection", lambda store, *, partition: collection)
    partition = MemPalaceBackend.native_partition_identity(
        workspace_id="ws",
        primary_block_id="block-a",
        catalog_revision_id="catalog:test",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="block-a", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )

    rows = native.rows(query="target", limit=1, partition=partition)

    assert [row["id"] for row in rows] == ["ranked"]
    assert {name for name, _kwargs in collection.calls} == {"query"}
    assert "where" not in collection.calls[0][1]


NativeBackend = HindsightBackend | MemPalaceBackend | MnemosyneBackend


def _backend_id(backend: NativeBackend) -> str:
    if isinstance(backend, MemPalaceBackend):
        return "mempalace"
    if isinstance(backend, MnemosyneBackend):
        return "mnemosyne"
    return "hindsight"


def _native_filter_calls(backend: NativeBackend) -> list[dict[str, Any]]:
    if isinstance(backend, HindsightBackend):
        return backend._provided_client.calls
    if isinstance(backend, MnemosyneBackend):
        return backend._native_memory.calls
    return backend._native.calls


def _retain_sidecar(backend: NativeBackend, records: list[MemoryBlockRecord]) -> None:
    for record in records:
        entry = backend._index.open_write_outbox(
            record=record,
            store=_backend_id(backend),
        )
        backend._index.retain_block(record)
        backend._index.complete_write_outbox(entry.commit_revision)


def _retain_projected_sidecar(
    backend: HindsightBackend | MemPalaceBackend,
    records: list[MemoryBlockRecord],
) -> None:
    for record in records:
        store = "palace.idea" if isinstance(backend, MemPalaceBackend) else "hindsight"
        entry = backend._index.open_write_outbox(record=record, store=store)
        backend._index.retain_block(record)
        native_id = (
            record.block_id
            if isinstance(backend, MemPalaceBackend)
            else f"UM-MEM:block:{record.block_id}"
        )
        backend._index.set_write_outbox_native_ref(entry.commit_revision, native_id)
        backend._index.complete_write_outbox(entry.commit_revision)


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight_backend, _mempalace_backend],
    ids=["hindsight", "mempalace"],
)
def test_scoped_native_recall_filters_before_top_k_and_returns_lower_global_target(
    tmp_path: Path,
    backend_factory: Callable[[Path, list[MemoryBlockRecord]], NativeBackend],
) -> None:
    records = [
        _record("distractor", "higher global distractor", primary_block_id="block-b"),
        _record("target", "lower global target", primary_block_id="block-a"),
    ]
    backend = backend_factory(tmp_path, records)
    assert isinstance(backend, (HindsightBackend, MemPalaceBackend))
    _retain_projected_sidecar(backend, records)
    try:
        recalled = backend.recall(
            {
                "query": "opaque",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["block-a"],
                "limit": 1,
            }
        )
        calls = _native_filter_calls(backend)
    finally:
        backend.close()

    assert [record.block_id for record in recalled] == ["target"]
    assert len(calls) == 1
    if isinstance(backend, HindsightBackend):
        assert calls[0]["tags"] == ["um_mem:block:block-a"]
        assert calls[0]["tags_match"] == "all_strict"
    elif isinstance(backend, MnemosyneBackend):
        assert calls[0]["source"] == "um_mem:block:block-a"
    else:
        assert calls[0]["partition_id"] == _record_partition_identity(records[1]).partition_id
    metadata = recalled[0].metadata
    assert metadata["native_logical_block_id"] == "block-a"
    assert metadata["native_rank"] == 1
    assert metadata["candidate_pool_ids"] == ["target"]
    assert metadata["selected_record_ids"] == ["target"]
    assert metadata["selection_mode"] == (
        "native_similarity" if isinstance(backend, MemPalaceBackend) else "block_rank_round"
    )


@pytest.mark.parametrize("leaf_count", [9, 100])
def test_mempalace_global_top_k_keeps_late_leaf_and_is_invariant_to_block_order(
    tmp_path: Path, leaf_count: int
) -> None:
    records = [
        _record(f"record-{index:03}", f"Fact {index}", primary_block_id=f"leaf-{index:03}")
        for index in range(leaf_count)
    ]
    backend = _mempalace_backend(tmp_path, records)
    _retain_projected_sidecar(backend, records)
    native = backend._native
    assert isinstance(native, _FakeMemPalaceNative)
    for index, row in enumerate(native._rows):
        row["score"] = index / leaf_count
    try:
        blocks = [record.block_membership.primary_block_id for record in records]
        results = [
            backend.recall(
                {
                    "query": "opaque query",
                    "logical_block_ids": order,
                    "workspace_id": "ws",
                    "scope_chain": ["workspace:ws"],
                    "limit": 8,
                }
            )
            for order in (blocks, list(reversed(blocks)), blocks[1:] + blocks[:1])
        ]
        expected_ids = [record.block_id for record in reversed(records[-8:])]
        assert [[record.block_id for record in result] for result in results] == [expected_ids] * 3
        assert all(
            record.metadata["selection_mode"] == "native_similarity" for record in results[0]
        )
        assert len(results[0][0].metadata["candidate_pool_ids"]) == leaf_count
    finally:
        backend.close()


def test_mnemosyne_session_partition_returns_target_below_global_top_k(
    tmp_path: Path,
) -> None:
    records = [
        *[
            _record(
                f"distractor-{index:02d}",
                f"higher global distractor {index}",
                primary_block_id="block-b",
            )
            for index in range(60)
        ],
        _record("target", "lower global target", primary_block_id="block-a"),
    ]
    backend, store = _partitioned_mnemosyne_backend(tmp_path, records)
    _retain_sidecar(backend, records)
    try:
        health = backend.health()
        assert health["native_ok"] is True
        assert health["strict_ok"] is True
        assert health["native_block_scoping_supported"] is True
        recalled = backend.recall(
            {
                "query": "lower global target",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["block-a"],
                "limit": 1,
            }
        )
    finally:
        backend.close()

    assert [record.block_id for record in recalled] == ["target"]
    target_partition = MnemosyneBackend._native_session_partition_id(
        "block-a",
        workspace_id="ws",
        catalog_revision_id="catalog:test",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="block-a", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )
    assert store.instances[target_partition].calls == [
        {
            "query": "lower global target",
            "top_k": 1,
            "source": "um_mem:block:block-a",
        }
    ]


def test_mnemosyne_multi_block_recall_is_rank_round_and_excludes_foreign_rows(
    tmp_path: Path,
) -> None:
    records = [
        _record("a1", "A first", primary_block_id="block-a"),
        _record("a2", "A second", primary_block_id="block-a"),
        _record("b1", "B first", primary_block_id="block-b"),
        _record("b2", "B second", primary_block_id="block-b"),
        _record("foreign", "foreign", primary_block_id="block-c"),
    ]
    backend, _store = _partitioned_mnemosyne_backend(tmp_path, records)
    _retain_sidecar(backend, records)
    try:
        recalled = backend.recall(
            {
                "query": "opaque",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["block-a", "block-b"],
                "limit": 4,
            }
        )
    finally:
        backend.close()

    assert [record.block_id for record in recalled] == ["a1", "b1", "a2", "b2"]
    assert all(record.block_id != "foreign" for record in recalled)


def test_mnemosyne_search_all_enumerates_existing_primary_partitions(
    tmp_path: Path,
) -> None:
    records = [
        _record("a1", "A first", primary_block_id="block-a"),
        _record("b1", "B first", primary_block_id="block-b"),
        _record("c1", "C first", primary_block_id="block-c"),
    ]
    backend, _store = _partitioned_mnemosyne_backend(tmp_path, records)
    _retain_sidecar(backend, records)
    try:
        recalled = backend.recall(
            {
                "query": "opaque",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "search_all": True,
                "limit": 3,
            }
        )
    finally:
        backend.close()

    assert [record.block_id for record in recalled] == ["a1", "b1", "c1"]
    assert [record.metadata["native_logical_block_id"] for record in recalled] == [
        "block-a",
        "block-b",
        "block-c",
    ]


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight_backend, _mempalace_backend],
    ids=["hindsight", "mempalace"],
)
def test_native_recall_requires_selected_blocks_or_explicit_search_all(
    tmp_path: Path,
    backend_factory: Callable[[Path, list[MemoryBlockRecord]], NativeBackend],
) -> None:
    record = _record(
        "scoped-only",
        "scoped",
        primary_block_id="workspace_episodes",
    )
    backend = backend_factory(tmp_path, [record])
    _retain_projected_sidecar(backend, [record])
    try:
        with pytest.raises(ValueError, match="logical_block_ids or search_all"):
            backend.recall(
                {
                    "query": "scoped",
                    "workspace_id": "ws",
                    "limit": 5,
                }
            )
    finally:
        backend.close()


def test_mempalace_health_exports_complete_per_block_metrics(tmp_path: Path) -> None:
    record = _record(
        "metrics-leaf",
        "metrics",
        primary_block_id="workspace_episodes",
    )
    backend = _mempalace_backend(tmp_path, [record])
    _retain_projected_sidecar(backend, [record])
    try:
        recalled = backend.recall(
            {
                "query": "metrics",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["workspace_episodes"],
                "limit": 1,
            }
        )
        assert [item.block_id for item in recalled] == [record.block_id]
        health = backend.health()
    finally:
        backend.close()

    [metrics] = health["block_metrics"]
    assert metrics["primary_block_id"] == "workspace_episodes"
    assert metrics["record_count"] == 1
    assert metrics["recall_count"] == 1
    assert metrics["hit_count"] == 1
    assert metrics["scoped_recall_count"] == 1
    assert metrics["broad_recall_count"] == 0
    assert metrics["health"] == "ready"
    assert metrics["partition_mode"] == "native_collection_family"


def test_mempalace_runtime_health_skips_exhaustive_partition_probe(tmp_path: Path) -> None:
    record = _record(
        "runtime-health",
        "health",
        primary_block_id="workspace_episodes",
    )
    backend = _mempalace_backend(tmp_path, [record])
    _retain_projected_sidecar(backend, [record])
    calls: list[tuple[object, ...]] = []
    original_health = backend._native.health

    def health(partitions: tuple[object, ...]) -> dict[str, Any]:
        calls.append(partitions)
        return original_health(partitions)

    backend._native.health = health
    try:
        runtime = backend.health()
        preflight = backend.preflight_health()
    finally:
        backend.close()

    assert runtime["native_ok"] is True
    assert preflight["native_ok"] is True
    runtime_timing = runtime["preflight_timing"]
    preflight_timing = preflight["preflight_timing"]
    assert runtime_timing["strict_partition_validation"] is False
    assert runtime_timing["lock_wait_ms"] == 0.0
    assert runtime_timing["partition_count"] == 1
    assert runtime_timing["health_body_ms"] >= 0
    assert preflight_timing["strict_partition_validation"] is True
    assert preflight_timing["lock_wait_ms"] >= 0
    assert preflight_timing["partition_count"] == 1
    assert preflight_timing["health_body_ms"] >= 0
    assert calls[0] == ()
    assert len(calls[1]) == 1


def test_hindsight_health_reports_native_bank_inventory_and_metrics(
    tmp_path: Path,
) -> None:
    record = _record(
        "hindsight-metrics",
        "metrics",
        primary_block_id="workspace_episodes",
    )
    backend = _hindsight_backend(tmp_path, [record])
    _retain_projected_sidecar(backend, [record])
    try:
        backend.recall(
            {
                "query": "metrics",
                "workspace_id": "ws",
                "logical_block_ids": ["workspace_episodes"],
                "limit": 1,
            }
        )
        health = backend.health()
    finally:
        backend.close()

    assert health["strict_ok"] is True
    assert health["native_partition_count"] == 1
    assert health["native_partition_mode"] == "native_bank"
    [metrics] = health["block_metrics"]
    assert metrics["record_count"] == 1
    assert metrics["recall_count"] == 1
    assert metrics["health"] == "ready"


def test_hindsight_runtime_health_does_not_block_writes_during_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _hindsight_backend(tmp_path, [])
    monkeypatch.setattr(backend._index, "wake_native_projection", lambda: None)
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)
    saved = client.remember(
        "durable source awaiting native extraction",
        requested_block_id="stable_project_facts",
        operation_id="pending",
    )
    try:
        assert saved.saved
        health = backend.health()
        assert health["native_ok"] is True
        assert health["strict_ok"] is True
        assert health["projection"]["pending"] == 1
        with pytest.raises(RuntimeError, match="native_projection_pending"):
            backend.recall(
                {
                    "query": "source",
                    "workspace_id": "ws",
                    "scope_chain": ["workspace:ws"],
                    "logical_block_ids": ["stable_project_facts"],
                    "limit": 1,
                }
            )
        assert backend.preflight_health()["strict_ok"] is False
    finally:
        backend.close()


def test_mnemosyne_runtime_health_leaves_exhaustive_validation_to_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record("health", "source in a native partition")
    backend, _ = _partitioned_mnemosyne_backend(tmp_path, [record])
    _retain_sidecar(backend, [record])
    calls = []

    def inspect_partitions() -> tuple[bool, str]:
        calls.append(True)
        return False, "native_projection_rebuild_required"

    monkeypatch.setattr(backend, "_partition_projection_ready", inspect_partitions)
    try:
        assert backend.health()["strict_ok"] is True
        assert calls == []
        assert backend.preflight_health()["strict_ok"] is False
        assert calls == [True]
    finally:
        backend.close()


def test_active_context_metrics_charge_only_the_rendered_leaf(
    tmp_path: Path,
) -> None:
    active = _record(
        "active-metrics",
        "active",
        primary_block_id="workspace_episodes",
    )
    passive = _record(
        "passive-metrics",
        "passive",
        primary_block_id="decisions_and_contracts",
    )
    backend = _mempalace_backend(tmp_path, [active, passive])
    _retain_projected_sidecar(backend, [active, passive])
    client = MemoryClient(
        tmp_path,
        workspace_id="ws",
        backend="mempalace",
        backend_instance=backend,
    )
    try:
        items = client.build_context(
            "active",
            logical_block_ids=("workspace_episodes",),
            scope_chain=("workspace:ws",),
            max_tokens=256,
            limit=1,
        )
        metrics = {row["primary_block_id"]: row for row in backend.health()["block_metrics"]}
    finally:
        backend.close()

    assert items
    assert metrics["workspace_episodes"]["rendered_tokens"] > 0
    assert metrics["decisions_and_contracts"]["rendered_tokens"] == 0


def test_mnemosyne_workspace_record_is_visible_from_a_later_host_session(
    tmp_path: Path,
) -> None:
    record = replace(_record("cross-session", "durable target"), run_id="session-a")
    backend, _store = _partitioned_mnemosyne_backend(tmp_path, [record])
    _retain_sidecar(backend, [record])
    try:
        recalled = backend.recall(
            {
                "query": "durable target",
                "workspace_id": "ws",
                "run_id": "session-b",
                "scope_chain": ["session:session-b", "workspace:ws"],
                "logical_block_ids": ["workspace_episodes"],
                "limit": 1,
            }
        )
    finally:
        backend.close()

    assert [item.block_id for item in recalled] == ["cross-session"]


def test_mnemosyne_superseded_native_row_is_removed_by_canonical_overlay(
    tmp_path: Path,
) -> None:
    records = [
        replace(_record("old", "old value"), status="superseded"),
        _record("current", "current value"),
    ]
    backend, _store = _partitioned_mnemosyne_backend(tmp_path, records)
    _retain_sidecar(backend, records)
    try:
        recalled = backend.recall(
            {
                "query": "value",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["workspace_episodes"],
                "limit": 2,
            }
        )
    finally:
        backend.close()

    assert [record.block_id for record in recalled] == ["current"]


def test_mnemosyne_old_native_projection_fails_closed(
    tmp_path: Path,
) -> None:
    record = _record("old-projection", "must rebuild")
    backend, _store = _partitioned_mnemosyne_backend(
        tmp_path,
        [record],
        projection_marker="block_partition_v1",
    )
    _retain_sidecar(backend, [record])
    try:
        with pytest.raises(ValueError, match="native_projection_rebuild_required"):
            backend.recall(
                {
                    "query": "must rebuild",
                    "workspace_id": "ws",
                    "scope_chain": ["workspace:ws"],
                    "logical_block_ids": ["workspace_episodes"],
                    "limit": 1,
                }
            )
    finally:
        backend.close()


def test_mnemosyne_write_uses_block_partition_session_scope_and_marker(
    tmp_path: Path,
) -> None:
    record = _record("partitioned-write", "write target", primary_block_id="block-a")
    backend, store = _partitioned_mnemosyne_backend(tmp_path, [])
    try:
        result = backend.retain_block(record)
    finally:
        backend.close()

    partition_id = MnemosyneBackend._native_session_partition_id(
        "block-a",
        workspace_id="ws",
        catalog_revision_id="catalog:test",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="block-a", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )
    call = store.instances[partition_id].remember_calls[0]
    assert result["ok"] is True
    assert call["scope"] == "session"
    assert call["source"] == "um_mem:block:block-a"
    assert call["metadata"]["session_partition_id"] == partition_id
    assert call["metadata"]["projection_marker"] == MNEMOSYNE_PARTITION_PROJECTION_MARKER


def test_mempalace_scoped_recall_overfetches_ranked_rows_past_stale_revision(
    tmp_path: Path,
) -> None:
    records = [
        replace(
            _record("stale", "best stale match", primary_block_id="block-a"), status="superseded"
        ),
        _record("current", "next current match", primary_block_id="block-a"),
    ]
    backend = _mempalace_backend(tmp_path, records)
    _retain_projected_sidecar(backend, records)
    try:
        recalled = backend.recall(
            {
                "query": "match",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["block-a"],
                "limit": 1,
            }
        )
        calls = list(backend._native.calls)
    finally:
        backend.close()

    assert [record.block_id for record in recalled] == ["current"]
    assert calls[0]["partition_id"] == _record_partition_identity(records[0]).partition_id
    assert calls[0]["limit"] > 1


def test_hindsight_scoped_recall_scans_ranked_hits_past_stale_revision(
    tmp_path: Path,
) -> None:
    records = [
        replace(
            _record("stale", "best stale match", primary_block_id="block-a"),
            status="superseded",
        ),
        _record("current", "next current match", primary_block_id="block-a"),
    ]
    backend = _hindsight_backend(tmp_path, records)
    _retain_projected_sidecar(backend, records)
    try:
        recalled = backend.recall(
            {
                "query": "match",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["block-a"],
                "limit": 1,
            }
        )
    finally:
        backend.close()

    assert [record.block_id for record in recalled] == ["current"]


def test_scoped_native_recall_does_not_apply_auto_extracted_exact_dimensions(
    tmp_path: Path,
) -> None:
    record = _record(
        "target",
        "the user decision and binding constraint",
        primary_block_id="workspace_episodes",
    )
    backend = _mempalace_backend(tmp_path, [record])
    _retain_projected_sidecar(backend, [record])
    client = MemoryClient(
        tmp_path,
        workspace_id="ws",
        backend="mempalace",
        backend_instance=backend,
    )
    try:
        recalled = client.recall(
            "sealed lifecycle case alpha-decision",
            logical_block_ids=("workspace_episodes",),
            scope_chain=("workspace:ws",),
            include_candidates=True,
            limit=1,
        )
    finally:
        backend.close()

    assert [hit.metadata["block_id"] for hit in recalled] == ["target"]


def test_real_mempalace_search_all_does_not_apply_auto_extracted_exact_dimensions(
    tmp_path: Path,
) -> None:
    record = replace(
        _record(
            "target",
            "the user decision and binding constraint",
            primary_block_id="decisions_and_contracts",
        ),
        title="sealed lifecycle case alpha-audit",
    )
    backend = MemPalaceBackend(tmp_path, workspace_id="ws")
    backend.retain_block(record)
    client = MemoryClient(
        tmp_path,
        workspace_id="ws",
        backend="mempalace",
        backend_instance=backend,
    )
    query = "sealed lifecycle case alpha-audit"
    try:
        scoped = client.recall(
            query,
            logical_block_ids=("decisions_and_contracts",),
            scope_chain=("workspace:ws",),
            include_candidates=True,
            limit=1,
        )
        global_hits = client.recall(
            query,
            search_all=True,
            scope_chain=("workspace:ws",),
            include_candidates=True,
            limit=1,
        )
    finally:
        backend.close()

    assert [hit.metadata["block_id"] for hit in scoped] == ["target"]
    assert [hit.metadata["block_id"] for hit in global_hits] == ["target"]


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight_backend, _mempalace_backend],
    ids=["hindsight", "mempalace"],
)
def test_multi_block_native_recall_uses_only_backend_qualified_score_comparison(
    tmp_path: Path,
    backend_factory: Callable[[Path, list[MemoryBlockRecord]], NativeBackend],
) -> None:
    records = [
        _record("a1", "A first", primary_block_id="block-a"),
        _record("a2", "A second", primary_block_id="block-a"),
        _record("b1", "B first", primary_block_id="block-b"),
        _record("b2", "B second", primary_block_id="block-b"),
    ]
    backend = backend_factory(tmp_path, records)
    assert isinstance(backend, (HindsightBackend, MemPalaceBackend))
    _retain_projected_sidecar(backend, records)
    try:
        recalled = backend.recall(
            {
                "query": "opaque",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["block-a", "block-b"],
                "limit": 4,
            }
        )
    finally:
        backend.close()

    expected = (
        ["a1", "a2", "b1", "b2"]
        if isinstance(backend, MemPalaceBackend)
        else ["a1", "b1", "a2", "b2"]
    )
    assert [record.block_id for record in recalled] == expected
    assert all(record.metadata["selected_record_ids"] == expected for record in recalled)
    assert all(record.metadata["candidate_pool_ids"] == expected for record in recalled)


def test_multi_block_context_records_complete_rank_round_decision_ledger(
    tmp_path: Path,
) -> None:
    records = [
        _record("a1", "A first", primary_block_id="workspace_episodes"),
        _record("a2", "A second", primary_block_id="workspace_episodes"),
        _record("b1", "B first", primary_block_id="decisions_and_contracts"),
        _record("b2", "B second", primary_block_id="decisions_and_contracts"),
    ]
    backend = _hindsight_backend(tmp_path, records)
    _retain_projected_sidecar(backend, records)
    client = MemoryClient(
        tmp_path,
        workspace_id="ws",
        backend="hindsight",
        backend_instance=backend,
    )
    try:
        items = client.build_context(
            "opaque",
            logical_block_ids=("workspace_episodes", "decisions_and_contracts"),
            scope_chain=("workspace:ws",),
            include_candidates=True,
            limit=4,
        )
        row = backend._index._conn.execute(
            "SELECT block_id FROM usage WHERE outcome='memory_decision'"
        ).fetchone()
        assert row is not None
        decision_id = str(row[0]).removeprefix("memory-decision-event:")
        trace = client.memory_decision_trace(decision_id)
    finally:
        backend.close()

    expected_ids = ("a1", "b1", "a2", "b2")
    assert items
    assert trace.decision.candidate_ids == expected_ids
    assert trace.decision.selected_candidate_ids == expected_ids
    prompt_payload = json.dumps(items, ensure_ascii=False, sort_keys=True)
    assert "candidate_pool_ids" not in prompt_payload
    assert "native_rank" not in prompt_payload


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight_backend, _mempalace_backend],
    ids=["hindsight", "mempalace"],
)
def test_scoped_recall_fails_closed_when_native_block_projection_is_missing(
    tmp_path: Path,
    backend_factory: Callable[[Path, list[MemoryBlockRecord]], NativeBackend],
) -> None:
    record = _record("old-projection", "sidecar record without native block projection")
    backend = backend_factory(tmp_path, [])
    _retain_sidecar(backend, [record])
    try:
        with pytest.raises(RuntimeError, match="native_projection_rebuild_required"):
            backend.recall(
                {
                    "query": "opaque",
                    "workspace_id": "ws",
                    "scope_chain": ["workspace:ws"],
                    "logical_block_ids": ["workspace_episodes"],
                    "limit": 1,
                }
            )
    finally:
        backend.close()


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight_backend, _mempalace_backend],
    ids=["hindsight", "mempalace"],
)
def test_scoped_semantic_miss_with_valid_projection_returns_empty(
    tmp_path: Path,
    backend_factory: Callable[[Path, list[MemoryBlockRecord]], NativeBackend],
) -> None:
    record = _record("projected", "a valid projected record")
    backend = backend_factory(tmp_path, [record])
    assert isinstance(backend, (HindsightBackend, MemPalaceBackend))
    _retain_projected_sidecar(backend, [record])
    try:
        recalled = backend.recall(
            {
                "query": "no semantic match",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["workspace_episodes"],
                "limit": 1,
            }
        )
    finally:
        backend.close()

    assert recalled == []


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight_backend, _mempalace_backend, _mnemosyne_backend],
    ids=["hindsight", "mempalace", "mnemosyne"],
)
def test_native_transport_errors_are_not_reported_as_empty_success(
    tmp_path: Path,
    backend_factory: Callable[[Path, list[MemoryBlockRecord]], NativeBackend],
) -> None:
    records = [_record("transport-probe", "native transport probe")]
    backend = backend_factory(tmp_path, records)
    if isinstance(backend, (HindsightBackend, MemPalaceBackend)):
        _retain_projected_sidecar(backend, records)
    else:
        _retain_sidecar(backend, records)

    def fail_native(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise OSError("native transport failed")

    if isinstance(backend, HindsightBackend):
        backend._provided_client.recall = fail_native
    elif isinstance(backend, MnemosyneBackend):
        backend._native_memory.recall = fail_native
    else:
        backend._native.rows = fail_native
    query = {
        "query": "opaque",
        "workspace_id": "ws",
        "scope_chain": ["workspace:ws"],
        "limit": 1,
        "logical_block_ids": ["workspace_episodes"],
    }
    try:
        with pytest.raises(OSError, match="native transport failed"):
            backend.recall(query)
    finally:
        backend.close()


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight_backend, _mempalace_backend, _mnemosyne_backend],
    ids=["hindsight", "mempalace", "mnemosyne"],
)
def test_missing_primary_membership_fails_before_any_backend_mutation(
    tmp_path: Path,
    backend_factory: Callable[[Path, list[MemoryBlockRecord]], NativeBackend],
) -> None:
    backend = backend_factory(tmp_path, [])
    record = replace(_record("missing-primary", "must not mutate"), block_membership=None)
    tables = ("write_outbox", "write_outbox_native_refs", "blocks")
    before = {
        table: int(backend._index._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in tables
    }
    try:
        with pytest.raises(ValueError, match="primary block membership"):
            backend.retain_block(record)
        after = {
            table: int(backend._index._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
    finally:
        backend.close()

    assert after == before


def test_mnemosyne_batch_validates_every_record_before_first_mutation(tmp_path: Path) -> None:
    backend = _mnemosyne_backend(tmp_path, [])
    backend._project_write_outbox = lambda _entry, record: f"native:{record.block_id}"
    records = [
        _record("valid-first", "must not be partially written"),
        replace(_record("invalid-second", "missing membership"), block_membership=None),
    ]
    tables = ("write_outbox", "write_outbox_native_refs", "blocks")
    before = {
        table: int(backend._index._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in tables
    }
    try:
        with pytest.raises(ValueError, match="primary block membership"):
            backend.retain_blocks(records)
        after = {
            table: int(backend._index._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
    finally:
        backend.close()

    assert after == before


def test_hindsight_rejects_caller_owned_reserved_block_tags(tmp_path: Path) -> None:
    record = replace(
        _record("reserved-tag", "must reject reserved tag"),
        tags=("um_mem:block:caller-owned",),
    )
    backend = _hindsight_backend(tmp_path, [])
    try:
        with pytest.raises(ValueError, match="reserved um_mem:block: prefix"):
            backend.retain_block(record)
        assert backend._index.get_block(record.block_id, workspace_id="ws") is None
    finally:
        backend.close()


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight_backend, _mempalace_backend],
    ids=["hindsight", "mempalace"],
)
def test_native_recall_preserves_backend_order_and_never_injects_sidecar_candidates(
    tmp_path: Path,
    backend_factory: Callable[[Path, list[MemoryBlockRecord]], NativeBackend],
) -> None:
    native_records = [
        _record("native-first", "backend ranked first"),
        _record("native-second", "backend ranked second"),
    ]
    sidecar_only = _record(
        "sidecar-only",
        "must not backfill incomplete native output",
        primary_block_id="foreign-block",
    )
    backend = backend_factory(tmp_path, native_records)
    assert isinstance(backend, (HindsightBackend, MemPalaceBackend))
    _retain_projected_sidecar(backend, native_records)
    _retain_sidecar(backend, [sidecar_only])
    assert not hasattr(backend._index, "recall_active")
    assert not hasattr(backend._index, "recall_candidates")
    try:
        recalled = backend.recall(
            {
                "query": "opaque query",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["workspace_episodes"],
                "include_candidates": True,
                "limit": 3,
            }
        )
    finally:
        backend.close()

    assert [record.block_id for record in recalled] == ["native-first", "native-second"]


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight_backend, _mempalace_backend],
    ids=["hindsight", "mempalace"],
)
def test_scoped_recall_evidence_keeps_native_ranking_inside_selected_block(
    tmp_path: Path,
    backend_factory: Callable[[Path, list[MemoryBlockRecord]], NativeBackend],
) -> None:
    records = [
        _record("z_relevant", "the requested semantic answer"),
        _record("a_irrelevant", "unrelated material"),
    ]
    backend = backend_factory(tmp_path, records)
    assert isinstance(backend, (HindsightBackend, MemPalaceBackend))
    _retain_projected_sidecar(backend, records)
    try:
        recalled = backend.recall_evidence(
            {
                "query": "requested semantic answer",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["workspace_episodes"],
                "limit": 1,
            }
        )
    finally:
        backend.close()

    assert [hit.metadata["block_id"] for hit in recalled] == ["z_relevant"]


@pytest.mark.parametrize(
    "backend_factory",
    [_hindsight_backend, _mempalace_backend],
    ids=["hindsight", "mempalace"],
)
def test_committed_agent_candidate_is_visible_cross_session_only_when_policy_includes_it(
    tmp_path: Path,
    backend_factory: Callable[[Path, list[MemoryBlockRecord]], NativeBackend],
) -> None:
    candidate = replace(
        _record("agent-candidate", "durable fact written through the agent tool"),
        status="candidate",
        trust_level="agent_reported",
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
        run_id="producer-session",
        evidence_refs=(),
        metadata={"evidence_kind": "agent_tool_claim"},
    )
    backend = backend_factory(tmp_path, [candidate])
    assert isinstance(backend, (HindsightBackend, MemPalaceBackend))
    _retain_projected_sidecar(backend, [candidate])
    base_query = {
        "query": "durable fact written through the agent tool",
        "workspace_id": "ws",
        "run_id": "consumer-session",
        "scope_chain": ["workspace:ws"],
        "logical_block_ids": ["workspace_episodes"],
        "limit": 1,
    }
    try:
        hidden = backend.recall_evidence(base_query)
        visible = backend.recall_evidence({**base_query, "include_candidates": True})
    finally:
        backend.close()

    assert hidden == []
    assert [hit.metadata["block_id"] for hit in visible] == ["agent-candidate"]
