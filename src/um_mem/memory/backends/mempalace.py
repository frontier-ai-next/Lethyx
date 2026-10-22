"""MemPalace durable backend adapter.

Public backend id is ``mempalace``. Physical Chroma/SQLite collection names may
still use the historical palace layout internally.
"""

import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chromadb.errors import InternalError as ChromaInternalError
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from filelock import FileLock, Timeout
from mempalace.backends.base import PalaceRef
from mempalace.backends.chroma import ChromaBackend, ChromaCollection
from mempalace.embedding import get_embedding_function
from tokenizers import Tokenizer  # type: ignore[import-untyped]

from um_mem.contracts import (
    ContextualUtilitySummary,
    EvidenceRef,
    MemoryBlockRecord,
    MemoryDecisionEvent,
    MemoryDecisionTrace,
    MemoryUsageEvent,
    OutcomeCreditEvent,
    ProspectiveObligation,
)
from um_mem.memory.backends.base import (
    MemoryBusyError,
    MemoryHit,
    MemoryQuery,
    NativeProjectionPending,
    ReconcileReport,
)
from um_mem.memory.backends.contracts import (
    BackendCapabilities,
    CanonicalHeadConflict,
    CanonicalHeadReservation,
    MemoryEdgeRecord,
    MemoryIndexNode,
    MemoryUsageRecord,
    NativePartitionIdentity,
    block_from_json,
    block_record_to_memory_hit,
    block_to_json,
)
from um_mem.memory.backends.query_policy import (
    NATIVE_PROJECTION_REBUILD_REQUIRED,
    ensure_record_is_storable,
    merge_hybrid_block_results,
    merge_native_block_results,
    native_primary_block_id,
    normalize_query_payload,
    record_allowed_for_query,
    record_id_sequence,
)
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex, WriteOutboxEntry
from um_mem.memory.dependencies import DependencyValidation
from um_mem.memory.kernel.telemetry import record_memory_event
from um_mem.memory.mutation import (
    AmendmentCommitReceipt,
    CanonicalMutationIdentity,
    MutationDecision,
    PublicationCommitReceipt,
)
from um_mem.memory.paths import (
    MANAGER_WORKSPACE_ID,
    palace_path_for,
    workspace_memory_root,
)
from um_mem.memory.prospective import ProspectiveTriggerContext, ProspectiveTriggerDecision

_LOG = logging.getLogger(__name__)
_MISSING_HNSW_READER = "Error creating hnsw segment reader: Nothing found on disk"
_NATIVE_PROJECTION_VISIBILITY_DELAYS_SECONDS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)


def _index_path(repo_root: Path, workspace_id: str) -> Path:
    return (
        workspace_memory_root(repo_root, workspace_id or MANAGER_WORKSPACE_ID)
        / "mempalace"
        / "UM_MEM_index.sqlite"
    )


_SNAPSHOT_SCHEMA_VERSION = 3
_SNAPSHOT_BATCH_SIZE = 128
_MEM_PALACE_OPERATION_LOCKS_GUARD = threading.Lock()
_MEM_PALACE_OPERATION_LOCKS: dict[str, threading.RLock] = {}
_MEM_PALACE_NATIVE_GATES: dict[str, "_PalaceOperationGate"] = {}
_PALACE_STORES = frozenset(
    {
        "palace.charter",
        "palace.lesson",
        "palace.idea",
        "palace.codeptr",
        "palace.skill_index",
        "palace.run",
        "palace.subtask",
        "palace.durable",
    }
)
_MEMPALACE_PARTITION_PROJECTION_MARKER = "mempalace_collection_family_v1"


def _mempalace_partition_identity(
    *,
    workspace_id: str,
    primary_block_id: str,
    catalog_revision_id: str,
    backend_partition_id: str,
) -> NativePartitionIdentity:
    workspace = str(workspace_id or "").strip()
    block_id = str(primary_block_id or "").strip()
    catalog_revision = str(catalog_revision_id or "").strip()
    partition_key = str(backend_partition_id or "").strip()
    if not workspace or not block_id or not catalog_revision or not partition_key:
        raise ValueError(
            "MemPalace native partition requires workspace, primary block, and catalog revision"
        )
    material = json.dumps(
        [workspace, partition_key],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return NativePartitionIdentity(
        backend_id="mempalace",
        workspace_id=workspace,
        primary_block_id=block_id,
        catalog_revision_id=catalog_revision,
        partition_id=f"mempalace:collection-family:{digest}",
        isolation_class="physical",
        partition_mode="native_collection_family",
    )


def _record_partition_identity(record: MemoryBlockRecord) -> NativePartitionIdentity:
    membership = record.block_membership
    if membership is None:
        raise ValueError("MemPalace native partition requires typed block membership")
    return _mempalace_partition_identity(
        workspace_id=record.workspace_id,
        primary_block_id=membership.primary_block_id,
        catalog_revision_id=membership.catalog_revision_id,
        backend_partition_id=membership.backend_partition_id,
    )


def _operation_lock_path(index_path: Path) -> Path:
    return index_path.with_suffix(".operation.lock")


def _operation_lock_for(index_path: Path) -> threading.RLock:
    key = str(index_path.resolve())
    with _MEM_PALACE_OPERATION_LOCKS_GUARD:
        lock = _MEM_PALACE_OPERATION_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _MEM_PALACE_OPERATION_LOCKS[key] = lock
        return lock


def _operation_lock_timeout_seconds() -> float:
    name = "UM_MEM_MEMPALACE_LOCK_TIMEOUT_SECONDS"
    try:
        timeout = float(os.environ.get(name, "30"))
    except ValueError as error:
        raise ValueError(f"{name} must be a number in (0, 3600]") from error
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError(f"{name} must be a finite number in (0, 3600]")
    return timeout


class _PalaceOperationGate:
    """One native process owner, concurrent operations, exclusive maintenance."""

    def __init__(self, index_path: Path) -> None:
        self.condition = threading.Condition(_operation_lock_for(index_path))
        self.file_lock = FileLock(str(_operation_lock_path(index_path)), thread_local=False)
        self.active = 0
        self.exclusive = False
        self.waiting_writers = 0
        self.releases: set[Callable[[], None]] = set()

    @contextmanager
    def hold(
        self, *, exclusive: bool, timeout: float, release: Callable[[], None] | None = None
    ) -> Iterator[float]:
        started = time.monotonic()
        deadline = started + timeout
        if not self.condition.acquire(timeout=timeout):
            raise MemoryBusyError("memory_operation_lock_timeout")
        try:
            if exclusive:
                self.waiting_writers += 1
            try:
                while (
                    self.exclusive
                    or (exclusive and self.active)
                    or (not exclusive and self.waiting_writers)
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise MemoryBusyError("memory_operation_lock_timeout")
                    self.condition.wait(remaining)
                if not self.active:
                    try:
                        self.file_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
                    except Timeout as exc:
                        raise MemoryBusyError("memory_operation_lock_timeout") from exc
                self.active += 1
                self.exclusive = exclusive
                if release is not None:
                    self.releases.add(release)
            finally:
                if exclusive:
                    self.waiting_writers -= 1
        finally:
            self.condition.release()
        try:
            yield (time.monotonic() - started) * 1000
        finally:
            with self.condition:
                self.active -= 1
                self.exclusive = False
                try:
                    if not self.active:
                        try:
                            for callback in self.releases:
                                callback()
                        finally:
                            self.releases.clear()
                            self.file_lock.release()
                finally:
                    self.condition.notify_all()


def _native_gate_for(index_path: Path) -> _PalaceOperationGate:
    key = str(index_path.resolve())
    # Construct outside the registry lock: construction resolves the existing process lock.
    gate = _PalaceOperationGate(index_path)
    with _MEM_PALACE_OPERATION_LOCKS_GUARD:
        return _MEM_PALACE_NATIVE_GATES.setdefault(key, gate)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_sqlite(source: sqlite3.Connection, destination: Path) -> None:
    destination_connection = sqlite3.connect(str(destination))
    try:
        source.backup(destination_connection)
    finally:
        destination_connection.close()


def _restore_sqlite(snapshot: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(str(snapshot))
    destination_connection = sqlite3.connect(str(destination))
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def _path_is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _store_for_block(record: MemoryBlockRecord) -> str:
    if record.kind in {"durable", "verification_report"}:
        return "palace.durable"
    if record.kind in {"lesson", "bkb_rule"}:
        return "palace.lesson"
    if record.kind == "codeptr":
        return "palace.codeptr"
    if record.kind == "skill_index":
        return "palace.skill_index"
    if record.kind == "run_state" or record.delivery_policy == "working_set":
        return "palace.run"
    return "palace.idea"


def _record_to_native_payload(
    record: MemoryBlockRecord, *, commit_revision: int | None = None
) -> tuple[str, dict[str, Any]]:
    source_backend = record.source_backend or "mempalace"
    logical_block_id = native_primary_block_id(record)
    metadata: dict[str, Any] = {
        "id": record.block_id,
        "store": _store_for_block(record),
        "tier": record.temperature or "warm",
        "scope": record.scope,
        "tags": ",".join(record.tags),
        "stage_id": "",
        "subtask_id": record.subtask_id,
        "run_id": record.run_id,
        "source_path": str(
            record.metadata.get("source_path")
            or record.metadata.get("evidence_id")
            or "memory_record"
        ),
        "verified": int(record.verification_status == "verified"),
        "workspace_id": record.workspace_id,
        "logical_block_id": logical_block_id,
        "projection_marker": _MEMPALACE_PARTITION_PROJECTION_MARKER,
        "native_partition_id": _record_partition_identity(record).partition_id,
        "catalog_revision_id": _record_partition_identity(record).catalog_revision_id,
        "kind": record.kind,
        "title": record.title,
        "status": record.status,
        "retention": record.retention,
        "delivery_policy": record.delivery_policy,
        "temperature": record.temperature,
        "authority": record.authority,
        "trust_level": record.trust_level,
        "verification_status": record.verification_status,
        "verification_methods": ",".join(record.verification_methods),
        "promotion_decision": record.promotion_decision,
        "source_backend": source_backend,
        "storage_backend": "mempalace",
        "evidence_refs_json": json.dumps(
            [asdict(ref) for ref in record.evidence_refs],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "external_refs_json": json.dumps(record.external_refs, ensure_ascii=False, sort_keys=True),
        "sensitivity": record.sensitivity,
        "content_mode": record.content_mode,
        "operational_risk": record.operational_risk,
        "metadata_json": json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
        "canonical_record_json": json.dumps(
            block_to_json(record), ensure_ascii=False, sort_keys=True
        ),
        "evidence_id": str(record.metadata.get("evidence_id") or ""),
        "producer": record.producer,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "expires_at": record.expires_at,
    }
    if commit_revision is not None:
        metadata["commit_revision"] = commit_revision
    return _store_for_block(record), metadata


def _native_projection_rows(
    record: MemoryBlockRecord, commit_revision: int
) -> tuple[str, list[tuple[str, str, dict[str, Any]]]]:
    store, metadata = _record_to_native_payload(record, commit_revision=commit_revision)
    rows = [(record.block_id, record.content, metadata)]
    embedding = get_embedding_function()
    if not isinstance(embedding, ONNXMiniLM_L6_V2):
        return store, rows
    tokenizer = Tokenizer.from_str(embedding.tokenizer.to_str())
    tokenizer.no_truncation()
    tokenizer.no_padding()
    offsets = [
        (start, end) for start, end in tokenizer.encode(record.content).offsets if end > start
    ]
    if len(offsets) <= 254:
        return store, rows
    partition = _record_partition_identity(record)
    start = 0
    ordinal = 0
    while start < len(offsets):
        end = min(len(offsets), start + 200)
        start_char = 0 if start == 0 else offsets[start][0]
        end_char = len(record.content) if end == len(offsets) else offsets[end - 1][1]
        start_byte = len(record.content[:start_char].encode("utf-8"))
        end_byte = len(record.content[:end_char].encode("utf-8"))
        rows.append(
            (
                f"{record.block_id}:window:{ordinal:04d}",
                record.content[start_char:end_char],
                {
                    "derived_window": 1,
                    "canonical_record_id": record.block_id,
                    "window_start_byte": start_byte,
                    "window_end_byte": end_byte,
                    "workspace_id": record.workspace_id,
                    "logical_block_id": partition.primary_block_id,
                    "native_partition_id": partition.partition_id,
                    "catalog_revision_id": partition.catalog_revision_id,
                    "projection_marker": _MEMPALACE_PARTITION_PROJECTION_MARKER,
                    "commit_revision": commit_revision,
                },
            )
        )
        if end == len(offsets):
            break
        start += 160
        ordinal += 1
    return store, rows


class _NativeMemPalace:
    """The sole MemPalace vendor boundary used by the UM-MEM adapter."""

    def __init__(self, root: Path, workspace_id: str) -> None:
        self._commit_lock = _operation_lock_for(root / ".native-write")
        self._backend = ChromaBackend()
        self._palace = PalaceRef(
            id=workspace_id,
            local_path=str(root),
        )

    @contextmanager
    def _commit_gate(self) -> Iterator[None]:
        if not self._commit_lock.acquire(timeout=_operation_lock_timeout_seconds()):
            raise MemoryBusyError("native_commit_lock_timeout")
        try:
            yield
        finally:
            self._commit_lock.release()

    @staticmethod
    def embedding_diagnostics(content: str) -> dict[str, Any]:
        embedding = get_embedding_function()
        if not isinstance(embedding, ONNXMiniLM_L6_V2):
            return {"model": embedding.name(), "status": "not_minilm"}
        tokenizer = Tokenizer.from_str(embedding.tokenizer.to_str())
        tokenizer.no_truncation()
        tokenizer.no_padding()
        length = len(tokenizer.encode(content).ids)
        return {
            "model": "all-MiniLM-L6-v2",
            "token_count": length,
            "window_tokens": 256,
            "window_exceeded": length > 256,
            "recommended_tokens": [80, 200],
            "record_content_preserved": True,
        }

    def collection(
        self,
        store: str,
        *,
        partition: NativePartitionIdentity,
    ) -> ChromaCollection:
        if store not in _PALACE_STORES:
            raise ValueError("unknown MemPalace collection")
        if partition.backend_id != "mempalace" or partition.workspace_id != self._palace.id:
            raise ValueError("MemPalace collection partition does not match its workspace")
        with self._commit_gate():
            return self._backend.get_collection(
                palace=self._palace,
                collection_name=self._collection_name(store, partition=partition),
                create=True,
                options={"hnsw_space": "cosine"},
            )

    @staticmethod
    def _collection_name(
        store: str,
        *,
        partition: NativePartitionIdentity,
    ) -> str:
        return f"{partition.collection_prefix}_{store.replace('.', '_')}"

    def add_record(self, record: MemoryBlockRecord, *, commit_revision: int) -> None:
        store, rows = _native_projection_rows(record, commit_revision)
        partition = _record_partition_identity(record)
        present = self.collection(store, partition=partition).get(
            ids=[row[0] for row in rows], include=["documents", "metadatas"]
        )
        expected = {row[0]: (row[1], row[2]) for row in rows}
        for native_id, document, metadata in zip(
            present.ids, present.documents, present.metadatas, strict=True
        ):
            expected_document, expected_metadata = expected[native_id]
            if document != expected_document or (
                expected_metadata.get("derived_window") == 1
                and any(
                    (metadata or {}).get(key) != expected_metadata[key]
                    for key in (
                        "canonical_record_id",
                        "window_start_byte",
                        "window_end_byte",
                        "native_partition_id",
                        "commit_revision",
                    )
                )
            ):
                raise ValueError("mempalace native projection row conflicts with canonical source")
        present_ids = set(present.ids)
        missing = [row for row in rows if row[0] not in present_ids]
        if not missing:
            return
        # MemPalace's mine lock permits one native commit per palace, even within
        # one process. Compute embeddings concurrently before entering that gate.
        embeddings = get_embedding_function()([row[1] for row in missing])
        with self._commit_gate():
            self.collection(store, partition=partition).add(
                ids=[row[0] for row in missing],
                documents=[row[1] for row in missing],
                metadatas=[row[2] for row in missing],
                embeddings=embeddings,
            )

    def _reopen(self) -> None:
        self._backend.close_palace(self._palace)
        self._backend.close()
        self._backend = ChromaBackend()

    def _rebuild_collection_index(
        self,
        store: str,
        *,
        partition: NativePartitionIdentity,
        rebuild_records: tuple[tuple[MemoryBlockRecord, int], ...] = (),
    ) -> ChromaCollection:
        collection = self.collection(store, partition=partition)
        embeddings: list[Any] | None
        try:
            snapshot = collection.get(
                include=["documents", "metadatas", "embeddings"],
            )
        except ChromaInternalError as error:
            if _MISSING_HNSW_READER not in str(error) or not rebuild_records:
                raise
            projected = [
                row
                for record, commit_revision in rebuild_records
                if _record_partition_identity(record) == partition
                for selected_store, rows in (_native_projection_rows(record, commit_revision),)
                if selected_store == store
                for row in rows
            ]
            ids = [row[0] for row in projected]
            documents = [row[1] for row in projected]
            metadatas = [row[2] for row in projected]
            embeddings = None
        else:
            ids = list(snapshot.ids)
            documents = list(snapshot.documents)
            metadatas = list(snapshot.metadatas)
            if snapshot.embeddings is None:
                raise RuntimeError("native collection snapshot is missing embeddings")
            embeddings = list(snapshot.embeddings)
        if not ids:
            raise RuntimeError("cannot rebuild a populated MemPalace collection without rows")

        collection_name = self._collection_name(store, partition=partition)
        self._backend.delete_collection(str(self._palace.local_path), collection_name)
        rebuilt = self.collection(store, partition=partition)
        add_payload: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            add_payload["embeddings"] = embeddings
        rebuilt.add(
            **add_payload,
        )
        return rebuilt

    def _run_with_hnsw_repair(
        self,
        store: str,
        *,
        partition: NativePartitionIdentity,
        operation: Callable[[ChromaCollection], Any],
        rebuild_records: tuple[tuple[MemoryBlockRecord, int], ...] = (),
    ) -> tuple[ChromaCollection, Any]:
        collection = self.collection(store, partition=partition)
        try:
            return collection, operation(collection)
        except ChromaInternalError as error:
            if _MISSING_HNSW_READER not in str(error):
                raise
        _LOG.warning("MemPalace reopened its native client after a missing HNSW reader")
        self._reopen()
        collection = self.collection(store, partition=partition)
        try:
            return collection, operation(collection)
        except ChromaInternalError as error:
            if _MISSING_HNSW_READER not in str(error):
                raise
        _LOG.warning(
            "MemPalace rebuilt one native collection after a persistent missing HNSW reader"
        )
        collection = self._rebuild_collection_index(
            store,
            partition=partition,
            rebuild_records=rebuild_records,
        )
        return collection, operation(collection)

    def rows(
        self,
        *,
        query: str,
        limit: int,
        partition: NativePartitionIdentity,
        rebuild_records: tuple[tuple[MemoryBlockRecord, int], ...] = (),
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        partition_stores = {
            _store_for_block(record)
            for record, _commit_revision in rebuild_records
            if _record_partition_identity(record) == partition
        }
        stores = sorted(partition_stores or _PALACE_STORES)
        for store in stores:
            collection, collection_count = self._run_with_hnsw_repair(
                store,
                partition=partition,
                operation=lambda selected: selected.count(),
                rebuild_records=rebuild_records,
            )
            if not collection_count:
                continue
            # A semantic query must stay on the ranked native query path even
            # for a one-row collection. Falling back to ``get`` would silently
            # change the recall contract from semantic selection to an
            # unranked collection scan.
            if query:
                collection, result = self._run_with_hnsw_repair(
                    store,
                    partition=partition,
                    operation=lambda selected: selected.query(
                        query_texts=[query],
                        n_results=limit,
                        include=["documents", "metadatas", "distances"],
                    ),
                    rebuild_records=rebuild_records,
                )
                ids = result.ids[0]
                documents = result.documents[0]
                metadatas = result.metadatas[0]
                distances = result.distances[0]
            else:
                result = collection.get(
                    limit=min(limit, collection_count),
                    include=["documents", "metadatas"],
                )
                ids = result.ids
                documents = result.documents
                metadatas = result.metadatas
                distances = [0.0] * len(ids)
            for index, native_id in enumerate(ids):
                metadata = dict(metadatas[index] or {})
                rows.append(
                    {
                        **metadata,
                        "id": str(native_id),
                        "content": str(documents[index]),
                        "store": store,
                        "score": (
                            1.0 - float(distances[index]) if distances[index] is not None else None
                        ),
                    }
                )
        rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        return rows[:limit]

    def lexical_rows(
        self,
        *,
        query: str,
        limit: int,
        partition: NativePartitionIdentity,
        rebuild_records: tuple[tuple[MemoryBlockRecord, int], ...] = (),
    ) -> list[dict[str, Any]]:
        # Chroma's native trigram FTS index cannot match tokens shorter than three
        # characters. Quote each term so caller text cannot become FTS syntax.
        terms = tuple(dict.fromkeys(term for term in query.split() if len(term) >= 3))
        if not terms or limit <= 0:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        stores = sorted(
            {
                _store_for_block(record)
                for record, _ in rebuild_records
                if _record_partition_identity(record) == partition
            }
            or _PALACE_STORES
        )
        names = {self._collection_name(store, partition=partition): store for store in stores}
        if self._palace.local_path is None:
            raise RuntimeError("MemPalace local storage path is unavailable")
        database = Path(self._palace.local_path) / "chroma.sqlite3"
        try:
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
                selected = connection.execute(
                    "SELECT e.embedding_id, c.name, bm25(embedding_fulltext_search) "
                    "FROM embedding_fulltext_search "
                    "JOIN embeddings e ON e.id = embedding_fulltext_search.rowid "
                    "JOIN segments s ON e.segment_id = s.id "
                    "JOIN collections c ON s.collection = c.id "
                    "WHERE embedding_fulltext_search MATCH ? AND c.name IN ("
                    + ",".join("?" for _ in names)
                    + ") ORDER BY bm25(embedding_fulltext_search), c.name, e.embedding_id LIMIT ?",
                    (expression, *names, limit),
                ).fetchall()
        except sqlite3.Error as error:
            raise RuntimeError("mempalace_fulltext_index_failed") from error
        fetched: dict[tuple[str, str], dict[str, Any]] = {}
        for name, store in names.items():
            ids = [row[0] for row in selected if row[1] == name]
            if not ids:
                continue
            result = self.collection(store, partition=partition).get(
                ids=ids,
                include=["documents", "metadatas"],
            )
            for index, native_id in enumerate(result.ids):
                fetched[(name, native_id)] = {
                    **dict(result.metadatas[index] or {}),
                    "id": str(native_id),
                    "content": str(result.documents[index]),
                    "store": store,
                }
        if any((name, native_id) not in fetched for native_id, name, _ in selected):
            raise RuntimeError("mempalace_fulltext_projection_missing")
        return [fetched[(name, native_id)] for native_id, name, _ in selected]

    def health(
        self,
        partitions: tuple[NativePartitionIdentity, ...] = (),
    ) -> dict[str, Any]:
        status = self._backend.health(self._palace)
        stores_ok: list[str] = []
        stores_fail: list[str] = []
        if status.ok:
            for partition in partitions:
                for store in sorted(_PALACE_STORES):
                    identity = f"{partition.partition_id}:{store}"
                    try:
                        self.collection(store, partition=partition).count()
                        stores_ok.append(identity)
                    except (OSError, RuntimeError, ValueError) as exc:
                        stores_fail.append(f"{identity}: {exc}")
        return {
            "ok": status.ok and not stores_fail,
            "detail": status.detail,
            "stores_ok": stores_ok,
            "stores_fail": stores_fail,
        }

    def release(self) -> None:
        """Release the process-local client before another process owns the palace."""

        self._backend.close_palace(self._palace)

    def close(self) -> None:
        self.release()
        self._backend.close()


class MemPalaceBackend:
    backend_id = "mempalace"

    @staticmethod
    def native_partition_identity(
        *,
        workspace_id: str,
        primary_block_id: str,
        catalog_revision_id: str,
        backend_partition_id: str,
    ) -> NativePartitionIdentity:
        return _mempalace_partition_identity(
            workspace_id=workspace_id,
            primary_block_id=primary_block_id,
            catalog_revision_id=catalog_revision_id,
            backend_partition_id=backend_partition_id,
        )

    def __init__(
        self,
        repo_root: Path,
        workspace_id: str = "",
        *,
        reconcile_pending_writes: bool = True,
    ) -> None:
        self._operation_lock_timeout_seconds = _operation_lock_timeout_seconds()
        self._repo_root = Path(repo_root)
        self._workspace_id = workspace_id
        self._native = _NativeMemPalace(
            palace_path_for(self._repo_root, workspace_id),
            workspace_id,
        )
        self._index = BackendSQLiteIndex(
            _index_path(self._repo_root, workspace_id),
            backend_id="mempalace",
            workspace_id=workspace_id,
            repo_root=self._repo_root,
        )
        self._operation_state = threading.local()
        self._native_operation_gate = _native_gate_for(self._index.path)
        if reconcile_pending_writes:
            self._index.replay_pending_canonical_heads()
        self._index.configure_native_projection(
            self._project_durable_record,
            native_gate=self.parallel_native_gate,
            start=reconcile_pending_writes,
        )

    @contextmanager
    def operation_gate(self, *, exclusive: bool = True) -> Iterator[float]:
        depth = int(getattr(self._operation_state, "depth", 0))
        if depth:
            self._operation_state.depth = depth + 1
            try:
                yield 0.0
            finally:
                self._operation_state.depth = depth
            return

        with self._native_operation_gate.hold(
            exclusive=exclusive,
            timeout=self._operation_lock_timeout_seconds,
            release=self._native.release,
        ) as lock_wait_ms:
            self._operation_state.depth = 1
            try:
                yield lock_wait_ms
            finally:
                self._operation_state.depth = 0

    def parallel_native_gate(self) -> AbstractContextManager[float]:
        return self.operation_gate(exclusive=False)

    def capabilities(self) -> BackendCapabilities:
        health = self.health()
        native_ok = health.get("native_ok") is True and health.get("effective_mode") == "native"
        return BackendCapabilities.effective(
            "mempalace",
            native_ok=native_ok,
            implemented=frozenset(
                {
                    "persistent",
                    "semantic_recall",
                    "fts_recall",
                    "graph_edges",
                    "lifecycle_mutation",
                    "workspace_isolation",
                    "run_subtask_isolation",
                    "value_tracking",
                    "export_import",
                    "exact_record_read",
                    "scoped_search",
                    "revision_history",
                    "revision_compare_and_swap",
                    "recoverable_writes",
                    "staged_topology_evolution",
                }
            ),
            supports_embedded=True,
        )

    def ensure_banks(self, *, workspace_id: str = "") -> dict[str, Any]:
        return {
            "ok": True,
            "backend": "mempalace",
            "workspace_id": workspace_id or self._workspace_id,
        }

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        with self.operation_gate(exclusive=False), self._index.projection_gate(record):
            return self._retain_block(record)

    def prepare_canonical_write(self, record: MemoryBlockRecord) -> dict[str, Any]:
        ensure_record_is_storable(record)
        entry = self._index.open_write_outbox(record=record, store=_store_for_block(record))
        return {"ok": True, "commit_revision": entry.commit_revision}

    def enqueue_native_projection(self, record: MemoryBlockRecord) -> dict[str, Any]:
        return self._index.enqueue_native_projection(record)

    def canonical_write_gate(self) -> AbstractContextManager[None]:
        return self._index.canonical_write_gate()

    def atomic_canonical_write_gate(self) -> AbstractContextManager[None]:
        return self._index.atomic_canonical_write_gate()

    def wake_native_projection(self) -> None:
        self._index.wake_native_projection()

    def _project_durable_record(self, record: MemoryBlockRecord) -> None:
        entry = self._index.latest_write_outbox_entry(
            block_id=record.block_id, workspace_id=record.workspace_id
        )
        if entry is None:
            raise ValueError("native projection outbox is missing")
        self._project_write_outbox(entry, record)

    def stage_topology_projection(self, record: MemoryBlockRecord) -> None:
        """Copy an immutable source into an unpublished leaf partition."""

        entry = self._index.latest_write_outbox_entry(
            block_id=record.block_id, workspace_id=record.workspace_id
        )
        if entry is None or entry.status != "committed":
            raise NativeProjectionPending("native_projection_pending")
        with self.operation_gate(exclusive=False), self._index.projection_gate(record):
            self._native.add_record(record, commit_revision=entry.commit_revision)
            if self._topology_native_record(record) != record:
                raise RuntimeError("staged MemPalace topology projection is invalid")

    def _topology_native_record(self, record: MemoryBlockRecord) -> MemoryBlockRecord | None:
        partition = _record_partition_identity(record)
        store = _store_for_block(record)
        native = self._native.collection(store, partition=partition).get(ids=[record.block_id])
        if len(native.ids) != 1 or len(native.documents) != 1 or len(native.metadatas) != 1:
            return None
        metadata = native.metadatas[0]
        if not isinstance(metadata, dict):
            return None
        return self._native_node_to_block(
            {
                **metadata,
                "id": str(native.ids[0]),
                "content": str(native.documents[0]),
                "store": store,
            }
        )

    def _retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        started_at = time.perf_counter()
        ensure_record_is_storable(record)
        outbox = self._index.open_write_outbox(
            record=record,
            store=_store_for_block(record),
        )
        record = self._index.write_outbox_record(outbox)
        result = self._index.retain_block(record)
        native_id = self._project_write_outbox(outbox, record)
        embedding_diagnostics = _NativeMemPalace.embedding_diagnostics(record.content)
        if embedding_diagnostics.get("window_exceeded"):
            _LOG.warning(
                "MemPalace embedding window exceeded: record=%s tokens=%s window=256; "
                "source content is preserved; the author should split the record by meaning",
                record.block_id,
                embedding_diagnostics["token_count"],
            )
        partition = _record_partition_identity(record)
        self._index.record_partition_operation(
            partition,
            operation="write",
            latency_ms=(time.perf_counter() - started_at) * 1_000,
            record_bytes=len(
                json.dumps(
                    block_to_json(record),
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ),
            run_id=record.run_id,
            profile_id=record.profile_id,
        )
        return {
            **result,
            "store": _store_for_block(record),
            "native_id": native_id,
            "native_written": True,
            "commit_revision": outbox.commit_revision,
            "embedding_diagnostics": embedding_diagnostics,
        }

    def retain_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        with self.operation_gate():
            return self._retain_blocks(records)

    def _retain_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        if not records:
            return {"ok": True, "n_written": 0, "native_written": 0}

        for record in records:
            ensure_record_is_storable(record)

        written = 0
        first_result: dict[str, Any] = {"ok": True}
        for record in records:
            result = self._retain_block(record)
            first_result = result if written == 0 else first_result
            if result.get("ok"):
                written += 1
        return {**first_result, "n_written": written, "native_written": written}

    def get_block(
        self,
        block_id: str,
        *,
        workspace_id: str = "",
    ) -> MemoryBlockRecord | None:
        target_workspace = str(workspace_id or self._workspace_id)
        return self._index.get_block(block_id, workspace_id=target_workspace)

    def exact_reference_block_ids(self, reference: str, *, workspace_id: str) -> tuple[str, ...]:
        return self._index.exact_reference_block_ids(reference, workspace_id=workspace_id)

    def scoped_record_ids(
        self,
        *,
        workspace_id: str,
        profile_id: str,
        scope_key: str,
        logical_block_id: str = "",
    ) -> tuple[str, ...]:
        return self._index.scoped_record_ids(
            workspace_id=workspace_id,
            profile_id=profile_id,
            scope_key=scope_key,
            logical_block_id=logical_block_id,
        )

    def validate_dependencies(
        self,
        record_id: str,
        *,
        workspace_id: str = "",
        profile_id: str = "default",
        as_of: str = "",
        max_nodes: int = 256,
        max_edges: int = 1024,
    ) -> DependencyValidation:
        return self._index.validate_dependencies(
            record_id,
            workspace_id=workspace_id,
            profile_id=profile_id,
            as_of=as_of,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def record_ids_for_logical_blocks(
        self,
        logical_block_ids: tuple[str, ...] | list[str],
        *,
        workspace_id: str = "",
        profile_id: str = "",
        historical: bool = False,
    ) -> tuple[str, ...]:
        return self._index.record_ids_for_logical_blocks(
            logical_block_ids,
            workspace_id=str(workspace_id or self._workspace_id),
            profile_id=profile_id,
            historical=historical,
        )

    def logical_block_descriptor_snapshot(
        self, logical_block_id: str, *, workspace_id: str, profile_id: str
    ) -> dict[str, Any]:
        return self._index.logical_block_descriptor_snapshot(
            logical_block_id, workspace_id=workspace_id, profile_id=profile_id
        )

    def creation_operation_receipt(
        self, operation_id: str, *, workspace_id: str
    ) -> dict[str, Any] | None:
        return self._index.creation_operation_receipt(operation_id, workspace_id=workspace_id)

    def descriptor_sync_gate(self, logical_block_id: str) -> AbstractContextManager[None]:
        return self._index.descriptor_sync_gate(logical_block_id)

    def _existing_native_partitions(
        self,
        *,
        logical_block_ids: tuple[str, ...] = (),
        profile_id: str = "",
        current_query: dict[str, Any] | None = None,
    ) -> tuple[NativePartitionIdentity, ...]:
        selected_order = {block_id: index for index, block_id in enumerate(logical_block_ids)}
        partitions = (
            self.native_partition_identity(
                workspace_id=self._workspace_id,
                primary_block_id=block_id,
                catalog_revision_id=catalog_revision_id,
                backend_partition_id=backend_partition_id,
            )
            for block_id, catalog_revision_id, backend_partition_id in self._index.native_partition_memberships(
                workspace_id=self._workspace_id,
                profile_id=profile_id,
                logical_block_ids=logical_block_ids,
            )
        )
        if current_query is not None:
            # Archived revisions retain their native partitions for exact history.
            # Ordinary recall must not open or repair partitions that cannot supply
            # a current, authorized result. Native ranking still selects relevance.
            identities = {
                partition
                for partition in partitions
                if any(
                    (record := self._index.get_block(record_id, workspace_id=self._workspace_id))
                    is not None
                    and _record_partition_identity(record) == partition
                    and self._record_allowed_for_query(record, current_query)
                    for record_id in self._index.record_ids_for_logical_blocks(
                        (partition.primary_block_id,),
                        workspace_id=self._workspace_id,
                        profile_id=profile_id,
                    )
                )
            }
        else:
            identities = set(partitions)
        return tuple(
            sorted(
                identities,
                key=lambda item: (
                    selected_order.get(item.primary_block_id, len(selected_order)),
                    item.primary_block_id,
                    item.catalog_revision_id,
                    item.partition_id,
                ),
            )
        )

    def _partition_rebuild_records(
        self,
        partition: NativePartitionIdentity,
    ) -> tuple[tuple[MemoryBlockRecord, int], ...]:
        records: list[tuple[MemoryBlockRecord, int]] = []
        for record_id in self._index.record_ids_for_logical_blocks(
            (partition.primary_block_id,), workspace_id=self._workspace_id
        ):
            record = self._index.get_block(record_id, workspace_id=self._workspace_id)
            if record is None or _record_partition_identity(record) != partition:
                continue
            entry = self._index.latest_write_outbox_entry(
                block_id=record.block_id,
                workspace_id=self._workspace_id,
            )
            if entry is None or entry.status != "committed":
                raise RuntimeError(NATIVE_PROJECTION_REBUILD_REQUIRED)
            records.append((record, entry.commit_revision))
        return tuple(records)

    def write_outbox_status(self, block_id: str) -> str | None:
        return self._index.write_outbox_status(block_id)

    def latest_write_outbox_entry(
        self, *, block_id: str, workspace_id: str = ""
    ) -> WriteOutboxEntry | None:
        return self._index.latest_write_outbox_entry(
            block_id=block_id, workspace_id=workspace_id or self._workspace_id
        )

    def _reconcile_pending_outboxes(self) -> None:
        if (
            not self._index.pending_canonical_head_reservations()
            and not self._index.pending_write_outbox()
        ):
            return
        with self.operation_gate():
            if self._index.pending_canonical_head_reservations():
                self._index.replay_pending_canonical_heads()
            if self._index.pending_write_outbox():
                self._reconcile_pending_write_outbox_locked()

    def reconcile(self, *, dry_run: bool = True) -> ReconcileReport:
        with self.operation_gate():
            initial_head_entries = self._index.reconcile_canonical_head_reservations()
            initial_write_entries = self._index.reconcile_write_outbox()
            initial_head_ids = {entry.commit_revision for entry in initial_head_entries}
            initial_write_ids = {entry.commit_revision for entry in initial_write_entries}
            initial_heads, initial_writes = self._index._reconcile_entry_statuses(
                canonical_head_revisions=initial_head_ids,
                write_revisions=initial_write_ids,
            )
            pending_heads = sum(initial_heads.get(key) == "pending" for key in initial_head_ids)
            pending_writes = sum(initial_writes.get(key) == "pending" for key in initial_write_ids)
            inventory_count = len(initial_head_ids) + len(initial_write_ids)
            initial_failed = sum(
                initial_heads.get(key) == "failed" for key in initial_head_ids
            ) + sum(initial_writes.get(key) == "failed" for key in initial_write_ids)
            initial_unresolved = sum(
                initial_heads.get(key) not in {"committed", "failed"} for key in initial_head_ids
            ) + sum(
                initial_writes.get(key) not in {"committed", "failed"} for key in initial_write_ids
            )
            initial_missing = sum(key not in initial_heads for key in initial_head_ids) + sum(
                key not in initial_writes for key in initial_write_ids
            )
            if dry_run:
                return ReconcileReport(
                    backend=self.backend_id,
                    dry_run=True,
                    pending_canonical_heads=pending_heads,
                    pending_write_outbox=pending_writes,
                    repaired=0,
                    remaining=initial_failed + initial_unresolved,
                    failures=initial_failed + initial_missing,
                    result_code=(
                        "failed"
                        if initial_failed + initial_missing
                        else ("inventory" if inventory_count else "clean")
                    ),
                    reason_code=(
                        "reconcile_failures"
                        if initial_failed + initial_missing
                        else ("pending_work" if inventory_count else "no_pending_work")
                    ),
                )
            stage_failures = 0
            try:
                self._index.replay_pending_canonical_heads()
            except (OSError, RuntimeError, ValueError, sqlite3.Error):
                stage_failures += 1
            try:
                self._reconcile_pending_write_outbox_locked()
            except (OSError, RuntimeError, ValueError, sqlite3.Error):
                stage_failures += 1
            final_heads, final_writes = self._index._reconcile_entry_statuses(
                canonical_head_revisions=initial_head_ids,
                write_revisions=initial_write_ids,
            )
            repaired = sum(final_heads.get(key) == "committed" for key in initial_head_ids) + sum(
                final_writes.get(key) == "committed" for key in initial_write_ids
            )
            failed = sum(final_heads.get(key) == "failed" for key in initial_head_ids) + sum(
                final_writes.get(key) == "failed" for key in initial_write_ids
            )
            unresolved = sum(
                final_heads.get(key) not in {"committed", "failed"} for key in initial_head_ids
            ) + sum(
                final_writes.get(key) not in {"committed", "failed"} for key in initial_write_ids
            )
            failures = (
                failed + unresolved + (1 if stage_failures and not (failed + unresolved) else 0)
            )
            remaining = failed + unresolved
            return ReconcileReport(
                backend=self.backend_id,
                dry_run=False,
                pending_canonical_heads=pending_heads,
                pending_write_outbox=pending_writes,
                repaired=repaired,
                remaining=remaining,
                failures=failures,
                result_code=("failed" if failures else ("repaired" if repaired else "clean")),
                reason_code=(
                    "reconcile_failures"
                    if failures
                    else ("reconcile_complete" if repaired else "no_pending_work")
                ),
            )

    def _reconcile_pending_write_outbox(self) -> None:
        with self.operation_gate():
            self._reconcile_pending_write_outbox_locked()

    def _reconcile_pending_write_outbox_locked(self) -> None:
        for entry in self._index.pending_write_outbox():
            try:
                record = self._index.write_outbox_record(entry)
                if self._index.canonical_head_status(record.block_id) not in {None, "committed"}:
                    continue
                if record.workspace_id != entry.workspace_id:
                    raise ValueError("selected backend pending write workspace mismatch")
                self._index.commit_write_outbox_record(entry)
                self._project_write_outbox(entry, record)
            except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
                raise RuntimeError("selected backend pending write reconcile failed") from exc

    def _project_write_outbox(
        self,
        entry: WriteOutboxEntry,
        record: MemoryBlockRecord,
    ) -> str:
        with self._index.projection_gate(record):
            return self._project_write_outbox_locked(entry, record)

    def _project_write_outbox_locked(
        self, entry: WriteOutboxEntry, record: MemoryBlockRecord
    ) -> str:
        self._native.add_record(record, commit_revision=entry.commit_revision)
        projected = self._native_record_for_write_outbox(entry)
        for delay_seconds in _NATIVE_PROJECTION_VISIBILITY_DELAYS_SECONDS:
            if projected is not None:
                break
            time.sleep(delay_seconds)
            projected = self._native_record_for_write_outbox(entry)
        if projected is None or projected != record:
            raise RuntimeError("selected backend native projection mismatch")
        actual = self._index.write_outbox_entry(
            block_id=entry.block_id,
            commit_revision=entry.commit_revision,
        )
        if actual is None:
            raise RuntimeError("selected backend write journal is unavailable")
        if actual.status == "pending":
            self._index.set_write_outbox_native_ref(
                entry.commit_revision,
                record.block_id,
            )
            self._index.complete_write_outbox(entry.commit_revision)
        elif actual.status != "committed":
            raise RuntimeError("selected backend write journal is not replayable")
        return record.block_id

    def _native_record_for_write_outbox(
        self,
        entry: WriteOutboxEntry,
    ) -> MemoryBlockRecord | None:
        if entry.store not in _PALACE_STORES:
            return None
        try:
            expected = self._index.write_outbox_record(entry)
        except ValueError:
            return None
        partition = _record_partition_identity(expected)
        native = self._native.collection(entry.store, partition=partition).get(ids=[entry.block_id])
        ids = list(native.ids)
        documents = list(native.documents)
        metadatas = list(native.metadatas)
        if len(ids) != 1 or len(documents) != 1 or len(metadatas) != 1:
            return None
        metadata = metadatas[0]
        if not isinstance(metadata, dict) or str(metadata.get("commit_revision") or "") != str(
            entry.commit_revision
        ):
            return None
        try:
            record = self._native_node_to_block(
                {
                    **metadata,
                    "id": str(ids[0]),
                    "content": str(documents[0]),
                    "store": entry.store,
                }
            )
        except ValueError:
            return None
        return record if record.block_id == entry.block_id else None

    def _require_current_block_projection(
        self,
        logical_block_id: str,
        *,
        query: dict[str, Any] | None = None,
    ) -> None:
        for record_id in self._index.record_ids_for_logical_blocks(
            (logical_block_id,),
            workspace_id=self._workspace_id,
            profile_id=str((query or {}).get("profile_id") or ""),
        ):
            if query is not None:
                record = self._index.get_block(record_id, workspace_id=self._workspace_id)
                if record is None or not self._record_allowed_for_query(record, query):
                    continue
            entry = self._index.latest_write_outbox_entry(
                block_id=record_id,
                workspace_id=self._workspace_id,
            )
            if entry is not None and entry.status == "pending":
                raise NativeProjectionPending("native_projection_pending")
            if (
                entry is None
                or entry.status != "committed"
                or self._index.write_outbox_native_id(entry.commit_revision) is None
            ):
                raise RuntimeError(NATIVE_PROJECTION_REBUILD_REQUIRED)
            expected = self._index.write_outbox_record(entry)
            current = self._index.get_block(record_id, workspace_id=self._workspace_id)
            if current is None:
                raise RuntimeError(NATIVE_PROJECTION_REBUILD_REQUIRED)
            if current.block_membership != expected.block_membership:
                projected = self._topology_native_record(current)
            else:
                projected = self._native_record_for_write_outbox(entry)
            if projected != current:
                raise RuntimeError(NATIVE_PROJECTION_REBUILD_REQUIRED)

    def purge_exact_native_artifacts(
        self,
        record: MemoryBlockRecord,
        *,
        allow_native_absent: bool = False,
    ) -> dict[str, Any]:
        with self.operation_gate():
            return self._purge_exact_native_artifacts(
                record,
                allow_native_absent=allow_native_absent,
            )

    def _purge_exact_native_artifacts(
        self,
        record: MemoryBlockRecord,
        *,
        allow_native_absent: bool = False,
    ) -> dict[str, Any]:
        """Remove only the exact selected-backend vector row."""
        workspace_id = self._workspace_id or record.workspace_id
        block_id = str(record.block_id or "").strip()
        if not block_id or record.workspace_id != workspace_id:
            return {"ok": False, "error": "native_record_workspace_mismatch"}

        partition = _record_partition_identity(record)
        collection = self._native.collection(
            _store_for_block(record),
            partition=partition,
        )
        before = collection.get(ids=[block_id])
        native_ids = [str(value) for value in before.ids]
        native_present = bool(native_ids)
        if native_present:
            if native_ids != [block_id]:
                return {"ok": False, "error": "native_exact_id_not_found"}
            metadata_rows = before.metadatas
            metadata = metadata_rows[0] if metadata_rows else {}
            if str(metadata.get("workspace_id") or "") != workspace_id:
                return {"ok": False, "error": "native_record_workspace_mismatch"}
        elif not allow_native_absent:
            return {"ok": False, "error": "native_exact_id_not_found"}

        if native_present:
            collection.delete(ids=[block_id])
        after = collection.get(ids=[block_id])
        if block_id in {str(value) for value in after.ids}:
            return {"ok": False, "error": "native_exact_id_not_deleted"}
        return {
            "ok": True,
            "native_exact_id_absent": True,
            "native_edges_absent": True,
        }

    def retain_index_node(self, node: MemoryIndexNode) -> dict[str, Any]:
        return self._index.retain_index_node(node)

    def get_index_node(self, node_id: str, *, workspace_id: str = "") -> MemoryIndexNode | None:
        return self._index.get_index_node(node_id, workspace_id=workspace_id or self._workspace_id)

    def select_index_nodes(
        self,
        *,
        workspace_id: str = "",
        node_type: str = "",
        tags: list[str] | tuple[str, ...] | set[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryIndexNode]:
        return self._index.select_index_nodes(
            workspace_id=workspace_id or self._workspace_id,
            node_type=node_type,
            tags=tags,
            limit=limit,
        )

    def search_index_nodes(
        self,
        *,
        workspace_id: str = "",
        terms: list[str] | tuple[str, ...] = (),
        node_types: list[str] | tuple[str, ...] = (),
        limit: int = 128,
    ) -> list[MemoryIndexNode]:
        return self._index.search_index_nodes(
            workspace_id=workspace_id or self._workspace_id,
            terms=terms,
            node_types=node_types,
            limit=limit,
        )

    def export_index_nodes(self) -> list[MemoryIndexNode]:
        return self._index.export_index_nodes()

    def create_prospective_obligation(
        self,
        obligation: ProspectiveObligation,
        *,
        event_id: str,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        with self.operation_gate():
            return self._index.create_prospective_obligation(
                obligation,
                event_id=event_id,
                idempotency_key=idempotency_key,
            )

    def get_prospective_obligation(self, obligation_id: str) -> ProspectiveObligation | None:
        with self.operation_gate():
            return self._index.get_prospective_obligation(obligation_id)

    def transition_prospective_obligation(
        self,
        obligation_id: str,
        *,
        expected_version: int,
        expected_status: str,
        status: str,
        evidence_refs: tuple[EvidenceRef, ...] | list[EvidenceRef],
        satisfaction_evidence_refs: tuple[EvidenceRef, ...] | list[EvidenceRef] = (),
        satisfaction_verifier: str = "",
        satisfaction_verifier_result: str = "",
        event_id: str,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        with self.operation_gate():
            return self._index.transition_prospective_obligation(
                obligation_id,
                expected_version=expected_version,
                expected_status=expected_status,
                status=status,
                evidence_refs=evidence_refs,
                satisfaction_evidence_refs=satisfaction_evidence_refs,
                satisfaction_verifier=satisfaction_verifier,
                satisfaction_verifier_result=satisfaction_verifier_result,
                event_id=event_id,
                idempotency_key=idempotency_key,
            )

    def prospective_trigger_candidates(
        self,
        context: ProspectiveTriggerContext,
        *,
        workspace_id: str = "",
        limit: int = 50,
    ) -> list[ProspectiveTriggerDecision]:
        with self.operation_gate():
            return self._index.prospective_trigger_candidates(
                context,
                workspace_id=workspace_id or self._workspace_id,
                limit=limit,
            )

    @staticmethod
    def _tags_from_any(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return tuple(tag.strip() for tag in value.split(",") if tag.strip())
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(str(tag).strip() for tag in value if str(tag).strip())
        return ()

    def _native_node_to_block(self, node: dict[str, Any]) -> MemoryBlockRecord:
        if node.get("derived_window") == 1:
            canonical_id = str(node.get("canonical_record_id") or "")
            record = self._index.get_block(canonical_id, workspace_id=self._workspace_id)
            if record is None:
                raise ValueError("mempalace window canonical record is unavailable")
            partition = _record_partition_identity(record)
            start = node.get("window_start_byte")
            end = node.get("window_end_byte")
            encoded = record.content.encode("utf-8")
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(encoded)
                or str(node.get("id") or "").split(":window:", 1)[0] != record.block_id
                or str(node.get("workspace_id") or "") != record.workspace_id
                or str(node.get("logical_block_id") or "") != partition.primary_block_id
                or str(node.get("native_partition_id") or "") != partition.partition_id
                or str(node.get("catalog_revision_id") or "") != partition.catalog_revision_id
                or str(node.get("projection_marker") or "")
                != _MEMPALACE_PARTITION_PROJECTION_MARKER
            ):
                raise ValueError("mempalace window identity is invalid")
            try:
                expected = encoded[start:end].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("mempalace window offsets are invalid") from exc
            if node.get("content") != expected:
                raise ValueError("mempalace window content does not match canonical source")
            return record
        raw = node.get("canonical_record_json")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("mempalace native canonical record is missing")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("mempalace native canonical record is invalid") from exc
        try:
            record = block_from_json(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("mempalace native canonical record is invalid") from exc
        native_id = str(node.get("id") or "")
        if native_id and native_id != record.block_id:
            raise ValueError("mempalace native canonical record id mismatch")
        content = node.get("content")
        if not isinstance(content, str) or content != record.content:
            raise ValueError("mempalace native canonical record content mismatch")
        if str(node.get("projection_marker") or "") != _MEMPALACE_PARTITION_PROJECTION_MARKER:
            raise ValueError(NATIVE_PROJECTION_REBUILD_REQUIRED)
        logical_block_id = str(node.get("logical_block_id") or "")
        if logical_block_id != native_primary_block_id(record):
            raise ValueError("mempalace native primary block mismatch")
        partition = _record_partition_identity(record)
        if str(node.get("native_partition_id") or "") != partition.partition_id:
            raise ValueError("mempalace native partition identity mismatch")
        if str(node.get("catalog_revision_id") or "") != partition.catalog_revision_id:
            raise ValueError("mempalace native catalog revision mismatch")
        return record

    @staticmethod
    def _record_allowed_for_query(record: MemoryBlockRecord, query: dict[str, Any]) -> bool:
        return record_allowed_for_query(record, query)

    @staticmethod
    def _query_payload(
        query: MemoryQuery | dict[str, Any], *, default_workspace_id: str = ""
    ) -> dict[str, Any]:
        return normalize_query_payload(query, default_workspace_id=default_workspace_id)

    def recall(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryBlockRecord]:
        profile_id = str(
            self._query_payload(query, default_workspace_id=self._workspace_id)["profile_id"]
        )
        with self.operation_gate(), self._index.topology_snapshot(profile_id):
            self.last_recall_trace: dict[str, Any] = {}
            started = time.perf_counter()
            succeeded = False
            try:
                result = self._recall(query)
                succeeded = True
                return result
            finally:
                trace = self.last_recall_trace
                trace["total_ms"] = (time.perf_counter() - started) * 1000
                record_memory_event(
                    self._repo_root,
                    event_type="native_retrieval_trace",
                    backend="mempalace",
                    workspace_id=self._workspace_id,
                    profile_id=str(trace.get("profile_id") or ""),
                    run_id=str(trace.get("run_id") or ""),
                    status="ok" if succeeded else "failed",
                    operation="recall",
                    duration_ms=trace["total_ms"],
                    data=trace,
                )

    def _recall(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryBlockRecord]:
        payload = self._query_payload(query, default_workspace_id=self._workspace_id)
        payload = self._index.apply_exact_query(payload)
        self.last_recall_trace.update(
            retrieval_mode=payload["retrieval_mode"],
            profile_id=payload["profile_id"],
            run_id=payload.get("run_id", ""),
            temporal_mode=payload["temporal_mode"],
            requested_block_ids=list(payload.get("logical_block_ids") or ()),
        )
        if payload["retrieval_mode"] == "hybrid" and record_id_sequence(payload):
            raise ValueError("hybrid retrieval does not apply to exact reads")
        if (
            str(payload.get("temporal_mode") or "current") == "historical"
            or payload.get("as_of")
            or payload.get("known_as_of")
        ):
            return self._index.recall(payload)
        text = str(payload.get("query") or "").strip()
        try:
            limit = max(1, int(payload.get("limit") or 10))
        except (TypeError, ValueError):
            limit = 10
        # Explicit record IDs are an exact-fetch contract. They must keep
        # candidate/passive selections available and preserve caller order.
        if record_id_sequence(payload):
            selected_rows: list[MemoryBlockRecord] = []
            for row in self._index.recall(payload):
                canonical_record = self.get_block(row.block_id)
                if canonical_record is not None:
                    selected_rows.append(canonical_record)
            return selected_rows[:limit]
        selected_blocks = tuple(str(value) for value in payload.get("logical_block_ids") or ())
        if not selected_blocks and payload.get("search_all") is not True:
            raise ValueError(
                "MemPalace native recall requires logical_block_ids or search_all=True"
            )
        partitions = self._existing_native_partitions(
            logical_block_ids=selected_blocks,
            profile_id=str(payload.get("profile_id") or "default"),
            current_query=payload,
        )
        if selected_blocks:
            for logical_block_id in selected_blocks:
                self._require_current_block_projection(logical_block_id, query=payload)
            # Exact projection checks use Chroma's metadata path before semantic
            # recall opens HNSW readers. End that client generation while the
            # palace-wide operation gate is still held so the vector query starts
            # from one coherent on-disk generation after any admitted projection.
            self._native.release()
        pools: list[tuple[str, list[tuple[MemoryBlockRecord, float | None]]]] = []
        lexical_pools: list[tuple[str, list[tuple[MemoryBlockRecord, float | None]]]] = []
        hybrid = payload["retrieval_mode"] == "hybrid"
        trace: dict[str, Any] = {
            **self.last_recall_trace,
            "retrieval_mode": payload["retrieval_mode"],
            "fusion": "rrf" if hybrid else "cosine_similarity" if text else "partition_rank",
            "selected_block_ids": [partition.primary_block_id for partition in partitions],
            "channels": [],
            "selected_records": [],
        }
        self.last_recall_trace = trace
        for partition in partitions:
            logical_block_id = partition.primary_block_id
            native_limit = 64 if hybrid else min(64, max(limit + 1, limit * 4))
            rebuild_records = self._partition_rebuild_records(partition)
            for channel in ("semantic", "lexical") if hybrid else ("semantic",):
                started_at = time.perf_counter()
                rows: list[dict[str, Any]] = []
                native_success = False
                channel_trace: dict[str, Any] = {
                    "block_id": logical_block_id,
                    "channel": channel,
                    "candidates": [],
                }
                trace["channels"].append(channel_trace)
                try:
                    search = (
                        self._native.lexical_rows if channel == "lexical" else self._native.rows
                    )
                    rows = search(
                        query=text,
                        limit=native_limit,
                        partition=partition,
                        rebuild_records=rebuild_records,
                    )
                    native_success = True
                finally:
                    channel_trace["elapsed_ms"] = (time.perf_counter() - started_at) * 1_000
                    channel_trace["success"] = native_success
                    self._index.record_partition_operation(
                        partition,
                        operation="recall",
                        latency_ms=channel_trace["elapsed_ms"],
                        hit_count=len(rows),
                        success=native_success,
                        search_mode="scoped" if selected_blocks else "search_all",
                        run_id=str(payload.get("run_id") or ""),
                        profile_id=str(payload.get("profile_id") or ""),
                    )
                pool: list[tuple[MemoryBlockRecord, float | None]] = []
                seen_record_ids: set[str] = set()
                for native_rank, native_payload in enumerate(rows, start=1):
                    native_row = self._native_node_to_block(dict(native_payload))
                    if _record_partition_identity(native_row) != partition:
                        raise ValueError("mempalace native row escaped its selected partition")
                    if native_primary_block_id(native_row) != logical_block_id:
                        raise ValueError("mempalace native scoped row escaped its primary block")
                    candidate_trace = {
                        "record_id": native_row.block_id,
                        "revision_id": native_row.revision_id,
                        "native_rank": native_rank,
                        "omission": "",
                    }
                    channel_trace["candidates"].append(candidate_trace)
                    if not self._record_allowed_for_query(native_row, payload):
                        candidate_trace["omission"] = "query_policy"
                        continue
                    canonical_record = self._index.overlay_native_hit(
                        native_row,
                        workspace_id=self._workspace_id,
                        query=payload,
                    )
                    if canonical_record is None:
                        candidate_trace["omission"] = "canonical_revision_or_dependencies"
                        continue
                    if canonical_record.block_id in seen_record_ids:
                        candidate_trace["omission"] = "duplicate_window"
                        continue
                    seen_record_ids.add(canonical_record.block_id)
                    score = native_payload.get("score") if channel == "semantic" else None
                    try:
                        native_score = float(score) if score is not None else None
                    except (TypeError, ValueError) as exc:
                        raise ValueError("mempalace native score is invalid") from exc
                    pool.append((canonical_record, native_score))
                    if not hybrid and len(pool) >= limit:
                        break
                (lexical_pools if channel == "lexical" else pools).append((logical_block_id, pool))
        if hybrid:
            selected = merge_hybrid_block_results(pools, lexical_pools, limit=limit)
        else:
            selected = merge_native_block_results(
                pools,
                limit=limit,
                score_mode="cosine_similarity" if text else "partition_rank",
            )
        trace["selected_records"] = [
            {
                "record_id": row.block_id,
                "revision_id": row.revision_id,
                "channel_ranks": row.metadata.get("channel_ranks", {}),
            }
            for row in selected
        ]
        return selected

    def expire_scope(self, scope: str, scope_id: str = "", *, run_id: str = "") -> int:
        with self.operation_gate():
            return self._index.expire_scope(scope, scope_id, run_id=run_id)

    def add_edge(self, edge: MemoryEdgeRecord) -> dict[str, Any]:
        return self._index.add_edge(edge)

    def reserve_canonical_head(
        self,
        *,
        workspace_id: str,
        canonical_key: str,
        predecessor_id: str,
        successor_id: str,
        successor_status: str,
        successor_authority: str,
        mutation_action: str = "insert",
        rollback_target_id: str = "",
        mutation: MutationDecision,
    ) -> CanonicalHeadReservation:
        return self._index.reserve_canonical_head(
            workspace_id=workspace_id,
            canonical_key=canonical_key,
            predecessor_id=predecessor_id,
            successor_id=successor_id,
            successor_status=successor_status,
            successor_authority=successor_authority,
            mutation_action=mutation_action,
            rollback_target_id=rollback_target_id,
            mutation=mutation,
        )

    def complete_canonical_head(
        self,
        reservation: CanonicalHeadReservation,
        mutation: MutationDecision,
    ) -> dict[str, Any]:
        entry = self._index.latest_write_outbox_entry(
            block_id=mutation.record.block_id,
            workspace_id=reservation.workspace_id,
        )
        if (
            entry is None
            or entry.status != "pending"
            or entry.store != _store_for_block(mutation.record)
        ):
            raise CanonicalHeadConflict("canonical_head_write_journal_missing")
        return self._index.complete_canonical_head(reservation, mutation)

    def fail_canonical_head(
        self,
        reservation: CanonicalHeadReservation,
        *,
        error: str,
    ) -> None:
        self._index.fail_canonical_head(reservation, error=error)

    def canonical_history(
        self,
        *,
        canonical_key: str,
        workspace_id: str = "",
    ) -> list[MemoryBlockRecord]:
        return self._index.canonical_history(
            canonical_key=canonical_key,
            workspace_id=workspace_id or self._workspace_id,
        )

    def publication_receipt(self, receipt_id: str) -> PublicationCommitReceipt | None:
        return self._index.publication_receipt(receipt_id)

    def amendment_receipt(self, receipt_id: str) -> AmendmentCommitReceipt | None:
        return self._index.amendment_receipt(receipt_id)

    def record_publication_receipt(self, receipt: PublicationCommitReceipt) -> None:
        self._index.record_publication_receipt(receipt)

    def canonical_mutation_identity(
        self,
        *,
        successor_id: str,
        workspace_id: str = "",
    ) -> CanonicalMutationIdentity | None:
        return self._index.canonical_mutation_identity(
            successor_id=successor_id,
            workspace_id=workspace_id or self._workspace_id,
        )

    def edges_for(self, block_id: str) -> list[MemoryEdgeRecord]:
        return self._index.edges_for(block_id)

    def export_edges(self) -> list[MemoryEdgeRecord]:
        return self._index.export_edges()

    def reverse_provenance_sources(self, derived_id: str) -> tuple[str, ...]:
        return self._index.reverse_provenance_sources(derived_id)

    def rebuild_memory_indexes(self, *, catalog: Any) -> dict[str, Any]:
        with self.operation_gate():
            return self._index.rebuild_memory_indexes(catalog=catalog)

    def refresh_memory_indexes(self, block_id: str) -> dict[str, Any]:
        with self.operation_gate():
            return self._index.refresh_memory_indexes(block_id)

    def record_usage(self, usage: MemoryUsageRecord) -> dict[str, Any]:
        with self.operation_gate():
            return self._index.record_usage(usage)

    def record_memory_decision(self, event: MemoryDecisionEvent) -> dict[str, Any]:
        with self.operation_gate():
            return self._index.record_memory_decision(event)

    def get_memory_decision(self, decision_id: str) -> MemoryDecisionEvent | None:
        with self.operation_gate():
            return self._index.get_memory_decision(decision_id)

    def record_memory_usage_event(self, event: MemoryUsageEvent) -> dict[str, Any]:
        with self.operation_gate():
            return self._index.record_memory_usage_event(event)

    def memory_decision_trace(self, decision_id: str) -> MemoryDecisionTrace:
        with self.operation_gate():
            return self._index.memory_decision_trace(decision_id)

    def record_outcome_credit(self, event: OutcomeCreditEvent) -> dict[str, Any]:
        with self.operation_gate():
            return self._index.record_outcome_credit(event)

    def outcome_credits(self, decision_id: str) -> tuple[OutcomeCreditEvent, ...]:
        with self.operation_gate():
            return self._index.outcome_credits(decision_id)

    def outcome_credits_for_memory(self, memory_id: str) -> tuple[OutcomeCreditEvent, ...]:
        with self.operation_gate():
            return self._index.outcome_credits_for_memory(memory_id)

    def contextual_utility_summary(self, **kwargs: Any) -> ContextualUtilitySummary:
        with self.operation_gate():
            return self._index.contextual_utility_summary(**kwargs)

    def value_stats(self, block_id: str) -> dict[str, Any]:
        return self._index.value_stats(block_id)

    def export_blocks(self) -> list[MemoryBlockRecord]:
        return self._index.export_blocks()

    def import_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        with self.operation_gate():
            return self._index.import_blocks(records)

    def compile_active_overlay(self, *, workspace_id: str = "", run_id: str = "") -> dict[str, Any]:
        return self._index.compile_active_overlay(
            workspace_id=workspace_id or self._workspace_id, run_id=run_id
        )

    def render_prompt_items(
        self, *, workspace_id: str = "", run_id: str = ""
    ) -> list[dict[str, Any]]:
        return self._index.render_prompt_items(
            workspace_id=workspace_id or self._workspace_id, run_id=run_id
        )

    def recall_evidence(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryHit]:
        return [block_record_to_memory_hit(row) for row in self.recall(query)]

    def record_rendered_tokens(
        self,
        rendered_tokens_by_record_id: dict[str, int],
    ) -> None:
        for record_id, rendered_tokens in rendered_tokens_by_record_id.items():
            record = self._index.get_block(
                record_id,
                workspace_id=self._workspace_id,
            )
            if record is None or record.block_membership is None:
                continue
            self._index.record_partition_operation(
                _record_partition_identity(record),
                operation="render",
                rendered_tokens=rendered_tokens,
            )

    def _health(self, *, verify_native_partitions: bool) -> dict[str, Any]:
        health_started = time.perf_counter()
        revisions = self._index.operational_revisions()
        partitions = self._existing_native_partitions()
        try:
            native = self._native.health(partitions if verify_native_partitions else ())
            if verify_native_partitions:
                for partition in partitions:
                    self._require_current_block_projection(partition.primary_block_id)
        except (OSError, RuntimeError, ValueError) as exc:
            native = {"ok": False, "error": str(exc)}
        native_ok = bool(native.get("ok", False)) and not bool(native.get("volatile_stub"))
        reason = (
            ""
            if native_ok
            else str(native.get("error") or native.get("stores_fail") or "native_unavailable")
        )
        metric_health = "ready" if native_ok else f"unavailable:{reason}"
        result = {
            **revisions,
            "ok": native_ok,
            "backend": "mempalace",
            "implementation": "mempalace_chroma",
            "selected": True,
            "native_ok": native_ok,
            "strict_ok": native_ok,
            "native_block_scoping_supported": native_ok,
            "native_block_scoping_reason": ("native_collection_family" if native_ok else reason),
            "native_partition_isolation_class": "physical",
            "native_partition_mode": "native_collection_family",
            "native_partition_count": len(partitions),
            "native_partitions": [partition.to_mapping() for partition in partitions],
            "block_metrics": [
                self._index.partition_metrics(
                    partition,
                    health=metric_health,
                ).to_mapping()
                for partition in partitions
            ],
            "effective_mode": "native" if native_ok else "unavailable",
            "reason": reason,
            "native": native,
        }
        result["preflight_timing"] = {
            "health_body_ms": round((time.perf_counter() - health_started) * 1000, 3),
            "partition_count": len(partitions),
            "strict_partition_validation": verify_native_partitions,
        }
        return result

    def health(self) -> dict[str, Any]:
        """Return a lightweight runtime readiness view without scanning every collection."""

        result = self._health(verify_native_partitions=False)
        result["projection"] = self._index.native_projection_health()
        timing = dict(result["preflight_timing"])
        timing["lock_wait_ms"] = 0.0
        result["preflight_timing"] = timing
        return result

    def preflight_health(self) -> dict[str, Any]:
        """Verify every materialized partition only at the explicit Preflight boundary."""

        with self.operation_gate() as lock_wait_ms:
            result = self._health(verify_native_partitions=True)
            timing = dict(result["preflight_timing"])
            timing["lock_wait_ms"] = lock_wait_ms
            result["preflight_timing"] = timing
            return result

    def recovery_preflight(self) -> dict[str, Any]:
        partitions = self._existing_native_partitions()
        native_health = self._native.health(partitions)
        available_stores = list(native_health.get("stores_ok") or [])
        expected_collections = {
            f"{partition.partition_id}:{store}"
            for partition in partitions
            for store in _PALACE_STORES
        }
        native_collections_ok = (
            bool(native_health.get("ok")) and set(available_stores) == expected_collections
        )
        if native_collections_ok:
            try:
                for partition in partitions:
                    self._require_current_block_projection(partition.primary_block_id)
            except (OSError, RuntimeError, ValueError):
                native_collections_ok = False

        sidecar_ok = self._index.path.is_file()
        if sidecar_ok:
            try:
                self.get_block("__um_mem_recovery_preflight__")
                self.get_index_node("__um_mem_recovery_preflight__")
                self.edges_for("__um_mem_recovery_preflight__")
                self.value_stats("__um_mem_recovery_preflight__")
            except (OSError, sqlite3.Error):
                sidecar_ok = False

        components = {
            "native_collections": {
                "ok": native_collections_ok,
                "available": available_stores,
                "partition_count": len(partitions),
                "isolation_class": "physical",
                "partition_mode": "native_collection_family",
            },
            "sidecar_index": {"ok": sidecar_ok},
        }
        manifest_complete = native_collections_ok and sidecar_ok
        return {
            "ok": manifest_complete,
            "backend": "mempalace",
            "manifest_complete": manifest_complete,
            "preflight_only": True,
            "backup_restore_performed": False,
            "snapshot_available": False,
            "restore_available": False,
            "components": components,
            "reason": "" if manifest_complete else "required_local_component_unavailable",
        }

    def _write_native_snapshot(self, stage_path: Path) -> dict[str, Any]:
        preflight = self.recovery_preflight()
        if not bool(preflight.get("ok")):
            raise RuntimeError("native_snapshot_preflight_unavailable")
        native_stores = sorted(_PALACE_STORES)
        native_partitions = self._existing_native_partitions()
        collections_path = stage_path / "native_collections.jsonl"
        collection_counts: dict[str, int] = {}

        with collections_path.open("w", encoding="utf-8", newline="\n") as handle:
            for partition in native_partitions:
                for store in native_stores:
                    collection = self._native.collection(store, partition=partition)
                    count = int(collection.count())
                    collection_key = f"{partition.partition_id}:{store}"
                    collection_counts[collection_key] = count
                    for offset in range(0, count, _SNAPSHOT_BATCH_SIZE):
                        page = collection.get(
                            limit=_SNAPSHOT_BATCH_SIZE,
                            offset=offset,
                            include=["documents", "metadatas", "embeddings"],
                        )
                        ids = list(page.ids)
                        documents = list(page.documents)
                        metadatas = list(page.metadatas)
                        embeddings = page.embeddings
                        if embeddings is None or not (
                            len(ids) == len(documents) == len(metadatas) == len(embeddings)
                        ):
                            raise ValueError("native_snapshot_collection_shape_invalid")
                        if not ids:
                            raise RuntimeError("native_snapshot_collection_page_empty")
                        for index, native_id in enumerate(ids):
                            document = documents[index]
                            metadata = metadatas[index]
                            raw_embedding = embeddings[index]
                            embedding = (
                                raw_embedding.tolist()
                                if hasattr(raw_embedding, "tolist")
                                else list(raw_embedding)
                            )
                            if (
                                not isinstance(native_id, str)
                                or not isinstance(document, str)
                                or (metadata is not None and not isinstance(metadata, dict))
                                or not isinstance(embedding, list)
                            ):
                                raise ValueError("native_snapshot_collection_value_invalid")
                            try:
                                vector = [float(value) for value in embedding]
                            except (TypeError, ValueError) as exc:
                                raise ValueError("native_snapshot_embedding_invalid") from exc
                            handle.write(
                                json.dumps(
                                    {
                                        "partition_id": partition.partition_id,
                                        "store": store,
                                        "id": native_id,
                                        "document": document,
                                        "metadata": metadata,
                                        "embedding": vector,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )

        _backup_sqlite(self._index._conn, stage_path / "mempalace_index.sqlite")
        return {
            "native_stores": native_stores,
            "native_partitions": [partition.to_mapping() for partition in native_partitions],
            "collection_counts": collection_counts,
        }

    def create_native_snapshot(self, snapshot_path: Path) -> dict[str, Any]:
        target = Path(snapshot_path).expanduser().resolve()
        palace_root = palace_path_for(self._repo_root, self._workspace_id)
        if target.exists():
            return {"ok": False, "error": "snapshot_target_exists"}
        if _path_is_inside(target, palace_root) or _path_is_inside(
            target,
            self._index.path.parent.resolve(),
        ):
            return {"ok": False, "error": "snapshot_target_not_authorized"}

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with self.operation_gate():
                stage_path = Path(
                    tempfile.mkdtemp(prefix=f".{target.name}.snapshot.", dir=target.parent)
                )
                # Pin this thread's WAL read view while native projection is fenced.
                # Other connections may still commit canonical writes immediately.
                self._index._conn.execute("BEGIN")
                try:
                    snapshot = self._write_native_snapshot(stage_path)
                finally:
                    self._index._conn.rollback()
                components = [
                    {"name": "native_collections", "file": "native_collections.jsonl"},
                    {"name": "sidecar_index", "file": "mempalace_index.sqlite"},
                ]
                for component in components:
                    component_path = stage_path / component["file"]
                    component["sha256"] = _file_sha256(component_path)
                manifest = {
                    "schema_version": _SNAPSHOT_SCHEMA_VERSION,
                    "backend": "mempalace",
                    "workspace_id": self._workspace_id,
                    "logical_native_export": True,
                    "native_stores": snapshot["native_stores"],
                    "native_partitions": snapshot["native_partitions"],
                    "collection_counts": snapshot["collection_counts"],
                    "components": components,
                }
                manifest_path = stage_path / "manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                manifest_sha256 = _file_sha256(manifest_path)
                os.replace(stage_path, target)
        except MemoryBusyError:
            return {"ok": False, "error": "snapshot_gate_unavailable"}
        except (OSError, sqlite3.Error, RuntimeError, TypeError, ValueError):
            return {"ok": False, "error": "native_snapshot_failed"}

        return {
            "ok": True,
            "backend": "mempalace",
            "logical_native_export": True,
            "manifest_sha256": manifest_sha256,
            "collection_count": sum(snapshot["collection_counts"].values()),
        }

    @classmethod
    def restore_native_snapshot(
        cls,
        snapshot_path: Path,
        *,
        repo_root: Path,
        workspace_id: str,
    ) -> "MemPalaceBackend":
        source = Path(snapshot_path).expanduser().resolve()
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("native_snapshot_manifest_missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("native_snapshot_manifest_invalid") from exc
        if not isinstance(manifest, dict):
            raise ValueError("native_snapshot_manifest_invalid")
        if (
            manifest.get("schema_version") != _SNAPSHOT_SCHEMA_VERSION
            or manifest.get("backend") != "mempalace"
            or manifest.get("workspace_id") != workspace_id
            or manifest.get("logical_native_export") is not True
        ):
            raise ValueError("native_snapshot_manifest_incompatible")

        components_raw = manifest.get("components")
        if not isinstance(components_raw, list):
            raise ValueError("native_snapshot_components_invalid")
        component_paths: dict[str, Path] = {}
        for component in components_raw:
            if not isinstance(component, dict):
                raise ValueError("native_snapshot_component_invalid")
            name = component.get("name")
            filename = component.get("file")
            digest = component.get("sha256")
            if (
                not isinstance(name, str)
                or not isinstance(filename, str)
                or not isinstance(digest, str)
                or Path(filename).name != filename
            ):
                raise ValueError("native_snapshot_component_invalid")
            component_path = source / filename
            if not component_path.is_file() or _file_sha256(component_path) != digest:
                raise ValueError("native_snapshot_component_checksum_invalid")
            component_paths[name] = component_path
        required_components = {
            "native_collections",
            "sidecar_index",
        }
        if set(component_paths) != required_components:
            raise ValueError("native_snapshot_components_incomplete")

        native_stores_raw = manifest.get("native_stores")
        native_partitions_raw = manifest.get("native_partitions")
        collection_counts_raw = manifest.get("collection_counts")
        if (
            not isinstance(native_stores_raw, list)
            or not isinstance(native_partitions_raw, list)
            or not isinstance(collection_counts_raw, dict)
        ):
            raise ValueError("native_snapshot_native_payload_invalid")
        native_stores = [str(store) for store in native_stores_raw]
        try:
            native_partitions = [
                NativePartitionIdentity(**dict(value))
                for value in native_partitions_raw
                if isinstance(value, dict)
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError("native_snapshot_native_payload_invalid") from exc
        if (
            len(native_stores) != len(set(native_stores))
            or not set(native_stores).issubset(_PALACE_STORES)
            or len(native_partitions) != len(native_partitions_raw)
            or len({item.partition_id for item in native_partitions}) != len(native_partitions)
            or any(
                item.backend_id != "mempalace" or item.workspace_id != workspace_id
                for item in native_partitions
            )
            or set(collection_counts_raw)
            != {
                f"{partition.partition_id}:{store}"
                for partition in native_partitions
                for store in native_stores
            }
        ):
            raise ValueError("native_snapshot_native_payload_invalid")
        collection_counts: dict[str, int] = {}
        for store, count in collection_counts_raw.items():
            if not isinstance(count, int) or count < 0:
                raise ValueError("native_snapshot_collection_count_invalid")
            collection_counts[str(store)] = count

        with component_paths["native_collections"].open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("native_snapshot_collection_row_invalid") from exc
                if not isinstance(row, dict):
                    raise ValueError("native_snapshot_collection_row_invalid")

        target_root = Path(repo_root).expanduser().resolve()
        target_index = _index_path(target_root, workspace_id)
        target_palace = palace_path_for(target_root, workspace_id)
        target_index.parent.mkdir(parents=True, exist_ok=True)
        wait_budget = _operation_lock_timeout_seconds()
        restored: MemPalaceBackend | None = None
        created_palace = False
        created_index = False
        try:
            with _native_gate_for(target_index).hold(exclusive=True, timeout=wait_budget):
                if target_palace.exists() or target_index.exists():
                    raise ValueError("native_snapshot_restore_target_not_fresh")
                created_index = True
                _restore_sqlite(component_paths["sidecar_index"], target_index)
                restored = cls(
                    target_root,
                    workspace_id=workspace_id,
                    reconcile_pending_writes=False,
                )
                created_palace = True
                restored._restore_native_collections(
                    component_paths["native_collections"],
                    native_stores=native_stores,
                    native_partitions=native_partitions,
                    collection_counts=collection_counts,
                )
                if not restored.recovery_preflight()["ok"]:
                    raise ValueError("native_snapshot_restore_preflight_failed")
            assert restored is not None
            restored._reconcile_pending_outboxes()
        except (OSError, sqlite3.Error, RuntimeError, TypeError, ValueError) as exc:
            if restored is not None:
                restored._native.close()
                restored._index.close()
            if created_index:
                for index_component in (
                    target_index,
                    target_index.with_name(f"{target_index.name}-wal"),
                    target_index.with_name(f"{target_index.name}-shm"),
                ):
                    index_component.unlink(missing_ok=True)
            if created_palace:
                shutil.rmtree(target_palace)
            if isinstance(exc, MemoryBusyError):
                raise RuntimeError("native_snapshot_restore_gate_unavailable") from exc
            raise

        assert restored is not None
        return restored

    def _restore_native_collections(
        self,
        collections_path: Path,
        *,
        native_stores: list[str],
        native_partitions: list[NativePartitionIdentity],
        collection_counts: dict[str, int],
    ) -> None:
        partition_by_id = {partition.partition_id: partition for partition in native_partitions}
        restored_counts = {
            f"{partition.partition_id}:{store}": 0
            for partition in native_partitions
            for store in native_stores
        }
        buffers: dict[str, dict[str, list[object]]] = {
            collection_key: {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
            }
            for collection_key in restored_counts
        }

        def flush(collection_key: str) -> None:
            buffer = buffers[collection_key]
            if not buffer["ids"]:
                return
            partition_id, store = collection_key.rsplit(":", 1)
            self._native.collection(
                store,
                partition=partition_by_id[partition_id],
            ).add(
                ids=buffer["ids"],
                documents=buffer["documents"],
                metadatas=buffer["metadatas"],
                embeddings=buffer["embeddings"],
            )
            restored_counts[collection_key] += len(buffer["ids"])
            buffers[collection_key] = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": [],
            }

        for partition in native_partitions:
            for store in native_stores:
                self._native.collection(store, partition=partition)
        with collections_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("native_snapshot_collection_row_invalid") from exc
                if not isinstance(row, dict):
                    raise ValueError("native_snapshot_collection_row_invalid")
                partition_id = row.get("partition_id")
                stored_store = row.get("store")
                native_id = row.get("id")
                document = row.get("document")
                metadata = row.get("metadata")
                embedding = row.get("embedding")
                if (
                    not isinstance(partition_id, str)
                    or not isinstance(stored_store, str)
                    or f"{partition_id}:{stored_store}" not in buffers
                    or not isinstance(native_id, str)
                    or not isinstance(document, str)
                    or (metadata is not None and not isinstance(metadata, dict))
                    or not isinstance(embedding, list)
                ):
                    raise ValueError("native_snapshot_collection_row_invalid")
                try:
                    vector = [float(value) for value in embedding]
                except (TypeError, ValueError) as exc:
                    raise ValueError("native_snapshot_embedding_invalid") from exc
                collection_key = f"{partition_id}:{stored_store}"
                buffer = buffers[collection_key]
                buffer["ids"].append(native_id)
                buffer["documents"].append(document)
                buffer["metadatas"].append(metadata)
                buffer["embeddings"].append(vector)
                if len(buffer["ids"]) >= _SNAPSHOT_BATCH_SIZE:
                    flush(collection_key)
        for collection_key in buffers:
            flush(collection_key)
        if restored_counts != collection_counts:
            raise ValueError("native_snapshot_collection_count_mismatch")

    def close(self) -> None:
        self._index.stop_native_projection()
        with self.operation_gate():
            self._native.close()
            self._index.close()
