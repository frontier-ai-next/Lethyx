"""Zvec native leaf collections over Lethyx's immutable canonical record index."""

import hashlib
import importlib.util
import math
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from filelock import FileLock

from lethyx.contracts import MemoryBlockRecord
from lethyx.memory.backends.base import (
    MemoryHit,
    MemoryQuery,
    NativeProjectionPending,
    ReconcileReport,
)
from lethyx.memory.backends.contracts import (
    BackendCapabilities,
    NativePartitionIdentity,
    block_record_to_memory_hit,
)
from lethyx.memory.backends.query_policy import (
    ensure_record_is_storable,
    merge_hybrid_block_results,
    merge_native_block_results,
    normalize_query_payload,
)
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex


class ZvecBackend(BackendSQLiteIndex):
    backend_id = "zvec"
    _window_chars = 480
    _window_step = 360
    # Zvec accepts at most 1024 documents per write; keep embedding memory bounded too.
    _write_batch_size = 256

    def __init__(
        self,
        repo_root: Path,
        workspace_id: str = "",
        *,
        reconcile_pending_writes: bool = True,
    ) -> None:
        if importlib.util.find_spec("zvec") is None:
            raise RuntimeError("native Zvec is unavailable; install lethyx[zvec]")
        import zvec  # type: ignore[import-not-found, import-untyped]

        self._zvec = zvec
        self.repo_root = Path(repo_root)
        self.workspace_id = workspace_id
        self.data_dir = (
            self.repo_root / "workspaces" / (workspace_id or "default") / ".memory" / "zvec"
        )
        self._collections: dict[str, Any] = {}
        self._collection_lock = threading.RLock()
        self._embedding_state = threading.local()
        super().__init__(
            self.data_dir / "LETHYX_index.sqlite",
            backend_id="zvec",
            workspace_id=workspace_id,
            repo_root=self.repo_root,
        )
        self._index = self
        if reconcile_pending_writes:
            self.replay_pending_canonical_heads()
        self.configure_native_projection(
            self._project_durable_record, start=reconcile_pending_writes
        )

    @staticmethod
    def native_partition_identity(
        primary_block_id: str,
        *,
        workspace_id: str,
        catalog_revision_id: str,
        backend_partition_id: str,
    ) -> NativePartitionIdentity:
        if not all((primary_block_id, workspace_id, catalog_revision_id, backend_partition_id)):
            raise ValueError("Zvec native partition requires exact leaf and workspace identity")
        digest = hashlib.sha256(
            f"{workspace_id}\0{backend_partition_id}".encode("utf-8")
        ).hexdigest()
        return NativePartitionIdentity(
            backend_id="zvec",
            workspace_id=workspace_id,
            primary_block_id=primary_block_id,
            catalog_revision_id=catalog_revision_id,
            partition_id=f"zvec:collection:{digest}",
            isolation_class="physical",
            partition_mode="native_collection",
        )

    def _partition(self, record: MemoryBlockRecord) -> NativePartitionIdentity:
        membership = record.block_membership
        if membership is None:
            raise ValueError("Zvec native record requires a leaf membership")
        return self.native_partition_identity(
            membership.primary_block_id,
            workspace_id=record.workspace_id,
            catalog_revision_id=membership.catalog_revision_id,
            backend_partition_id=membership.backend_partition_id,
        )

    def _collection(self, partition: NativePartitionIdentity, *, create: bool) -> Any | None:
        key = partition.partition_id.rsplit(":", 1)[-1]
        path = self.data_dir / "partitions" / key
        with self._collection_lock:
            cached = self._collections.get(key)
            if cached is not None:
                return cached
            if not path.exists() and not create:
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".open.lock", timeout=30):
                if path.exists():
                    collection = self._zvec.open(path=str(path))
                else:
                    schema = self._zvec.CollectionSchema(
                        name="lethyx_leaf",
                        fields=[
                            self._zvec.FieldSchema("record_id", self._zvec.DataType.STRING),
                            self._zvec.FieldSchema("source_sha256", self._zvec.DataType.STRING),
                            self._zvec.FieldSchema("commit_revision", self._zvec.DataType.INT64),
                            self._zvec.FieldSchema("chunk_start", self._zvec.DataType.INT64),
                            self._zvec.FieldSchema("chunk_end", self._zvec.DataType.INT64),
                            self._zvec.FieldSchema(
                                "content",
                                self._zvec.DataType.STRING,
                                index_param=self._zvec.FtsIndexParam(
                                    tokenizer_name="standard", filters=["lowercase"]
                                ),
                            ),
                        ],
                        vectors=[
                            self._zvec.VectorSchema(
                                "embedding",
                                self._zvec.DataType.VECTOR_FP32,
                                384,
                                index_param=self._zvec.HnswIndexParam(
                                    metric_type=self._zvec.MetricType.COSINE
                                ),
                            )
                        ],
                    )
                    collection = self._zvec.create_and_open(path=str(path), schema=schema)
            self._collections[key] = collection
            return collection

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embedder = getattr(self._embedding_state, "embedder", None)
        if embedder is None:
            embedder = ONNXMiniLM_L6_V2()
            self._embedding_state.embedder = embedder
        vectors = embedder(texts)
        if len(vectors) != len(texts) or any(len(vector) != 384 for vector in vectors):
            raise RuntimeError("Zvec embedding dimension mismatch")
        return [list(map(float, vector)) for vector in vectors]

    @classmethod
    def _chunks(cls, record: MemoryBlockRecord) -> list[tuple[str, int, int, str]]:
        content = record.content
        spans = [
            (start, min(start + cls._window_chars, len(content)))
            for start in range(0, max(1, len(content)), cls._window_step)
        ]
        return [
            (
                hashlib.sha256(f"{record.block_id}\0{start}\0{end}".encode()).hexdigest(),
                start,
                end,
                content[start:end],
            )
            for start, end in spans
        ]

    def _write_native(self, record: MemoryBlockRecord, commit_revision: int) -> str:
        partition = self._partition(record)
        collection = self._collection(partition, create=True)
        assert collection is not None
        chunks = self._chunks(record)
        source_hash = hashlib.sha256(record.content.encode("utf-8")).hexdigest()
        for offset in range(0, len(chunks), self._write_batch_size):
            batch = chunks[offset : offset + self._write_batch_size]
            existing = collection.fetch(ids=[doc_id for doc_id, _, _, _ in batch])
            missing = []
            for doc_id, start, end, content in batch:
                fetched = existing.get(doc_id)
                if fetched is not None:
                    fields = fetched.fields or {}
                    if (
                        fields.get("record_id") != record.block_id
                        or fields.get("source_sha256") != source_hash
                        or fields.get("commit_revision") != commit_revision
                        or fields.get("chunk_start") != start
                        or fields.get("chunk_end") != end
                        or fields.get("content") != content
                    ):
                        raise RuntimeError("Zvec native projection conflicts with immutable source")
                else:
                    missing.append((doc_id, start, end, content))
            if missing:
                vectors = self._embed([content for _, _, _, content in missing])
                documents = [
                    self._zvec.Doc(
                        id=doc_id,
                        fields={
                            "record_id": record.block_id,
                            "source_sha256": source_hash,
                            "commit_revision": commit_revision,
                            "chunk_start": start,
                            "chunk_end": end,
                            "content": content,
                        },
                        vectors={"embedding": vector},
                    )
                    for (doc_id, start, end, content), vector in zip(missing, vectors, strict=True)
                ]
                statuses = collection.upsert(documents)
                if (
                    not isinstance(statuses, list)
                    or len(statuses) != len(documents)
                    or any(not status.ok() for status in statuses)
                ):
                    raise RuntimeError("Zvec native upsert failed")
                collection.flush()
            verified = collection.fetch(ids=[doc_id for doc_id, _, _, _ in batch])
            for doc_id, _, _, content in batch:
                fetched = verified.get(doc_id)
                if fetched is None or (fetched.fields or {}).get("content") != content:
                    raise RuntimeError("Zvec native projection verification failed")
        return chunks[0][0]

    def _project_durable_record(self, record: MemoryBlockRecord) -> None:
        entry = self.latest_write_outbox_entry(
            block_id=record.block_id, workspace_id=record.workspace_id
        )
        if entry is None or entry.store != "zvec":
            raise RuntimeError("Zvec native projection outbox is missing")
        with self.projection_gate(record):
            if entry.status == "pending":
                self.mark_write_outbox_projection_started(entry.commit_revision)
            native_id = self._write_native(record, entry.commit_revision)
            if entry.status == "pending":
                self.set_write_outbox_native_ref(entry.commit_revision, native_id)
                self.complete_write_outbox(entry.commit_revision)

    def stage_topology_projection(self, record: MemoryBlockRecord) -> None:
        entry = self.latest_write_outbox_entry(
            block_id=record.block_id, workspace_id=record.workspace_id
        )
        if entry is None or entry.status != "committed":
            raise NativeProjectionPending("native_projection_pending")
        with self.projection_gate(record):
            self._write_native(record, entry.commit_revision)

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        ensure_record_is_storable(record)
        entry = self.open_write_outbox(record=record, store="zvec")
        stored = self.write_outbox_record(entry)
        result = super().retain_block(stored)
        self._project_durable_record(stored)
        return {**result, "native": True, "commit_revision": entry.commit_revision}

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities.effective(
            "zvec",
            native_ok=self.health()["native_ok"],
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

    def health(self) -> dict[str, Any]:
        projection = self.native_projection_health()
        return {
            "ok": True,
            "backend": "zvec",
            "native_ok": True,
            "strict_ok": projection["failed"] == 0 and projection["attention"] == 0,
            "effective_mode": "native",
            "native_block_scoping_supported": True,
            "implementation": "zvec_leaf_collections",
            "workspace_id": self.workspace_id,
            "native_projection": projection,
        }

    def ensure_banks(self, *, workspace_id: str = "") -> dict[str, Any]:
        return {"ok": True, "backend": "zvec", "workspace_id": workspace_id or self.workspace_id}

    def reconcile(self, *, dry_run: bool = True) -> ReconcileReport:
        before = self.native_projection_health()
        if not dry_run:
            self.replay_pending_canonical_heads()
            self.wake_native_projection()
        after = self.native_projection_health()
        return ReconcileReport(
            backend="zvec",
            dry_run=dry_run,
            pending_canonical_heads=len(self.pending_canonical_head_reservations()),
            pending_write_outbox=before["pending"],
            repaired=max(0, before["pending"] - after["pending"]),
            remaining=after["pending"],
            failures=after["failed"] + after["attention"],
            result_code="pending" if after["pending"] else "ok",
            reason_code="native_projection" if after["pending"] else "drained",
        )

    def recall(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryBlockRecord]:
        payload = self.apply_exact_query(
            normalize_query_payload(query, default_workspace_id=self.workspace_id)
        )
        with self.topology_snapshot(str(payload.get("profile_id") or "default")):
            if (
                payload.get("record_ids")
                or payload.get("known_as_of")
                or payload.get("as_of")
                or payload.get("temporal_mode") == "historical"
            ):
                return super().recall({**payload, "retrieval_mode": "semantic"})
            blocks = tuple(str(value) for value in payload.get("logical_block_ids") or ())
            if not blocks and payload.get("search_all") is not True:
                raise ValueError("Zvec native recall requires logical_block_ids or search_all=True")
            if not str(payload.get("query") or "").strip():
                return super().recall({**payload, "retrieval_mode": "semantic"})
            memberships = self.native_partition_memberships(
                workspace_id=self.workspace_id,
                profile_id=str(payload.get("profile_id") or "default"),
                logical_block_ids=blocks,
            )
            order = {block_id: position for position, block_id in enumerate(blocks)}
            memberships = tuple(
                sorted(memberships, key=lambda row: (order.get(row[0], len(order)), row))
            )
            query_text = str(payload["query"])
            vector = self._embed([query_text])[0]
            limit = int(payload.get("limit") or 10)
            candidate_limit = max(64, limit * 4)
            dense_pools: list[tuple[str, list[tuple[MemoryBlockRecord, float | None]]]] = []
            lexical_pools: list[tuple[str, list[tuple[MemoryBlockRecord, float | None]]]] = []
            for block_id, catalog_revision, partition_id in memberships:
                partition = self.native_partition_identity(
                    block_id,
                    workspace_id=self.workspace_id,
                    catalog_revision_id=catalog_revision,
                    backend_partition_id=partition_id,
                )
                collection = self._collection(partition, create=False)
                if collection is None:
                    raise NativeProjectionPending("native_projection_pending")
                started = time.perf_counter()
                dense_rows = collection.query(
                    queries=self._zvec.Query(field_name="embedding", vector=vector),
                    topk=candidate_limit,
                )
                lexical_rows = (
                    collection.query(
                        queries=self._zvec.Query(
                            field_name="content", fts=self._zvec.Fts(match_string=query_text)
                        ),
                        topk=candidate_limit,
                    )
                    if payload["retrieval_mode"] == "hybrid"
                    else []
                )
                for rows, destination in ((dense_rows, dense_pools), (lexical_rows, lexical_pools)):
                    pool: list[tuple[MemoryBlockRecord, float | None]] = []
                    seen: set[str] = set()
                    for row in rows:
                        fields = row.fields or {}
                        record_id = str(fields.get("record_id") or "")
                        if not record_id or record_id in seen:
                            continue
                        seen.add(record_id)
                        record = self.get_block(record_id, workspace_id=self.workspace_id)
                        if (
                            record is None
                            or self._partition(record).partition_id != partition.partition_id
                        ):
                            continue
                        if hashlib.sha256(record.content.encode("utf-8")).hexdigest() != fields.get(
                            "source_sha256"
                        ):
                            raise RuntimeError(
                                "Zvec native source hash differs from canonical record"
                            )
                        admitted = self.overlay_native_hit(
                            record, workspace_id=self.workspace_id, query=payload
                        )
                        if admitted is None:
                            continue
                        score = float(row.score) if row.score is not None else math.inf
                        if not math.isfinite(score):
                            raise ValueError("Zvec native score is not finite")
                        pool.append((admitted, 1.0 / (1.0 + max(0.0, score))))
                    destination.append((block_id, pool))
                self.record_partition_operation(
                    partition,
                    operation="recall",
                    latency_ms=(time.perf_counter() - started) * 1000,
                    hit_count=len(dense_pools[-1][1]),
                    search_mode="scoped" if blocks else "search_all",
                    run_id=str(payload.get("run_id") or ""),
                    profile_id=str(payload.get("profile_id") or "default"),
                )
            if payload["retrieval_mode"] == "hybrid":
                return merge_hybrid_block_results(dense_pools, lexical_pools, limit=limit)
            return merge_native_block_results(
                dense_pools, limit=limit, score_mode="cosine_similarity"
            )

    def recall_evidence(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryHit]:
        hits = []
        for record in self.recall(query):
            hit = block_record_to_memory_hit(record)
            hits.append(replace(hit, metadata={**hit.metadata, "source_backend": "zvec"}))
        return hits

    def record_rendered_tokens(self, rendered_tokens_by_record_id: dict[str, int]) -> None:
        for record_id, count in rendered_tokens_by_record_id.items():
            record = self.get_block(record_id, workspace_id=self.workspace_id)
            if record is not None and record.block_membership is not None:
                self.record_partition_operation(
                    self._partition(record), operation="render", rendered_tokens=count
                )

    def close(self) -> None:
        self.stop_native_projection()
        with self._collection_lock:
            for collection in self._collections.values():
                collection.close()
            self._collections.clear()
        super().close()
