"""Small persistent backend-owned index for memory backend parity.

The index stores UM-MEM-neutral records for adapters that need parity metadata
outside their native content store. It is intentionally not a MemPalace mirror;
it is owned by the selected backend implementation directory.
"""

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from filelock import FileLock, Timeout

from um_mem.contracts import (
    ConsolidationCommitReceipt,
    ContextualUtilitySummary,
    EvidenceRef,
    MemoryBlockRecord,
    MemoryDecisionEvent,
    MemoryDecisionTrace,
    MemoryUsageEvent,
    OutcomeCreditEvent,
    ProspectiveObligation,
    ProspectiveTelemetry,
    ProspectiveTriggerPredicate,
    RecordBlockMembership,
    TransferEnvelope,
    is_protected_memory,
)
from um_mem.contracts.memory import (
    ProspectiveRecurrencePolicy,
    ProspectiveScope,
    ProspectiveStatus,
    _closed_value,
)
from um_mem.llm_budget import estimate_tokens
from um_mem.memory.backends.base import MemoryQuery
from um_mem.memory.backends.contracts import (
    ALLOWED_EDGE_TYPES,
    CanonicalHeadConflict,
    CanonicalHeadReservation,
    EdgeType,
    MemoryEdgeRecord,
    MemoryIndexNode,
    MemoryUsageRecord,
    NativePartitionIdentity,
    NativePartitionMetrics,
    block_from_json,
    block_to_json,
    index_node_from_json,
    index_node_to_json,
)
from um_mem.memory.backends.query_policy import (
    ensure_record_is_storable,
    normalize_query_payload,
    query_sequence,
    record_allowed_for_query,
    record_id_sequence,
    record_ids,
)
from um_mem.memory.blocks.catalog import SemanticCatalog, sync_block_descriptor_index
from um_mem.memory.dependencies import DependencyIssue, DependencyValidation
from um_mem.memory.indexing import compile_memory_index_document
from um_mem.memory.kernel.telemetry import (
    current_memory_profile_context,
    current_memory_profile_span_id,
    memory_profile_summary,
    record_memory_event,
)
from um_mem.memory.mutation import (
    AmendmentCommitReceipt,
    CanonicalMutationIdentity,
    MutationDecision,
    PublicationCommitReceipt,
)
from um_mem.memory.paths import workspace_memory_root
from um_mem.memory.prospective import (
    ProspectiveTriggerContext,
    ProspectiveTriggerDecision,
    evaluate_prospective_trigger,
    prospective_trigger_index_node,
)

_MAX_RECENT_SUBSTRING_FALLBACK = 128
SQLITE_INDEX_SCHEMA_VERSION = 16
_EXACT_QUERY_TERMS = {
    "paths": "path",
    "symbols": "symbol",
    "entities": "entity",
    "config_keys": "config_key",
    "endpoints": "endpoint",
    "artifact_hashes": "artifact_hash",
}
_CURRENT_SCHEMA_TABLES = frozenset(
    {
        "schema_metadata",
        "blocks",
        "index_nodes",
        "edges",
        "usage",
        "canonical_payloads",
        "write_outbox",
        "write_outbox_native_refs",
        "canonical_heads",
        "canonical_head_outbox",
        "canonical_mutations",
        "canonical_mutation_receipts",
        "canonical_mutation_invalidations",
        "prospective_obligations",
        "prospective_obligation_events",
        "memory_exact_terms",
        "memory_temporal_index",
        "memory_provenance_links",
        "record_block_memberships",
        "native_partition_events",
        "memory_fts",
        "memory_fts_data",
        "memory_fts_idx",
        "memory_fts_content",
        "memory_fts_docsize",
        "memory_fts_config",
    }
)
_CANONICAL_BLOCK_COLUMNS = frozenset(
    {
        "block_id",
        "workspace_id",
        "run_id",
        "stage_id",
        "subtask_id",
        "scope",
        "status",
        "retention",
        "delivery_policy",
        "temperature",
        "authority",
        "trust_level",
        "verification_status",
        "promotion_decision",
        "content",
        "payload_json",
        "updated_at",
        "committed_at",
    }
)
_CANONICAL_HEAD_OUTBOX_COLUMNS = frozenset(
    {
        "commit_revision",
        "workspace_id",
        "canonical_key_sha256",
        "predecessor_id",
        "successor_id",
        "successor_status",
        "successor_authority",
        "mutation_action",
        "rollback_target_id",
        "payload_id",
        "payload_sha256",
        "status",
        "error",
        "created_at",
        "updated_at",
    }
)
_WRITE_OUTBOX_COLUMNS = frozenset(
    {
        "commit_revision",
        "block_id",
        "workspace_id",
        "store",
        "payload_id",
        "payload_sha256",
        "status",
        "error",
        "created_at",
        "updated_at",
    }
)
_CANONICAL_MUTATION_COLUMNS = frozenset(
    {
        "commit_revision",
        "workspace_id",
        "canonical_key_sha256",
        "predecessor_id",
        "successor_id",
        "mutation_action",
        "rollback_target_id",
        "record_payload_id",
        "record_payload_sha256",
        "recorded_at",
        "committed_at",
    }
)
_CANONICAL_PAYLOAD_COLUMNS = frozenset(
    {"payload_id", "payload_sha256", "payload_kind", "payload_json", "created_at"}
)
_PROSPECTIVE_OBLIGATION_COLUMNS = frozenset(
    {
        "obligation_id",
        "scope",
        "status",
        "version",
        "payload_id",
        "payload_sha256",
        "receipt_id",
        "updated_at",
    }
)
_PROSPECTIVE_EVENT_COLUMNS = frozenset(
    {
        "obligation_id",
        "event_id",
        "idempotency_key",
        "expected_version",
        "from_status",
        "to_status",
        "payload_id",
        "payload_sha256",
        "receipt_id",
        "created_at",
    }
)
_MEMORY_EXACT_TERM_COLUMNS = frozenset({"workspace_id", "term_type", "term_value", "block_id"})
_MEMORY_TEMPORAL_COLUMNS = frozenset(
    {
        "workspace_id",
        "block_id",
        "canonical_key_sha256",
        "revision_id",
        "valid_from",
        "valid_to",
        "recorded_at",
        "is_current",
    }
)
_MEMORY_PROVENANCE_COLUMNS = frozenset({"workspace_id", "derived_id", "raw_source_ref", "block_id"})
_RECORD_BLOCK_MEMBERSHIP_COLUMNS = frozenset(
    {
        "workspace_id",
        "record_id",
        "record_revision_id",
        "membership_revision_id",
        "logical_block_id",
        "role",
        "ordinal",
        "catalog_revision_id",
        "backend_partition_id",
        "valid_from",
        "valid_to",
        "is_current",
    }
)
_NATIVE_PARTITION_EVENT_COLUMNS = frozenset(
    {
        "id",
        "workspace_id",
        "primary_block_id",
        "catalog_revision_id",
        "partition_id",
        "isolation_class",
        "partition_mode",
        "operation",
        "latency_ms",
        "hit_count",
        "record_bytes",
        "rendered_tokens",
        "success",
        "search_mode",
        "recorded_at",
    }
)

_CREATE = """
CREATE TABLE schema_metadata (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE blocks (
    block_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    subtask_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL,
    retention TEXT NOT NULL,
    delivery_policy TEXT NOT NULL,
    temperature TEXT NOT NULL,
    authority TEXT NOT NULL,
    trust_level TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    promotion_decision TEXT NOT NULL,
    content TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    committed_at TEXT NOT NULL
);
CREATE INDEX idx_blocks_workspace ON blocks(workspace_id, status, authority);
CREATE INDEX idx_blocks_scope ON blocks(workspace_id, scope, run_id, stage_id, subtask_id);
CREATE INDEX idx_blocks_committed ON blocks(workspace_id, committed_at, block_id);
CREATE TABLE index_nodes (
    node_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    title TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_index_nodes_workspace_type ON index_nodes(workspace_id, node_type);
CREATE INDEX idx_index_nodes_workspace_updated ON index_nodes(workspace_id, updated_at);
CREATE TABLE edges (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(workspace_id, src_id, dst_id, edge_type)
);
CREATE INDEX idx_edges_src ON edges(workspace_id, src_id);
CREATE INDEX idx_edges_dst ON edges(workspace_id, dst_id);
CREATE TABLE usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE canonical_payloads (
    payload_id TEXT PRIMARY KEY,
    payload_sha256 TEXT NOT NULL UNIQUE,
    payload_kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE write_outbox (
    commit_revision INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    store TEXT NOT NULL,
    payload_id TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_write_outbox_pending
ON write_outbox(block_id) WHERE status='pending';
CREATE INDEX idx_write_outbox_workspace_status
ON write_outbox(workspace_id, status, commit_revision);
CREATE TABLE write_outbox_native_refs (
    commit_revision INTEGER PRIMARY KEY,
    native_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(commit_revision) REFERENCES write_outbox(commit_revision)
);
CREATE TABLE canonical_heads (
    workspace_id TEXT NOT NULL,
    canonical_key_sha256 TEXT NOT NULL,
    block_id TEXT NOT NULL,
    commit_revision INTEGER NOT NULL,
    PRIMARY KEY(workspace_id, canonical_key_sha256)
);
CREATE TABLE canonical_head_outbox (
    commit_revision INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    canonical_key_sha256 TEXT NOT NULL,
    predecessor_id TEXT NOT NULL,
    successor_id TEXT NOT NULL,
    successor_status TEXT NOT NULL,
    successor_authority TEXT NOT NULL,
    mutation_action TEXT NOT NULL,
    rollback_target_id TEXT NOT NULL,
    payload_id TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_canonical_head_outbox_pending
ON canonical_head_outbox(workspace_id, canonical_key_sha256) WHERE status='pending';
CREATE INDEX idx_canonical_head_outbox_successor
ON canonical_head_outbox(successor_id, status, commit_revision);
CREATE TABLE canonical_mutations (
    commit_revision INTEGER PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    canonical_key_sha256 TEXT NOT NULL,
    predecessor_id TEXT NOT NULL,
    successor_id TEXT NOT NULL UNIQUE,
    mutation_action TEXT NOT NULL,
    rollback_target_id TEXT NOT NULL,
    record_payload_id TEXT NOT NULL,
    record_payload_sha256 TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    FOREIGN KEY(commit_revision) REFERENCES canonical_head_outbox(commit_revision)
);
CREATE INDEX idx_canonical_mutations_lookup
ON canonical_mutations(workspace_id, canonical_key_sha256, committed_at, commit_revision);
CREATE TABLE canonical_mutation_receipts (
    commit_revision INTEGER NOT NULL,
    receipt_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(commit_revision, receipt_id),
    FOREIGN KEY(commit_revision) REFERENCES canonical_mutations(commit_revision)
);
CREATE TABLE canonical_mutation_invalidations (
    commit_revision INTEGER NOT NULL,
    view_id TEXT NOT NULL,
    PRIMARY KEY(commit_revision, view_id),
    FOREIGN KEY(commit_revision) REFERENCES canonical_mutations(commit_revision)
);
CREATE TABLE prospective_obligations (
    obligation_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    status TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload_id TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    receipt_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE prospective_obligation_events (
    obligation_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    expected_version INTEGER NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    payload_id TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    receipt_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(obligation_id, event_id),
    UNIQUE(obligation_id, idempotency_key)
);
CREATE TABLE memory_exact_terms (
    workspace_id TEXT NOT NULL,
    term_type TEXT NOT NULL,
    term_value TEXT NOT NULL,
    block_id TEXT NOT NULL,
    PRIMARY KEY(workspace_id, term_type, term_value, block_id)
);
CREATE INDEX idx_memory_exact_lookup
ON memory_exact_terms(workspace_id, term_type, term_value, block_id);
CREATE INDEX idx_memory_exact_block
ON memory_exact_terms(block_id, term_type, term_value);
CREATE TABLE memory_temporal_index (
    workspace_id TEXT NOT NULL,
    block_id TEXT PRIMARY KEY,
    canonical_key_sha256 TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    is_current INTEGER NOT NULL CHECK(is_current IN (0, 1))
);
CREATE INDEX idx_memory_temporal_current
ON memory_temporal_index(workspace_id, is_current, canonical_key_sha256, valid_from, valid_to);
CREATE INDEX idx_memory_temporal_revision
ON memory_temporal_index(workspace_id, revision_id, block_id);
CREATE TABLE memory_provenance_links (
    workspace_id TEXT NOT NULL,
    derived_id TEXT NOT NULL,
    raw_source_ref TEXT NOT NULL,
    block_id TEXT NOT NULL,
    PRIMARY KEY(workspace_id, derived_id, raw_source_ref, block_id)
);
CREATE INDEX idx_memory_provenance_reverse
ON memory_provenance_links(workspace_id, derived_id, raw_source_ref);
CREATE TABLE record_block_memberships (
    workspace_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    record_revision_id TEXT NOT NULL,
    membership_revision_id TEXT NOT NULL,
    logical_block_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('primary', 'secondary')),
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    catalog_revision_id TEXT NOT NULL,
    backend_partition_id TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT NOT NULL,
    is_current INTEGER NOT NULL CHECK(is_current IN (0, 1)),
    PRIMARY KEY(record_id, logical_block_id)
);
CREATE UNIQUE INDEX idx_record_block_membership_primary
ON record_block_memberships(record_id) WHERE role='primary';
CREATE INDEX idx_record_block_membership_lookup
ON record_block_memberships(workspace_id, logical_block_id, is_current, record_id);
CREATE TABLE native_partition_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    primary_block_id TEXT NOT NULL,
    catalog_revision_id TEXT NOT NULL,
    partition_id TEXT NOT NULL,
    isolation_class TEXT NOT NULL,
    partition_mode TEXT NOT NULL,
    operation TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    hit_count INTEGER NOT NULL,
    record_bytes INTEGER,
    rendered_tokens INTEGER NOT NULL,
    success INTEGER NOT NULL CHECK(success IN (0, 1)),
    search_mode TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX idx_native_partition_events_identity
ON native_partition_events(
    workspace_id, primary_block_id, catalog_revision_id, partition_id, id
);
CREATE VIRTUAL TABLE memory_fts USING fts5(
    block_id UNINDEXED,
    workspace_id UNINDEXED,
    title,
    content,
    tags,
    tokenize='unicode61'
);
"""


def _parse_time(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _record_token_cost(record: MemoryBlockRecord) -> int:
    return estimate_tokens("\n".join(part for part in (record.title, record.content) if part))


def _truncate_record_to_budget(
    record: MemoryBlockRecord, token_budget: int
) -> MemoryBlockRecord | None:
    budget = max(0, int(token_budget or 0))
    if budget <= 0:
        return None
    original_tokens = _record_token_cost(record)
    if original_tokens <= budget:
        return record

    title = record.title
    content = record.content
    if not content:
        content = title
        title = ""
    if title and estimate_tokens(title + "\n") >= budget:
        title = ""

    low = 0
    high = len(content)
    while low < high:
        middle = (low + high + 1) // 2
        rendered = "\n".join(part for part in (title, content[:middle]) if part)
        if estimate_tokens(rendered) <= budget:
            low = middle
        else:
            high = middle - 1
    truncated_content = content[:low]
    if not truncated_content and content:
        return None
    return replace(
        record,
        title=title,
        content=truncated_content,
        metadata={
            **dict(record.metadata or {}),
            "retrieval_truncated": True,
            "retrieval_original_tokens": original_tokens,
            "retrieval_visible_tokens": _record_token_cost(
                replace(record, title=title, content=truncated_content)
            ),
        },
    )


@dataclass(frozen=True)
class WriteOutboxEntry:
    commit_revision: int
    block_id: str
    workspace_id: str
    store: str
    payload_id: str
    payload_sha256: str
    status: str
    error: str
    created_at: str
    updated_at: str

    @property
    def projection_retry_exhausted(self) -> bool:
        if self.status != "pending" or not self.error:
            return False
        try:
            state = json.loads(self.error)
        except json.JSONDecodeError:
            return False
        return (
            isinstance(state, dict)
            and state.get("kind") == "native_projection"
            and type(state.get("attempts")) is int
            and state["attempts"] >= 3
        )


class BackendSQLiteIndex:
    def __init__(
        self,
        path: Path,
        *,
        backend_id: str,
        workspace_id: str = "",
        repo_root: Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.backend_id = backend_id
        self.workspace_id = workspace_id
        self._repo_root = Path(repo_root) if repo_root is not None else None
        self._thread_connection = threading.local()
        self._connections_lock = threading.Lock()
        self._connections: list[sqlite3.Connection] = []
        self._closed = False
        self._projection_callback: Callable[[MemoryBlockRecord], object] | None = None
        self._native_projection_gate: Callable[[], AbstractContextManager[object]] = nullcontext
        self._projection_wake = threading.Event()
        self._projection_stop = threading.Event()
        self._projection_thread: threading.Thread | None = None
        self._projection_error = ""
        self._projection_active = 0
        self._projection_lock_state = threading.local()
        self._projection_start_lock = threading.Lock()
        self._projection_workers: set[threading.Thread] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        initialization_lock = FileLock(f"{self.path}.initialize.lock", timeout=10.0)
        with initialization_lock:
            if self.path.exists():
                self._validate_existing_schema(self.path)
                connection = sqlite3.connect(
                    f"{self.path.resolve().as_uri()}?mode=rw",
                    uri=True,
                    check_same_thread=False,
                    timeout=10.0,
                )
                self._thread_connection.connection = connection
                self._connections.append(connection)
                self._conn.execute("PRAGMA busy_timeout=10000")
                self._conn.execute("PRAGMA synchronous=FULL")
                return
            connection = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10.0)
            self._thread_connection.connection = connection
            self._connections.append(connection)
            try:
                self._conn.execute("PRAGMA busy_timeout=10000")
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=FULL")
                self._conn.executescript(_CREATE)
                self._conn.execute(
                    "INSERT INTO schema_metadata(key, value) VALUES(?, ?)",
                    ("schema_version", SQLITE_INDEX_SCHEMA_VERSION),
                )
                self._conn.commit()
            except Exception:
                self._conn.close()
                raise

    @property
    def _conn(self) -> sqlite3.Connection:
        """A transaction belongs to one worker; WAL coordinates independent writers."""
        if self._closed:
            raise RuntimeError("memory index is closed")
        connection = getattr(self._thread_connection, "connection", None)
        if connection is None:
            with self._connections_lock:
                if self._closed:
                    raise RuntimeError("memory index is closed")
                connection = sqlite3.connect(
                    f"{self.path.resolve().as_uri()}?mode=rw",
                    uri=True,
                    check_same_thread=False,
                    timeout=10.0,
                )
                connection.execute("PRAGMA busy_timeout=10000")
                connection.execute("PRAGMA synchronous=FULL")
                self._connections.append(connection)
                self._thread_connection.connection = connection
        return connection

    def close(self) -> None:
        self.stop_native_projection()
        self._close_connections()

    def _close_connections(self) -> None:
        with self._connections_lock:
            self._closed = True
            for connection in self._connections:
                connection.close()
            self._connections.clear()

    def close_thread_connection(self) -> None:
        connection = getattr(self._thread_connection, "connection", None)
        with self._connections_lock:
            if connection in self._connections:
                connection.close()
                self._connections.remove(connection)
            for catalog in getattr(self._thread_connection, "catalog_connections", {}).values():
                if catalog in self._connections:
                    catalog.close()
                    self._connections.remove(catalog)
        if connection is not None:
            del self._thread_connection.connection
        self._thread_connection.catalog_connections = {}

    def _topology_catalog_connection(self, profile_id: str) -> sqlite3.Connection | None:
        if self._repo_root is None or not self.workspace_id:
            return None
        key = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()
        path = (
            workspace_memory_root(self._repo_root, self.workspace_id) / "catalogs" / f"{key}.sqlite"
        )
        if not path.exists():
            return None
        connections = getattr(self._thread_connection, "catalog_connections", None)
        if connections is None:
            connections = {}
            self._thread_connection.catalog_connections = connections
        if profile_id not in connections:
            connection = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode=ro",
                uri=True,
                check_same_thread=False,
                timeout=10.0,
            )
            with self._connections_lock:
                self._connections.append(connection)
            connections[profile_id] = connection
        return connections[profile_id]

    @contextmanager
    def topology_snapshot(self, profile_id: str) -> Iterator[None]:
        """Pin one published membership generation without blocking a catalog writer."""

        catalog = self._topology_catalog_connection(profile_id)
        if catalog is None:
            yield
            return
        depths = getattr(self._thread_connection, "topology_snapshot_depths", None)
        if depths is None:
            depths = {}
            self._thread_connection.topology_snapshot_depths = depths
        depth = depths.get(profile_id, 0)
        if depth:
            depths[profile_id] = depth + 1
            try:
                yield
            finally:
                depths[profile_id] = depth
            return
        catalog.execute("BEGIN")
        catalog.execute(
            "SELECT value FROM catalog_metadata WHERE key='catalog_revision'"
        ).fetchone()
        depths[profile_id] = 1
        try:
            yield
        finally:
            depths.pop(profile_id, None)
            catalog.rollback()

    def _published_membership(self, record: MemoryBlockRecord) -> RecordBlockMembership | None:
        catalog = self._topology_catalog_connection(record.profile_id)
        if catalog is None or record.block_membership is None:
            return None
        row = catalog.execute(
            "SELECT block_id,backend_partition_id,catalog_revision_id,record_revision_id "
            "FROM published_record_memberships WHERE record_id=? AND workspace_id=? "
            "AND profile_id=?",
            (record.block_id, record.workspace_id, record.profile_id),
        ).fetchone()
        if row is None or str(row[3]) != record.revision_id:
            return None
        original = record.block_membership
        return replace(
            original,
            primary_block_id=str(row[0]),
            backend_partition_id=str(row[1]),
            catalog_revision_id=str(row[2]),
            secondary_block_ids=tuple(
                block_id for block_id in original.secondary_block_ids if block_id != str(row[0])
            ),
            membership_revision_id="",
        )

    def effective_record(self, record: MemoryBlockRecord) -> MemoryBlockRecord:
        membership = self._published_membership(record)
        return replace(record, block_membership=membership) if membership is not None else record

    @staticmethod
    def _projection_partition(record: MemoryBlockRecord) -> str:
        membership = record.block_membership
        if membership is None:
            raise ValueError("native projection requires primary block membership")
        return hashlib.sha256(
            json.dumps(
                [
                    record.workspace_id,
                    record.profile_id,
                    membership.backend_partition_id,
                ]
            ).encode()
        ).hexdigest()

    @contextmanager
    def canonical_write_gate(self) -> Iterator[None]:
        """Fence recovery against live reserve/prepare/commit, never native I/O."""
        with FileLock(f"{self.path}.canonical.lock", timeout=10.0):
            yield

    @contextmanager
    def descriptor_sync_gate(self, logical_block_id: str) -> Iterator[None]:
        """Serialize only one leaf's derived descriptor, outside the canonical lane."""

        if not logical_block_id:
            raise ValueError("descriptor sync requires a logical block ID")
        key = hashlib.sha256(logical_block_id.encode("utf-8")).hexdigest()
        with FileLock(f"{self.path}.descriptor-{key}.lock", timeout=10.0):
            yield

    @contextmanager
    def atomic_canonical_write_gate(self) -> Iterator[None]:
        """Publish the canonical head, indexes and write outbox in one commit."""
        with self.canonical_write_gate(), self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            self._thread_connection.atomic_canonical_write = True
            try:
                yield
            finally:
                self._thread_connection.atomic_canonical_write = False

    @contextmanager
    def _canonical_transaction_step(self) -> Iterator[None]:
        if getattr(self._thread_connection, "atomic_canonical_write", False):
            yield
        else:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                yield

    @contextmanager
    def projection_gate(
        self, record: MemoryBlockRecord, *, timeout: float = 10.0
    ) -> Iterator[None]:
        key = self._projection_partition(record)
        held: set[str] = getattr(self._projection_lock_state, "held", set())
        if key in held:
            yield
            return
        with FileLock(f"{self.path}.projection-{key}.lock", timeout=timeout):
            self._projection_lock_state.held = held | {key}
            try:
                yield
            finally:
                self._projection_lock_state.held = held

    def configure_native_projection(
        self,
        project: Callable[[MemoryBlockRecord], object],
        *,
        native_gate: Callable[[], AbstractContextManager[object]] = nullcontext,
        start: bool = True,
    ) -> None:
        if self._projection_callback is not None:
            raise RuntimeError("native projection worker is already configured")
        self._projection_callback = project
        self._native_projection_gate = native_gate
        if start:
            self.wake_native_projection()

    def stop_native_projection(self) -> None:
        self._projection_stop.set()
        self.wake_native_projection()
        if self._projection_thread is not None:
            self._projection_thread.join()
            self._projection_thread = None
        # Adapters close native resources immediately after this returns.
        # The dispatcher alone cannot fence active partition workers.
        with self._projection_start_lock:
            workers = tuple(self._projection_workers)
        for worker in workers:
            worker.join()
        with self._projection_start_lock:
            self._projection_workers.difference_update(workers)
        self._projection_active = 0

    def enqueue_native_projection(self, record: MemoryBlockRecord) -> dict[str, Any]:
        if self._projection_callback is None:
            return self.retain_block(record)
        entry = self.latest_write_outbox_entry(
            block_id=record.block_id, workspace_id=record.workspace_id
        )
        if (
            entry is None
            or self.get_block(record.block_id, workspace_id=record.workspace_id) is None
        ):
            raise ValueError("native projection requires a durable canonical record and outbox")
        self.wake_native_projection()
        pending = entry.status != "committed"
        return {
            "ok": True,
            "saved": True,
            "block_id": record.block_id,
            "backend": self.backend_id,
            "commit_revision": entry.commit_revision,
            "durable_status": "committed",
            "projection_pending": pending,
            "native_written": not pending,
            "projection_status": "pending" if pending else "committed",
            "projection_revision": 0 if pending else entry.commit_revision,
            "retryable": pending,
        }

    def wake_native_projection(self) -> None:
        with self._projection_start_lock:
            if (
                self._projection_callback is not None
                and self._projection_thread is None
                and not self._projection_stop.is_set()
            ):
                self._projection_thread = threading.Thread(
                    target=self._run_native_projection,
                    name="um-mem-projection",
                    daemon=True,
                )
                self._projection_thread.start()
        self._projection_wake.set()

    @staticmethod
    def _projection_retry_state(entry: WriteOutboxEntry) -> tuple[int, float]:
        if not entry.error:
            return 0, 0.0
        try:
            state = json.loads(entry.error)
        except json.JSONDecodeError:
            return 0, 0.0
        if not isinstance(state, dict) or state.get("kind") != "native_projection":
            return 0, 0.0
        return int(state["attempts"]), float(state["retry_at"])

    def _project_partition(self, key: str, first: MemoryBlockRecord) -> None:
        waiting_at = time.perf_counter()
        try:
            with self._native_projection_gate(), self.projection_gate(first, timeout=0):
                lock_wait_ms = (time.perf_counter() - waiting_at) * 1000
                for entry in self.pending_write_outbox():
                    if self._projection_stop.is_set():
                        break
                    record = self.write_outbox_record(entry)
                    if self._projection_partition(record) != key:
                        continue
                    # A prepared but uncommitted predecessor must not be overtaken.
                    head_status = self.canonical_head_status(record.block_id)
                    if head_status == "failed":
                        self.fail_write_outbox(entry.commit_revision, error="canonical_head_failed")
                        continue
                    if head_status not in {None, "committed"}:
                        break
                    attempts, retry_at = self._projection_retry_state(entry)
                    if entry.projection_retry_exhausted or retry_at > time.time():
                        break
                    started = time.perf_counter()
                    queued_ms = max(
                        0.0,
                        (time.time() - datetime.fromisoformat(entry.created_at).timestamp()) * 1000,
                    )
                    success = False
                    try:
                        if (
                            self._conn.execute(
                                "SELECT 1 FROM blocks WHERE block_id=? AND workspace_id=?",
                                (record.block_id, record.workspace_id),
                            ).fetchone()
                            is None
                        ):
                            with self.canonical_write_gate():
                                self.commit_write_outbox_record(entry)
                        assert self._projection_callback is not None
                        self._projection_callback(record)
                        if self.write_outbox_status(record.block_id) != "committed":
                            raise RuntimeError(
                                "native projection returned without a committed receipt"
                            )
                        success = True
                    except Exception as exc:  # noqa: BLE001 -- native SDK exceptions cross this worker boundary
                        # Persist every provider failure before bounded retry or operator repair.
                        error = json.dumps(
                            {
                                "kind": "native_projection",
                                "attempts": attempts + 1,
                                "retry_at": time.time() + min(30.0, 0.5 * 4**attempts),
                                "error_type": type(exc).__name__,
                                "error_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                            }
                        )
                        with self._conn:
                            self._conn.execute(
                                "UPDATE write_outbox SET error=?,updated_at=? WHERE commit_revision=? AND status='pending'",
                                (
                                    error,
                                    datetime.now(UTC).isoformat(),
                                    entry.commit_revision,
                                ),
                            )
                        break
                    finally:
                        duration_ms = (time.perf_counter() - started) * 1000
                        if self._repo_root is not None:
                            record_memory_event(
                                self._repo_root,
                                event_type="native_projection",
                                workspace_id=record.workspace_id,
                                profile_id=record.profile_id,
                                run_id=record.run_id,
                                backend=self.backend_id,
                                status="ok" if success else "failed",
                                operation="native_projection",
                                duration_ms=duration_ms,
                                data={
                                    "operation_id": str(record.metadata.get("operation_id") or ""),
                                    "record_id": record.block_id,
                                    "commit_revision": entry.commit_revision,
                                    "projection_revision": entry.commit_revision if success else 0,
                                    "queue_wait_ms": queued_ms,
                                    "lock_wait_ms": lock_wait_ms,
                                    "native_write_ms": duration_ms,
                                    "native_projection_lag_ms": queued_ms + duration_ms,
                                    "retry_count": attempts,
                                    "source_bytes": len(record.content.encode("utf-8")),
                                },
                            )
        except Timeout:
            return  # Another process owns this partition; its receipt remains authoritative.
        finally:
            self.close_thread_connection()

    def _run_native_projection(self) -> None:
        active: dict[str, threading.Thread] = {}
        while not self._projection_stop.is_set():
            self._projection_wake.wait(0.1 if active else 0.5)
            self._projection_wake.clear()
            if self._projection_stop.is_set():
                break
            try:
                for key, worker in list(active.items()):
                    if not worker.is_alive():
                        del active[key]
                        with self._projection_start_lock:
                            self._projection_workers.discard(worker)
                seen: set[str] = set()
                for entry in self.pending_write_outbox():
                    if len(active) >= 4:
                        break
                    record = self.write_outbox_record(entry)
                    key = self._projection_partition(record)
                    if self.canonical_head_status(record.block_id) == "failed":
                        self.fail_write_outbox(entry.commit_revision, error="canonical_head_failed")
                        continue
                    if key in seen or key in active:
                        continue
                    seen.add(key)
                    _attempts, retry_at = self._projection_retry_state(entry)
                    if entry.projection_retry_exhausted or retry_at > time.time():
                        continue
                    if self.canonical_head_status(record.block_id) not in {None, "committed"}:
                        continue
                    worker = threading.Thread(
                        target=self._project_partition,
                        args=(key, record),
                        name=f"um-mem-native-{key[:8]}",
                        daemon=True,
                    )
                    active[key] = worker
                    with self._projection_start_lock:
                        self._projection_workers.add(worker)
                    worker.start()
                self._projection_active = len(active)
                self._projection_error = ""
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
                self._projection_error = type(exc).__name__
        self._projection_active = sum(worker.is_alive() for worker in active.values())

    def native_projection_health(self) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT status,error,created_at FROM write_outbox WHERE status!='committed'"
        ).fetchall()
        pending = [row for row in rows if row[0] == "pending"]
        attention = 0
        for row in pending:
            try:
                state = json.loads(row[1] or "{}")
            except json.JSONDecodeError:
                state = {}
            if (
                isinstance(state, dict)
                and state.get("kind") == "native_projection"
                and int(state.get("attempts", 0)) >= 3
            ):
                attention += 1
        oldest = min(
            (datetime.fromisoformat(row[2]).timestamp() for row in pending), default=time.time()
        )
        return {
            "pending": len(pending),
            "failed": len(rows) - len(pending),
            "attention": attention,
            "active_partitions": self._projection_active,
            "max_workers": 4,
            "worker_error": self._projection_error,
            "oldest_pending_ms": max(0.0, (time.time() - oldest) * 1000),
            "recovery": "reconcile" if attention else "automatic",
        }

    def operational_revisions(self) -> dict[str, str]:
        """Return content-free schema and data-generation revisions.

        The index revision uses SQLite generation counters plus database/WAL
        file metadata. It changes with local or external durable writes without
        scanning or exposing record content, canonical keys, or subject ids.
        """

        paths = (self.path, Path(f"{self.path}-wal"))
        file_generations: list[tuple[str, int, int]] = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                file_generations.append((path.suffix, 0, 0))
            else:
                file_generations.append((path.suffix, stat.st_size, stat.st_mtime_ns))
        payload = json.dumps(
            {
                "data_version": int(self._conn.execute("PRAGMA data_version").fetchone()[0]),
                "file_generations": file_generations,
                "freelist_count": int(self._conn.execute("PRAGMA freelist_count").fetchone()[0]),
                "page_count": int(self._conn.execute("PRAGMA page_count").fetchone()[0]),
                "total_changes": self._conn.total_changes,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        revision = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        return {
            "schema_revision": f"um-mem-backend-index-schema:{SQLITE_INDEX_SCHEMA_VERSION}",
            "index_revision": revision,
            "canonical_snapshot_sha256": revision,
        }

    @staticmethod
    def _validate_existing_schema(path: Path) -> None:
        try:
            conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise ValueError("existing memory SQLite store cannot be validated") from exc
        try:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            if "schema_metadata" not in tables:
                raise ValueError(
                    "existing memory SQLite store is unversioned; create a fresh store"
                )
            version_rows = conn.execute(
                "SELECT key, value FROM schema_metadata ORDER BY key"
            ).fetchall()
            if version_rows != [("schema_version", SQLITE_INDEX_SCHEMA_VERSION)]:
                raise ValueError(
                    "existing memory SQLite store schema version is unsupported; "
                    f"expected {SQLITE_INDEX_SCHEMA_VERSION}; create a fresh store"
                )
            if tables != _CURRENT_SCHEMA_TABLES:
                raise ValueError("existing memory SQLite store schema tables are unsupported")
            expected_columns = {
                "blocks": _CANONICAL_BLOCK_COLUMNS,
                "canonical_payloads": _CANONICAL_PAYLOAD_COLUMNS,
                "write_outbox": _WRITE_OUTBOX_COLUMNS,
                "canonical_head_outbox": _CANONICAL_HEAD_OUTBOX_COLUMNS,
                "canonical_mutations": _CANONICAL_MUTATION_COLUMNS,
                "prospective_obligations": _PROSPECTIVE_OBLIGATION_COLUMNS,
                "prospective_obligation_events": _PROSPECTIVE_EVENT_COLUMNS,
                "memory_exact_terms": _MEMORY_EXACT_TERM_COLUMNS,
                "memory_temporal_index": _MEMORY_TEMPORAL_COLUMNS,
                "memory_provenance_links": _MEMORY_PROVENANCE_COLUMNS,
                "record_block_memberships": _RECORD_BLOCK_MEMBERSHIP_COLUMNS,
                "native_partition_events": _NATIVE_PARTITION_EVENT_COLUMNS,
            }
            for table, expected in expected_columns.items():
                columns = {
                    str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                if columns != expected:
                    raise ValueError(f"existing memory SQLite {table} schema is unsupported")
        except sqlite3.Error as exc:
            raise ValueError("existing memory SQLite store schema is unreadable") from exc
        finally:
            conn.close()

    def _sync_record_block_membership(
        self,
        record: MemoryBlockRecord,
        *,
        is_current: int,
        projected_valid_to: str,
    ) -> None:
        self._conn.execute(
            "DELETE FROM record_block_memberships WHERE record_id=?",
            (record.block_id,),
        )
        membership = record.block_membership
        if membership is None:
            return
        if (
            membership.record_id != record.block_id
            or membership.record_revision_id != record.revision_id
            or membership.workspace_id != record.workspace_id
            or not membership.membership_revision_id
        ):
            raise ValueError("record block membership identity does not match canonical record")
        rows = (
            (membership.primary_block_id, "primary", 0),
            *(
                (block_id, "secondary", ordinal)
                for ordinal, block_id in enumerate(membership.secondary_block_ids, start=1)
            ),
        )
        self._conn.executemany(
            """
            INSERT INTO record_block_memberships(
                workspace_id, record_id, record_revision_id, membership_revision_id,
                logical_block_id, role, ordinal, catalog_revision_id,
                backend_partition_id, valid_from, valid_to, is_current
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    record.workspace_id,
                    record.block_id,
                    record.revision_id,
                    membership.membership_revision_id,
                    block_id,
                    role,
                    ordinal,
                    membership.catalog_revision_id,
                    membership.backend_partition_id,
                    membership.valid_from,
                    projected_valid_to,
                    is_current,
                )
                for block_id, role, ordinal in rows
            ],
        )

    def _sync_memory_indexes(
        self,
        record: MemoryBlockRecord,
        *,
        transfer_envelope: TransferEnvelope | None = None,
    ) -> None:
        document = compile_memory_index_document(
            record,
            transfer_envelope=transfer_envelope,
        )
        self._conn.execute("DELETE FROM memory_exact_terms WHERE block_id=?", (record.block_id,))
        self._conn.executemany(
            """
            INSERT INTO memory_exact_terms(workspace_id, term_type, term_value, block_id)
            VALUES(?,?,?,?)
            """,
            [
                (record.workspace_id, term.term_type, term.term_value, record.block_id)
                for term in document.exact_terms
            ],
        )
        self._conn.execute(
            "DELETE FROM memory_provenance_links WHERE block_id=?",
            (record.block_id,),
        )
        self._conn.executemany(
            """
            INSERT INTO memory_provenance_links(
                workspace_id, derived_id, raw_source_ref, block_id
            ) VALUES(?,?,?,?)
            """,
            [
                (record.workspace_id, derived_id, raw_ref, record.block_id)
                for derived_id, raw_ref in document.reverse_provenance_links
            ],
        )
        for node in document.structure_nodes:
            existing = self.get_index_node(node.node_id, workspace_id=record.workspace_id)
            if existing is not None:
                related = tuple(
                    dict.fromkeys(
                        (
                            *existing.metadata.get("related_record_ids", ()),
                            *node.metadata.get("related_record_ids", ()),
                        )
                    )
                )
                node = replace(
                    node,
                    created_at=existing.created_at,
                    metadata={**node.metadata, "related_record_ids": list(related)},
                )
            self._upsert_index_node(node)

        head = None
        canonical_key_sha256 = ""
        if record.canonical_key:
            canonical_key_sha256 = self._canonical_key_sha256(record.canonical_key)
            head = self._conn.execute(
                """
                SELECT block_id FROM canonical_heads
                WHERE workspace_id=? AND canonical_key_sha256=?
                """,
                (record.workspace_id, canonical_key_sha256),
            ).fetchone()
        is_current = int(
            (head is None or str(head[0]) == record.block_id)
            and record.status
            not in {"quarantined", "superseded", "deprecated", "retracted", "expired"}
        )
        # Keep the correction boundary in derived indexes even when the source
        # has no event start and cannot carry a fully bounded record interval.
        projected_valid_to = record.valid_to or (
            self._successor_valid_from(record) if not is_current else ""
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memory_temporal_index(
                workspace_id, block_id, canonical_key_sha256, revision_id,
                valid_from, valid_to, recorded_at, is_current
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                record.workspace_id,
                record.block_id,
                canonical_key_sha256,
                record.revision_id,
                record.valid_from,
                projected_valid_to,
                record.recorded_at,
                is_current,
            ),
        )
        self._sync_record_block_membership(
            record, is_current=is_current, projected_valid_to=projected_valid_to
        )
        self._conn.execute("DELETE FROM memory_fts WHERE block_id=?", (record.block_id,))
        self._conn.execute(
            """
            INSERT INTO memory_fts(block_id, workspace_id, title, content, tags)
            VALUES(?,?,?,?,?)
            """,
            (
                record.block_id,
                record.workspace_id,
                record.title,
                record.content,
                " ".join(record.tags),
            ),
        )

    def refresh_memory_indexes(
        self,
        block_id: str,
        *,
        transfer_envelope: TransferEnvelope | None = None,
    ) -> dict[str, Any]:
        record = self.get_block(block_id, workspace_id=self.workspace_id)
        if record is None:
            raise ValueError("memory index refresh requires a visible canonical record")
        with self._conn:
            self._sync_memory_indexes(record, transfer_envelope=transfer_envelope)
        return {"ok": True, "block_id": block_id, "backend": self.backend_id}

    def rebuild_memory_indexes(self, *, catalog: SemanticCatalog) -> dict[str, Any]:
        """Rebuild record-derived indexes without mutating canonical records."""

        if not isinstance(catalog, SemanticCatalog):
            raise TypeError("memory index rebuild requires a SemanticCatalog")
        records = sorted(
            (
                self._project_valid_to(record)
                for record in self.export_blocks()
                if not self.workspace_id or record.workspace_id == self.workspace_id
            ),
            key=lambda record: record.block_id,
        )
        workspace_id = self.workspace_id
        with self._conn:
            self._conn.execute(
                "DELETE FROM memory_exact_terms WHERE workspace_id=?",
                (workspace_id,),
            )
            self._conn.execute(
                "DELETE FROM memory_temporal_index WHERE workspace_id=?",
                (workspace_id,),
            )
            self._conn.execute(
                "DELETE FROM memory_provenance_links WHERE workspace_id=?",
                (workspace_id,),
            )
            self._conn.execute(
                "DELETE FROM record_block_memberships WHERE workspace_id=?",
                (workspace_id,),
            )
            self._conn.execute(
                "DELETE FROM memory_fts WHERE workspace_id=?",
                (workspace_id,),
            )
            self._conn.execute(
                """
                DELETE FROM index_nodes
                WHERE workspace_id=?
                  AND node_type IN ('project_structure', 'block_descriptor', 'exact_facet')
                """,
                (workspace_id,),
            )
            for record in records:
                self._sync_memory_indexes(record)
        for record in records:
            sync_block_descriptor_index(
                self,
                catalog=catalog,
                record=record,
                workspace_id=workspace_id,
                update_reason="rebuild",
            )
        return {
            "ok": True,
            "backend": self.backend_id,
            "workspace_id": workspace_id,
            "record_count": len(records),
            "index_sha256": self._memory_index_sha256(),
        }

    def _memory_index_sha256(self) -> str:
        workspace_id = self.workspace_id
        payload = {
            "exact": self._conn.execute(
                """
                SELECT term_type, term_value, block_id FROM memory_exact_terms
                WHERE workspace_id=? ORDER BY term_type, term_value, block_id
                """,
                (workspace_id,),
            ).fetchall(),
            "temporal": self._conn.execute(
                """
                SELECT block_id, canonical_key_sha256, revision_id, valid_from, valid_to,
                       recorded_at, is_current
                FROM memory_temporal_index WHERE workspace_id=? ORDER BY block_id
                """,
                (workspace_id,),
            ).fetchall(),
            "provenance": self._conn.execute(
                """
                SELECT derived_id, raw_source_ref, block_id FROM memory_provenance_links
                WHERE workspace_id=? ORDER BY derived_id, raw_source_ref, block_id
                """,
                (workspace_id,),
            ).fetchall(),
            "memberships": self._conn.execute(
                """
                SELECT record_id, record_revision_id, membership_revision_id,
                       logical_block_id, role, ordinal, catalog_revision_id,
                       backend_partition_id, valid_from, valid_to, is_current
                FROM record_block_memberships
                WHERE workspace_id=?
                ORDER BY record_id, ordinal, logical_block_id
                """,
                (workspace_id,),
            ).fetchall(),
            "text": self._conn.execute(
                """
                SELECT block_id, title, content, tags FROM memory_fts
                WHERE workspace_id=? ORDER BY block_id
                """,
                (workspace_id,),
            ).fetchall(),
            "nodes": self._conn.execute(
                """
                SELECT node_id, payload_json FROM index_nodes
                WHERE workspace_id=?
                  AND node_type IN ('project_structure', 'block_descriptor', 'exact_facet')
                ORDER BY node_id
                """,
                (workspace_id,),
            ).fetchall(),
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def exact_index_terms(self, block_id: str) -> dict[str, tuple[str, ...]]:
        rows = self._conn.execute(
            """
            SELECT term_type, term_value FROM memory_exact_terms
            WHERE block_id=? ORDER BY term_type, term_value
            """,
            (block_id,),
        ).fetchall()
        grouped: dict[str, list[str]] = {}
        for term_type, term_value in rows:
            grouped.setdefault(str(term_type), []).append(str(term_value))
        return {key: tuple(values) for key, values in grouped.items()}

    def exact_reference_block_ids(self, reference: str, *, workspace_id: str) -> tuple[str, ...]:
        rows = self._conn.execute(
            """
            SELECT block_id FROM memory_exact_terms
            WHERE workspace_id=? AND term_type='evidence_ref' AND term_value=?
            ORDER BY block_id
            """,
            (workspace_id, reference),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def scoped_record_ids(
        self,
        *,
        workspace_id: str,
        profile_id: str,
        scope_key: str,
        logical_block_id: str = "",
    ) -> tuple[str, ...]:
        params: list[str] = [workspace_id, profile_id, workspace_id, scope_key]
        rows = self._conn.execute(
            """
            SELECT profile.block_id FROM memory_exact_terms AS profile
            WHERE profile.workspace_id=? AND profile.term_type='profile_id'
              AND profile.term_value=?
              AND EXISTS (
                  SELECT 1 FROM memory_exact_terms AS scope
                  WHERE scope.workspace_id=? AND scope.block_id=profile.block_id
                    AND scope.term_type='scope_key' AND scope.term_value=?
              )
            ORDER BY profile.block_id
            """,
            params,
        ).fetchall()
        ids = tuple(str(row[0]) for row in rows)
        if not logical_block_id:
            return ids
        selected = set(
            self.record_ids_for_logical_blocks(
                (logical_block_id,), workspace_id=workspace_id, profile_id=profile_id
            )
        )
        return tuple(record_id for record_id in ids if record_id in selected)

    def project_structure_nodes(self, block_id: str) -> list[MemoryIndexNode]:
        order = {"repository": 0, "package": 1, "component": 2, "file": 3, "symbol": 4}
        nodes = [
            node
            for node in self.select_index_nodes(
                workspace_id=self.workspace_id,
                node_type="project_structure",
                limit=500,
            )
            if block_id in node.metadata.get("related_record_ids", ())
        ]
        return sorted(
            nodes,
            key=lambda node: order.get(str(node.metadata.get("structure_kind")), 99),
        )

    def reverse_provenance_sources(self, derived_id: str) -> tuple[str, ...]:
        rows = self._conn.execute(
            """
            SELECT DISTINCT raw_source_ref FROM memory_provenance_links
            WHERE workspace_id=? AND derived_id=? ORDER BY raw_source_ref
            """,
            (self.workspace_id, derived_id),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def temporal_index_entries(self, canonical_key: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT block_id, revision_id, valid_from, valid_to, recorded_at, is_current
            FROM memory_temporal_index
            WHERE workspace_id=? AND canonical_key_sha256=?
            ORDER BY valid_from, recorded_at, block_id
            """,
            (self.workspace_id, self._canonical_key_sha256(canonical_key)),
        ).fetchall()
        return [
            {
                "block_id": str(row[0]),
                "revision_id": str(row[1]),
                "valid_from": str(row[2]),
                "valid_to": str(row[3]),
                "recorded_at": str(row[4]),
                "is_current": bool(row[5]),
            }
            for row in rows
        ]

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        ensure_record_is_storable(record)
        if not record.workspace_id and self.workspace_id:
            record = replace(record, workspace_id=self.workspace_id)
        reservation = self._pending_canonical_head_reservation(record.block_id)
        payload = block_to_json(record)
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._conn.execute("BEGIN IMMEDIATE")
        existing = self._conn.execute(
            "SELECT payload_json FROM blocks WHERE block_id=?", (record.block_id,)
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload_json:
                self._conn.rollback()
                raise ValueError("canonical memory revisions are immutable")
            try:
                self._sync_memory_indexes(record)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return {
                "ok": True,
                "saved": True,
                "replayed": True,
                "block_id": record.block_id,
                "backend": self.backend_id,
            }
        try:
            self._conn.execute(
                """
                INSERT INTO blocks(
                    block_id, workspace_id, run_id, stage_id, subtask_id, scope,
                    status, retention, delivery_policy, temperature, authority,
                    trust_level, verification_status, promotion_decision,
                    content, payload_json, updated_at, committed_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.block_id,
                    record.workspace_id,
                    record.run_id,
                    record.stage_id,
                    record.subtask_id,
                    record.scope,
                    record.status,
                    record.retention,
                    record.delivery_policy,
                    record.temperature,
                    record.authority,
                    record.trust_level,
                    record.verification_status,
                    record.promotion_decision,
                    record.content,
                    payload_json,
                    record.updated_at,
                    datetime.now(UTC).isoformat(),
                ),
            )
            canonical_key = record.canonical_key
            if canonical_key and record.status == "accepted" and reservation is None:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO canonical_heads(
                        workspace_id, canonical_key_sha256, block_id, commit_revision
                    ) VALUES(?,?,?,0)
                    """,
                    (
                        record.workspace_id,
                        self._canonical_key_sha256(canonical_key),
                        record.block_id,
                    ),
                )
            self._sync_memory_indexes(record)
            evicted = self._evict_over_cap()
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        result: dict[str, Any] = {
            "ok": True,
            "saved": True,
            "block_id": record.block_id,
            "backend": self.backend_id,
        }
        if evicted:
            result["evicted"] = evicted
        return result

    def prepare_canonical_write(self, record: MemoryBlockRecord) -> dict[str, Any]:
        ensure_record_is_storable(record)
        entry = self.open_write_outbox(record=record, store=self.backend_id)
        return {"ok": True, "commit_revision": entry.commit_revision}

    def get_block(self, block_id: str, *, workspace_id: str = "") -> MemoryBlockRecord | None:
        canonical_status = self.canonical_head_status(block_id)
        if canonical_status is not None and canonical_status != "committed":
            return None
        write_status = self.write_outbox_status(block_id)
        if write_status == "failed":
            return None
        pending_write = write_status == "pending"
        if pending_write and canonical_status != "committed":
            return None
        row = self._conn.execute(
            "SELECT workspace_id, payload_json FROM blocks WHERE block_id=?",
            (str(block_id),),
        ).fetchone()
        if row is None:
            return None
        if workspace_id and row[0] and row[0] != workspace_id:
            return None
        data = json.loads(row[1])
        record = block_from_json(data)
        return self.effective_record(self._project_valid_to(record))

    def _successor_valid_from(self, record: MemoryBlockRecord) -> str:
        if not record.canonical_key:
            return ""
        row = self._conn.execute(
            """
            SELECT successor_id, record_payload_id, record_payload_sha256
            FROM canonical_mutations
            WHERE workspace_id=? AND canonical_key_sha256=? AND predecessor_id=?
            ORDER BY commit_revision
            LIMIT 1
            """,
            (
                record.workspace_id or self.workspace_id,
                self._canonical_key_sha256(record.canonical_key),
                record.block_id,
            ),
        ).fetchone()
        if row is None:
            return ""
        successor = block_from_json(
            self._load_payload(str(row[1]), str(row[2]), payload_kind="record")
        )
        if successor.block_id != str(row[0]):
            raise ValueError("canonical mutation record identity is invalid")
        return successor.valid_from

    def _project_valid_to(self, record: MemoryBlockRecord) -> MemoryBlockRecord:
        boundary = self._successor_valid_from(record)
        boundary_time = _parse_time(boundary)
        explicit_time = _parse_time(record.valid_to)
        if boundary_time is None or (
            record.valid_to and explicit_time is not None and explicit_time <= boundary_time
        ):
            return record
        if not record.valid_from:
            return record
        return replace(
            record,
            valid_to=boundary,
        )

    def overlay_native_hit(
        self,
        record: MemoryBlockRecord,
        *,
        workspace_id: str = "",
        query: dict[str, Any] | None = None,
    ) -> MemoryBlockRecord | None:
        """Return a native hit only when its current sidecar contract is valid."""

        target_workspace = workspace_id or self.workspace_id
        if target_workspace and record.workspace_id != target_workspace:
            return None
        if str((query or {}).get("temporal_mode") or "current") != "current":
            return None
        if str((query or {}).get("as_of") or "").strip():
            return None
        journal_row = self._conn.execute(
            """
            SELECT commit_revision, block_id, workspace_id, store, payload_id, payload_sha256,
                   status, error, created_at, updated_at
            FROM write_outbox
            WHERE block_id=? AND workspace_id=? AND status='committed'
            ORDER BY commit_revision DESC LIMIT 1
            """,
            (record.block_id, target_workspace),
        ).fetchone()
        if journal_row is None:
            return None
        entry = self._outbox_from_row(journal_row)
        stored = self.get_block(record.block_id, workspace_id=target_workspace)
        if stored is None:
            return None
        try:
            journal_record = self.write_outbox_record(entry)
        except ValueError:
            return None
        if self.effective_record(journal_record) != stored or stored != record:
            return None
        now = datetime.now(UTC)
        valid_from = _parse_time(stored.valid_from)
        valid_to = _parse_time(stored.valid_to)
        if (
            stored.status not in {"accepted", "candidate"}
            or (valid_from is not None and valid_from > now)
            or (valid_to is not None and valid_to <= now)
            or self._is_expired(stored, now=now)
        ):
            return None
        if is_protected_memory(stored.sensitivity, stored.content_mode, stored.operational_risk):
            return None
        payload = dict(query or {})
        payload["workspace_id"] = target_workspace
        payload["temporal_mode"] = "current"
        if not record_allowed_for_query(stored, payload):
            return None
        if stored.canonical_key:
            head = self._conn.execute(
                """
                SELECT block_id FROM canonical_heads
                WHERE workspace_id=? AND canonical_key_sha256=?
                """,
                (stored.workspace_id, self._canonical_key_sha256(stored.canonical_key)),
            ).fetchone()
            if head is None or str(head[0]) != stored.block_id:
                return None
        if (
            self.validate_dependencies(
                stored.block_id, workspace_id=stored.workspace_id, profile_id=stored.profile_id
            ).status
            != "valid"
        ):
            return None
        fields = {
            name: getattr(stored, name)
            for name in MemoryBlockRecord.__dataclass_fields__
            if name != "metadata"
        }
        fields["metadata"] = {**dict(record.metadata), **dict(stored.metadata)}
        return replace(record, **fields)

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
        """Check a bounded exact derivation graph without changing accepted history."""
        if type(max_nodes) is not int or not 1 <= max_nodes <= 4096:
            raise ValueError("dependency max_nodes must be between 1 and 4096")
        if type(max_edges) is not int or not 1 <= max_edges <= 16384:
            raise ValueError("dependency max_edges must be between 1 and 16384")
        target_workspace = workspace_id or self.workspace_id
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("dependency validation requires an exact record ID")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("dependency validation requires a profile ID")
        if not isinstance(as_of, str):
            raise ValueError("dependency as_of requires an ISO-8601 timestamp")
        instant = _parse_time(as_of) if as_of else datetime.now(UTC)
        if instant is None:
            raise ValueError("dependency as_of requires an ISO-8601 timestamp")
        pending: list[tuple[str, frozenset[str]]] = [(record_id, frozenset())]
        visited: set[str] = set()
        checked: list[str] = []
        issues: list[DependencyIssue] = []
        edge_count = 0
        incomplete = False
        while pending:
            current_id, ancestors = pending.pop()
            if current_id in ancestors:
                issues.append(DependencyIssue(current_id, "dependency_cycle"))
                incomplete = True
                continue
            if current_id in visited:
                continue
            if len(visited) >= max_nodes:
                issues.append(DependencyIssue(record_id, "dependency_budget_exhausted"))
                incomplete = True
                break
            visited.add(current_id)
            source = self.get_block(current_id, workspace_id=target_workspace)
            if source is None:
                # An index-only witness has no canonical record lifecycle. Its
                # evidence quality remains the explicit verifier's responsibility.
                if (
                    current_id != record_id
                    and self.get_index_node(current_id, workspace_id=target_workspace) is not None
                ):
                    continue
                issues.append(DependencyIssue(current_id, "record_unavailable"))
                continue
            if source.profile_id != profile_id:
                issues.append(DependencyIssue(current_id, "profile_mismatch"))
                continue
            checked.append(current_id)
            if source.status not in {"accepted", "candidate"}:
                issues.append(DependencyIssue(current_id, "lifecycle_invalid"))
            if (
                source.canonical_key
                and not as_of
                and not self.record_matches_temporal(source, {"temporal_mode": "current"})
            ):
                issues.append(DependencyIssue(current_id, "revision_not_current"))
            if not self.record_matches_temporal(
                source, {"temporal_mode": "historical", "as_of": instant.isoformat()}
            ):
                issues.append(DependencyIssue(current_id, "outside_validity"))
            rows = self._conn.execute(
                """
                SELECT payload_json FROM edges
                WHERE workspace_id=? AND src_id=? AND edge_type=?
                ORDER BY dst_id LIMIT ?
                """,
                (target_workspace, current_id, EdgeType.DERIVED_FROM, max_edges - edge_count + 1),
            ).fetchall()
            edge_count += len(rows)
            if edge_count > max_edges:
                issues.append(DependencyIssue(record_id, "dependency_budget_exhausted"))
                incomplete = True
                break
            for edge in self._edge_records(rows):
                created_at = _parse_time(edge.created_at)
                if as_of and created_at is not None and created_at > instant:
                    continue
                pending.append((edge.dst_id, ancestors | {current_id}))
        return DependencyValidation(
            record_id=record_id,
            workspace_id=target_workspace,
            profile_id=profile_id,
            as_of=instant.isoformat(),
            status="incomplete" if incomplete else ("invalidated" if issues else "valid"),
            checked_record_ids=tuple(checked),
            issues=tuple(dict.fromkeys(issues)),
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    @staticmethod
    def _payload_identity(payload_kind: str, payload_json: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{payload_kind}\x00{payload_json}".encode()).hexdigest()
        return f"{payload_kind}:{digest}", digest

    def _store_payload(self, payload_kind: str, payload_json: str) -> tuple[str, str]:
        payload_id, payload_sha256 = self._payload_identity(payload_kind, payload_json)
        row = self._conn.execute(
            """
            SELECT payload_sha256, payload_kind, payload_json
            FROM canonical_payloads WHERE payload_id=?
            """,
            (payload_id,),
        ).fetchone()
        if row is not None:
            if tuple(str(value) for value in row) != (
                payload_sha256,
                payload_kind,
                payload_json,
            ):
                raise ValueError("canonical payload identity mismatch")
            return payload_id, payload_sha256
        self._conn.execute(
            """
            INSERT INTO canonical_payloads(
                payload_id, payload_sha256, payload_kind, payload_json, created_at
            ) VALUES(?,?,?,?,?)
            """,
            (
                payload_id,
                payload_sha256,
                payload_kind,
                payload_json,
                datetime.now(UTC).isoformat(),
            ),
        )
        return payload_id, payload_sha256

    def _load_payload(
        self,
        payload_id: str,
        payload_sha256: str,
        *,
        payload_kind: str,
    ) -> Any:
        row = self._conn.execute(
            """
            SELECT payload_sha256, payload_kind, payload_json
            FROM canonical_payloads WHERE payload_id=?
            """,
            (payload_id,),
        ).fetchone()
        if row is None:
            raise ValueError("canonical payload is missing")
        stored_sha256, stored_kind, payload_json = (str(value) for value in row)
        expected_id, expected_sha256 = self._payload_identity(stored_kind, payload_json)
        if (
            stored_kind != payload_kind
            or payload_id != expected_id
            or payload_sha256 != stored_sha256
            or stored_sha256 != expected_sha256
        ):
            raise ValueError("canonical payload linkage is invalid")
        try:
            return json.loads(payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("canonical payload is invalid") from exc

    @staticmethod
    def _outbox_from_row(row: tuple[Any, ...]) -> WriteOutboxEntry:
        return WriteOutboxEntry(
            commit_revision=int(row[0]),
            block_id=str(row[1]),
            workspace_id=str(row[2]),
            store=str(row[3]),
            payload_id=str(row[4]),
            payload_sha256=str(row[5]),
            status=str(row[6]),
            error=str(row[7]),
            created_at=str(row[8]),
            updated_at=str(row[9]),
        )

    def _record_payload_identity(self, record: MemoryBlockRecord) -> tuple[str, str, str]:
        payload_json = json.dumps(
            block_to_json(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        payload_id, payload_sha256 = self._payload_identity("record", payload_json)
        return payload_id, payload_sha256, payload_json

    @staticmethod
    def _write_retry_equivalent(
        existing: MemoryBlockRecord,
        candidate: MemoryBlockRecord,
    ) -> bool:
        existing_payload = block_to_json(existing)
        candidate_payload = block_to_json(candidate)
        for field_name in ("created_at", "updated_at"):
            existing_payload.pop(field_name, None)
            candidate_payload.pop(field_name, None)
        return existing_payload == candidate_payload

    def open_write_outbox(
        self,
        *,
        record: MemoryBlockRecord,
        store: str,
    ) -> WriteOutboxEntry:
        ensure_record_is_storable(record)
        if not record.workspace_id and self.workspace_id:
            record = replace(record, workspace_id=self.workspace_id)
        if not record.block_id or not store:
            raise ValueError("write outbox requires block and store")
        payload_id, payload_sha256, payload_json = self._record_payload_identity(record)
        now = datetime.now(UTC).isoformat()
        with self._canonical_transaction_step():
            row = self._conn.execute(
                """
                SELECT commit_revision, block_id, workspace_id, store, payload_id, payload_sha256,
                       status, error, created_at, updated_at
                FROM write_outbox WHERE block_id=? ORDER BY commit_revision DESC LIMIT 1
                """,
                (record.block_id,),
            ).fetchone()
            if row is not None:
                entry = self._outbox_from_row(row)
                if entry.workspace_id != record.workspace_id or entry.store != store:
                    raise ValueError(
                        "selected backend write identity conflicts with existing journal"
                    )
                existing_record = block_from_json(
                    self._load_payload(
                        entry.payload_id,
                        entry.payload_sha256,
                        payload_kind="record",
                    )
                )
                if (
                    entry.payload_id != payload_id or entry.payload_sha256 != payload_sha256
                ) and not self._write_retry_equivalent(existing_record, record):
                    raise ValueError(
                        "selected backend write identity conflicts with existing journal"
                    )
                return entry
            self._store_payload("record", payload_json)
            cursor = self._conn.execute(
                """
                INSERT INTO write_outbox(
                    block_id, workspace_id, store, payload_id, payload_sha256,
                    status, error, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.block_id,
                    record.workspace_id,
                    store,
                    payload_id,
                    payload_sha256,
                    "pending",
                    "",
                    now,
                    now,
                ),
            )
        if cursor.lastrowid is None:
            raise RuntimeError("write journal insert produced no revision")
        return WriteOutboxEntry(
            commit_revision=cursor.lastrowid,
            block_id=record.block_id,
            workspace_id=record.workspace_id,
            store=store,
            payload_id=payload_id,
            payload_sha256=payload_sha256,
            status="pending",
            error="",
            created_at=now,
            updated_at=now,
        )

    def pending_write_outbox(self) -> list[WriteOutboxEntry]:
        rows = self._conn.execute(
            """
            SELECT commit_revision, block_id, workspace_id, store, payload_id, payload_sha256,
                   status, error, created_at, updated_at
            FROM write_outbox
            WHERE status='pending'
            ORDER BY commit_revision
            """
        ).fetchall()
        return [self._outbox_from_row(row) for row in rows]

    def reconcile_write_outbox(self) -> list[WriteOutboxEntry]:
        rows = self._conn.execute(
            """
            SELECT commit_revision, block_id, workspace_id, store, payload_id, payload_sha256,
                   status, error, created_at, updated_at
            FROM write_outbox
            WHERE status IN ('pending', 'failed')
            ORDER BY commit_revision
            """
        ).fetchall()
        return [self._outbox_from_row(row) for row in rows]

    def _reconcile_entry_statuses(
        self,
        *,
        canonical_head_revisions: set[int],
        write_revisions: set[int],
    ) -> tuple[dict[int, str], dict[int, str]]:
        def statuses(table: str, revisions: set[int]) -> dict[int, str]:
            if not revisions:
                return {}
            placeholders = ",".join("?" * len(revisions))
            rows = self._conn.execute(
                f"SELECT commit_revision, status FROM {table} "
                f"WHERE commit_revision IN ({placeholders})",
                sorted(revisions),
            ).fetchall()
            return {int(row[0]): str(row[1]) for row in rows}

        return (
            statuses("canonical_head_outbox", canonical_head_revisions),
            statuses("write_outbox", write_revisions),
        )

    def write_outbox_status(self, block_id: str) -> str | None:
        row = self._conn.execute(
            """
            SELECT status FROM write_outbox
            WHERE block_id=?
            ORDER BY commit_revision DESC
            LIMIT 1
            """,
            (block_id,),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def write_outbox_entry(
        self,
        *,
        block_id: str,
        commit_revision: int,
    ) -> WriteOutboxEntry | None:
        row = self._conn.execute(
            """
            SELECT commit_revision, block_id, workspace_id, store, payload_id, payload_sha256,
                   status, error, created_at, updated_at
            FROM write_outbox
            WHERE block_id=? AND commit_revision=?
            """,
            (block_id, commit_revision),
        ).fetchone()
        if row is None:
            return None
        return self._outbox_from_row(row)

    def latest_write_outbox_entry(
        self,
        *,
        block_id: str,
        workspace_id: str = "",
    ) -> WriteOutboxEntry | None:
        params: list[Any] = [block_id]
        workspace_clause = ""
        if workspace_id:
            workspace_clause = " AND workspace_id=?"
            params.append(workspace_id)
        row = self._conn.execute(
            f"""
            SELECT commit_revision, block_id, workspace_id, store, payload_id, payload_sha256,
                   status, error, created_at, updated_at
            FROM write_outbox
            WHERE block_id=?{workspace_clause}
            ORDER BY commit_revision DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        return self._outbox_from_row(row) if row is not None else None

    def write_outbox_record(self, entry: WriteOutboxEntry) -> MemoryBlockRecord:
        actual = self.write_outbox_entry(
            block_id=entry.block_id,
            commit_revision=entry.commit_revision,
        )
        if actual is None or (
            actual.workspace_id != entry.workspace_id
            or actual.store != entry.store
            or actual.payload_id != entry.payload_id
            or actual.payload_sha256 != entry.payload_sha256
        ):
            raise ValueError("selected backend write journal identity is invalid")
        payload = self._load_payload(
            actual.payload_id,
            actual.payload_sha256,
            payload_kind="record",
        )
        try:
            record = block_from_json(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("selected backend write payload is invalid") from exc
        if record.block_id != actual.block_id or record.workspace_id != actual.workspace_id:
            raise ValueError("selected backend write payload identity is invalid")
        return record

    def mark_write_outbox_projection_started(self, commit_revision: int) -> None:
        """Persist the native-call boundary before an irreversible provider write."""

        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT status,error FROM write_outbox WHERE commit_revision=?",
                (commit_revision,),
            ).fetchone()
            if row is None or str(row[0]) != "pending":
                raise RuntimeError("native projection requires a pending write outbox")
            if not row[1]:
                self._conn.execute(
                    "UPDATE write_outbox SET error=?,updated_at=? WHERE commit_revision=?",
                    (
                        json.dumps(
                            {
                                "kind": "native_projection",
                                "attempts": 1,
                                "retry_at": 0,
                                "state": "started",
                            }
                        ),
                        datetime.now(UTC).isoformat(),
                        commit_revision,
                    ),
                )

    def commit_write_outbox_record(self, entry: WriteOutboxEntry) -> dict[str, Any]:
        actual = self.write_outbox_entry(
            block_id=entry.block_id,
            commit_revision=entry.commit_revision,
        )
        if actual is None or actual.status not in {"pending", "committed"}:
            raise RuntimeError("selected backend write journal cannot be committed")
        record = self.write_outbox_record(actual)
        if actual.status == "committed":
            return {
                "ok": True,
                "saved": True,
                "replayed": True,
                "block_id": record.block_id,
                "backend": self.backend_id,
            }
        return self.retain_block(record)

    def set_write_outbox_native_ref(self, commit_revision: int, native_id: str) -> None:
        native_id = native_id.strip()
        if not native_id:
            raise ValueError("selected backend native reference requires an id")
        now = datetime.now(UTC).isoformat()
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            outbox = self._conn.execute(
                "SELECT status FROM write_outbox WHERE commit_revision=?",
                (commit_revision,),
            ).fetchone()
            if outbox is None or str(outbox[0]) != "pending":
                raise RuntimeError("selected backend native reference requires a pending outbox")
            existing = self._conn.execute(
                "SELECT native_id FROM write_outbox_native_refs WHERE commit_revision=?",
                (commit_revision,),
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    """
                    INSERT INTO write_outbox_native_refs(commit_revision, native_id, created_at)
                    VALUES(?,?,?)
                    """,
                    (commit_revision, native_id, now),
                )
            elif str(existing[0]) != native_id:
                raise RuntimeError("selected backend native reference is immutable")

    def write_outbox_native_id(self, commit_revision: int) -> str | None:
        row = self._conn.execute(
            "SELECT native_id FROM write_outbox_native_refs WHERE commit_revision=?",
            (commit_revision,),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def committed_write_outbox_native_id(
        self, *, block_id: str, workspace_id: str, store: str
    ) -> str | None:
        row = self._conn.execute(
            """
            SELECT native_refs.native_id
            FROM write_outbox AS outbox
            JOIN write_outbox_native_refs AS native_refs
              ON native_refs.commit_revision=outbox.commit_revision
            WHERE outbox.block_id=? AND outbox.workspace_id=? AND outbox.store=?
              AND outbox.status='committed'
            ORDER BY outbox.commit_revision DESC
            LIMIT 1
            """,
            (block_id, workspace_id, store),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def complete_write_outbox(self, commit_revision: int) -> None:
        now = datetime.now(UTC).isoformat()
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE write_outbox
                SET status='committed', error='', updated_at=?
                WHERE commit_revision=? AND status='pending'
                """,
                (now, commit_revision),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("selected backend write outbox completion rejected")

    def fail_write_outbox(self, commit_revision: int, *, error: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE write_outbox
                SET status='failed', error=?, updated_at=?
                WHERE commit_revision=? AND status='pending'
                """,
                (error, now, commit_revision),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("selected backend write outbox failure transition rejected")

    @staticmethod
    def _prospective_payload_json(obligation: ProspectiveObligation) -> str:
        return json.dumps(
            {
                "obligation_id": obligation.obligation_id,
                "scope": obligation.scope,
                "scope_id": obligation.scope_id,
                "evidence_refs": [asdict(ref) for ref in obligation.evidence_refs],
                "required_intervention": obligation.required_intervention,
                "trigger_predicates": [asdict(value) for value in obligation.trigger_predicates],
                "semantic_trigger_cues": list(obligation.semantic_trigger_cues),
                "earliest_at": obligation.earliest_at,
                "deadline_at": obligation.deadline_at,
                "expires_at": obligation.expires_at,
                "recurrence_policy": obligation.recurrence_policy,
                "recurrence_interval_seconds": obligation.recurrence_interval_seconds,
                "suppress_predicates": [asdict(value) for value in obligation.suppress_predicates],
                "cancel_predicates": [asdict(value) for value in obligation.cancel_predicates],
                "satisfaction_verifier": obligation.satisfaction_verifier,
                "satisfaction_verifier_result": obligation.satisfaction_verifier_result,
                "satisfaction_evidence_refs": [
                    asdict(ref) for ref in obligation.satisfaction_evidence_refs
                ],
                "telemetry": asdict(obligation.telemetry),
                "status": obligation.status,
                "transition_from": obligation.transition_from,
                "transition_evidence_refs": [
                    asdict(ref) for ref in obligation.transition_evidence_refs
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _prospective_from_payload(payload: Any) -> ProspectiveObligation:
        if not isinstance(payload, dict):
            raise ValueError("prospective obligation payload is invalid")
        try:
            verifier_result = str(payload.get("satisfaction_verifier_result", ""))
            if verifier_result not in {"", "passed", "failed"}:
                raise ValueError("prospective satisfaction verifier result is invalid")
            return ProspectiveObligation(
                obligation_id=str(payload["obligation_id"]),
                scope=cast(
                    ProspectiveScope,
                    _closed_value(
                        str(payload["scope"]), field_name="scope", shape=ProspectiveScope
                    ),
                ),
                scope_id=str(payload.get("scope_id", "")),
                evidence_refs=tuple(
                    EvidenceRef.from_mapping(dict(value)) for value in payload["evidence_refs"]
                ),
                required_intervention=str(payload.get("required_intervention", "")),
                trigger_predicates=tuple(
                    ProspectiveTriggerPredicate(**dict(value))
                    for value in payload.get("trigger_predicates", ())
                ),
                semantic_trigger_cues=tuple(payload.get("semantic_trigger_cues", ())),
                earliest_at=str(payload.get("earliest_at", "")),
                deadline_at=str(payload.get("deadline_at", "")),
                expires_at=str(payload.get("expires_at", "")),
                recurrence_policy=cast(
                    ProspectiveRecurrencePolicy,
                    _closed_value(
                        str(payload.get("recurrence_policy", "once")),
                        field_name="recurrence_policy",
                        shape=ProspectiveRecurrencePolicy,
                    ),
                ),
                recurrence_interval_seconds=int(payload.get("recurrence_interval_seconds", 0)),
                suppress_predicates=tuple(
                    ProspectiveTriggerPredicate(**dict(value))
                    for value in payload.get("suppress_predicates", ())
                ),
                cancel_predicates=tuple(
                    ProspectiveTriggerPredicate(**dict(value))
                    for value in payload.get("cancel_predicates", ())
                ),
                satisfaction_verifier=str(payload.get("satisfaction_verifier", "")),
                satisfaction_verifier_result=cast(Literal["", "passed", "failed"], verifier_result),
                satisfaction_evidence_refs=tuple(
                    EvidenceRef.from_mapping(dict(value))
                    for value in payload.get("satisfaction_evidence_refs", ())
                ),
                telemetry=ProspectiveTelemetry(**dict(payload.get("telemetry", {}))),
                status=cast(
                    ProspectiveStatus,
                    _closed_value(
                        str(payload["status"]), field_name="status", shape=ProspectiveStatus
                    ),
                ),
                transition_from=(
                    cast(
                        ProspectiveStatus,
                        _closed_value(
                            payload["transition_from"],
                            field_name="transition_from",
                            shape=ProspectiveStatus,
                        ),
                    )
                    if payload["transition_from"] is not None
                    else None
                ),
                transition_evidence_refs=tuple(
                    EvidenceRef.from_mapping(dict(value))
                    for value in payload["transition_evidence_refs"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("prospective obligation payload is invalid") from exc

    @staticmethod
    def _prospective_receipt_id(
        obligation_id: str,
        event_id: str,
        version: int,
        payload_sha256: str,
    ) -> str:
        material = f"{obligation_id}\x00{event_id}\x00{version}\x00{payload_sha256}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _prospective_result(
        obligation: ProspectiveObligation,
        *,
        version: int,
        receipt_id: str,
        replayed: bool,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "obligation": obligation,
            "obligation_id": obligation.obligation_id,
            "status": obligation.status,
            "version": version,
            "receipt_id": receipt_id,
            "replayed": replayed,
        }

    def _upsert_index_node(self, node: MemoryIndexNode) -> None:
        payload = index_node_to_json(node)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO index_nodes(
                node_id, workspace_id, node_type, title, tags_json, payload_json, updated_at
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                node.node_id,
                node.workspace_id,
                node.node_type,
                node.title,
                json.dumps(list(node.tags), ensure_ascii=False, sort_keys=True),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                node.updated_at,
            ),
        )

    def _sync_prospective_trigger_index(
        self,
        obligation: ProspectiveObligation,
        *,
        version: int,
    ) -> None:
        workspace_id = self.workspace_id
        node = prospective_trigger_index_node(
            obligation,
            workspace_id=workspace_id,
            version=version,
        )
        existing = self.get_index_node(node.node_id, workspace_id=workspace_id)
        if existing is not None:
            node = replace(node, created_at=existing.created_at)
        self._upsert_index_node(node)

    def get_prospective_obligation(self, obligation_id: str) -> ProspectiveObligation | None:
        row = self._conn.execute(
            """
            SELECT scope, status, version, payload_id, payload_sha256, receipt_id
            FROM prospective_obligations WHERE obligation_id=?
            """,
            (obligation_id,),
        ).fetchone()
        if row is None:
            return None
        payload = self._load_payload(
            str(row[3]), str(row[4]), payload_kind="prospective_obligation"
        )
        obligation = self._prospective_from_payload(payload)
        if (
            obligation.obligation_id != obligation_id
            or obligation.scope != str(row[0])
            or obligation.status != str(row[1])
        ):
            raise ValueError("prospective obligation current state linkage is invalid")
        event = self._conn.execute(
            """
            SELECT event_id, expected_version, to_status, payload_id, payload_sha256, receipt_id
            FROM prospective_obligation_events
            WHERE obligation_id=? ORDER BY expected_version DESC LIMIT 1
            """,
            (obligation_id,),
        ).fetchone()
        if (
            event is None
            or int(row[2]) < 0
            or int(row[2]) != int(event[1]) + 1
            or str(event[2]) != obligation.status
            or str(event[3]) != str(row[3])
            or str(event[4]) != str(row[4])
            or str(event[5]) != str(row[5])
            or str(event[5])
            != self._prospective_receipt_id(
                obligation_id,
                str(event[0]),
                int(row[2]),
                str(row[4]),
            )
        ):
            raise ValueError("prospective obligation event linkage is invalid")
        return obligation

    def create_prospective_obligation(
        self,
        obligation: ProspectiveObligation,
        *,
        event_id: str,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        if obligation.status != "pending":
            raise ValueError("prospective obligation creation requires pending state")
        event_id = event_id.strip()
        key = idempotency_key.strip() or event_id
        if not event_id:
            raise ValueError("prospective obligation event_id is required")
        payload_json = self._prospective_payload_json(obligation)
        payload_id, payload_sha256 = self._payload_identity("prospective_obligation", payload_json)
        receipt_id = self._prospective_receipt_id(
            obligation.obligation_id, event_id, 0, payload_sha256
        )
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            events = self._conn.execute(
                """
                SELECT event_id, idempotency_key, expected_version, from_status, to_status,
                       payload_id, payload_sha256, receipt_id
                FROM prospective_obligation_events
                WHERE obligation_id=? AND (event_id=? OR idempotency_key=?)
                """,
                (obligation.obligation_id, event_id, key),
            ).fetchall()
            if events:
                if len(events) != 1 or tuple(str(value) for value in events[0][:7]) != (
                    event_id,
                    key,
                    "-1",
                    "",
                    "pending",
                    payload_id,
                    payload_sha256,
                ):
                    raise CanonicalHeadConflict("prospective_obligation_idempotency_conflict")
                if str(events[0][7]) != receipt_id:
                    raise CanonicalHeadConflict("prospective_obligation_receipt_conflict")
                replayed = self._prospective_from_payload(
                    self._load_payload(
                        str(events[0][5]),
                        str(events[0][6]),
                        payload_kind="prospective_obligation",
                    )
                )
                current_row = self._conn.execute(
                    """
                    SELECT scope, status, version, payload_id, payload_sha256, receipt_id
                    FROM prospective_obligations WHERE obligation_id=?
                    """,
                    (obligation.obligation_id,),
                ).fetchone()
                if current_row is None:
                    raise CanonicalHeadConflict("prospective_obligation_state_mismatch")
                current = self._prospective_from_payload(
                    self._load_payload(
                        str(current_row[3]),
                        str(current_row[4]),
                        payload_kind="prospective_obligation",
                    )
                )
                if (
                    replayed != obligation
                    or current != obligation
                    or str(current_row[0]) != obligation.scope
                    or str(current_row[1]) != obligation.status
                    or int(current_row[2]) != 0
                    or str(current_row[3]) != payload_id
                    or str(current_row[4]) != payload_sha256
                    or str(current_row[5]) != str(events[0][7])
                ):
                    raise CanonicalHeadConflict("prospective_obligation_state_mismatch")
                self._sync_prospective_trigger_index(obligation, version=0)
                return self._prospective_result(
                    obligation,
                    version=0,
                    receipt_id=str(events[0][7]),
                    replayed=True,
                )
            if (
                self._conn.execute(
                    "SELECT 1 FROM prospective_obligations WHERE obligation_id=?",
                    (obligation.obligation_id,),
                ).fetchone()
                is not None
            ):
                raise CanonicalHeadConflict("prospective_obligation_exists")
            self._store_payload("prospective_obligation", payload_json)
            now = datetime.now(UTC).isoformat()
            self._conn.execute(
                """
                INSERT INTO prospective_obligations(
                    obligation_id, scope, status, version, payload_id, payload_sha256,
                    receipt_id, updated_at
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    obligation.obligation_id,
                    obligation.scope,
                    obligation.status,
                    0,
                    payload_id,
                    payload_sha256,
                    receipt_id,
                    now,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO prospective_obligation_events(
                    obligation_id, event_id, idempotency_key, expected_version, from_status,
                    to_status, payload_id, payload_sha256, receipt_id, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    obligation.obligation_id,
                    event_id,
                    key,
                    -1,
                    "",
                    "pending",
                    payload_id,
                    payload_sha256,
                    receipt_id,
                    now,
                ),
            )
            self._sync_prospective_trigger_index(obligation, version=0)
        return self._prospective_result(
            obligation, version=0, receipt_id=receipt_id, replayed=False
        )

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
        event_id = event_id.strip()
        key = idempotency_key.strip() or event_id
        if not event_id:
            raise ValueError("prospective obligation event_id is required")
        refs = tuple(
            ref if isinstance(ref, EvidenceRef) else EvidenceRef.from_mapping(dict(ref))
            for ref in evidence_refs
        )
        satisfaction_refs = tuple(
            ref if isinstance(ref, EvidenceRef) else EvidenceRef.from_mapping(dict(ref))
            for ref in satisfaction_evidence_refs
        )
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            events = self._conn.execute(
                """
                SELECT event_id, idempotency_key, expected_version, from_status, to_status,
                       payload_id, payload_sha256, receipt_id
                FROM prospective_obligation_events
                WHERE obligation_id=? AND (event_id=? OR idempotency_key=?)
                """,
                (obligation_id, event_id, key),
            ).fetchall()
            if events:
                if len(events) != 1:
                    raise CanonicalHeadConflict("prospective_obligation_idempotency_conflict")
                event = events[0]
                replay_payload = self._load_payload(
                    str(event[5]), str(event[6]), payload_kind="prospective_obligation"
                )
                replayed = self._prospective_from_payload(replay_payload)
                if (
                    str(event[0]) != event_id
                    or str(event[1]) != key
                    or int(event[2]) != expected_version
                    or str(event[3]) != expected_status
                    or str(event[4]) != status
                    or replayed.transition_from != expected_status
                    or replayed.status != status
                    or replayed.transition_evidence_refs != refs
                    or replayed.satisfaction_evidence_refs != satisfaction_refs
                    or (
                        satisfaction_verifier
                        and replayed.satisfaction_verifier != satisfaction_verifier
                    )
                    or replayed.satisfaction_verifier_result != satisfaction_verifier_result
                    or str(event[7])
                    != self._prospective_receipt_id(
                        obligation_id,
                        event_id,
                        expected_version + 1,
                        str(event[6]),
                    )
                ):
                    raise CanonicalHeadConflict("prospective_obligation_idempotency_conflict")
                current_row = self._conn.execute(
                    """
                    SELECT status, version, payload_id, payload_sha256, receipt_id
                    FROM prospective_obligations WHERE obligation_id=?
                    """,
                    (obligation_id,),
                ).fetchone()
                replay_version = expected_version + 1
                if current_row is None or int(current_row[1]) < replay_version:
                    raise CanonicalHeadConflict("prospective_obligation_state_mismatch")
                current_version = int(current_row[1])
                if current_version == replay_version:
                    if (
                        str(current_row[0]) != status
                        or str(current_row[2]) != str(event[5])
                        or str(current_row[3]) != str(event[6])
                        or str(current_row[4]) != str(event[7])
                    ):
                        raise CanonicalHeadConflict("prospective_obligation_state_mismatch")
                    self._sync_prospective_trigger_index(replayed, version=replay_version)
                else:
                    current = self.get_prospective_obligation(obligation_id)
                    continuation = self._conn.execute(
                        """
                        SELECT expected_version, from_status, to_status
                        FROM prospective_obligation_events
                        WHERE obligation_id=? AND expected_version>=? AND expected_version<?
                        ORDER BY expected_version
                        """,
                        (obligation_id, replay_version, current_version),
                    ).fetchall()
                    version = replay_version
                    chained_status: str = status
                    for next_version, from_status, to_status in continuation:
                        if int(next_version) != version or str(from_status) != chained_status:
                            raise CanonicalHeadConflict("prospective_obligation_state_mismatch")
                        version += 1
                        chained_status = str(to_status)
                    if (
                        current is None
                        or version != current_version
                        or chained_status != current.status
                    ):
                        raise CanonicalHeadConflict("prospective_obligation_state_mismatch")
                return self._prospective_result(
                    replayed,
                    version=replay_version,
                    receipt_id=str(event[7]),
                    replayed=True,
                )
            row = self._conn.execute(
                """
                SELECT scope, status, version, payload_id, payload_sha256
                FROM prospective_obligations WHERE obligation_id=?
                """,
                (obligation_id,),
            ).fetchone()
            if row is None:
                raise CanonicalHeadConflict("prospective_obligation_missing")
            if int(row[2]) != expected_version or str(row[1]) != expected_status:
                raise CanonicalHeadConflict("prospective_obligation_stale")
            current = self._prospective_from_payload(
                self._load_payload(str(row[3]), str(row[4]), payload_kind="prospective_obligation")
            )
            if (
                current.obligation_id != obligation_id
                or current.scope != str(row[0])
                or current.status != expected_status
            ):
                raise ValueError("prospective obligation current state linkage is invalid")
            if satisfaction_verifier_result not in {"", "passed", "failed"}:
                raise ValueError("prospective satisfaction verifier result is invalid")
            successor = current.transition_to(
                cast(
                    ProspectiveStatus,
                    _closed_value(status, field_name="status", shape=ProspectiveStatus),
                ),
                evidence_refs=refs,
                satisfaction_evidence_refs=satisfaction_refs,
                satisfaction_verifier=satisfaction_verifier,
                satisfaction_verifier_result=cast(
                    Literal["", "passed", "failed"], satisfaction_verifier_result
                ),
            )
            payload_json = self._prospective_payload_json(successor)
            payload_id, payload_sha256 = self._payload_identity(
                "prospective_obligation", payload_json
            )
            receipt_id = self._prospective_receipt_id(
                obligation_id, event_id, expected_version + 1, payload_sha256
            )
            self._store_payload("prospective_obligation", payload_json)
            now = datetime.now(UTC).isoformat()
            cursor = self._conn.execute(
                """
                UPDATE prospective_obligations
                SET status=?, version=?, payload_id=?, payload_sha256=?, receipt_id=?, updated_at=?
                WHERE obligation_id=? AND version=? AND status=?
                """,
                (
                    successor.status,
                    expected_version + 1,
                    payload_id,
                    payload_sha256,
                    receipt_id,
                    now,
                    obligation_id,
                    expected_version,
                    expected_status,
                ),
            )
            if cursor.rowcount != 1:
                raise CanonicalHeadConflict("prospective_obligation_stale")
            self._conn.execute(
                """
                INSERT INTO prospective_obligation_events(
                    obligation_id, event_id, idempotency_key, expected_version, from_status,
                    to_status, payload_id, payload_sha256, receipt_id, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    obligation_id,
                    event_id,
                    key,
                    expected_version,
                    expected_status,
                    status,
                    payload_id,
                    payload_sha256,
                    receipt_id,
                    now,
                ),
            )
            self._sync_prospective_trigger_index(
                successor,
                version=expected_version + 1,
            )
        return self._prospective_result(
            successor,
            version=expected_version + 1,
            receipt_id=receipt_id,
            replayed=False,
        )

    def prospective_trigger_candidates(
        self,
        context: ProspectiveTriggerContext,
        *,
        workspace_id: str = "",
        limit: int = 50,
    ) -> list[ProspectiveTriggerDecision]:
        if not isinstance(context, ProspectiveTriggerContext):
            raise ValueError("context must use ProspectiveTriggerContext")
        target_workspace = workspace_id or self.workspace_id
        nodes = self.select_index_nodes(
            workspace_id=target_workspace,
            node_type="prospective_trigger",
            limit=limit,
        )
        decisions: list[ProspectiveTriggerDecision] = []
        for node in nodes:
            obligation_id = str(node.metadata.get("obligation_id", ""))
            if (
                not obligation_id
                or node.workspace_id != target_workspace
                or node.metadata.get("host_action_authorized") is not False
            ):
                raise ValueError("prospective trigger index linkage is invalid")
            obligation = self.get_prospective_obligation(obligation_id)
            state_row = self._conn.execute(
                "SELECT status, version FROM prospective_obligations WHERE obligation_id=?",
                (obligation_id,),
            ).fetchone()
            if obligation is None or (
                state_row is None
                or node.metadata.get("version") != int(state_row[1])
                or node.metadata.get("status") != str(state_row[0])
                or node.metadata.get("status") != obligation.status
                or not isinstance(node.metadata.get("version"), int)
            ):
                raise ValueError("prospective trigger index state is invalid")
            decision = evaluate_prospective_trigger(obligation, context)
            if decision.matched:
                decisions.append(decision)
        return decisions

    @staticmethod
    def _canonical_key_sha256(canonical_key: str) -> str:
        return hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _mutation_to_json(
        mutation: MutationDecision,
        *,
        record_payload_id: str,
        record_payload_sha256: str,
    ) -> str:
        payload = {
            "record_payload_id": record_payload_id,
            "record_payload_sha256": record_payload_sha256,
            "superseded_block_id": mutation.superseded_block_id,
            "action": mutation.action,
            "edge": asdict(mutation.edge) if mutation.edge is not None else None,
            "rollback_target_id": mutation.rollback_target_id,
            "affected_view_ids": list(mutation.affected_view_ids),
            "consolidation_receipt": (
                asdict(mutation.consolidation_receipt)
                if mutation.consolidation_receipt is not None
                else None
            ),
            "publication_receipt": asdict(mutation.publication_receipt)
            if mutation.publication_receipt
            else None,
            "amendment_receipt": asdict(mutation.amendment_receipt)
            if mutation.amendment_receipt
            else None,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def amendment_receipt(self, receipt_id: str) -> AmendmentCommitReceipt | None:
        row = self._conn.execute(
            "SELECT payload_json FROM canonical_mutation_receipts WHERE receipt_id=?",
            (receipt_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            receipt = AmendmentCommitReceipt(**json.loads(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("canonical amendment receipt is invalid") from exc
        if receipt.receipt_id != receipt_id or receipt.workspace_id != self.workspace_id:
            raise ValueError("canonical amendment receipt identity mismatch")
        return receipt

    def publication_receipt(self, receipt_id: str) -> PublicationCommitReceipt | None:
        row = self._conn.execute(
            "SELECT payload_json FROM canonical_mutation_receipts WHERE receipt_id=?",
            (receipt_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            receipt = PublicationCommitReceipt(**json.loads(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("canonical publication receipt is invalid") from exc
        if receipt.receipt_id != receipt_id or receipt.workspace_id != self.workspace_id:
            raise ValueError("canonical publication receipt identity mismatch")
        return receipt

    def creation_operation_receipt(
        self, operation_id: str, *, workspace_id: str
    ) -> dict[str, Any] | None:
        """Resolve a committed creation when the separate caller response was lost."""

        if not workspace_id or (self.workspace_id and workspace_id != self.workspace_id):
            raise ValueError("creation receipt requires the selected workspace")
        operation_hash = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        row = self._conn.execute(
            "SELECT payload_json FROM canonical_mutation_receipts WHERE receipt_id=?",
            (f"creation:{operation_hash}",),
        ).fetchone()
        if row is None:
            return None
        try:
            receipt = json.loads(str(row[0]))
        except json.JSONDecodeError as exc:
            raise ValueError("canonical creation receipt is invalid") from exc
        if (
            not isinstance(receipt, dict)
            or set(receipt)
            != {"record_id", "revision_id", "record_payload_sha256", "operation_id_sha256"}
            or receipt["operation_id_sha256"] != operation_hash
            or not isinstance(receipt["record_id"], str)
            or not receipt["record_id"]
        ):
            raise ValueError("canonical creation receipt identity mismatch")
        record_id = str(receipt["record_id"])
        record = self.get_block(record_id, workspace_id=workspace_id)
        if record is None:
            return None
        outbox = self.latest_write_outbox_entry(block_id=record_id, workspace_id=workspace_id)
        if (
            record.metadata.get("operation_id") != operation_id
            or record.revision_id != receipt["revision_id"]
            or outbox is None
            or outbox.payload_sha256 != receipt["record_payload_sha256"]
            or self.canonical_head_status(record_id) != "committed"
        ):
            raise ValueError("canonical creation receipt does not match the durable record")
        return {
            "record_id": record_id,
            "profile_id": record.profile_id,
            "revision_id": record.revision_id,
            "commit_revision": outbox.commit_revision,
            "projection_status": outbox.status,
        }

    def record_publication_receipt(self, receipt: PublicationCommitReceipt) -> None:
        if (
            not isinstance(receipt, PublicationCommitReceipt)
            or receipt.workspace_id != self.workspace_id
        ):
            raise ValueError("publication receipt requires an exact workspace")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self.publication_receipt(receipt.receipt_id)
            if existing is not None:
                if existing != receipt:
                    raise ValueError("publication receipt conflict")
            else:
                record = self.get_block(receipt.record_id)
                if (
                    record is None
                    or record.revision_id != receipt.revision_id
                    or record.profile_id != receipt.profile_id
                ):
                    raise ValueError("publication receipt record mismatch")
                row = self._conn.execute(
                    "SELECT commit_revision FROM canonical_mutations WHERE successor_id=? AND workspace_id=?",
                    (receipt.record_id, receipt.workspace_id),
                ).fetchone()
                if row is None:
                    raise ValueError("publication receipt requires a committed canonical record")
                self._conn.execute(
                    "INSERT INTO canonical_mutation_receipts(commit_revision, receipt_id, payload_json) VALUES(?,?,?)",
                    (
                        row[0],
                        receipt.receipt_id,
                        json.dumps(asdict(receipt), ensure_ascii=False, sort_keys=True),
                    ),
                )
            self._conn.commit()
        except (ValueError, RuntimeError, sqlite3.Error):
            self._conn.rollback()
            raise

    def _mutation_from_json(self, payload_json: str) -> MutationDecision:
        try:
            payload = json.loads(payload_json)
            expected = {
                "record_payload_id",
                "record_payload_sha256",
                "superseded_block_id",
                "action",
                "edge",
                "rollback_target_id",
                "affected_view_ids",
                "consolidation_receipt",
                "publication_receipt",
                "amendment_receipt",
            }
            if not isinstance(payload, dict) or set(payload) != expected:
                raise ValueError("canonical mutation payload fields are invalid")
            edge_payload = payload["edge"]
            if edge_payload is not None and not isinstance(edge_payload, dict):
                raise ValueError("canonical mutation edge payload is invalid")
            receipt_payload = payload["consolidation_receipt"]
            if receipt_payload is not None and not isinstance(receipt_payload, dict):
                raise ValueError("canonical mutation receipt payload is invalid")
            view_ids = payload["affected_view_ids"]
            if not isinstance(view_ids, list) or not all(
                isinstance(value, str) for value in view_ids
            ):
                raise ValueError("canonical mutation invalidation payload is invalid")
            record = block_from_json(
                self._load_payload(
                    str(payload["record_payload_id"]),
                    str(payload["record_payload_sha256"]),
                    payload_kind="record",
                )
            )
            return MutationDecision(
                record=record,
                superseded_block_id=str(payload["superseded_block_id"]),
                action=str(payload["action"]),
                edge=MemoryEdgeRecord(**edge_payload) if edge_payload is not None else None,
                rollback_target_id=str(payload["rollback_target_id"]),
                affected_view_ids=tuple(view_ids),
                consolidation_receipt=(
                    ConsolidationCommitReceipt(
                        receipt_id=str(receipt_payload["receipt_id"]),
                        evidence_refs=tuple(
                            EvidenceRef.from_mapping(dict(value))
                            for value in receipt_payload["evidence_refs"]
                        ),
                        committed_ids=tuple(
                            str(value) for value in receipt_payload["committed_ids"]
                        ),
                    )
                    if receipt_payload is not None
                    else None
                ),
                publication_receipt=(
                    PublicationCommitReceipt(**payload["publication_receipt"])
                    if payload["publication_receipt"] is not None
                    else None
                ),
                amendment_receipt=(
                    AmendmentCommitReceipt(**payload["amendment_receipt"])
                    if payload["amendment_receipt"] is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("canonical mutation payload is invalid") from exc

    def _pending_canonical_head_reservation(
        self,
        successor_id: str,
    ) -> CanonicalHeadReservation | None:
        row = self._conn.execute(
            """
            SELECT commit_revision, workspace_id, canonical_key_sha256, predecessor_id,
                   successor_id, successor_status, successor_authority
            FROM canonical_head_outbox
            WHERE successor_id=? AND status='pending'
            ORDER BY commit_revision DESC
            LIMIT 1
            """,
            (successor_id,),
        ).fetchone()
        if row is None:
            return None
        return CanonicalHeadReservation(
            commit_revision=int(row[0]),
            workspace_id=str(row[1]),
            canonical_key_sha256=str(row[2]),
            predecessor_id=str(row[3]),
            successor_id=str(row[4]),
            successor_status=str(row[5]),
            successor_authority=str(row[6]),
        )

    def pending_canonical_head_reservations(self) -> list[CanonicalHeadReservation]:
        rows = self._conn.execute(
            """
            SELECT commit_revision, workspace_id, canonical_key_sha256, predecessor_id,
                   successor_id, successor_status, successor_authority
            FROM canonical_head_outbox
            WHERE status='pending'
            ORDER BY commit_revision
            """
        ).fetchall()
        return [
            CanonicalHeadReservation(
                commit_revision=int(row[0]),
                workspace_id=str(row[1]),
                canonical_key_sha256=str(row[2]),
                predecessor_id=str(row[3]),
                successor_id=str(row[4]),
                successor_status=str(row[5]),
                successor_authority=str(row[6]),
            )
            for row in rows
        ]

    def reconcile_canonical_head_reservations(self) -> list[CanonicalHeadReservation]:
        rows = self._conn.execute(
            """
            SELECT commit_revision, workspace_id, canonical_key_sha256, predecessor_id,
                   successor_id, successor_status, successor_authority
            FROM canonical_head_outbox
            WHERE status IN ('pending', 'failed')
            ORDER BY commit_revision
            """
        ).fetchall()
        return [
            CanonicalHeadReservation(
                commit_revision=int(row[0]),
                workspace_id=str(row[1]),
                canonical_key_sha256=str(row[2]),
                predecessor_id=str(row[3]),
                successor_id=str(row[4]),
                successor_status=str(row[5]),
                successor_authority=str(row[6]),
            )
            for row in rows
        ]

    def pending_canonical_head_mutation(
        self,
        reservation: CanonicalHeadReservation,
    ) -> MutationDecision | None:
        row = self._conn.execute(
            """
            SELECT payload_id, payload_sha256 FROM canonical_head_outbox
            WHERE commit_revision=? AND workspace_id=? AND canonical_key_sha256=?
              AND predecessor_id=? AND successor_id=? AND status='pending'
            """,
            (
                reservation.commit_revision,
                reservation.workspace_id,
                reservation.canonical_key_sha256,
                reservation.predecessor_id,
                reservation.successor_id,
            ),
        ).fetchone()
        if row is None:
            return None
        payload = self._load_payload(str(row[0]), str(row[1]), payload_kind="mutation")
        return self._mutation_from_json(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )

    def replay_pending_canonical_heads(self) -> None:
        with self.canonical_write_gate():
            self._replay_pending_canonical_heads_locked()

    def _replay_pending_canonical_heads_locked(self) -> None:
        for reservation in self.pending_canonical_head_reservations():
            try:
                mutation = self.pending_canonical_head_mutation(reservation)
            except ValueError as exc:
                self.fail_canonical_head(
                    reservation,
                    error="canonical_mutation_payload_invalid",
                )
                raise RuntimeError("pending canonical mutation payload is invalid") from exc
            if mutation is None:
                self.fail_canonical_head(
                    reservation,
                    error="canonical_mutation_payload_missing",
                )
                raise RuntimeError("pending canonical mutation payload is unavailable")
            journal = self._conn.execute(
                """
                SELECT commit_revision, block_id, workspace_id, store, payload_id, payload_sha256,
                       status, error, created_at, updated_at
                FROM write_outbox
                WHERE block_id=? AND workspace_id=? AND status IN ('pending', 'committed')
                ORDER BY commit_revision DESC LIMIT 1
                """,
                (mutation.record.block_id, reservation.workspace_id),
            ).fetchone()
            if journal is None:
                self.fail_canonical_head(
                    reservation,
                    error="canonical_write_outbox_missing",
                )
                continue
            try:
                journal_record = self.write_outbox_record(self._outbox_from_row(journal))
            except ValueError as exc:
                self.fail_canonical_head(
                    reservation,
                    error="canonical_write_outbox_invalid",
                )
                raise RuntimeError("pending canonical mutation write journal is invalid") from exc
            if journal_record != mutation.record:
                self.fail_canonical_head(
                    reservation,
                    error="canonical_write_outbox_mismatch",
                )
                raise RuntimeError("pending canonical mutation write journal mismatches payload")
            try:
                self.complete_canonical_head(reservation, mutation)
            except (CanonicalHeadConflict, OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
                self.fail_canonical_head(reservation, error=str(exc))
                raise RuntimeError("pending canonical mutation replay failed") from exc

    def canonical_head_status(self, successor_id: str) -> str | None:
        row = self._conn.execute(
            """
            SELECT status FROM canonical_head_outbox
            WHERE successor_id=?
            ORDER BY commit_revision DESC
            LIMIT 1
            """,
            (successor_id,),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def canonical_history(
        self,
        *,
        canonical_key: str,
        workspace_id: str = "",
    ) -> list[MemoryBlockRecord]:
        key_hash = self._canonical_key_sha256(canonical_key)
        rows = self._conn.execute(
            """
            SELECT successor_id, record_payload_id, record_payload_sha256
            FROM canonical_mutations
            WHERE workspace_id=? AND canonical_key_sha256=?
            ORDER BY commit_revision
            """,
            (workspace_id or self.workspace_id, key_hash),
        ).fetchall()
        history: list[MemoryBlockRecord] = []
        for successor_id, payload_id, payload_sha256 in rows:
            record = block_from_json(
                self._load_payload(str(payload_id), str(payload_sha256), payload_kind="record")
            )
            if record.block_id != str(successor_id):
                raise ValueError("canonical mutation record identity is invalid")
            history.append(self._project_valid_to(record))
        return history

    def canonical_mutation_identity(
        self,
        *,
        successor_id: str,
        workspace_id: str = "",
    ) -> CanonicalMutationIdentity | None:
        row = self._conn.execute(
            """
            SELECT mutation_action, predecessor_id, successor_id, rollback_target_id,
                   record_payload_id, record_payload_sha256
            FROM canonical_mutations
            WHERE successor_id=? AND workspace_id=?
            """,
            (successor_id, workspace_id or self.workspace_id),
        ).fetchone()
        if row is None:
            return None
        record = block_from_json(
            self._load_payload(str(row[4]), str(row[5]), payload_kind="record")
        )
        if record.block_id != str(row[2]):
            raise ValueError("canonical mutation record identity is invalid")
        return CanonicalMutationIdentity(
            action=str(row[0]),
            predecessor_id=str(row[1]),
            successor_id=str(row[2]),
            rollback_target_id=str(row[3]),
        )

    def record_matches_temporal(self, record: MemoryBlockRecord, payload: dict[str, Any]) -> bool:
        temporal_mode = str(payload.get("temporal_mode") or "current")
        revision_id = str(payload.get("revision_id") or "").strip()
        as_of = str(payload.get("as_of") or "").strip()
        known_as_of = str(payload.get("known_as_of") or "").strip()
        if as_of:
            event_bound = _parse_time(as_of)
            if event_bound is None:
                raise ValueError("as_of requires an ISO-8601 timestamp")
            event_start = _parse_time(record.valid_from)
            event_end = _parse_time(record.valid_to)
            if not known_as_of and not record.valid_from:
                # An undated predecessor still ends at an explicitly dated correction.
                # Do not synthesize a start date or leak a future correction into a
                # historical knowledge snapshot.
                successor_start = _parse_time(self._successor_valid_from(record))
                if successor_start is not None and (
                    event_end is None or successor_start < event_end
                ):
                    event_end = successor_start
            expiration = _parse_time(record.expires_at)
            # Provenance/ingestion time is not an event-validity boundary.
            # Unknown event dates remain eligible, explicitly labelled below.
            if (
                (event_start is not None and event_bound < event_start)
                or (event_end is not None and event_bound >= event_end)
                or (expiration is not None and event_bound >= expiration)
            ):
                return False
        if known_as_of:
            known_bound = _parse_time(known_as_of)
            if known_bound is None:
                raise ValueError("known_as_of requires an ISO-8601 timestamp")
            cutoff = known_bound.isoformat()
            committed = self._conn.execute(
                "SELECT 1 FROM blocks WHERE block_id=? AND workspace_id=? AND committed_at<=?",
                (record.block_id, record.workspace_id or self.workspace_id, cutoff),
            ).fetchone()
            if committed is None:
                return False
            if revision_id and record.revision_id != revision_id:
                return False
            if record.canonical_key:
                key_hash = self._canonical_key_sha256(record.canonical_key)
                head = self._conn.execute(
                    "SELECT successor_id FROM canonical_mutations "
                    "WHERE workspace_id=? AND canonical_key_sha256=? AND committed_at<=? "
                    "ORDER BY commit_revision DESC LIMIT 1",
                    (record.workspace_id or self.workspace_id, key_hash, cutoff),
                ).fetchone()
                if head is None:
                    head = self._conn.execute(
                        "SELECT temporal.block_id FROM memory_temporal_index temporal "
                        "JOIN blocks ON blocks.block_id=temporal.block_id "
                        "WHERE temporal.workspace_id=? AND temporal.canonical_key_sha256=? "
                        "AND blocks.committed_at<=? "
                        "ORDER BY blocks.committed_at DESC, blocks.block_id LIMIT 1",
                        (record.workspace_id or self.workspace_id, key_hash, cutoff),
                    ).fetchone()
                if head is None or str(head[0]) != record.block_id:
                    return False
            return True
        if temporal_mode == "historical":
            return not (revision_id and record.revision_id != revision_id)
        if not record.canonical_key:
            return not revision_id
        key_hash = self._canonical_key_sha256(record.canonical_key)
        if as_of:
            bound = _parse_time(as_of)
            if bound is None:
                raise ValueError("current as_of requires an ISO-8601 timestamp")
            rows = self._conn.execute(
                """
                SELECT successor_id, record_payload_id, record_payload_sha256
                FROM canonical_mutations
                WHERE workspace_id=? AND canonical_key_sha256=?
                ORDER BY commit_revision
                """,
                (record.workspace_id or self.workspace_id, key_hash),
            ).fetchall()
            current_id = ""
            for successor_id, payload_id, payload_sha256 in rows:
                historical = block_from_json(
                    self._load_payload(str(payload_id), str(payload_sha256), payload_kind="record")
                )
                if historical.block_id != str(successor_id):
                    raise ValueError("canonical mutation record identity is invalid")
                historical = self._project_valid_to(historical)
                valid_from = _parse_time(historical.valid_from)
                valid_to = _parse_time(historical.valid_to)
                if (valid_from is None or valid_from <= bound) and (
                    valid_to is None or bound < valid_to
                ):
                    current_id = str(successor_id)
            return current_id == record.block_id
        head = self._conn.execute(
            """
            SELECT block_id FROM canonical_heads
            WHERE workspace_id=? AND canonical_key_sha256=?
            """,
            (record.workspace_id or self.workspace_id, key_hash),
        ).fetchone()
        return head is not None and str(head[0]) == record.block_id

    def fail_canonical_head(
        self,
        reservation: CanonicalHeadReservation,
        *,
        error: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._canonical_transaction_step():
            cursor = self._conn.execute(
                """
                UPDATE canonical_head_outbox
                SET status='failed', error=?, updated_at=?
                WHERE commit_revision=? AND status='pending'
                """,
                (error, now, reservation.commit_revision),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("canonical head failure transition rejected")
            self._conn.execute(
                "UPDATE write_outbox SET status='failed',error=?,updated_at=? "
                "WHERE block_id=? AND workspace_id=? AND status='pending'",
                (error, now, reservation.successor_id, reservation.workspace_id),
            )

    def fail_pending_canonical_head_for_successor(self, successor_id: str, *, error: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._conn:
            self._conn.execute(
                """
                UPDATE canonical_head_outbox
                SET status='failed', error=?, updated_at=?
                WHERE successor_id=? AND status='pending'
                """,
                (error, now, successor_id),
            )
            self._conn.execute(
                "UPDATE write_outbox SET status='failed',error=?,updated_at=? "
                "WHERE block_id=? AND status='pending' AND EXISTS "
                "(SELECT 1 FROM canonical_head_outbox WHERE successor_id=? AND status='failed')",
                (error, now, successor_id, successor_id),
            )

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
        canonical_key = canonical_key.strip()
        predecessor_id = predecessor_id.strip()
        successor_id = successor_id.strip()
        if not canonical_key or not successor_id:
            raise ValueError("canonical head reservation requires key and successor id")
        if predecessor_id == successor_id:
            raise ValueError("canonical head successor must differ from predecessor")
        successor = mutation.record
        if (
            successor.workspace_id != workspace_id
            or successor.canonical_key != canonical_key
            or successor.block_id != successor_id
            or successor.status != successor_status
            or successor.authority != successor_authority
            or mutation.superseded_block_id != predecessor_id
            or mutation.action != mutation_action
            or mutation.rollback_target_id != rollback_target_id
        ):
            raise ValueError("canonical head reservation does not match the mutation envelope")
        key_hash = self._canonical_key_sha256(canonical_key)
        now = datetime.now(UTC).isoformat()
        with self._canonical_transaction_step():
            pending = self._conn.execute(
                """
                SELECT commit_revision FROM canonical_head_outbox
                WHERE workspace_id=? AND canonical_key_sha256=? AND status='pending'
                """,
                (workspace_id, key_hash),
            ).fetchone()
            if pending is not None:
                raise CanonicalHeadConflict("canonical_head_pending")
            head = self._conn.execute(
                """
                SELECT block_id FROM canonical_heads
                WHERE workspace_id=? AND canonical_key_sha256=?
                """,
                (workspace_id, key_hash),
            ).fetchone()
            if predecessor_id:
                if head is None or str(head[0]) != predecessor_id:
                    raise CanonicalHeadConflict("canonical_head_conflict")
            elif head is not None:
                raise CanonicalHeadConflict("canonical_head_conflict")
            record_payload_id, record_payload_sha256, record_payload_json = (
                self._record_payload_identity(successor)
            )
            self._store_payload("record", record_payload_json)
            mutation_json = self._mutation_to_json(
                mutation,
                record_payload_id=record_payload_id,
                record_payload_sha256=record_payload_sha256,
            )
            payload_id, payload_sha256 = self._store_payload("mutation", mutation_json)
            cursor = self._conn.execute(
                """
                INSERT INTO canonical_head_outbox(
                    workspace_id, canonical_key_sha256, predecessor_id, successor_id,
                    successor_status, successor_authority, mutation_action, rollback_target_id,
                    payload_id, payload_sha256, status, error, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    workspace_id,
                    key_hash,
                    predecessor_id,
                    successor_id,
                    successor_status,
                    successor_authority,
                    mutation_action,
                    rollback_target_id,
                    payload_id,
                    payload_sha256,
                    "pending",
                    "",
                    now,
                    now,
                ),
            )
        if cursor.lastrowid is None:
            raise RuntimeError("canonical reservation insert produced no revision")
        return CanonicalHeadReservation(
            commit_revision=cursor.lastrowid,
            workspace_id=workspace_id,
            canonical_key_sha256=key_hash,
            predecessor_id=predecessor_id,
            successor_id=successor_id,
            successor_status=successor_status,
            successor_authority=successor_authority,
        )

    def complete_canonical_head(
        self,
        reservation: CanonicalHeadReservation,
        mutation: MutationDecision,
    ) -> dict[str, Any]:
        successor = mutation.record
        if (
            successor.block_id != reservation.successor_id
            or successor.workspace_id != reservation.workspace_id
            or self._canonical_key_sha256(successor.canonical_key)
            != reservation.canonical_key_sha256
        ):
            raise ValueError("canonical mutation does not match reservation")
        if mutation.superseded_block_id != reservation.predecessor_id:
            raise ValueError("canonical mutation predecessor does not match reservation")
        edge = mutation.edge
        if edge is not None and (
            edge.edge_type != "supersedes"
            or edge.src_id != successor.block_id
            or edge.dst_id != reservation.predecessor_id
            or (edge.workspace_id and edge.workspace_id != reservation.workspace_id)
        ):
            raise ValueError("canonical mutation edge does not match reservation")
        with self._canonical_transaction_step():
            now = datetime.now(UTC).isoformat()
            journal = self._conn.execute(
                """
                SELECT commit_revision, block_id, workspace_id, store, payload_id, payload_sha256,
                       status, error, created_at, updated_at
                FROM write_outbox
                WHERE block_id=? AND workspace_id=? AND status IN ('pending', 'committed')
                ORDER BY commit_revision DESC LIMIT 1
                """,
                (successor.block_id, reservation.workspace_id),
            ).fetchone()
            if journal is None:
                raise CanonicalHeadConflict("canonical_head_write_journal_missing")
            try:
                journal_record = self.write_outbox_record(self._outbox_from_row(journal))
            except ValueError as exc:
                raise CanonicalHeadConflict("canonical_head_write_journal_invalid") from exc
            if journal_record != successor:
                raise CanonicalHeadConflict("canonical_head_write_journal_mismatch")
            row = self._conn.execute(
                """
                SELECT workspace_id, canonical_key_sha256, predecessor_id, successor_id,
                       successor_status, successor_authority, mutation_action, rollback_target_id,
                       payload_id, payload_sha256, status
                FROM canonical_head_outbox
                WHERE commit_revision=?
                """,
                (reservation.commit_revision,),
            ).fetchone()
            if row is None:
                raise RuntimeError("canonical head reservation is unavailable")
            if (
                tuple(str(value) for value in row[:8])
                != (
                    reservation.workspace_id,
                    reservation.canonical_key_sha256,
                    reservation.predecessor_id,
                    reservation.successor_id,
                    reservation.successor_status,
                    reservation.successor_authority,
                    mutation.action,
                    mutation.rollback_target_id,
                )
                or str(row[10]) != "pending"
            ):
                raise CanonicalHeadConflict("canonical_head_reservation_rejected")
            record_payload_id, record_payload_sha256, record_payload_json = (
                self._record_payload_identity(successor)
            )
            expected_mutation_json = self._mutation_to_json(
                mutation,
                record_payload_id=record_payload_id,
                record_payload_sha256=record_payload_sha256,
            )
            expected_payload_id, expected_payload_sha256 = self._payload_identity(
                "mutation", expected_mutation_json
            )
            if str(row[8]) != expected_payload_id or str(row[9]) != expected_payload_sha256:
                raise CanonicalHeadConflict("canonical_head_reservation_rejected")
            self._load_payload(str(row[8]), str(row[9]), payload_kind="mutation")
            stored_record = self._load_payload(
                record_payload_id,
                record_payload_sha256,
                payload_kind="record",
            )
            if stored_record != json.loads(record_payload_json):
                raise ValueError("canonical mutation record payload is invalid")
            head = self._conn.execute(
                """
                SELECT block_id FROM canonical_heads
                WHERE workspace_id=? AND canonical_key_sha256=?
                """,
                (reservation.workspace_id, reservation.canonical_key_sha256),
            ).fetchone()
            if reservation.predecessor_id:
                if head is None or str(head[0]) != reservation.predecessor_id:
                    raise CanonicalHeadConflict("canonical_head_conflict")
            elif head is not None:
                raise CanonicalHeadConflict("canonical_head_conflict")
            payload_json = json.dumps(block_to_json(successor), ensure_ascii=False, sort_keys=True)
            existing_block = self._conn.execute(
                "SELECT workspace_id, payload_json FROM blocks WHERE block_id=?",
                (successor.block_id,),
            ).fetchone()
            if existing_block is None:
                self._conn.execute(
                    """
                    INSERT INTO blocks(
                        block_id, workspace_id, run_id, stage_id, subtask_id, scope,
                        status, retention, delivery_policy, temperature, authority,
                        trust_level, verification_status, promotion_decision,
                        content, payload_json, updated_at, committed_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        successor.block_id,
                        successor.workspace_id,
                        successor.run_id,
                        successor.stage_id,
                        successor.subtask_id,
                        successor.scope,
                        successor.status,
                        successor.retention,
                        successor.delivery_policy,
                        successor.temperature,
                        successor.authority,
                        successor.trust_level,
                        successor.verification_status,
                        successor.promotion_decision,
                        successor.content,
                        payload_json,
                        successor.updated_at,
                        now,
                    ),
                )
            elif (
                str(existing_block[0]) != successor.workspace_id
                or str(existing_block[1]) != payload_json
            ):
                raise CanonicalHeadConflict("canonical_mutation_record_conflict")
            self._conn.execute(
                """
                INSERT INTO canonical_mutations(
                    commit_revision, workspace_id, canonical_key_sha256, predecessor_id,
                    successor_id, mutation_action, rollback_target_id,
                    record_payload_id, record_payload_sha256, recorded_at, committed_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    reservation.commit_revision,
                    reservation.workspace_id,
                    reservation.canonical_key_sha256,
                    reservation.predecessor_id,
                    successor.block_id,
                    mutation.action,
                    mutation.rollback_target_id,
                    record_payload_id,
                    record_payload_sha256,
                    successor.recorded_at,
                    now,
                ),
            )
            if edge is not None:
                self._conn.execute(
                    """
                    INSERT INTO edges(src_id, dst_id, edge_type, workspace_id, payload_json)
                    VALUES(?,?,?,?,?)
                    """,
                    (
                        edge.src_id,
                        edge.dst_id,
                        edge.edge_type,
                        edge.workspace_id or reservation.workspace_id,
                        json.dumps(asdict(edge), ensure_ascii=False, sort_keys=True),
                    ),
                )
            dependency_source_id = (
                mutation.rollback_target_id
                if mutation.action == "rollback"
                else reservation.predecessor_id
                if mutation.action == "promote"
                else ""
            )
            if (
                mutation.action == "correction"
                and successor.metadata.get("correction_target_id") == reservation.predecessor_id
                and successor.metadata.get("approved_correction_candidate_id")
            ):
                source_id = str(successor.metadata["approved_correction_candidate_id"])
                source = self.get_block(source_id)
                if (
                    source is None
                    or source.workspace_id != reservation.workspace_id
                    or source.profile_id != successor.profile_id
                    or source.status != "candidate"
                    or source_id != successor.metadata.get("promotion_source_id")
                    or source.revision_id != successor.metadata.get("promotion_source_revision_id")
                    or source.content != successor.content
                ):
                    raise CanonicalHeadConflict("correction_promotion_source_mismatch")
                source_head = self._conn.execute(
                    "SELECT block_id FROM canonical_heads WHERE workspace_id=? AND canonical_key_sha256=?",
                    (reservation.workspace_id, self._canonical_key_sha256(source.canonical_key)),
                ).fetchone()
                if source_head is None or str(source_head[0]) != source_id:
                    raise CanonicalHeadConflict("correction_promotion_source_changed")
                dependency_source_id = source_id
            if dependency_source_id:
                dependency_rows = self._conn.execute(
                    """
                    SELECT payload_json FROM edges
                    WHERE workspace_id=? AND src_id=? AND edge_type=?
                    ORDER BY dst_id
                    """,
                    (reservation.workspace_id, dependency_source_id, EdgeType.DERIVED_FROM),
                ).fetchall()
                for dependency in self._edge_records(dependency_rows):
                    inherited = replace(dependency, src_id=successor.block_id, created_at=now)
                    self._conn.execute(
                        """
                        INSERT INTO edges(src_id, dst_id, edge_type, workspace_id, payload_json)
                        VALUES(?,?,?,?,?)
                        """,
                        (
                            inherited.src_id,
                            inherited.dst_id,
                            inherited.edge_type,
                            inherited.workspace_id,
                            json.dumps(asdict(inherited), ensure_ascii=False, sort_keys=True),
                        ),
                    )
            receipt = mutation.consolidation_receipt
            if receipt is not None:
                self._conn.execute(
                    """
                    INSERT INTO canonical_mutation_receipts(commit_revision, receipt_id, payload_json)
                    VALUES(?,?,?)
                    """,
                    (
                        reservation.commit_revision,
                        receipt.receipt_id,
                        json.dumps(asdict(receipt), ensure_ascii=False, sort_keys=True),
                    ),
                )
            publication = mutation.publication_receipt
            if publication is not None:
                self._conn.execute(
                    "INSERT INTO canonical_mutation_receipts(commit_revision, receipt_id, payload_json) VALUES(?,?,?)",
                    (
                        reservation.commit_revision,
                        publication.receipt_id,
                        json.dumps(asdict(publication), ensure_ascii=False, sort_keys=True),
                    ),
                )
            amendment = mutation.amendment_receipt
            if amendment is not None:
                self._conn.execute(
                    "INSERT INTO canonical_mutation_receipts(commit_revision, receipt_id, payload_json) VALUES(?,?,?)",
                    (
                        reservation.commit_revision,
                        amendment.receipt_id,
                        json.dumps(asdict(amendment), ensure_ascii=False, sort_keys=True),
                    ),
                )
            operation_id = str(successor.metadata.get("operation_id") or "")
            if mutation.action == "insert" and operation_id:
                operation_hash = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
                self._conn.execute(
                    "INSERT INTO canonical_mutation_receipts(commit_revision, receipt_id, payload_json) VALUES(?,?,?)",
                    (
                        reservation.commit_revision,
                        f"creation:{operation_hash}",
                        json.dumps(
                            {
                                "record_id": successor.block_id,
                                "revision_id": successor.revision_id,
                                "record_payload_sha256": record_payload_sha256,
                                "operation_id_sha256": operation_hash,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
            for view_id in mutation.affected_view_ids:
                self._conn.execute(
                    """
                    INSERT INTO canonical_mutation_invalidations(commit_revision, view_id)
                    VALUES(?,?)
                    """,
                    (reservation.commit_revision, view_id),
                )
            if reservation.predecessor_id:
                head_cursor = self._conn.execute(
                    """
                    UPDATE canonical_heads
                    SET block_id=?, commit_revision=?
                    WHERE workspace_id=? AND canonical_key_sha256=? AND block_id=?
                    """,
                    (
                        successor.block_id,
                        reservation.commit_revision,
                        reservation.workspace_id,
                        reservation.canonical_key_sha256,
                        reservation.predecessor_id,
                    ),
                )
                if head_cursor.rowcount != 1:
                    raise CanonicalHeadConflict("canonical_head_conflict")
            else:
                self._conn.execute(
                    """
                    INSERT INTO canonical_heads(
                        workspace_id, canonical_key_sha256, block_id, commit_revision
                    ) VALUES(?,?,?,?)
                    """,
                    (
                        reservation.workspace_id,
                        reservation.canonical_key_sha256,
                        successor.block_id,
                        reservation.commit_revision,
                    ),
                )
            if reservation.predecessor_id:
                boundary = successor.valid_from
                self._conn.execute(
                    """
                    UPDATE memory_temporal_index
                    SET is_current=0,
                        valid_to=CASE WHEN valid_to='' THEN ? ELSE valid_to END
                    WHERE block_id=?
                    """,
                    (boundary, reservation.predecessor_id),
                )
                self._conn.execute(
                    """
                    UPDATE record_block_memberships
                    SET is_current=0,
                        valid_to=CASE WHEN valid_to='' THEN ? ELSE valid_to END
                    WHERE record_id=?
                    """,
                    (boundary, reservation.predecessor_id),
                )
            self._sync_memory_indexes(successor)
            cursor = self._conn.execute(
                """
                UPDATE canonical_head_outbox
                SET status='committed', error='', updated_at=?
                WHERE commit_revision=? AND status='pending'
                """,
                (now, reservation.commit_revision),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("canonical head completion rejected")
            self._evict_over_cap()
        return {
            "ok": True,
            "commit_revision": reservation.commit_revision,
            "block_id": reservation.successor_id,
            "predecessor_id": reservation.predecessor_id,
            "backend": self.backend_id,
        }

    def retain_index_node(self, node: MemoryIndexNode) -> dict[str, Any]:
        if not node.workspace_id and self.workspace_id:
            node = replace(node, workspace_id=self.workspace_id)
        with self._conn:
            self._upsert_index_node(node)
        return {"ok": True, "node_id": node.node_id, "backend": self.backend_id}

    def get_index_node(self, node_id: str, *, workspace_id: str = "") -> MemoryIndexNode | None:
        row = self._conn.execute(
            "SELECT workspace_id, payload_json FROM index_nodes WHERE node_id=?",
            (str(node_id),),
        ).fetchone()
        if row is None:
            return None
        if workspace_id and row[0] and row[0] != workspace_id:
            return None
        return index_node_from_json(json.loads(row[1]))

    def select_index_nodes(
        self,
        *,
        workspace_id: str = "",
        node_type: str = "",
        tags: list[str] | tuple[str, ...] | set[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryIndexNode]:
        wanted_tags = {str(tag) for tag in (tags or ()) if str(tag)}
        clauses = []
        params: list[Any] = []
        target_workspace = workspace_id or self.workspace_id
        if target_workspace:
            clauses.append("workspace_id=?")
            params.append(target_workspace)
        if node_type:
            clauses.append("node_type=?")
            params.append(node_type)
        if wanted_tags:
            tag_clauses = ["instr(tags_json, ?) > 0" for _ in wanted_tags]
            clauses.append("(" + " OR ".join(tag_clauses) + ")")
            params.extend(json.dumps(tag, ensure_ascii=False) for tag in sorted(wanted_tags))
        sql = "SELECT payload_json FROM index_nodes"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC, node_id ASC LIMIT ?"
        params.append(max(1, int(limit or 50)))
        out: list[MemoryIndexNode] = []
        for (payload_json,) in self._conn.execute(sql, params).fetchall():
            node = index_node_from_json(json.loads(payload_json))
            if wanted_tags and not wanted_tags.intersection(str(tag) for tag in node.tags):
                continue
            out.append(node)
        return out

    def search_index_nodes(
        self,
        *,
        workspace_id: str = "",
        terms: list[str] | tuple[str, ...] = (),
        node_types: list[str] | tuple[str, ...] = (),
        limit: int = 128,
    ) -> list[MemoryIndexNode]:
        """Search the existing structural index without scanning canonical records."""

        normalized_terms = tuple(
            dict.fromkeys(str(term).strip().casefold() for term in terms if str(term).strip())
        )[:64]
        normalized_types = tuple(
            dict.fromkeys(
                str(node_type).strip() for node_type in node_types if str(node_type).strip()
            )
        )
        if not normalized_terms:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        target_workspace = workspace_id or self.workspace_id
        if target_workspace:
            clauses.append("workspace_id=?")
            params.append(target_workspace)
        if normalized_types:
            placeholders = ",".join("?" for _ in normalized_types)
            clauses.append(f"node_type IN ({placeholders})")
            params.extend(normalized_types)
        term_clauses: list[str] = []
        for term in normalized_terms:
            term_clauses.append("(instr(lower(title), ?) > 0 OR instr(lower(payload_json), ?) > 0)")
            params.extend((term, term))
        clauses.append("(" + " OR ".join(term_clauses) + ")")
        sql = "SELECT payload_json FROM index_nodes WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC, node_id ASC LIMIT ?"
        params.append(max(1, int(limit or 128)))
        return [
            index_node_from_json(json.loads(payload_json))
            for (payload_json,) in self._conn.execute(sql, params).fetchall()
        ]

    def export_index_nodes(self) -> list[MemoryIndexNode]:
        rows = self._conn.execute(
            "SELECT payload_json FROM index_nodes ORDER BY node_id"
        ).fetchall()
        return [index_node_from_json(json.loads(payload_json)) for (payload_json,) in rows]

    def record_ids_for_logical_blocks(
        self,
        logical_block_ids: tuple[str, ...] | list[str],
        *,
        workspace_id: str = "",
        profile_id: str = "",
        historical: bool = False,
    ) -> tuple[str, ...]:
        block_ids = tuple(
            dict.fromkeys(
                str(block_id).strip() for block_id in logical_block_ids if str(block_id).strip()
            )
        )
        target_workspace = str(workspace_id or self.workspace_id).strip()
        if not block_ids or not target_workspace:
            return ()
        placeholders = ",".join("?" for _ in block_ids)
        clauses = [
            "membership.workspace_id=?",
            "membership.role='primary'",
            f"membership.logical_block_id IN ({placeholders})",
            (
                "NOT EXISTS ("
                "SELECT 1 FROM canonical_head_outbox AS head_outbox "
                "WHERE head_outbox.successor_id=membership.record_id "
                "AND head_outbox.status<>'committed'"
                ")"
            ),
        ]
        if not historical:
            clauses.append("membership.is_current=1")
        rows = self._conn.execute(
            "SELECT DISTINCT membership.record_id "
            "FROM record_block_memberships AS membership WHERE "
            + " AND ".join(clauses)
            + " ORDER BY membership.record_id",
            (target_workspace, *block_ids),
        ).fetchall()
        matching_ids = tuple(str(row[0]) for row in rows)
        selected_profile = str(profile_id or "").strip()
        profiles = (
            (selected_profile,)
            if selected_profile
            else tuple(
                str(row[0])
                for row in self._conn.execute(
                    "SELECT DISTINCT json_extract(payload_json, '$.profile_id') FROM blocks "
                    "WHERE workspace_id=?",
                    (target_workspace,),
                ).fetchall()
                if row[0]
            )
        )
        moved: set[str] = set()
        added: set[str] = set()
        for selected in profiles:
            catalog = self._topology_catalog_connection(selected)
            if catalog is None:
                continue
            for offset in range(0, len(matching_ids), 400):
                chunk = matching_ids[offset : offset + 400]
                if not chunk:
                    continue
                selected_rows = catalog.execute(
                    "SELECT record_id,block_id FROM published_record_memberships "
                    "WHERE workspace_id=? AND profile_id=? AND record_id IN ("
                    + ",".join("?" for _ in chunk)
                    + ")",
                    (target_workspace, selected, *chunk),
                ).fetchall()
                moved.update(str(row[0]) for row in selected_rows if str(row[1]) not in block_ids)
            assigned = catalog.execute(
                "SELECT record_id FROM published_record_memberships "
                "WHERE workspace_id=? AND profile_id=? AND block_id IN (" + placeholders + ")",
                (target_workspace, selected, *block_ids),
            ).fetchall()
            assigned_ids = tuple(str(row[0]) for row in assigned)
            for offset in range(0, len(assigned_ids), 400):
                chunk = assigned_ids[offset : offset + 400]
                current_clause = "" if historical else " AND temporal.is_current=1"
                visible = self._conn.execute(
                    "SELECT temporal.block_id FROM memory_temporal_index AS temporal "
                    "WHERE temporal.workspace_id=? AND temporal.block_id IN ("
                    + ",".join("?" for _ in chunk)
                    + ")"
                    + current_clause,
                    (target_workspace, *chunk),
                ).fetchall()
                added.update(str(row[0]) for row in visible)
        matching_ids = tuple(sorted((set(matching_ids) - moved) | added))
        if not selected_profile:
            return matching_ids
        visible_profile_ids: set[str] = set()
        for offset in range(0, len(matching_ids), 400):
            chunk = matching_ids[offset : offset + 400]
            rows = self._conn.execute(
                "SELECT block_id FROM blocks WHERE workspace_id=? "
                "AND json_extract(payload_json, '$.profile_id')=? "
                "AND block_id IN (" + ",".join("?" for _ in chunk) + ")",
                (target_workspace, selected_profile, *chunk),
            ).fetchall()
            visible_profile_ids.update(str(row[0]) for row in rows)
        return tuple(record_id for record_id in matching_ids if record_id in visible_profile_ids)

    def native_partition_memberships(
        self,
        *,
        workspace_id: str,
        profile_id: str = "",
        logical_block_ids: tuple[str, ...] = (),
    ) -> tuple[tuple[str, str, str], ...]:
        """Return distinct persisted leaf identities without decoding canonical records."""

        if not workspace_id or (self.workspace_id and workspace_id != self.workspace_id):
            raise ValueError("native partition lookup requires the selected workspace")
        selected = tuple(dict.fromkeys(logical_block_ids))
        clauses = ["membership.workspace_id=?", "membership.role='primary'"]
        params: list[str] = [workspace_id]
        if profile_id:
            clauses.append("json_extract(record.payload_json, '$.profile_id')=?")
            params.append(profile_id)
        if selected:
            clauses.append(
                "membership.logical_block_id IN (" + ",".join("?" for _ in selected) + ")"
            )
            params.extend(selected)
        rows = self._conn.execute(
            "SELECT DISTINCT membership.logical_block_id, membership.catalog_revision_id, "
            "membership.backend_partition_id "
            "FROM record_block_memberships AS membership "
            "JOIN blocks AS record ON record.block_id=membership.record_id "
            "WHERE " + " AND ".join(clauses),
            params,
        ).fetchall()
        identities = {(str(row[0]), str(row[1]), str(row[2])) for row in rows}
        profiles = (
            (profile_id,)
            if profile_id
            else tuple(
                str(row[0])
                for row in self._conn.execute(
                    "SELECT DISTINCT json_extract(payload_json, '$.profile_id') FROM blocks "
                    "WHERE workspace_id=?",
                    (workspace_id,),
                ).fetchall()
                if row[0]
            )
        )
        for selected_profile in profiles:
            catalog = self._topology_catalog_connection(selected_profile)
            if catalog is None:
                continue
            archived = {
                str(row[0])
                for row in catalog.execute(
                    "SELECT block_id FROM topology_blocks WHERE workspace_id=? "
                    "AND profile_id=? AND status!='active'",
                    (workspace_id, selected_profile),
                ).fetchall()
            }
            identities = {identity for identity in identities if identity[0] not in archived}
            selected_clause = (
                " AND block_id IN (" + ",".join("?" for _ in selected) + ")" if selected else ""
            )
            projected = catalog.execute(
                "SELECT DISTINCT block_id,catalog_revision_id,backend_partition_id "
                "FROM published_record_memberships WHERE workspace_id=? AND profile_id=?"
                + selected_clause,
                (workspace_id, selected_profile, *selected),
            ).fetchall()
            identities.update(
                (str(row[0]), str(row[1]), str(row[2]))
                for row in projected
                if str(row[0]) not in archived
            )
        return tuple(sorted(identities))

    def logical_block_descriptor_snapshot(
        self, logical_block_id: str, *, workspace_id: str, profile_id: str
    ) -> dict[str, Any]:
        """Count current records in SQLite and return only a bounded descriptor sample."""

        if (
            not logical_block_id
            or not workspace_id
            or (self.workspace_id and workspace_id != self.workspace_id)
            or not profile_id
        ):
            raise ValueError("descriptor snapshot requires an exact block and scope")
        catalog = self._topology_catalog_connection(profile_id)
        evolved = (
            catalog.execute(
                "SELECT 1 FROM published_record_memberships WHERE workspace_id=? "
                "AND profile_id=? AND block_id=? "
                "UNION SELECT 1 FROM topology_blocks WHERE workspace_id=? "
                "AND profile_id=? AND block_id=? AND status='archived' LIMIT 1",
                (
                    workspace_id,
                    profile_id,
                    logical_block_id,
                    workspace_id,
                    profile_id,
                    logical_block_id,
                ),
            ).fetchone()
            if catalog is not None
            else None
        )
        if evolved is not None:
            ids = self.record_ids_for_logical_blocks(
                (logical_block_id,), workspace_id=workspace_id, profile_id=profile_id
            )
            evolved_rows: list[tuple[str, str, str, str]] = []
            for offset in range(0, len(ids), 400):
                chunk = ids[offset : offset + 400]
                rows = self._conn.execute(
                    "SELECT block_id,json_extract(payload_json, '$.revision_id'),"
                    "json_extract(payload_json, '$.title'),updated_at FROM blocks "
                    "WHERE workspace_id=? AND status NOT IN "
                    "('superseded','deprecated','retracted','quarantined','expired') "
                    "AND block_id IN (" + ",".join("?" for _ in chunk) + ")",
                    (workspace_id, *chunk),
                ).fetchall()
                evolved_rows.extend(
                    (str(row[0]), str(row[1]), str(row[2] or ""), str(row[3])) for row in rows
                )
            evolved_preview = sorted(evolved_rows)[:64]
            evolved_latest = (
                max(evolved_rows, key=lambda row: (row[3], row[0])) if evolved_rows else None
            )
            dimensions = {
                "path": "paths",
                "symbol": "symbols",
                "entity": "entities",
                "config_key": "config_keys",
                "endpoint": "endpoints",
                "artifact_hash": "artifact_hashes",
            }
            present: set[str] = set()
            for offset in range(0, len(evolved_rows), 400):
                evolved_chunk = evolved_rows[offset : offset + 400]
                rows = self._conn.execute(
                    "SELECT DISTINCT term_type FROM memory_exact_terms WHERE workspace_id=? "
                    "AND block_id IN (" + ",".join("?" for _ in evolved_chunk) + ") "
                    "AND term_type IN ('path','symbol','entity','config_key','endpoint','artifact_hash')",
                    (workspace_id, *(row[0] for row in evolved_chunk)),
                ).fetchall()
                present.update(str(row[0]) for row in rows)
            return {
                "record_count": len(evolved_rows),
                "related_block_ids": tuple(row[0] for row in evolved_preview),
                "source_revision_ids": tuple(row[1] for row in evolved_preview),
                "truncated": len(evolved_rows) > len(evolved_preview),
                "latest_record_title": evolved_latest[2] if evolved_latest else "",
                "latest_updated_at": evolved_latest[3] if evolved_latest else "",
                "exact_dimensions": tuple(sorted(dimensions[key] for key in present)),
            }
        tables = """
            FROM record_block_memberships AS membership
            JOIN blocks AS record ON record.block_id=membership.record_id
        """
        criteria = """
            WHERE membership.workspace_id=? AND membership.logical_block_id=?
              AND membership.role='primary' AND membership.is_current=1
              AND record.status NOT IN
                  ('superseded','deprecated','retracted','quarantined','expired')
              AND json_extract(record.payload_json, '$.profile_id')=?
              AND NOT EXISTS (
                  SELECT 1 FROM canonical_head_outbox AS head
                  WHERE head.successor_id=membership.record_id AND head.status<>'committed'
              )
        """
        scope = (workspace_id, logical_block_id, profile_id)
        current = tables + criteria
        count = int(self._conn.execute("SELECT COUNT(*) " + current, scope).fetchone()[0])
        preview = self._conn.execute(
            "SELECT membership.record_id,membership.record_revision_id "
            + current
            + " ORDER BY membership.record_id LIMIT 64",
            scope,
        ).fetchall()
        latest = self._conn.execute(
            "SELECT json_extract(record.payload_json, '$.title'), record.updated_at "
            + current
            + " ORDER BY record.updated_at DESC,record.block_id DESC LIMIT 1",
            scope,
        ).fetchone()
        dimensions = {
            "path": "paths",
            "symbol": "symbols",
            "entity": "entities",
            "config_key": "config_keys",
            "endpoint": "endpoints",
            "artifact_hash": "artifact_hashes",
        }
        present_rows = self._conn.execute(
            "SELECT DISTINCT term.term_type "
            + tables
            + " JOIN memory_exact_terms AS term ON term.block_id=membership.record_id "
            "AND term.workspace_id=membership.workspace_id "
            + criteria
            + " AND term.term_type IN ('path','symbol','entity','config_key','endpoint','artifact_hash')",
            scope,
        ).fetchall()
        return {
            "record_count": count,
            "related_block_ids": tuple(str(row[0]) for row in preview),
            "source_revision_ids": tuple(str(row[1]) for row in preview),
            "truncated": count > len(preview),
            "latest_record_title": str(latest[0] or "") if latest else "",
            "latest_updated_at": str(latest[1] or "") if latest else "",
            "exact_dimensions": tuple(sorted(dimensions[str(row[0])] for row in present_rows)),
        }

    @staticmethod
    def _exact_query_groups(payload: dict[str, Any]) -> dict[str, tuple[str, ...]]:
        groups: dict[str, tuple[str, ...]] = {}
        for query_field, term_type in _EXACT_QUERY_TERMS.items():
            raw = payload.get(query_field) or ()
            values = (raw,) if isinstance(raw, str) else raw
            normalized = tuple(
                dict.fromkeys(str(value).strip() for value in values if str(value).strip())
            )
            if normalized:
                groups[term_type] = normalized
        facets = payload.get("facets") or {}
        if isinstance(facets, dict):
            for term_type, raw in facets.items():
                values = (raw,) if isinstance(raw, str) else raw
                normalized = tuple(
                    dict.fromkeys(str(value).strip() for value in values if str(value).strip())
                )
                if normalized:
                    groups[str(term_type)] = normalized
        return groups

    def exact_query_block_ids(
        self,
        payload: dict[str, Any],
        *,
        workspace_id: str = "",
    ) -> tuple[bool, set[str]]:
        groups = self._exact_query_groups(payload)
        if not groups:
            return False, set()
        target_workspace = workspace_id or self.workspace_id
        matched: set[str] | None = None
        for term_type, values in groups.items():
            placeholders = ",".join("?" for _ in values)
            params: list[Any] = [target_workspace, term_type, *values]
            rows = self._conn.execute(
                f"""
                SELECT DISTINCT block_id FROM memory_exact_terms
                WHERE workspace_id=? AND term_type=? AND term_value IN ({placeholders})
                """,
                params,
            ).fetchall()
            group_ids = {str(row[0]) for row in rows}
            matched = group_ids if matched is None else matched.intersection(group_ids)
            if not matched:
                break
        return True, matched or set()

    def apply_exact_query(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Compile exact fields into the canonical block-selection contract.

        Native providers do not share UM-MEM's exact term schema.  Turning an
        exact query into backend-owned block ids before native recall keeps the
        same strict semantics for every selected backend and prevents semantic
        hits outside the requested facets from leaking into the result.
        """

        applied, matched = self.exact_query_block_ids(
            payload,
            workspace_id=str(payload.get("workspace_id") or self.workspace_id),
        )
        if not applied:
            return payload
        caller_order = record_id_sequence(payload)
        if caller_order:
            selected = [block_id for block_id in caller_order if block_id in matched]
        else:
            selected = sorted(matched)
        if not selected:
            selected = ["__um_mem_no_exact_index_match__"]
        return {**payload, "record_ids": selected}

    @staticmethod
    def _workspace_scope_filter(payload: dict[str, Any]) -> tuple[str, ...]:
        scope_chain = query_sequence(payload, "scope_chain")
        # A project id is lineage inside one workspace, not a group of
        # workspaces, so only the user-global level may widen the physical
        # scan.  Every other scope keeps the workspace filter, and the client
        # re-checks the record workspace before a hit is returned.
        if any(scope_key.startswith("user_global:") for scope_key in scope_chain):
            return ()
        return tuple(
            scope_key.split(":", 1)[1]
            for scope_key in scope_chain
            if scope_key.startswith("workspace:")
        )

    def _fts_candidates(
        self,
        text: str,
        *,
        workspace_ids: tuple[str, ...],
        limit: int,
    ) -> tuple[list[str], dict[str, float]] | None:
        tokens = tuple(dict.fromkeys(token for token in re.findall(r"\w+", text) if token))
        if not tokens:
            return [], {}
        match = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        clauses = ["memory_fts MATCH ?"]
        params: list[Any] = [match]
        if workspace_ids:
            placeholders = ",".join("?" for _ in workspace_ids)
            clauses.append(f"workspace_id IN ({placeholders})")
            params.extend(workspace_ids)
        params.append(max(1, int(limit)))
        try:
            rows = self._conn.execute(
                "SELECT block_id, bm25(memory_fts) FROM memory_fts WHERE "
                + " AND ".join(clauses)
                + " ORDER BY bm25(memory_fts), rowid DESC LIMIT ?",
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            return None
        return [str(row[0]) for row in rows], {str(row[0]): float(row[1]) for row in rows}

    def recall(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryBlockRecord]:
        payload = normalize_query_payload(query, default_workspace_id=self.workspace_id)
        if payload["retrieval_mode"] == "hybrid":
            raise ValueError("hybrid retrieval is unsupported by sqlite_index")
        return self._recall_payload(payload)

    def _recall_payload(
        self,
        payload: dict[str, Any],
    ) -> list[MemoryBlockRecord]:
        text = str(payload.get("query") or "").strip().lower()
        workspace_id = str(payload.get("workspace_id") or self.workspace_id or "")
        caller_selected_ids = record_ids(payload)
        exact_applied, exact_ids = self.exact_query_block_ids(
            payload,
            workspace_id=workspace_id,
        )
        selected_ids = set(caller_selected_ids)
        if exact_applied:
            selected_ids = selected_ids.intersection(exact_ids) if selected_ids else exact_ids
        selected_sequence = record_id_sequence(payload)
        selected_order = {block_id: order for order, block_id in enumerate(selected_sequence)}
        limit = max(1, int(payload.get("limit") or 10))
        bounded_scan_limit = max(limit, 2000)
        raw_token_budget = payload.get("max_tokens")
        token_budget = max(0, int(raw_token_budget or 0))
        workspace_scope_filter = self._workspace_scope_filter(payload)
        history_query = (
            str(payload.get("temporal_mode") or "current") == "historical"
            or bool(str(payload.get("as_of") or "").strip())
            or bool(payload.get("known_as_of"))
        )
        retrieval_channel = ""
        fts_scores: dict[str, float] = {}
        if text and not caller_selected_ids:
            fts = self._fts_candidates(
                text,
                workspace_ids=workspace_scope_filter,
                limit=bounded_scan_limit,
            )
            if fts is None:
                retrieval_channel = "recent_substring_fallback"
                bounded_scan_limit = min(
                    _MAX_RECENT_SUBSTRING_FALLBACK,
                    max(limit, bounded_scan_limit),
                )
            else:
                retrieval_channel = "sqlite_fts5_bm25"
                fts_ids, fts_scores = fts
                fts_id_set = set(fts_ids)
                selected_ids = (
                    selected_ids.intersection(fts_id_set) if exact_applied else fts_id_set
                )
                if not selected_ids:
                    selected_ids = {"__um_mem_no_index_match__"}
                selected_order = {block_id: order for order, block_id in enumerate(fts_ids)}
        elif exact_applied:
            retrieval_channel = "exact_inverted_index"
            if not selected_ids:
                selected_ids = {"__um_mem_no_index_match__"}

        clauses: list[str] = []
        params: list[Any] = []
        if workspace_scope_filter:
            placeholders = ",".join("?" for _ in workspace_scope_filter)
            clauses.append(f"temporal.workspace_id IN ({placeholders})")
            params.extend(workspace_scope_filter)
        if selected_ids:
            placeholders = ",".join("?" for _ in selected_ids)
            clauses.append(f"temporal.block_id IN ({placeholders})")
            params.extend(sorted(selected_ids))
        if not history_query:
            clauses.append("temporal.is_current=1")
        if payload.get("known_as_of"):
            known_bound = _parse_time(str(payload["known_as_of"]))
            if known_bound is None:
                raise ValueError("known_as_of requires an ISO-8601 timestamp")
            clauses.append("blocks.committed_at<=?")
            params.append(known_bound.isoformat())
        sql = (
            "SELECT blocks.payload_json FROM memory_temporal_index temporal "
            "JOIN blocks ON blocks.block_id=temporal.block_id"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if history_query:
            sql += " ORDER BY temporal.recorded_at DESC, temporal.block_id"
        else:
            sql += " ORDER BY blocks.updated_at DESC"
        if not selected_ids:
            sql += " LIMIT ?"
            params.append(bounded_scan_limit)
        rows = self._conn.execute(sql, params).fetchall()
        tokens = [tok for tok in re.split(r"\W+", text) if tok]
        scored: list[tuple[int, int, MemoryBlockRecord, list[str]]] = []
        now = datetime.now(UTC)
        for order, row in enumerate(rows):
            block = block_from_json(json.loads(row[0]))
            if history_query and not payload.get("known_as_of"):
                block = self._project_valid_to(block)
            if not history_query:
                canonical_status = self.canonical_head_status(block.block_id)
                if canonical_status is not None and canonical_status != "committed":
                    continue
            if not history_query and self._is_expired(block, now=now):
                continue
            if not self.record_matches_temporal(block, payload):
                continue
            if not record_allowed_for_query(block, payload):
                continue
            if (
                not payload.get("known_as_of")
                and (not history_query or payload.get("as_of"))
                and self.validate_dependencies(
                    block.block_id,
                    workspace_id=block.workspace_id,
                    profile_id=block.profile_id,
                    as_of=str(payload.get("as_of") or ""),
                ).status
                != "valid"
            ):
                continue
            score = 0
            matched_terms: list[str] = []
            explicit_selected = bool(
                block.block_id in caller_selected_ids
                or (exact_applied and block.block_id in exact_ids)
            )
            if tokens:
                haystack = (block.content + " " + block.title + " " + " ".join(block.tags)).lower()
                matched_terms = [tok for tok in tokens if tok in haystack]
                score = len(matched_terms)
                if score == 0 and not explicit_selected:
                    continue
                if explicit_selected:
                    score = max(score, len(tokens), 1)
            scored.append(
                (
                    score,
                    selected_order.get(block.block_id, order),
                    block,
                    matched_terms,
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected: list[MemoryBlockRecord] = []
        used_tokens = 0
        for rank, (score, _, block, matched_terms) in enumerate(scored, start=1):
            retrieval_score = (float(score) / float(len(tokens))) if tokens else 1.0 / float(rank)
            block = replace(
                block,
                metadata={
                    **dict(block.metadata or {}),
                    **(
                        {"event_time_status": "unknown"}
                        if payload.get("as_of") and not block.valid_from
                        else {}
                    ),
                    "retrieval_backend": self.backend_id,
                    "retrieval_rank": rank,
                    "retrieval_score": retrieval_score,
                    "retrieval_score_raw": score,
                    "retrieval_matched_terms": matched_terms[:20],
                    **({"retrieval_channel": retrieval_channel} if retrieval_channel else {}),
                    **(
                        {"retrieval_bm25": fts_scores[block.block_id]}
                        if block.block_id in fts_scores
                        else {}
                    ),
                    **(
                        {"retrieval_scan_limit": bounded_scan_limit}
                        if retrieval_channel == "recent_substring_fallback"
                        else {}
                    ),
                },
            )
            block_tokens = _record_token_cost(block)
            if token_budget and used_tokens + block_tokens > token_budget:
                remaining = token_budget - used_tokens
                if selected:
                    continue
                truncated = _truncate_record_to_budget(block, remaining)
                if truncated is None:
                    continue
                block = truncated
                block_tokens = _record_token_cost(block)
            selected.append(block)
            used_tokens += block_tokens
            if len(selected) >= limit:
                break
        return selected

    def expire_scope(self, scope: str, scope_id: str = "", *, run_id: str = "") -> int:
        return 0

    def add_edge(self, edge: MemoryEdgeRecord) -> dict[str, Any]:
        src = str(edge.src_id or "").strip()
        dst = str(edge.dst_id or "").strip()
        if not src or not dst:
            raise ValueError(
                f"add_edge requires non-empty src/dst ids (got {edge.src_id!r} -> {edge.dst_id!r})"
            )
        if edge.edge_type not in ALLOWED_EDGE_TYPES:
            raise ValueError(
                f"unknown edge type {edge.edge_type!r}; allowed: {sorted(ALLOWED_EDGE_TYPES)}"
            )
        workspace_id = edge.workspace_id or self.workspace_id
        if self.workspace_id and workspace_id != self.workspace_id:
            raise ValueError("memory graph edge workspace does not match selected backend")
        if src == dst:
            raise ValueError("memory graph edges cannot be self-referential")
        src_record = self.get_block(src, workspace_id=workspace_id)
        dst_record = self.get_block(dst, workspace_id=workspace_id)
        src_node = (
            None if src_record is not None else self.get_index_node(src, workspace_id=workspace_id)
        )
        dst_node = (
            None if dst_record is not None else self.get_index_node(dst, workspace_id=workspace_id)
        )
        if (src_record is None and src_node is None) or (dst_record is None and dst_node is None):
            raise ValueError("memory graph edge endpoints must exist in the selected workspace")
        if edge.edge_type in {EdgeType.SUPERSEDES, EdgeType.REVISES}:
            if src_record is None or dst_record is None:
                raise ValueError("revision edges require canonical memory record endpoints")
            src_membership = src_record.block_membership
            dst_membership = dst_record.block_membership
            if (
                src_record.canonical_key != dst_record.canonical_key
                or src_membership is None
                or dst_membership is None
            ):
                raise ValueError("revision edges cannot cross canonical keys")
        acyclic_edge_types = {
            EdgeType.DERIVED_FROM,
            EdgeType.SUPERSEDES,
            EdgeType.REVISES,
        }
        if edge.edge_type in acyclic_edge_types:
            pending = [dst]
            visited: set[str] = set()
            while pending:
                current = pending.pop()
                if current == src:
                    raise ValueError("memory graph provenance cycle rejected")
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(
                    item.dst_id
                    for item in self.edges_for(current)
                    if item.src_id == current and item.edge_type in acyclic_edge_types
                )
        edge = replace(edge, src_id=src, dst_id=dst, workspace_id=workspace_id)
        payload = asdict(edge)
        payload["src_id"] = src
        payload["dst_id"] = dst
        payload["evidence_refs"] = [asdict(ref) for ref in edge.evidence_refs]
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        existing = self._conn.execute(
            """
            SELECT payload_json FROM edges
            WHERE workspace_id=? AND src_id=? AND dst_id=? AND edge_type=?
            """,
            (workspace_id, src, dst, edge.edge_type),
        ).fetchone()
        if existing is not None:
            existing_payload = json.loads(str(existing[0]))
            candidate_payload = dict(payload)
            existing_payload.pop("created_at", None)
            candidate_payload.pop("created_at", None)
            if existing_payload != candidate_payload:
                raise ValueError("memory graph edges are immutable")
            return {
                "ok": True,
                "replayed": True,
                "src_id": src,
                "dst_id": dst,
                "edge_type": edge.edge_type,
                "backend": self.backend_id,
            }
        self._conn.execute(
            "INSERT INTO edges(src_id,dst_id,edge_type,workspace_id,payload_json) VALUES(?,?,?,?,?)",
            (
                src,
                dst,
                edge.edge_type,
                workspace_id,
                payload_json,
            ),
        )
        self._conn.commit()
        return {
            "ok": True,
            "src_id": src,
            "dst_id": dst,
            "edge_type": edge.edge_type,
            "backend": self.backend_id,
        }

    def edges_for(self, block_id: str) -> list[MemoryEdgeRecord]:
        rows = self._conn.execute(
            """
            SELECT payload_json FROM edges
            WHERE workspace_id=? AND (src_id=? OR dst_id=?)
            ORDER BY rowid
            """,
            (self.workspace_id, block_id, block_id),
        ).fetchall()
        return self._edge_records(rows)

    @staticmethod
    def _edge_records(rows: list[tuple[Any, ...]]) -> list[MemoryEdgeRecord]:
        out: list[MemoryEdgeRecord] = []
        for (payload_json,) in rows:
            data = json.loads(payload_json)
            data["evidence_refs"] = tuple(data.get("evidence_refs") or ())
            out.append(
                MemoryEdgeRecord(
                    **{k: v for k, v in data.items() if k in MemoryEdgeRecord.__dataclass_fields__}
                )
            )
        return out

    def export_edges(self) -> list[MemoryEdgeRecord]:
        rows = self._conn.execute(
            """
            SELECT payload_json FROM edges
            WHERE workspace_id=?
            ORDER BY src_id, dst_id, edge_type
            """,
            (self.workspace_id,),
        ).fetchall()
        return self._edge_records(rows)

    def record_partition_operation(
        self,
        partition: NativePartitionIdentity,
        *,
        operation: str,
        latency_ms: float = 0.0,
        hit_count: int = 0,
        record_bytes: int | None = None,
        rendered_tokens: int = 0,
        success: bool = True,
        search_mode: str = "",
        run_id: str = "",
        profile_id: str = "",
    ) -> None:
        allowed_operations = {
            "write",
            "recall",
            "render",
            "consolidation",
            "compaction",
            "eviction",
            "expiry",
        }
        if partition.backend_id != self.backend_id:
            raise ValueError("native partition metrics backend does not match selected backend")
        if self.workspace_id and partition.workspace_id != self.workspace_id:
            raise ValueError("native partition metrics workspace does not match selected backend")
        if operation not in allowed_operations:
            raise ValueError("native partition metrics operation is unsupported")
        if search_mode not in {"", "scoped", "search_all"}:
            raise ValueError("native partition metrics search_mode is unsupported")
        if operation != "recall" and search_mode:
            raise ValueError("native partition search_mode is only valid for recall")
        latency = float(latency_ms)
        hits = int(hit_count)
        tokens = int(rendered_tokens)
        if latency < 0 or hits < 0 or tokens < 0:
            raise ValueError("native partition operation metrics cannot be negative")
        if record_bytes is not None and int(record_bytes) < 0:
            raise ValueError("native partition record_bytes cannot be negative")
        recorded_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._conn.execute(
            """
            INSERT INTO native_partition_events(
                workspace_id, primary_block_id, catalog_revision_id, partition_id,
                isolation_class, partition_mode, operation, latency_ms, hit_count,
                record_bytes, rendered_tokens, success, search_mode, recorded_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                partition.workspace_id,
                partition.primary_block_id,
                partition.catalog_revision_id,
                partition.partition_id,
                partition.isolation_class,
                partition.partition_mode,
                operation,
                latency,
                hits,
                int(record_bytes) if record_bytes is not None else None,
                tokens,
                int(bool(success)),
                search_mode,
                recorded_at,
            ),
        )
        self._conn.commit()
        profile_context = current_memory_profile_context()
        record_memory_event(
            self._repo_root,
            event_type="memory_profile",
            workspace_id=partition.workspace_id,
            profile_id=profile_id or profile_context["profile_id"],
            user_id=profile_context["user_id"],
            chat_id=profile_context["chat_id"],
            run_id=run_id or profile_context["run_id"],
            backend=self.backend_id,
            status="ok" if success else "failed",
            operation=f"backend_partition.{operation}",
            component=f"backend.{self.backend_id}.partition",
            outcome=search_mode or operation,
            reason="native_partition",
            duration_ms=latency,
            parent_span_id=current_memory_profile_span_id(),
            data={
                "input": memory_profile_summary(
                    {
                        "partition": asdict(partition),
                        "search_mode": search_mode,
                    }
                ),
                "output": memory_profile_summary(
                    {
                        "hit_count": hits,
                        "record_bytes": record_bytes,
                        "rendered_tokens": tokens,
                        "success": success,
                    }
                ),
            },
        )

    @staticmethod
    def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = max(1, int(len(ordered) * percentile + 0.999999999))
        return round(float(ordered[min(rank, len(ordered)) - 1]), 3)

    def partition_metrics(
        self,
        partition: NativePartitionIdentity,
        *,
        health: str = "unknown",
    ) -> NativePartitionMetrics:
        if partition.backend_id != self.backend_id:
            raise ValueError("native partition metrics backend does not match selected backend")
        identity = (
            partition.workspace_id,
            partition.primary_block_id,
            partition.catalog_revision_id,
            partition.partition_id,
        )
        record_count = int(
            self._conn.execute(
                """
                SELECT COUNT(*)
                FROM record_block_memberships AS membership
                JOIN blocks ON blocks.block_id=membership.record_id
                WHERE membership.workspace_id=?
                  AND membership.logical_block_id=?
                  AND membership.catalog_revision_id=?
                  AND membership.role='primary'
                  AND membership.is_current=1
                """,
                identity[:3],
            ).fetchone()[0]
        )
        rows = self._conn.execute(
            """
            SELECT operation, latency_ms, hit_count, record_bytes, rendered_tokens,
                   success, search_mode, recorded_at
            FROM native_partition_events
            WHERE workspace_id=? AND primary_block_id=?
              AND catalog_revision_id=? AND partition_id=?
            ORDER BY id
            """,
            identity,
        ).fetchall()
        write_count = sum(str(row[0]) == "write" for row in rows)
        recall_rows = [row for row in rows if str(row[0]) == "recall"]
        hit_count = sum(int(row[2]) for row in recall_rows if bool(row[5]))
        zero_hit_count = sum(bool(row[5]) and int(row[2]) == 0 for row in recall_rows)
        failure_count = sum(not bool(row[5]) for row in rows)
        byte_values = [int(row[3]) for row in rows if row[3] is not None]
        latencies = [float(row[1]) for row in rows if float(row[1]) > 0]
        last_accessed_at = str(rows[-1][7]) if rows else ""
        operation_counts = {
            operation: sum(str(row[0]) == operation for row in rows)
            for operation in ("consolidation", "compaction", "eviction", "expiry")
        }
        return NativePartitionMetrics(
            identity=partition,
            record_count=record_count,
            write_count=write_count,
            recall_count=len(recall_rows),
            hit_count=hit_count,
            zero_hit_count=zero_hit_count,
            failure_count=failure_count,
            record_bytes=sum(byte_values) if byte_values else None,
            latency_p50_ms=self._nearest_rank_percentile(latencies, 0.50),
            latency_p95_ms=self._nearest_rank_percentile(latencies, 0.95),
            rendered_tokens=sum(int(row[4]) for row in rows),
            consolidation_count=operation_counts["consolidation"],
            compaction_count=operation_counts["compaction"],
            eviction_count=operation_counts["eviction"],
            expiry_count=operation_counts["expiry"],
            scoped_recall_count=sum(str(row[6]) == "scoped" for row in recall_rows),
            broad_recall_count=sum(str(row[6]) == "search_all" for row in recall_rows),
            last_accessed_at=last_accessed_at,
            health=str(health or "unknown"),
        )

    def record_usage(self, usage: MemoryUsageRecord) -> dict[str, Any]:
        if usage.helped or usage.hurt or usage.outcome in {"helped", "hurt"}:
            raise ValueError("helped or hurt credit requires verified OutcomeCreditEvent")
        if usage.block_id.startswith(
            ("memory-decision-event:", "memory-usage-event:", "outcome-credit-event:")
        ) or (
            usage.outcome in {"memory_decision", "outcome_credit"}
            or usage.outcome.startswith("memory_usage:")
        ):
            raise ValueError("unverified usage record cannot use the decision journal namespace")
        payload = asdict(usage)
        payload["evidence_refs"] = [asdict(ref) for ref in usage.evidence_refs]
        self._conn.execute(
            "INSERT INTO usage(block_id,workspace_id,outcome,payload_json) VALUES(?,?,?,?)",
            (
                usage.block_id,
                usage.workspace_id or self.workspace_id,
                usage.outcome,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._conn.commit()
        return {"ok": True, "block_id": usage.block_id, "backend": self.backend_id}

    @staticmethod
    def _decision_event_envelope(
        event: MemoryDecisionEvent | MemoryUsageEvent | OutcomeCreditEvent,
    ) -> str:
        payload = asdict(event)
        payload["evidence_refs"] = [asdict(ref) for ref in event.evidence_refs]
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return json.dumps(
            {
                "payload": payload,
                "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _decision_event_payload(envelope_json: str) -> dict[str, Any]:
        try:
            envelope = json.loads(envelope_json)
        except json.JSONDecodeError as exc:
            raise ValueError("memory decision journal payload is invalid") from exc
        if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
            raise ValueError("memory decision journal payload is invalid")
        payload = dict(envelope["payload"])
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if envelope.get("payload_sha256") != expected:
            raise ValueError("memory decision journal payload hash mismatch")
        return payload

    @classmethod
    def _decision_from_envelope(cls, envelope_json: str) -> MemoryDecisionEvent:
        payload = cls._decision_event_payload(envelope_json)
        payload["candidate_ids"] = tuple(payload.get("candidate_ids", ()))
        payload["selected_candidate_ids"] = tuple(payload.get("selected_candidate_ids", ()))
        payload["evidence_refs"] = tuple(
            EvidenceRef.from_mapping(dict(value)) for value in payload.get("evidence_refs", ())
        )
        return MemoryDecisionEvent(**payload)

    @classmethod
    def _usage_event_from_envelope(cls, envelope_json: str) -> MemoryUsageEvent:
        payload = cls._decision_event_payload(envelope_json)
        payload["evidence_refs"] = tuple(
            EvidenceRef.from_mapping(dict(value)) for value in payload.get("evidence_refs", ())
        )
        return MemoryUsageEvent(**payload)

    @classmethod
    def _outcome_credit_from_envelope(cls, envelope_json: str) -> OutcomeCreditEvent:
        payload = cls._decision_event_payload(envelope_json)
        payload["memory_ids"] = tuple(payload.get("memory_ids", ()))
        payload["applied_usage_event_ids"] = tuple(payload.get("applied_usage_event_ids", ()))
        payload["evidence_refs"] = tuple(
            EvidenceRef.from_mapping(dict(value)) for value in payload.get("evidence_refs", ())
        )
        return OutcomeCreditEvent(**payload)

    def record_memory_decision(self, event: MemoryDecisionEvent) -> dict[str, Any]:
        if not isinstance(event, MemoryDecisionEvent):
            raise ValueError("event must use MemoryDecisionEvent")
        if event.workspace_id != self.workspace_id:
            raise ValueError("memory decision workspace mismatch")
        block_id = f"memory-decision-event:{event.decision_id}"
        envelope = self._decision_event_envelope(event)
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute(
                """
                SELECT payload_json FROM usage
                WHERE block_id=? AND workspace_id=? AND outcome='memory_decision'
                """,
                (block_id, event.workspace_id),
            ).fetchall()
            if rows:
                if len(rows) != 1 or str(rows[0][0]) != envelope:
                    raise CanonicalHeadConflict("memory_decision_idempotency_conflict")
                replayed = self._decision_from_envelope(str(rows[0][0]))
                if replayed != event:
                    raise CanonicalHeadConflict("memory_decision_idempotency_conflict")
                return {
                    "ok": True,
                    "decision_id": event.decision_id,
                    "backend": self.backend_id,
                    "replayed": True,
                }
            self._conn.execute(
                "INSERT INTO usage(block_id,workspace_id,outcome,payload_json) VALUES(?,?,?,?)",
                (block_id, event.workspace_id, "memory_decision", envelope),
            )
        return {
            "ok": True,
            "decision_id": event.decision_id,
            "backend": self.backend_id,
            "replayed": False,
        }

    def get_memory_decision(self, decision_id: str) -> MemoryDecisionEvent | None:
        decision_id = decision_id.strip()
        if not decision_id:
            raise ValueError("decision_id is required")
        rows = self._conn.execute(
            """
            SELECT payload_json FROM usage
            WHERE block_id=? AND workspace_id=? AND outcome='memory_decision'
            """,
            (f"memory-decision-event:{decision_id}", self.workspace_id),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("memory decision journal linkage is invalid")
        event = self._decision_from_envelope(str(rows[0][0]))
        if event.decision_id != decision_id or event.workspace_id != self.workspace_id:
            raise ValueError("memory decision journal linkage is invalid")
        return event

    def record_memory_usage_event(self, event: MemoryUsageEvent) -> dict[str, Any]:
        if not isinstance(event, MemoryUsageEvent):
            raise ValueError("event must use MemoryUsageEvent")
        if event.workspace_id != self.workspace_id:
            raise ValueError("memory usage workspace mismatch")
        decision = self.get_memory_decision(event.decision_id)
        if decision is None:
            raise CanonicalHeadConflict("memory_decision_missing")
        if (
            event.task_signature != decision.task_signature
            or event.action_trace_id != decision.action_trace_id
        ):
            raise ValueError("memory usage task signature linkage is invalid")
        if event.memory_id not in decision.candidate_ids:
            raise ValueError("memory usage available candidate linkage is invalid")
        if event.usage_type in {"selected", "rendered", "cited", "applied"} and (
            event.memory_id not in decision.selected_candidate_ids
        ):
            raise ValueError("memory usage selected candidate linkage is invalid")
        block_id = f"memory-usage-event:{event.event_id}"
        outcome = f"memory_usage:{event.usage_type}"
        envelope = self._decision_event_envelope(event)
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute(
                """
                SELECT payload_json FROM usage
                WHERE block_id=? AND workspace_id=? AND outcome=?
                """,
                (block_id, event.workspace_id, outcome),
            ).fetchall()
            if rows:
                if len(rows) != 1 or str(rows[0][0]) != envelope:
                    raise CanonicalHeadConflict("memory_usage_idempotency_conflict")
                replayed = self._usage_event_from_envelope(str(rows[0][0]))
                if replayed != event:
                    raise CanonicalHeadConflict("memory_usage_idempotency_conflict")
                return {
                    "ok": True,
                    "event_id": event.event_id,
                    "backend": self.backend_id,
                    "replayed": True,
                }
            if event.usage_type in {"applied", "not_applied"}:
                opposite = "not_applied" if event.usage_type == "applied" else "applied"
                contradictory = self._conn.execute(
                    """
                    SELECT 1 FROM usage WHERE workspace_id=? AND outcome=?
                      AND json_extract(payload_json, '$.payload.decision_id')=?
                      AND json_extract(payload_json, '$.payload.memory_id')=?
                    LIMIT 1
                    """,
                    (
                        self.workspace_id,
                        f"memory_usage:{opposite}",
                        event.decision_id,
                        event.memory_id,
                    ),
                ).fetchone()
                if contradictory is not None:
                    raise ValueError("memory usage contains contradictory action-use observations")
            sequence_row = self._conn.execute(
                """
                SELECT block_id FROM usage
                WHERE workspace_id=? AND outcome LIKE 'memory_usage:%'
                  AND json_extract(payload_json, '$.payload.decision_id')=?
                  AND json_extract(payload_json, '$.payload.sequence')=?
                """,
                (event.workspace_id, event.decision_id, event.sequence),
            ).fetchone()
            if sequence_row is not None:
                raise CanonicalHeadConflict("memory_usage_sequence_conflict")
            self._conn.execute(
                "INSERT INTO usage(block_id,workspace_id,outcome,payload_json) VALUES(?,?,?,?)",
                (block_id, event.workspace_id, outcome, envelope),
            )
        return {
            "ok": True,
            "event_id": event.event_id,
            "backend": self.backend_id,
            "replayed": False,
        }

    def memory_decision_trace(self, decision_id: str) -> MemoryDecisionTrace:
        decision = self.get_memory_decision(decision_id)
        if decision is None:
            raise CanonicalHeadConflict("memory_decision_missing")
        rows = self._conn.execute(
            """
            SELECT payload_json FROM usage
            WHERE workspace_id=? AND outcome LIKE 'memory_usage:%'
              AND json_extract(payload_json, '$.payload.decision_id')=?
            ORDER BY CAST(json_extract(payload_json, '$.payload.sequence') AS INTEGER), block_id
            """,
            (self.workspace_id, decision.decision_id),
        ).fetchall()
        events = tuple(self._usage_event_from_envelope(str(row[0])) for row in rows)
        return MemoryDecisionTrace(decision=decision, usage_events=events)

    def _memory_usage_event(self, event_id: str) -> MemoryUsageEvent | None:
        rows = self._conn.execute(
            """
            SELECT payload_json FROM usage
            WHERE block_id=? AND workspace_id=? AND outcome LIKE 'memory_usage:%'
            """,
            (f"memory-usage-event:{event_id}", self.workspace_id),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("memory usage journal linkage is invalid")
        event = self._usage_event_from_envelope(str(rows[0][0]))
        if event.event_id != event_id or event.workspace_id != self.workspace_id:
            raise ValueError("memory usage journal linkage is invalid")
        return event

    def record_outcome_credit(self, event: OutcomeCreditEvent) -> dict[str, Any]:
        if not isinstance(event, OutcomeCreditEvent):
            raise ValueError("event must use OutcomeCreditEvent")
        if event.workspace_id != self.workspace_id:
            raise ValueError("outcome credit workspace mismatch")
        decision = self.get_memory_decision(event.decision_id)
        if decision is None:
            raise CanonicalHeadConflict("memory_decision_missing")
        if (
            event.task_signature != decision.task_signature
            or event.action_trace_id != decision.action_trace_id
        ):
            raise ValueError("outcome credit task and action linkage is invalid")
        usage_events: list[MemoryUsageEvent] = []
        for event_id in event.applied_usage_event_ids:
            usage = self._memory_usage_event(event_id)
            if usage is None or usage.decision_id != event.decision_id:
                raise ValueError("outcome credit usage linkage is invalid")
            usage_events.append(usage)
        if usage_events and not {usage.memory_id for usage in usage_events}.issubset(
            event.memory_ids
        ):
            raise ValueError("outcome credit memory linkage is invalid")
        if event.outcome in {"helped", "hurt"} and (
            not usage_events
            or any(usage.usage_type != "applied" for usage in usage_events)
            or {usage.memory_id for usage in usage_events} != set(event.memory_ids)
        ):
            raise ValueError("outcome credit requires exact applied usage linkage")
        block_id = f"outcome-credit-event:{event.credit_id}"
        envelope = self._decision_event_envelope(event)
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute(
                """
                SELECT payload_json FROM usage
                WHERE block_id=? AND workspace_id=? AND outcome='outcome_credit'
                """,
                (block_id, event.workspace_id),
            ).fetchall()
            if rows:
                if len(rows) != 1 or str(rows[0][0]) != envelope:
                    raise CanonicalHeadConflict("outcome_credit_idempotency_conflict")
                replayed = self._outcome_credit_from_envelope(str(rows[0][0]))
                if replayed != event:
                    raise CanonicalHeadConflict("outcome_credit_idempotency_conflict")
                return {
                    "ok": True,
                    "credit_id": event.credit_id,
                    "backend": self.backend_id,
                    "replayed": True,
                }
            if event.outcome in {"helped", "hurt"}:
                placeholders = ",".join("?" for _ in event.applied_usage_event_ids)
                duplicate = self._conn.execute(
                    f"""
                    SELECT 1
                    FROM usage AS journal
                    JOIN json_each(
                        journal.payload_json,
                        '$.payload.applied_usage_event_ids'
                    ) AS applied
                    WHERE journal.workspace_id=?
                      AND journal.outcome='outcome_credit'
                      AND json_extract(journal.payload_json, '$.payload.outcome')
                          IN ('helped', 'hurt')
                      AND applied.value IN ({placeholders})
                    LIMIT 1
                    """,
                    (event.workspace_id, *event.applied_usage_event_ids),
                ).fetchone()
                if duplicate is not None:
                    raise CanonicalHeadConflict("outcome_credit_applied_usage_conflict")
            self._conn.execute(
                "INSERT INTO usage(block_id,workspace_id,outcome,payload_json) VALUES(?,?,?,?)",
                (block_id, event.workspace_id, "outcome_credit", envelope),
            )
        return {
            "ok": True,
            "credit_id": event.credit_id,
            "backend": self.backend_id,
            "replayed": False,
        }

    def outcome_credits(self, decision_id: str) -> tuple[OutcomeCreditEvent, ...]:
        target = str(decision_id or "").strip()
        if not target:
            raise ValueError("decision_id is required")
        rows = self._conn.execute(
            """
            SELECT payload_json FROM usage
            WHERE workspace_id=?
              AND outcome='outcome_credit'
              AND json_extract(payload_json, '$.payload.decision_id')=?
            ORDER BY id
            """,
            (self.workspace_id, target),
        ).fetchall()
        return tuple(self._outcome_credit_from_envelope(str(row[0])) for row in rows)

    def outcome_credits_for_memory(self, memory_id: str) -> tuple[OutcomeCreditEvent, ...]:
        target = str(memory_id or "").strip()
        if not target:
            raise ValueError("memory_id is required")
        rows = self._conn.execute(
            """
            SELECT journal.payload_json FROM usage AS journal
            JOIN json_each(journal.payload_json, '$.payload.memory_ids') AS memory
            WHERE journal.workspace_id=? AND journal.outcome='outcome_credit'
              AND memory.value=?
            ORDER BY journal.id
            """,
            (self.workspace_id, target),
        ).fetchall()
        return tuple(self._outcome_credit_from_envelope(str(row[0])) for row in rows)

    def contextual_utility_summary(
        self,
        *,
        memory_id: str,
        task_family: str,
        stage: str,
        project_id: str,
        host_id: str,
        profile_id: str,
        minimum_support: int,
        minimum_independent_outcomes: int,
    ) -> ContextualUtilitySummary:
        dimensions = {
            "memory_id": memory_id,
            "task_family": task_family,
            "stage": stage,
            "project_id": project_id,
            "host_id": host_id,
            "profile_id": profile_id,
        }
        for field_name, value in dimensions.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            dimensions[field_name] = value.strip()
        for field_name, minimum in (
            ("minimum_support", minimum_support),
            ("minimum_independent_outcomes", minimum_independent_outcomes),
        ):
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        rows = self._conn.execute(
            """
            SELECT payload_json FROM usage
            WHERE workspace_id=? AND outcome='outcome_credit'
              AND json_extract(payload_json, '$.payload.task_family')=?
              AND json_extract(payload_json, '$.payload.stage')=?
              AND json_extract(payload_json, '$.payload.project_id')=?
              AND json_extract(payload_json, '$.payload.host_id')=?
              AND json_extract(payload_json, '$.payload.profile_id')=?
              AND EXISTS (
                  SELECT 1 FROM json_each(
                      json_extract(payload_json, '$.payload.memory_ids')
                  ) WHERE value=?
              )
            ORDER BY json_extract(payload_json, '$.payload.created_at'), block_id
            """,
            (
                self.workspace_id,
                dimensions["task_family"],
                dimensions["stage"],
                dimensions["project_id"],
                dimensions["host_id"],
                dimensions["profile_id"],
                dimensions["memory_id"],
            ),
        ).fetchall()
        credits = tuple(self._outcome_credit_from_envelope(str(row[0])) for row in rows)
        helped = sum(credit.outcome == "helped" for credit in credits)
        hurt = sum(credit.outcome == "hurt" for credit in credits)
        neutral = sum(credit.outcome == "neutral" for credit in credits)
        verified = tuple(credit for credit in credits if credit.outcome in {"helped", "hurt"})
        identity = "|".join(
            (
                self.workspace_id,
                dimensions["memory_id"],
                dimensions["task_family"],
                dimensions["stage"],
                dimensions["project_id"],
                dimensions["host_id"],
                dimensions["profile_id"],
            )
        )
        return ContextualUtilitySummary(
            summary_id=f"contextual-utility:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}",
            workspace_id=self.workspace_id,
            memory_id=dimensions["memory_id"],
            task_family=dimensions["task_family"],
            stage=dimensions["stage"],
            project_id=dimensions["project_id"],
            host_id=dimensions["host_id"],
            profile_id=dimensions["profile_id"],
            minimum_support=minimum_support,
            minimum_independent_outcomes=minimum_independent_outcomes,
            helped_count=helped,
            hurt_count=hurt,
            neutral_count=neutral,
            outcome_diversity=len({credit.independence_domain for credit in verified}),
            oldest_credit_at=credits[0].created_at if credits else "",
            newest_credit_at=credits[-1].created_at if credits else "",
        )

    def value_stats(self, block_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM usage WHERE block_id=?",
            (block_id,),
        ).fetchone()
        return {
            "block_id": block_id,
            "uses_count": int(row[0]) if row is not None else 0,
            "helped_count": 0,
            "hurt_count": 0,
        }

    def export_blocks(self) -> list[MemoryBlockRecord]:
        rows = self._conn.execute("SELECT payload_json FROM blocks ORDER BY block_id").fetchall()
        return [block_from_json(json.loads(payload_json)) for (payload_json,) in rows]

    def import_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        count = 0
        for record in records:
            target = (
                replace(record, workspace_id=self.workspace_id) if self.workspace_id else record
            )
            self.retain_block(target)
            count += 1
        return {"ok": True, "imported": count, "backend": self.backend_id}

    def compile_active_overlay(self, *, workspace_id: str = "", run_id: str = "") -> dict[str, Any]:
        query: dict[str, Any] = {
            "workspace_id": workspace_id or self.workspace_id,
            "limit": 500,
            "include_candidates": True,
        }
        if run_id:
            query["run_id"] = run_id
        rows = self.recall(query)
        directives = [
            row.block_id
            for row in rows
            if row.authority == "directive" and row.status == "accepted"
        ]
        candidates = [
            row.block_id
            for row in rows
            if row.status == "candidate" or row.delivery_policy == "reflection_only"
        ]
        return {
            "backend": self.backend_id,
            "directive_blocks": directives,
            "candidate_blocks": candidates,
        }

    def render_prompt_items(
        self, *, workspace_id: str = "", run_id: str = ""
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"workspace_id": workspace_id or self.workspace_id, "limit": 500}
        if run_id:
            query["run_id"] = run_id
        rows = self.recall(query)
        rendered: list[dict[str, Any]] = []
        for row in rows:
            if row.status != "accepted":
                continue
            if is_protected_memory(row.sensitivity, row.content_mode, row.operational_risk):
                continue
            rendered.append(
                {
                    "id": row.block_id,
                    "text": row.content,
                    "authority": row.authority,
                    "source_backend": self.backend_id,
                    "sensitivity": row.sensitivity,
                    "content_mode": row.content_mode,
                    "operational_risk": row.operational_risk,
                }
            )
        return rendered

    def _evict_over_cap(self) -> int:
        return 0

    def _delete_blocks(self, block_ids: set[str] | list[str], *, commit: bool = True) -> int:
        return 0

    @staticmethod
    def _is_expired(block: MemoryBlockRecord, *, now: datetime) -> bool:
        expires_at = _parse_time(block.expires_at)
        return expires_at is not None and expires_at <= now
