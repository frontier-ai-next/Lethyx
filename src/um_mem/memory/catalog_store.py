import base64
import binascii
import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from filelock import FileLock

from um_mem.contracts.memory import _closed_value
from um_mem.memory.active_mask import ActiveMemoryMask
from um_mem.memory.blocks.models import MemoryBlock
from um_mem.memory.paths import workspace_memory_root

BlockLifecycle = Literal["active", "archived"]
BlockEdgeType = Literal[
    "related_to",
    "supports",
    "contradicts",
    "derived_from",
    "supersedes",
    "applies_to",
    "learned_from",
    "validated_by",
]
MaintenanceObservationType = Literal[
    "capacity_pressure",
    "low_usage",
    "high_revision_count",
    "many_conflicts",
    "active_reference_present",
    "consolidation_available",
]

BLOCK_EDGE_TYPES = frozenset(
    {
        "related_to",
        "supports",
        "contradicts",
        "derived_from",
        "supersedes",
        "applies_to",
        "learned_from",
        "validated_by",
    }
)


class CatalogRevisionConflict(ValueError):
    """A catalog writer observed a stale catalog revision."""


MAINTENANCE_OBSERVATION_TYPES = frozenset(
    {
        "capacity_pressure",
        "low_usage",
        "high_revision_count",
        "many_conflicts",
        "active_reference_present",
        "consolidation_available",
    }
)

_SCHEMA_VERSION = 2
_CREATE = """
CREATE TABLE catalog_metadata (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE topology_blocks (
    block_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'archived')),
    revision INTEGER NOT NULL CHECK(revision >= 1),
    protected INTEGER NOT NULL CHECK(protected IN (0, 1)),
    backend_partition_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_topology_blocks_parent
ON topology_blocks(workspace_id, profile_id, parent_id, status, block_id);
CREATE TABLE topology_edges (
    edge_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK(revision >= 1),
    created_at TEXT NOT NULL,
    UNIQUE(workspace_id, profile_id, src_id, dst_id, edge_type)
);
CREATE INDEX idx_topology_edges_src
ON topology_edges(workspace_id, profile_id, src_id, edge_type);
CREATE INDEX idx_topology_edges_dst
ON topology_edges(workspace_id, profile_id, dst_id, edge_type);
CREATE TABLE active_masks (
    mask_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK(revision >= 1),
    status TEXT NOT NULL CHECK(status IN ('active', 'suspended', 'expired')),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_active_masks_scope
ON active_masks(workspace_id, profile_id, session_id, task_id, updated_at, mask_id);
CREATE TABLE maintenance_observations (
    observation_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    observation_type TEXT NOT NULL,
    values_json TEXT NOT NULL,
    threshold_revision TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_maintenance_observations_block
ON maintenance_observations(workspace_id, profile_id, block_id, created_at);
CREATE TABLE catalog_mutations (
    mutation_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    previous_revision INTEGER NOT NULL,
    next_revision INTEGER NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    affected_ids_json TEXT NOT NULL,
    operation TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    output_fingerprint TEXT NOT NULL,
    validator_result TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_catalog_mutations_revision
ON catalog_mutations(workspace_id, profile_id, next_revision);
CREATE TABLE topology_tombstones (
    block_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    deleted_at TEXT NOT NULL
);
CREATE TABLE topology_evolutions (
    operation_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('prepared', 'projections_ready', 'published', 'aborted')),
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    abort_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (workspace_id, profile_id, operation_id)
);
CREATE INDEX idx_topology_evolutions_state
ON topology_evolutions(workspace_id, profile_id, state, created_at);
CREATE TABLE published_record_memberships (
    record_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    record_revision_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    backend_partition_id TEXT NOT NULL,
    catalog_revision_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    published_at TEXT NOT NULL,
    FOREIGN KEY(workspace_id, profile_id, operation_id)
        REFERENCES topology_evolutions(workspace_id, profile_id, operation_id)
);
CREATE INDEX idx_published_memberships_block
ON published_record_memberships(workspace_id, profile_id, block_id, record_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _required_text(value: object, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _scope_component(value: object, *, field_name: str) -> str:
    normalized = _required_text(value, field_name=field_name)
    if normalized in {".", ".."} or any(
        separator in normalized for separator in ("/", "\\", "\x00")
    ):
        raise ValueError(f"{field_name} is invalid")
    return normalized


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_json(value).encode("ascii")).hexdigest()


def _memory_block_from_json(payload_json: str) -> MemoryBlock:
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):
        raise ValueError("stored topology block payload must be a mapping")
    return MemoryBlock(**payload)


@dataclass(frozen=True)
class TopologyBlockState:
    block: MemoryBlock
    workspace_id: str
    profile_id: str
    revision: int
    status: BlockLifecycle
    protected: bool
    backend_partition_id: str
    created_at: str
    updated_at: str

    @property
    def block_id(self) -> str:
        return self.block.block_id

    def to_mapping(self) -> dict[str, Any]:
        return {
            **asdict(self.block),
            "workspace_id": self.workspace_id,
            "profile_id": self.profile_id,
            "revision": self.revision,
            "status": self.status,
            "protected": self.protected,
            "backend_partition_id": self.backend_partition_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class BlockGraphEdge:
    edge_id: str
    workspace_id: str
    profile_id: str
    src_id: str
    dst_id: str
    edge_type: BlockEdgeType
    reason: str
    revision: int
    created_at: str

    def __post_init__(self) -> None:
        for field_name in (
            "edge_id",
            "workspace_id",
            "profile_id",
            "src_id",
            "dst_id",
            "reason",
            "created_at",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if self.edge_type not in BLOCK_EDGE_TYPES:
            raise ValueError("block edge type is unsupported")
        if self.src_id == self.dst_id:
            raise ValueError("block graph edges cannot be self-referential")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("block edge revision must be a positive int")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TopologyNavigation:
    states: tuple[TopologyBlockState, ...]
    path_ids: tuple[str, ...]
    child_ids: dict[str, tuple[str, ...]]
    edges: dict[str, tuple[BlockGraphEdge, ...]]
    catalog_revision: int
    child_counts: dict[str, int]
    next_cursors: dict[str, str]


def navigation_cursor(
    workspace_id: str, profile_id: str, block_id: str, revision: str, after: tuple[str, str]
) -> str:
    raw = json.dumps([workspace_id, profile_id, block_id, revision, *after], separators=(",", ":"))
    token = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
    if len(token) > 4096:
        raise ValueError("navigation cursor exceeds size limit")
    return token


def read_navigation_cursor(
    cursor: str, *, workspace_id: str, profile_id: str, block_id: str, revision: str | None
) -> tuple[str, str]:
    if not isinstance(cursor, str) or len(cursor) > 4096:
        raise ValueError("invalid navigation cursor")
    if not cursor:
        return "", ""
    try:
        payload = json.loads(
            base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        )
    except (ValueError, binascii.Error, UnicodeError) as exc:
        raise ValueError("invalid navigation cursor") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 6
        or any(not isinstance(item, str) for item in payload)
    ):
        raise ValueError("invalid navigation cursor")
    if payload[:3] != [workspace_id, profile_id, block_id]:
        raise ValueError("navigation cursor scope does not match")
    if revision is not None and payload[3] != revision:
        raise ValueError("navigation cursor revision changed; restart from the first page")
    return payload[4], payload[5]


@dataclass(frozen=True)
class MaintenanceObservation:
    observation_id: str
    workspace_id: str
    profile_id: str
    block_id: str
    observation_type: MaintenanceObservationType
    values: dict[str, int | float]
    threshold_revision: str
    evidence_ids: tuple[str, ...]
    created_at: str

    def __post_init__(self) -> None:
        for field_name in (
            "observation_id",
            "workspace_id",
            "profile_id",
            "block_id",
            "threshold_revision",
            "created_at",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if self.observation_type not in MAINTENANCE_OBSERVATION_TYPES:
            raise ValueError("maintenance observation type is unsupported")
        if not isinstance(self.values, dict) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in self.values.values()
        ):
            raise ValueError("maintenance observation values must be numeric facts")
        object.__setattr__(self, "values", dict(self.values))
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(
                dict.fromkeys(
                    _required_text(value, field_name="evidence_id") for value in self.evidence_ids
                )
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlockEvolutionTarget:
    name: str
    synopsis: str
    record_ids: tuple[str, ...]
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None
    token_budget: int = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, field_name="name"))
        object.__setattr__(
            self,
            "synopsis",
            _required_text(self.synopsis, field_name="synopsis"),
        )
        object.__setattr__(
            self,
            "record_ids",
            tuple(
                dict.fromkeys(
                    _required_text(record_id, field_name="record_id")
                    for record_id in self.record_ids
                )
            ),
        )
        object.__setattr__(
            self,
            "tags",
            tuple(dict.fromkeys(_required_text(tag, field_name="tag") for tag in self.tags)),
        )
        metadata = {} if self.metadata is None else self.metadata
        if not isinstance(metadata, dict):
            raise ValueError("block evolution metadata must be a mapping")
        _json(metadata)
        object.__setattr__(self, "metadata", dict(metadata))
        if type(self.token_budget) is not int or self.token_budget < 0:
            raise ValueError("block evolution token_budget must be a non-negative int")


@dataclass(frozen=True)
class BlockMergeSource:
    block_id: str
    expected_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "block_id",
            _required_text(self.block_id, field_name="block_id"),
        )
        if type(self.expected_revision) is not int or self.expected_revision < 1:
            raise ValueError("merge source expected_revision must be a positive int")


class WorkspaceCatalogStore:
    def __init__(self, repo_root: str | Path, workspace_id: str, profile_id: str) -> None:
        self.repo_root = Path(repo_root)
        self.workspace_id = _scope_component(workspace_id, field_name="workspace_id")
        self.profile_id = _scope_component(profile_id, field_name="profile_id")
        profile_key = hashlib.sha256(self.profile_id.encode("utf-8")).hexdigest()
        self.path = (
            workspace_memory_root(self.repo_root, self.workspace_id)
            / "catalogs"
            / f"{profile_key}.sqlite"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        initialization_lock = FileLock(f"{self.path}.initialize.lock", timeout=10.0)
        with initialization_lock:
            self._conn = sqlite3.connect(str(self.path), timeout=10.0, check_same_thread=False)
            try:
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.execute("PRAGMA busy_timeout=10000")
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=FULL")
                self._initialize()
            except Exception:
                self._conn.close()
                raise

    def close(self) -> None:
        self._conn.close()

    def _begin_catalog_write(self) -> None:
        """Acquire the single SQLite writer lane before reading its revision."""

        self._conn.execute("BEGIN IMMEDIATE")

    def _initialize(self) -> None:
        tables = {
            str(row[0])
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if not tables:
            with self._conn:
                self._conn.executescript(_CREATE)
                self._conn.executemany(
                    "INSERT INTO catalog_metadata(key, value) VALUES (?, ?)",
                    (("schema_version", _SCHEMA_VERSION), ("catalog_revision", 0)),
                )
        row = self._conn.execute(
            "SELECT value FROM catalog_metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None or int(row[0]) != _SCHEMA_VERSION:
            raise RuntimeError("workspace catalog schema mismatch; clean rebuild required")
        with self._conn:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_topology_navigation_order ON topology_blocks"
                "(workspace_id, profile_id, parent_id, status, created_at, block_id)"
            )
        existing = self._conn.execute("SELECT block_id FROM topology_blocks LIMIT 1").fetchone()
        if existing is None:
            self._create_root()

    def _create_root(self) -> None:
        now = _now()
        root = MemoryBlock(
            block_id=str(uuid.uuid4()),
            block_type="workspace_profile_root",
            temperature="warm",
            scope="workspace",
            authority="advisory",
            delivery_policy="on_demand",
            retention="durable",
            name="Memory",
            guidance={
                "synopsis": "Protected root for this workspace and profile.",
                "use_when": (),
                "avoid_when": (),
            },
            index_only=True,
            agent_writable=False,
            owner="UM-MEM",
            producer="catalog",
            consumers=("memory_agent",),
            recall_path="hybrid_recall",
            lifecycle_policy={
                "initial_status": "active",
                "mutable": False,
                "terminal_statuses": ("archived",),
            },
            isolation_policy={"dimension": "workspace", "cross_workspace": False},
            token_budget=0,
        )
        with self._conn:
            self._begin_catalog_write()
            self._conn.execute(
                """
                INSERT INTO topology_blocks(
                    block_id, workspace_id, profile_id, parent_id, status, revision,
                    protected, backend_partition_id, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, '', 'active', 1, 1, '', ?, ?, ?)
                """,
                (
                    root.block_id,
                    self.workspace_id,
                    self.profile_id,
                    _json(asdict(root)),
                    now,
                    now,
                ),
            )
            self._record_mutation(
                previous_revision=0,
                operation="create_root",
                actor="system",
                reason="initialize workspace/profile catalog",
                affected_ids=(root.block_id,),
                input_payload={},
                output_payload=asdict(root),
            )

    def catalog_revision(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM catalog_metadata WHERE key='catalog_revision'"
        ).fetchone()
        if row is None:
            raise RuntimeError("workspace catalog revision is missing")
        return int(row[0])

    def topology_evolution(self, operation_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT state, payload_json, payload_sha256, abort_reason FROM topology_evolutions "
            "WHERE operation_id=? AND workspace_id=? AND profile_id=?",
            (operation_id, self.workspace_id, self.profile_id),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[1]))
        if not isinstance(payload, dict) or _fingerprint(payload) != str(row[2]):
            raise RuntimeError("topology evolution journal payload is invalid")
        return {
            "operation_id": operation_id,
            "state": str(row[0]),
            "payload": payload,
            "abort_reason": str(row[3]),
        }

    def pending_topology_evolutions(self) -> tuple[dict[str, Any], ...]:
        rows = self._conn.execute(
            "SELECT operation_id,state,payload_json,payload_sha256 FROM topology_evolutions "
            "WHERE workspace_id=? AND profile_id=? "
            "AND state IN ('prepared','projections_ready') "
            "ORDER BY created_at,operation_id",
            (self.workspace_id, self.profile_id),
        ).fetchall()
        pending = []
        for row in rows:
            payload = json.loads(str(row[2]))
            if not isinstance(payload, dict) or _fingerprint(payload) != str(row[3]):
                raise RuntimeError("topology evolution journal payload is invalid")
            pending.append(
                {
                    "operation_id": str(row[0]),
                    "state": str(row[1]),
                    "operation": str(payload["operation"]),
                }
            )
        return tuple(pending)

    def prepare_topology_evolution(
        self, operation_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        operation_id = _required_text(operation_id, field_name="operation_id")
        if payload.get("operation") not in {"split", "merge"}:
            raise ValueError("topology evolution operation is invalid")
        if (
            not isinstance(payload.get("sources"), dict)
            or not isinstance(payload.get("targets"), list)
            or not isinstance(payload.get("assignments"), dict)
        ):
            raise ValueError("topology evolution mapping is invalid")
        digest = _fingerprint(payload)
        now = _now()
        with self._conn:
            self._begin_catalog_write()
            existing = self.topology_evolution(operation_id)
            if existing is not None:
                if _fingerprint(existing["payload"]) != digest:
                    raise ValueError("operation_id already binds a different topology evolution")
                return existing
            pending = self._conn.execute(
                "SELECT payload_json FROM topology_evolutions "
                "WHERE workspace_id=? AND profile_id=? "
                "AND state IN ('prepared','projections_ready')",
                (self.workspace_id, self.profile_id),
            ).fetchall()
            if any(
                set(json.loads(str(row[0]))["sources"]).intersection(payload["sources"])
                for row in pending
            ):
                raise ValueError("source block already has a pending topology evolution")
            self._conn.execute(
                "INSERT INTO topology_evolutions(operation_id,workspace_id,profile_id,"
                "state,payload_json,payload_sha256,created_at,updated_at) "
                "VALUES(?,?,?,'prepared',?,?,?,?)",
                (
                    operation_id,
                    self.workspace_id,
                    self.profile_id,
                    _json(payload),
                    digest,
                    now,
                    now,
                ),
            )
        return {"operation_id": operation_id, "state": "prepared", "payload": payload}

    def mark_topology_projections_ready(self, operation_id: str) -> None:
        with self._conn:
            self._begin_catalog_write()
            evolution = self.topology_evolution(operation_id)
            if evolution is None:
                raise ValueError("topology evolution operation was not found")
            if evolution["state"] == "published":
                return
            self._conn.execute(
                "UPDATE topology_evolutions SET state='projections_ready',updated_at=? "
                "WHERE operation_id=? AND workspace_id=? AND profile_id=? "
                "AND state='prepared'",
                (_now(), operation_id, self.workspace_id, self.profile_id),
            )

    def abort_topology_evolution(self, operation_id: str, *, reason: str) -> None:
        """Retain the exact journal while releasing an irrecoverable source claim."""

        reason = _required_text(reason, field_name="reason")
        with self._conn:
            self._begin_catalog_write()
            evolution = self.topology_evolution(operation_id)
            if evolution is None:
                raise ValueError("topology evolution operation was not found")
            if evolution["state"] == "published":
                raise ValueError("published topology evolution cannot be aborted")
            self._conn.execute(
                "UPDATE topology_evolutions SET state='aborted',abort_reason=?,updated_at=? "
                "WHERE operation_id=? AND workspace_id=? AND profile_id=? "
                "AND state IN ('prepared','projections_ready')",
                (reason, _now(), operation_id, self.workspace_id, self.profile_id),
            )

    def published_record_membership(
        self, record_id: str, *, record_revision_id: str = ""
    ) -> tuple[str, str, str] | None:
        row = self._conn.execute(
            "SELECT block_id,backend_partition_id,catalog_revision_id,record_revision_id "
            "FROM published_record_memberships WHERE record_id=? AND workspace_id=? "
            "AND profile_id=?",
            (record_id, self.workspace_id, self.profile_id),
        ).fetchone()
        if row is None:
            return None
        if record_revision_id and str(row[3]) != record_revision_id:
            return None
        return str(row[0]), str(row[1]), str(row[2])

    def published_memberships_for_blocks(
        self, block_ids: tuple[str, ...]
    ) -> dict[str, tuple[str, str, str, str]]:
        if not block_ids:
            return {}
        placeholders = ",".join("?" for _ in block_ids)
        rows = self._conn.execute(
            "SELECT record_id,record_revision_id,block_id,backend_partition_id,"
            "catalog_revision_id FROM published_record_memberships "
            "WHERE workspace_id=? AND profile_id=? AND block_id IN (" + placeholders + ")",
            (self.workspace_id, self.profile_id, *block_ids),
        ).fetchall()
        return {str(row[0]): (str(row[1]), str(row[2]), str(row[3]), str(row[4])) for row in rows}

    def publish_topology_evolution(self, operation_id: str) -> dict[str, Any]:
        """Publish one prepared generation and its exact membership pointer atomically."""

        with self._conn:
            self._begin_catalog_write()
            evolution = self.topology_evolution(operation_id)
            if evolution is None:
                raise ValueError("topology evolution operation was not found")
            if evolution["state"] == "published":
                return evolution
            if evolution["state"] != "projections_ready":
                raise ValueError("topology native projections are not ready")
            payload = evolution["payload"]
            sources = payload["sources"]
            targets = payload["targets"]
            assignments = payload["assignments"]
            if not sources or not targets or not assignments:
                raise ValueError("topology evolution requires nonempty exact mappings")
            for source_id, expected_revision in sources.items():
                state = self.get_block(source_id)
                if (
                    state is None
                    or state.status != "active"
                    or state.revision != expected_revision
                    or state.protected
                    or state.block.index_only
                    or self.block_has_active_mask_reference(source_id)
                ):
                    raise CatalogRevisionConflict(
                        "topology evolution source changed before publication"
                    )
                edges = self.edges_for(source_id)
                if payload["operation"] == "split" and any(
                    edge.src_id != source_id or edge.edge_type != "derived_from" for edge in edges
                ):
                    raise CatalogRevisionConflict("split source graph edges changed")
                if payload["operation"] == "merge" and any(
                    edge.dst_id == source_id for edge in edges
                ):
                    raise CatalogRevisionConflict("merge source incoming edges changed")
            target_ids = {str(item["block"]["block_id"]) for item in targets}
            if len(target_ids) != len(targets) or any(
                self.get_block(target_id) is not None for target_id in target_ids
            ):
                raise CatalogRevisionConflict("topology evolution target already exists")
            if {item["target_block_id"] for item in assignments.values()} != target_ids:
                raise ValueError("topology evolution assignments must cover all targets")
            now = _now()
            previous_revision = self.catalog_revision()
            for item in targets:
                block = MemoryBlock(**item["block"])
                parent = self.get_block(block.parent_id)
                if parent is None or parent.status != "active" or not parent.block.index_only:
                    raise CatalogRevisionConflict("topology evolution parent changed")
                self._conn.execute(
                    "INSERT INTO topology_blocks(block_id,workspace_id,profile_id,parent_id,"
                    "status,revision,protected,backend_partition_id,payload_json,created_at,updated_at) "
                    "VALUES(?,?,?,?,'active',1,0,?,?,?,?)",
                    (
                        block.block_id,
                        self.workspace_id,
                        self.profile_id,
                        block.parent_id,
                        item["backend_partition_id"],
                        _json(asdict(block)),
                        now,
                        now,
                    ),
                )
                for source_id in sources:
                    self._conn.execute(
                        "INSERT INTO topology_edges(edge_id,workspace_id,profile_id,src_id,dst_id,"
                        "edge_type,reason,revision,created_at) "
                        "VALUES(?,?,?,?,?,'derived_from',?,1,?)",
                        (
                            str(uuid.uuid4()),
                            self.workspace_id,
                            self.profile_id,
                            block.block_id,
                            source_id,
                            payload["reason"],
                            now,
                        ),
                    )
            for source_id, expected_revision in sources.items():
                cursor = self._conn.execute(
                    "UPDATE topology_blocks SET status='archived',revision=revision+1,"
                    "updated_at=? WHERE block_id=? AND revision=? AND status='active'",
                    (now, source_id, expected_revision),
                )
                if cursor.rowcount != 1:
                    raise CatalogRevisionConflict("topology evolution source CAS conflict")
            for record_id, assignment in assignments.items():
                target = next(
                    item
                    for item in targets
                    if item["block"]["block_id"] == assignment["target_block_id"]
                )
                previous = self._conn.execute(
                    "SELECT workspace_id,profile_id FROM published_record_memberships "
                    "WHERE record_id=?",
                    (record_id,),
                ).fetchone()
                if previous is not None and tuple(previous) != (
                    self.workspace_id,
                    self.profile_id,
                ):
                    raise CatalogRevisionConflict("topology membership scope changed")
                self._conn.execute(
                    "INSERT INTO published_record_memberships(record_id,workspace_id,profile_id,"
                    "record_revision_id,block_id,backend_partition_id,catalog_revision_id,"
                    "operation_id,published_at) VALUES(?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(record_id) DO UPDATE SET record_revision_id=excluded.record_revision_id,"
                    "block_id=excluded.block_id,backend_partition_id=excluded.backend_partition_id,"
                    "catalog_revision_id=excluded.catalog_revision_id,operation_id=excluded.operation_id,"
                    "published_at=excluded.published_at",
                    (
                        record_id,
                        self.workspace_id,
                        self.profile_id,
                        assignment["record_revision_id"],
                        assignment["target_block_id"],
                        target["backend_partition_id"],
                        f"topology:{operation_id}",
                        operation_id,
                        now,
                    ),
                )
            self._conn.execute(
                "UPDATE topology_evolutions SET state='published',updated_at=? "
                "WHERE operation_id=? AND workspace_id=? AND profile_id=? "
                "AND state='projections_ready'",
                (now, operation_id, self.workspace_id, self.profile_id),
            )
            self._record_mutation(
                previous_revision=previous_revision,
                operation=f"{payload['operation']}_blocks",
                actor=payload["actor"],
                reason=payload["reason"],
                affected_ids=(*sources, *target_ids),
                input_payload={"sources": sources, "assignments": assignments},
                output_payload={"targets": targets, "operation_id": operation_id},
            )
        published = self.topology_evolution(operation_id)
        if published is None:
            raise RuntimeError("published topology evolution is missing")
        return published

    def _record_mutation(
        self,
        *,
        previous_revision: int,
        operation: str,
        actor: str,
        reason: str,
        affected_ids: tuple[str, ...],
        input_payload: object,
        output_payload: object,
    ) -> int:
        next_revision = previous_revision + 1
        cursor = self._conn.execute(
            "UPDATE catalog_metadata SET value=? WHERE key='catalog_revision' AND value=?",
            (next_revision, previous_revision),
        )
        if cursor.rowcount != 1:
            raise CatalogRevisionConflict("catalog revision conflict")
        self._conn.execute(
            """
            INSERT INTO catalog_mutations(
                mutation_id, workspace_id, profile_id, previous_revision, next_revision,
                actor, reason, affected_ids_json, operation, input_fingerprint,
                output_fingerprint, validator_result, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted', ?)
            """,
            (
                str(uuid.uuid4()),
                self.workspace_id,
                self.profile_id,
                previous_revision,
                next_revision,
                _required_text(actor, field_name="actor"),
                _required_text(reason, field_name="reason"),
                _json(affected_ids),
                _required_text(operation, field_name="operation"),
                _fingerprint(input_payload),
                _fingerprint(output_payload),
                _now(),
            ),
        )
        return next_revision

    @staticmethod
    def _block_state(row: tuple[Any, ...]) -> TopologyBlockState:
        status = str(row[4])
        if status not in {"active", "archived"}:
            raise ValueError("stored block status is unsupported")
        return TopologyBlockState(
            block=_memory_block_from_json(str(row[8])),
            workspace_id=str(row[1]),
            profile_id=str(row[2]),
            revision=int(row[5]),
            status=cast(BlockLifecycle, status),
            protected=bool(row[6]),
            backend_partition_id=str(row[7]),
            created_at=str(row[9]),
            updated_at=str(row[10]),
        )

    def list_blocks(self, *, include_archived: bool = False) -> list[TopologyBlockState]:
        where = "" if include_archived else " AND status='active'"
        rows = self._conn.execute(
            """
            SELECT block_id, workspace_id, profile_id, parent_id, status, revision,
                   protected, backend_partition_id, payload_json, created_at, updated_at
            FROM topology_blocks
            WHERE workspace_id=? AND profile_id=?
            """
            + where
            + " ORDER BY created_at, block_id",
            (self.workspace_id, self.profile_id),
        ).fetchall()
        return [self._block_state(row) for row in rows]

    def get_block(self, block_id: str) -> TopologyBlockState | None:
        row = self._conn.execute(
            """
            SELECT block_id, workspace_id, profile_id, parent_id, status, revision,
                   protected, backend_partition_id, payload_json, created_at, updated_at
            FROM topology_blocks
            WHERE workspace_id=? AND profile_id=? AND block_id=?
            """,
            (self.workspace_id, self.profile_id, str(block_id or "").strip()),
        ).fetchone()
        return self._block_state(row) if row is not None else None

    def navigation(
        self,
        *,
        block_id: str = "",
        depth: int = 1,
        include_edges: bool = False,
        child_limit: int = 50,
        cursor: str = "",
    ) -> TopologyNavigation:
        """Read one visible branch in a consistent snapshot without decoding other branches."""
        if type(depth) is not int or not 1 <= depth <= 8:
            raise ValueError("navigation depth must be an integer from 1 to 8")
        if type(child_limit) is not int or not 1 <= child_limit <= 200:
            raise ValueError("navigation child limit must be an integer from 1 to 200")
        with self._conn:
            self._conn.execute("BEGIN")
            revision = self.catalog_revision()
            path: list[str] = []
            current_id = block_id
            while current_id:
                state = self.get_block(current_id)
                if state is None or state.status != "active":
                    read_navigation_cursor(
                        cursor,
                        workspace_id=self.workspace_id,
                        profile_id=self.profile_id,
                        block_id=block_id,
                        revision=None,
                    )
                    return TopologyNavigation((), (), {}, {}, revision, {}, {})
                if current_id in path:
                    raise ValueError("memory block hierarchy contains a cycle")
                path.append(current_id)
                current_id = state.block.parent_id
            after = read_navigation_cursor(
                cursor,
                workspace_id=self.workspace_id,
                profile_id=self.profile_id,
                block_id=block_id,
                revision=str(revision),
            )
            if block_id:
                pending = [(block_id, 1)]
            else:
                pending = [
                    (str(row[0]), 1)
                    for row in self._conn.execute(
                        "SELECT block_id FROM topology_blocks "
                        "WHERE workspace_id=? AND profile_id=? AND parent_id='' AND status='active' "
                        "ORDER BY created_at, block_id",
                        (self.workspace_id, self.profile_id),
                    )
                ]
            states: list[TopologyBlockState] = []
            children: dict[str, tuple[str, ...]] = {}
            edges: dict[str, tuple[BlockGraphEdge, ...]] = {}
            counts: dict[str, int] = {}
            cursors: dict[str, str] = {}
            for selected_id, level in pending:
                state = self.get_block(selected_id)
                if state is None or state.status != "active":
                    raise ValueError("navigation encountered an unavailable block")
                states.append(state)
                scope = (self.workspace_id, self.profile_id, selected_id)
                counts[selected_id] = int(
                    self._conn.execute(
                        "SELECT COUNT(*) FROM topology_blocks WHERE workspace_id=? AND profile_id=? "
                        "AND parent_id=? AND status='active'",
                        scope,
                    ).fetchone()[0]
                )
                # Child summaries disclose branch size, not another unbounded list of grandchildren.
                page_limit = child_limit if level < depth or depth == 1 else 0
                if level < depth:
                    page_limit = min(page_limit, max(0, 201 - len(pending)))
                page_after = after if selected_id == block_id else ("", "")
                rows = self._conn.execute(
                    "SELECT created_at, block_id FROM topology_blocks "
                    "WHERE workspace_id=? AND profile_id=? AND parent_id=? AND status='active' "
                    "AND (created_at,block_id)>(?,?) ORDER BY created_at, block_id LIMIT ?",
                    (*scope, *page_after, page_limit + 1),
                ).fetchall()
                shown = rows[:page_limit]
                children[selected_id] = tuple(str(row[1]) for row in shown)
                cursors[selected_id] = (
                    navigation_cursor(
                        self.workspace_id,
                        self.profile_id,
                        selected_id,
                        str(revision),
                        (str(shown[-1][0]), str(shown[-1][1])) if shown else page_after,
                    )
                    if len(rows) > page_limit
                    else ""
                )
                if include_edges:
                    edges[selected_id] = tuple(self.edges_for(selected_id))
                if level < depth:
                    pending.extend((child_id, level + 1) for child_id in children[selected_id])
            return TopologyNavigation(
                tuple(states), tuple(reversed(path)), children, edges, revision, counts, cursors
            )

    def create_block(
        self,
        block: MemoryBlock,
        *,
        backend_partition_id: str,
        actor: str,
        reason: str,
        protected: bool = False,
    ) -> TopologyBlockState:
        if self.get_block(block.block_id) is not None:
            raise ValueError("memory block already exists")
        now = _now()
        with self._conn:
            self._begin_catalog_write()
            previous_catalog_revision = self.catalog_revision()
            self._conn.execute(
                """
                INSERT INTO topology_blocks(
                    block_id, workspace_id, profile_id, parent_id, status, revision,
                    protected, backend_partition_id, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'active', 1, ?, ?, ?, ?, ?)
                """,
                (
                    block.block_id,
                    self.workspace_id,
                    self.profile_id,
                    block.parent_id,
                    int(protected),
                    backend_partition_id,
                    _json(asdict(block)),
                    now,
                    now,
                ),
            )
            self._record_mutation(
                previous_revision=previous_catalog_revision,
                operation="create_block",
                actor=actor,
                reason=reason,
                affected_ids=(block.block_id,),
                input_payload={"parent_id": block.parent_id},
                output_payload=asdict(block),
            )
        created = self.get_block(block.block_id)
        if created is None:
            raise RuntimeError("created topology block is missing")
        return created

    def replace_block(
        self,
        state: TopologyBlockState,
        block: MemoryBlock,
        *,
        expected_revision: int,
        status: BlockLifecycle | None,
        actor: str,
        reason: str,
        operation: str,
        allow_protected: bool = False,
        backend_partition_id: str | None = None,
    ) -> TopologyBlockState:
        if state.protected and not allow_protected:
            raise ValueError("protected root cannot be amended")
        if state.revision != expected_revision:
            raise ValueError("expected_revision does not match the current block revision")
        next_status = state.status if status is None else status
        if next_status not in {"active", "archived"}:
            raise ValueError("block lifecycle is unsupported")
        now = _now()
        with self._conn:
            self._begin_catalog_write()
            previous_catalog_revision = self.catalog_revision()
            cursor = self._conn.execute(
                """
                UPDATE topology_blocks
                SET parent_id=?, status=?, revision=?, payload_json=?, updated_at=?,
                    backend_partition_id=?
                WHERE block_id=? AND workspace_id=? AND profile_id=? AND revision=?
                """,
                (
                    block.parent_id,
                    next_status,
                    state.revision + 1,
                    _json(asdict(block)),
                    now,
                    (
                        state.backend_partition_id
                        if backend_partition_id is None
                        else str(backend_partition_id)
                    ),
                    state.block_id,
                    self.workspace_id,
                    self.profile_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("expected_revision does not match the current block revision")
            self._record_mutation(
                previous_revision=previous_catalog_revision,
                operation=operation,
                actor=actor,
                reason=reason,
                affected_ids=(state.block_id,),
                input_payload=state.to_mapping(),
                output_payload={**asdict(block), "status": next_status},
            )
        updated = self.get_block(state.block_id)
        if updated is None:
            raise RuntimeError("updated topology block is missing")
        return updated

    def protect_block(
        self,
        state: TopologyBlockState,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
    ) -> TopologyBlockState:
        if state.revision != expected_revision:
            raise ValueError("expected_revision does not match the current block revision")
        if state.protected:
            return state
        now = _now()
        with self._conn:
            self._begin_catalog_write()
            previous_catalog_revision = self.catalog_revision()
            cursor = self._conn.execute(
                """
                UPDATE topology_blocks
                SET protected=1, revision=?, updated_at=?
                WHERE block_id=? AND workspace_id=? AND profile_id=? AND revision=?
                """,
                (
                    state.revision + 1,
                    now,
                    state.block_id,
                    self.workspace_id,
                    self.profile_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("expected_revision does not match the current block revision")
            self._record_mutation(
                previous_revision=previous_catalog_revision,
                operation="protect_block",
                actor=actor,
                reason=reason,
                affected_ids=(state.block_id,),
                input_payload=state.to_mapping(),
                output_payload={**state.to_mapping(), "protected": True},
            )
        protected = self.get_block(state.block_id)
        if protected is None:
            raise RuntimeError("protected topology block is missing")
        return protected

    def delete_block(
        self,
        state: TopologyBlockState,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        if state.protected:
            raise ValueError("protected root cannot be deleted")
        if state.revision != expected_revision:
            raise ValueError("expected_revision does not match the current block revision")
        deleted_at = _now()
        tombstone = {
            "block_id": state.block_id,
            "workspace_id": self.workspace_id,
            "profile_id": self.profile_id,
            "revision": state.revision,
            "reason": reason,
            "deleted_at": deleted_at,
            "fingerprint": _fingerprint(state.to_mapping()),
        }
        with self._conn:
            self._begin_catalog_write()
            previous_catalog_revision = self.catalog_revision()
            cursor = self._conn.execute(
                """
                DELETE FROM topology_blocks
                WHERE block_id=? AND workspace_id=? AND profile_id=? AND revision=?
                """,
                (state.block_id, self.workspace_id, self.profile_id, expected_revision),
            )
            if cursor.rowcount != 1:
                raise ValueError("expected_revision does not match the current block revision")
            self._conn.execute(
                """
                INSERT INTO topology_tombstones(
                    block_id, workspace_id, profile_id, payload_json, deleted_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    state.block_id,
                    self.workspace_id,
                    self.profile_id,
                    _json(tombstone),
                    deleted_at,
                ),
            )
            self._record_mutation(
                previous_revision=previous_catalog_revision,
                operation="delete_block",
                actor=actor,
                reason=reason,
                affected_ids=(state.block_id,),
                input_payload=state.to_mapping(),
                output_payload=tombstone,
            )
        return tombstone

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        *,
        edge_type: BlockEdgeType,
        reason: str,
        actor: str,
    ) -> BlockGraphEdge:
        if edge_type not in BLOCK_EDGE_TYPES:
            raise ValueError("block edge type is unsupported")
        src = self.get_block(src_id)
        dst = self.get_block(dst_id)
        if src is None or dst is None:
            raise ValueError("block edge endpoints must exist in the same workspace/profile")
        edge = BlockGraphEdge(
            edge_id=str(uuid.uuid4()),
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            src_id=src.block_id,
            dst_id=dst.block_id,
            edge_type=edge_type,
            reason=reason,
            revision=1,
            created_at=_now(),
        )
        with self._conn:
            self._begin_catalog_write()
            previous_catalog_revision = self.catalog_revision()
            try:
                self._conn.execute(
                    """
                    INSERT INTO topology_edges(
                        edge_id, workspace_id, profile_id, src_id, dst_id, edge_type,
                        reason, revision, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge.edge_id,
                        edge.workspace_id,
                        edge.profile_id,
                        edge.src_id,
                        edge.dst_id,
                        edge.edge_type,
                        edge.reason,
                        edge.revision,
                        edge.created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("block edge already exists") from exc
            self._record_mutation(
                previous_revision=previous_catalog_revision,
                operation="add_block_edge",
                actor=actor,
                reason=reason,
                affected_ids=(src.block_id, dst.block_id, edge.edge_id),
                input_payload={"src_id": src.block_id, "dst_id": dst.block_id},
                output_payload=edge.to_mapping(),
            )
        return edge

    @staticmethod
    def _edge(row: tuple[Any, ...]) -> BlockGraphEdge:
        edge_type = str(row[5])
        if edge_type not in BLOCK_EDGE_TYPES:
            raise ValueError("stored block edge type is unsupported")
        return BlockGraphEdge(
            edge_id=str(row[0]),
            workspace_id=str(row[1]),
            profile_id=str(row[2]),
            src_id=str(row[3]),
            dst_id=str(row[4]),
            edge_type=cast(BlockEdgeType, edge_type),
            reason=str(row[6]),
            revision=int(row[7]),
            created_at=str(row[8]),
        )

    def edges_for(self, block_id: str) -> list[BlockGraphEdge]:
        rows = self._conn.execute(
            """
            SELECT edge_id, workspace_id, profile_id, src_id, dst_id, edge_type,
                   reason, revision, created_at
            FROM topology_edges
            WHERE workspace_id=? AND profile_id=? AND (src_id=? OR dst_id=?)
            ORDER BY created_at, edge_id
            """,
            (self.workspace_id, self.profile_id, block_id, block_id),
        ).fetchall()
        return [self._edge(row) for row in rows]

    def list_edges(self) -> list[BlockGraphEdge]:
        """Return the complete content-free graph for this workspace profile."""

        rows = self._conn.execute(
            """
            SELECT edge_id, workspace_id, profile_id, src_id, dst_id, edge_type,
                   reason, revision, created_at
            FROM topology_edges
            WHERE workspace_id=? AND profile_id=?
            ORDER BY created_at, edge_id
            """,
            (self.workspace_id, self.profile_id),
        ).fetchall()
        return [self._edge(row) for row in rows]

    def remove_edge(
        self,
        edge_id: str,
        *,
        expected_revision: int,
        reason: str,
        actor: str,
    ) -> BlockGraphEdge:
        row = self._conn.execute(
            """
            SELECT edge_id, workspace_id, profile_id, src_id, dst_id, edge_type,
                   reason, revision, created_at
            FROM topology_edges
            WHERE workspace_id=? AND profile_id=? AND edge_id=?
            """,
            (self.workspace_id, self.profile_id, edge_id),
        ).fetchone()
        if row is None:
            raise ValueError("block edge was not found")
        edge = self._edge(row)
        if edge.revision != expected_revision:
            raise ValueError("expected_revision does not match the current edge revision")
        with self._conn:
            self._begin_catalog_write()
            previous_catalog_revision = self.catalog_revision()
            self._conn.execute(
                "DELETE FROM topology_edges WHERE edge_id=? AND revision=?",
                (edge.edge_id, expected_revision),
            )
            self._record_mutation(
                previous_revision=previous_catalog_revision,
                operation="remove_block_edge",
                actor=actor,
                reason=reason,
                affected_ids=(edge.src_id, edge.dst_id, edge.edge_id),
                input_payload=edge.to_mapping(),
                output_payload={"removed": True},
            )
        return edge

    def put_mask(
        self,
        mask: ActiveMemoryMask,
        *,
        expected_revision: int | None = None,
    ) -> ActiveMemoryMask:
        if mask.workspace_id != self.workspace_id or mask.profile_id != self.profile_id:
            raise ValueError("active memory mask workspace/profile mismatch")
        with self._conn:
            self._begin_catalog_write()
            current = self.get_mask(mask.mask_id)
            if expected_revision is None:
                if current is not None:
                    raise ValueError("active memory mask already exists")
                self._conn.execute(
                    """
                    INSERT INTO active_masks(
                        mask_id, workspace_id, profile_id, session_id, task_id,
                        revision, status, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mask.mask_id,
                        mask.workspace_id,
                        mask.profile_id,
                        mask.session_id,
                        mask.task_id,
                        mask.revision,
                        mask.status,
                        _json(mask.to_mapping()),
                        mask.created_at,
                        mask.updated_at,
                    ),
                )
                return mask
            if current is None or current.revision != expected_revision:
                raise ValueError("expected_revision does not match the current mask revision")
            cursor = self._conn.execute(
                """
                UPDATE active_masks
                SET session_id=?, task_id=?, revision=?, status=?, payload_json=?, updated_at=?
                WHERE mask_id=? AND revision=?
                """,
                (
                    mask.session_id,
                    mask.task_id,
                    mask.revision,
                    mask.status,
                    _json(mask.to_mapping()),
                    mask.updated_at,
                    mask.mask_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("expected_revision does not match the current mask revision")
        return mask

    def get_mask(self, mask_id: str) -> ActiveMemoryMask | None:
        row = self._conn.execute(
            """
            SELECT payload_json FROM active_masks
            WHERE workspace_id=? AND profile_id=? AND mask_id=?
            """,
            (self.workspace_id, self.profile_id, str(mask_id or "").strip()),
        ).fetchone()
        return ActiveMemoryMask.from_mapping(json.loads(str(row[0]))) if row is not None else None

    def current_mask(self, *, session_id: str, task_id: str) -> ActiveMemoryMask | None:
        row = self._conn.execute(
            """
            SELECT payload_json FROM active_masks
            WHERE workspace_id=? AND profile_id=? AND session_id=? AND task_id=?
            ORDER BY updated_at DESC, mask_id DESC
            LIMIT 1
            """,
            (self.workspace_id, self.profile_id, session_id, task_id),
        ).fetchone()
        return ActiveMemoryMask.from_mapping(json.loads(str(row[0]))) if row is not None else None

    def has_live_mask_for_scope(
        self,
        *,
        session_id: str,
        task_id: str,
        exclude_mask_id: str = "",
    ) -> bool:
        rows = self._conn.execute(
            """
            SELECT payload_json FROM active_masks
            WHERE workspace_id=? AND profile_id=? AND session_id=? AND task_id=?
              AND status IN ('active', 'suspended')
            """,
            (self.workspace_id, self.profile_id, session_id, task_id),
        ).fetchall()
        excluded = str(exclude_mask_id or "").strip()
        masks = (ActiveMemoryMask.from_mapping(json.loads(str(row[0]))) for row in rows)
        return any(mask.mask_id != excluded and not mask.is_expired() for mask in masks)

    def block_has_active_mask_reference(self, block_id: str) -> bool:
        rows = self._conn.execute(
            """
            SELECT payload_json FROM active_masks
            WHERE workspace_id=? AND profile_id=? AND status IN ('active', 'suspended')
            """,
            (self.workspace_id, self.profile_id),
        ).fetchall()
        masks = (ActiveMemoryMask.from_mapping(json.loads(str(row[0]))) for row in rows)
        return any(not mask.is_expired() and block_id in mask.selected_block_ids for mask in masks)

    def add_observation(
        self,
        *,
        block_id: str,
        observation_type: MaintenanceObservationType,
        values: dict[str, int | float],
        threshold_revision: str,
        evidence_ids: tuple[str, ...],
    ) -> MaintenanceObservation:
        if self.get_block(block_id) is None:
            raise ValueError("maintenance observation block was not found")
        observation = MaintenanceObservation(
            observation_id=str(uuid.uuid4()),
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            block_id=block_id,
            observation_type=observation_type,
            values=values,
            threshold_revision=threshold_revision,
            evidence_ids=evidence_ids,
            created_at=_now(),
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO maintenance_observations(
                    observation_id, workspace_id, profile_id, block_id,
                    observation_type, values_json, threshold_revision,
                    evidence_ids_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.observation_id,
                    observation.workspace_id,
                    observation.profile_id,
                    observation.block_id,
                    observation.observation_type,
                    _json(observation.values),
                    observation.threshold_revision,
                    _json(observation.evidence_ids),
                    observation.created_at,
                ),
            )
        return observation

    def observations(self, block_id: str) -> list[MaintenanceObservation]:
        rows = self._conn.execute(
            """
            SELECT observation_id, workspace_id, profile_id, block_id,
                   observation_type, values_json, threshold_revision,
                   evidence_ids_json, created_at
            FROM maintenance_observations
            WHERE workspace_id=? AND profile_id=? AND block_id=?
            ORDER BY created_at, observation_id
            """,
            (self.workspace_id, self.profile_id, block_id),
        ).fetchall()
        return [
            MaintenanceObservation(
                observation_id=str(row[0]),
                workspace_id=str(row[1]),
                profile_id=str(row[2]),
                block_id=str(row[3]),
                observation_type=cast(
                    MaintenanceObservationType,
                    _closed_value(
                        str(row[4]), field_name="observation_type", shape=MaintenanceObservationType
                    ),
                ),
                values=dict(json.loads(str(row[5]))),
                threshold_revision=str(row[6]),
                evidence_ids=tuple(json.loads(str(row[7]))),
                created_at=str(row[8]),
            )
            for row in rows
        ]
