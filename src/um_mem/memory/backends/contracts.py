"""Backend-neutral memory contracts.

These models describe UM-MEM memory semantics independently from any physical
storage implementation.  Backends may store the records natively or in a
backend-owned sidecar, but runtime authority is carried by these typed fields.
"""

import time
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, cast, get_args

from um_mem.contracts import (
    BackendId,
    EvidenceRef,
    MemoryBlockRecord,
    RecordBlockMembership,
    json_ready,
)
from um_mem.contracts.memory import (
    CURRENT_SCHEMA_VERSION,
    DeliveryPolicy,
    MemoryAuthority,
    MemoryRetention,
    MemoryStatus,
    _closed_value,
    compact_turn_child_claims,
    validate_memory_safety_axes,
)
from um_mem.memory.backends.base import MemoryHit

_ALLOWED_BACKENDS = frozenset(get_args(BackendId))
_AUTHORITY_CHANGING_STATUSES = frozenset({"accepted", "superseded", "deprecated", "retracted"})
_CANONICAL_RECORD_FIELDS = frozenset(MemoryBlockRecord.__dataclass_fields__)


class EdgeType:
    CONTAINS = "contains"
    DERIVED_FROM = "derived_from"
    CITES = "cites"
    TESTS = "tests"
    IMPLEMENTS = "implements"
    SUPERSEDES = "supersedes"
    SUPPORTS = "supports"
    VERIFIED_BY = "verified_by"
    VERIFIES = "verifies"
    REFUTES = "refutes"
    REVISES = "revises"
    CONTRADICTS = "contradicts"
    OBSERVED_IN = "observed_in"
    PRODUCED_BY = "produced_by"
    PRODUCED_IN = "produced_in"
    RELATED_TO = "related_to"
    REFERENCES_FILE = "references_file"
    FROM_SUBTASK = "from_subtask"
    TRIGGERED_BY_ERROR = "triggered_by_error"
    FLAGGED_BY = "flagged_by"
    BLOCKS = "blocks"
    APPLIED_REFLECTION = "applied_reflection"
    MENTIONS_SYMBOL = "mentions_symbol"


ALLOWED_EDGE_TYPES: frozenset[str] = frozenset(
    {
        EdgeType.CONTAINS,
        EdgeType.DERIVED_FROM,
        EdgeType.CITES,
        EdgeType.TESTS,
        EdgeType.IMPLEMENTS,
        EdgeType.SUPERSEDES,
        EdgeType.SUPPORTS,
        EdgeType.VERIFIED_BY,
        EdgeType.VERIFIES,
        EdgeType.REFUTES,
        EdgeType.REVISES,
        EdgeType.CONTRADICTS,
        EdgeType.OBSERVED_IN,
        EdgeType.PRODUCED_BY,
        EdgeType.PRODUCED_IN,
        EdgeType.RELATED_TO,
        EdgeType.REFERENCES_FILE,
        EdgeType.FROM_SUBTASK,
        EdgeType.TRIGGERED_BY_ERROR,
        EdgeType.FLAGGED_BY,
        EdgeType.BLOCKS,
        EdgeType.APPLIED_REFLECTION,
        EdgeType.MENTIONS_SYMBOL,
    }
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _evidence_from_any(value: Any) -> EvidenceRef:
    if isinstance(value, EvidenceRef):
        return value
    if isinstance(value, dict):
        return EvidenceRef.from_mapping(value)
    raise ValueError("evidence_refs must contain typed EvidenceRef values")


def _evidence_tuple(values: Any) -> tuple[EvidenceRef, ...]:
    if not values:
        return ()
    return tuple(_evidence_from_any(item) for item in values)


@dataclass(frozen=True)
class BackendCapabilities:
    """Effective operations of the selected backend, including its owned indexes.

    Recovery of one revision is distinct from atomic visibility of a whole batch.
    Cancellation means an interruptible backend operation, not a caller timeout.
    """

    backend_id: BackendId
    persistent: bool = False
    semantic_recall: bool = False
    fts_recall: bool = False
    graph_edges: bool = False
    lifecycle_mutation: bool = False
    workspace_isolation: bool = False
    run_subtask_isolation: bool = False
    value_tracking: bool = False
    export_import: bool = False
    exact_record_read: bool = False
    scoped_search: bool = False
    revision_history: bool = False
    revision_compare_and_swap: bool = False
    recoverable_writes: bool = False
    atomic_record_reassignment: bool = False
    staged_topology_evolution: bool = False
    atomic_batch_publication: bool = False
    batch_read: bool = False
    cancellation: bool = False
    requires_external_service: bool = False
    supports_embedded: bool = False

    def __post_init__(self) -> None:
        if self.backend_id not in _ALLOWED_BACKENDS:
            raise ValueError(f"backend_id must be one of {sorted(_ALLOWED_BACKENDS)}")
        for descriptor in fields(self):
            if descriptor.name != "backend_id" and type(getattr(self, descriptor.name)) is not bool:
                raise ValueError(f"backend capability {descriptor.name} must be a boolean")

    @classmethod
    def _feature_names(cls) -> frozenset[str]:
        return frozenset(field.name for field in fields(cls)) - {
            "backend_id",
            "requires_external_service",
            "supports_embedded",
        }

    @property
    def enabled_features(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self._feature_names() if getattr(self, name)))

    @classmethod
    def effective(
        cls,
        backend_id: BackendId,
        *,
        native_ok: bool,
        implemented: frozenset[str],
        requires_external_service: bool = False,
        supports_embedded: bool = False,
    ) -> "BackendCapabilities":
        if type(native_ok) is not bool:
            raise ValueError("backend native readiness must be a boolean")
        feature_names = cls._feature_names()
        unknown = implemented - feature_names
        if unknown:
            raise ValueError(f"unknown backend capabilities: {sorted(unknown)}")
        values = {name: bool(native_ok and name in implemented) for name in feature_names}
        return cls(
            backend_id=backend_id,
            **values,
            requires_external_service=bool(native_ok and requires_external_service),
            supports_embedded=bool(native_ok and supports_embedded),
        )


@dataclass(frozen=True)
class NativePartitionIdentity:
    backend_id: BackendId
    workspace_id: str
    primary_block_id: str
    catalog_revision_id: str = field(compare=False)
    partition_id: str
    isolation_class: str
    partition_mode: str

    def __post_init__(self) -> None:
        if self.backend_id not in _ALLOWED_BACKENDS:
            raise ValueError(f"backend_id must be one of {sorted(_ALLOWED_BACKENDS)}")
        for field_name in (
            "workspace_id",
            "primary_block_id",
            "catalog_revision_id",
            "partition_id",
            "partition_mode",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"native partition {field_name} is required")
        if self.isolation_class not in {"physical", "native_tag"}:
            raise ValueError("native partition isolation_class is unsupported")

    @property
    def collection_prefix(self) -> str:
        if self.backend_id != "mempalace":
            raise ValueError("collection_prefix is only defined for MemPalace partitions")
        digest = self.partition_id.rsplit(":", 1)[-1]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("MemPalace partition id must end in a SHA-256 digest")
        return f"ummem_{digest}"

    def to_mapping(self) -> dict[str, str]:
        return {
            "backend_id": self.backend_id,
            "workspace_id": self.workspace_id,
            "primary_block_id": self.primary_block_id,
            "catalog_revision_id": self.catalog_revision_id,
            "partition_id": self.partition_id,
            "isolation_class": self.isolation_class,
            "partition_mode": self.partition_mode,
        }


@dataclass(frozen=True)
class NativePartitionMetrics:
    identity: NativePartitionIdentity
    record_count: int = 0
    write_count: int = 0
    recall_count: int = 0
    hit_count: int = 0
    zero_hit_count: int = 0
    failure_count: int = 0
    record_bytes: int | None = None
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    rendered_tokens: int = 0
    consolidation_count: int = 0
    compaction_count: int = 0
    eviction_count: int = 0
    expiry_count: int = 0
    scoped_recall_count: int = 0
    broad_recall_count: int = 0
    last_accessed_at: str = ""
    health: str = "unknown"

    def __post_init__(self) -> None:
        integer_fields = (
            "record_count",
            "write_count",
            "recall_count",
            "hit_count",
            "zero_hit_count",
            "failure_count",
            "rendered_tokens",
            "consolidation_count",
            "compaction_count",
            "eviction_count",
            "expiry_count",
            "scoped_recall_count",
            "broad_recall_count",
        )
        if any(int(getattr(self, field_name)) < 0 for field_name in integer_fields):
            raise ValueError("native partition metric counters cannot be negative")
        if self.record_bytes is not None and self.record_bytes < 0:
            raise ValueError("native partition record_bytes cannot be negative")
        if self.latency_p50_ms < 0 or self.latency_p95_ms < 0:
            raise ValueError("native partition latency cannot be negative")

    def to_mapping(self) -> dict[str, Any]:
        return {
            **self.identity.to_mapping(),
            "record_count": self.record_count,
            "write_count": self.write_count,
            "recall_count": self.recall_count,
            "hit_count": self.hit_count,
            "zero_hit_count": self.zero_hit_count,
            "failure_count": self.failure_count,
            "record_bytes": self.record_bytes,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "rendered_tokens": self.rendered_tokens,
            "consolidation_count": self.consolidation_count,
            "compaction_count": self.compaction_count,
            "eviction_count": self.eviction_count,
            "expiry_count": self.expiry_count,
            "scoped_recall_count": self.scoped_recall_count,
            "broad_recall_count": self.broad_recall_count,
            "last_accessed_at": self.last_accessed_at,
            "health": self.health,
        }


@dataclass(frozen=True)
class MemoryIndexNode:
    node_id: str
    title: str
    node_type: str
    workspace_id: str = ""
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.node_id:
            object.__setattr__(self, "node_id", f"idx_{uuid.uuid4().hex[:16]}")
        if not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        now = _now()
        if not self.created_at:
            object.__setattr__(self, "created_at", now)
        if not self.updated_at:
            object.__setattr__(self, "updated_at", now)


@dataclass(frozen=True)
class MemoryEdgeRecord:
    src_id: str
    dst_id: str
    edge_type: str
    workspace_id: str = ""
    run_id: str = ""
    subtask_id: str = ""
    reason: str = ""
    weight: float = 1.0
    evidence_refs: tuple[EvidenceRef, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", _now())
        object.__setattr__(self, "evidence_refs", _evidence_tuple(self.evidence_refs))


class CanonicalHeadConflict(RuntimeError):
    """Raised when a canonical successor does not match the committed head."""


@dataclass(frozen=True)
class CanonicalHeadReservation:
    commit_revision: int
    workspace_id: str
    canonical_key_sha256: str
    predecessor_id: str
    successor_id: str
    successor_status: str
    successor_authority: str


@dataclass(frozen=True)
class MemoryUsageRecord:
    block_id: str
    workspace_id: str = ""
    run_id: str = ""
    subtask_id: str = ""
    included: bool = False
    skipped: bool = False
    blocked: bool = False
    outcome: str = "neutral"
    helped: bool = False
    hurt: bool = False
    evidence_refs: tuple[EvidenceRef, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", _now())
        if not isinstance(self.evidence_refs, tuple):
            object.__setattr__(self, "evidence_refs", _evidence_tuple(self.evidence_refs))


def validate_status_transition(
    current: MemoryBlockRecord,
    status: str,
    *,
    reason: str = "",
    evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...] | None = None,
    authority: str | None = None,
    retention: str | None = None,
    delivery_policy: str | None = None,
) -> None:
    if status not in get_args(MemoryStatus):
        raise ValueError(f"unknown status: {status}")
    refs = tuple(evidence_refs or ())
    if status in _AUTHORITY_CHANGING_STATUSES and not reason:
        raise ValueError("authority-changing status mutation requires reason")
    replace(
        current,
        status=cast(MemoryStatus, _closed_value(status, field_name="status", shape=MemoryStatus)),
        authority=cast(
            MemoryAuthority,
            _closed_value(
                authority or current.authority, field_name="authority", shape=MemoryAuthority
            ),
        ),
        retention=cast(
            MemoryRetention,
            _closed_value(
                retention or current.retention, field_name="retention", shape=MemoryRetention
            ),
        ),
        delivery_policy=cast(
            DeliveryPolicy,
            _closed_value(
                delivery_policy or current.delivery_policy,
                field_name="delivery_policy",
                shape=DeliveryPolicy,
            ),
        ),
        evidence_refs=refs or current.evidence_refs,
    )


def memory_event_to_block_record(event: Any, *, backend_id: str = "") -> MemoryBlockRecord:
    data = event if isinstance(event, dict) else getattr(event, "__dict__", {})
    metadata = dict(data.get("metadata") or {})
    source_path = str(data.get("source_path") or "").strip()
    if source_path and not str(metadata.get("source_path") or "").strip():
        metadata["source_path"] = source_path
    sensitivity, content_mode, operational_risk = validate_memory_safety_axes(
        data.get("sensitivity"), data.get("content_mode"), data.get("operational_risk")
    )
    return MemoryBlockRecord(
        block_id=str(
            data.get("block_id") or data.get("canonical_id") or data.get("event_id") or ""
        ),
        content=str(data.get("content") or ""),
        block_membership=(
            RecordBlockMembership.from_mapping(data["block_membership"])
            if data.get("block_membership") is not None
            else None
        ),
        canonical_key=str(data.get("canonical_key") or ""),
        revision_id=str(data.get("revision_id") or ""),
        duplicate_cluster_id=str(data.get("duplicate_cluster_id") or ""),
        recorded_at=str(data.get("recorded_at") or ""),
        title=str(data.get("title") or ""),
        kind=str(data.get("memory_kind") or data.get("kind") or "observation"),
        status=data.get("status", "candidate"),
        retention=data.get("retention", "durable"),
        delivery_policy=data.get("delivery_policy", "on_demand"),
        temperature=data.get("temperature", "warm"),
        authority=data.get("authority", "advisory"),
        trust_level=data.get("trust_level", "agent_reported"),
        scope=data.get("scope", "workspace"),
        verification_status=(data.get("verification_status", "unverified")),
        verification_methods=tuple(data.get("verification_methods") or ()),
        promotion_decision=(data.get("promotion_decision", "not_requested")),
        user_id=str(data.get("user_id") or ""),
        workspace_id=str(data.get("workspace_id") or ""),
        profile_id=str(data.get("profile_id") or "default"),
        run_id=str(data.get("run_id") or ""),
        stage_id=str(data.get("stage_id") or ""),
        subtask_id=str(data.get("subtask_id") or ""),
        session_id=data.get("session_id", ""),
        task_id=data.get("task_id", ""),
        parent_session_id=data.get("parent_session_id", ""),
        artifact_id=data.get("artifact_id", ""),
        artifact_revision=data.get("artifact_revision", ""),
        valid_from=data.get("valid_from", ""),
        valid_to=data.get("valid_to", ""),
        project_id=data.get("project_id", ""),
        repository_id=data.get("repository_id", ""),
        checkout=data.get("checkout", ""),
        worktree=data.get("worktree", ""),
        branch=data.get("branch", ""),
        revision=data.get("revision", ""),
        dirty_state=data.get("dirty_state", ""),
        schema_version=data.get("schema_version", CURRENT_SCHEMA_VERSION),
        hard_policy_version=data.get("hard_policy_version", 1),
        semantic_catalog_version=data.get("semantic_catalog_version", ""),
        producer=str(data.get("producer") or data.get("agent_kind") or "agent"),
        source_backend=str(data.get("source_backend") or "") or backend_id,
        tags=tuple(data.get("tags") or ()),
        evidence_refs=_evidence_tuple(data.get("evidence_refs") or ()),
        external_refs=dict(data.get("external_refs") or {}),
        sensitivity=sensitivity,
        content_mode=content_mode,
        operational_risk=operational_risk,
        child_claims=data.get("child_claims", ()),
        metadata=metadata,
    )


def _memory_hit_score(record: MemoryBlockRecord) -> float:
    metadata = dict(record.metadata or {})
    for key in ("retrieval_score", "score", "native_score"):
        if key not in metadata:
            continue
        try:
            return float(metadata[key])
        except (TypeError, ValueError):
            continue
    return 1.0


def block_record_to_memory_hit(record: MemoryBlockRecord) -> MemoryHit:
    return MemoryHit(
        text=record.content,
        source=f"{record.source_backend or 'memory'}:{record.block_id}",
        score=_memory_hit_score(record),
        kind=record.kind,
        tags=list(record.tags),
        metadata={
            **dict(record.metadata),
            "block_id": record.block_id,
            "title": record.title,
            "canonical_key": record.canonical_key,
            "revision_id": record.revision_id,
            "duplicate_cluster_id": record.duplicate_cluster_id,
            "recorded_at": record.recorded_at,
            "source_backend": record.source_backend,
            "producer": record.producer,
            "status": record.status,
            "retention": record.retention,
            "delivery_policy": record.delivery_policy,
            "temperature": record.temperature,
            "authority": record.authority,
            "trust_level": record.trust_level,
            "scope": record.scope,
            "verification_status": record.verification_status,
            "verification_methods": list(record.verification_methods),
            "promotion_decision": record.promotion_decision,
            "evidence_refs": json_ready(record.evidence_refs),
            "user_id": record.user_id,
            "workspace_id": record.workspace_id,
            "profile_id": record.profile_id,
            "run_id": record.run_id,
            "subtask_id": record.subtask_id,
            "session_id": record.session_id,
            "task_id": record.task_id,
            "parent_session_id": record.parent_session_id,
            "artifact_id": record.artifact_id,
            "artifact_revision": record.artifact_revision,
            "valid_from": record.valid_from,
            "valid_to": record.valid_to,
            "expires_at": record.expires_at,
            "project_id": record.project_id,
            "repository_id": record.repository_id,
            "checkout": record.checkout,
            "worktree": record.worktree,
            "branch": record.branch,
            "revision": record.revision,
            "dirty_state": record.dirty_state,
            "schema_version": record.schema_version,
            "hard_policy_version": record.hard_policy_version,
            "semantic_catalog_version": record.semantic_catalog_version,
            "sensitivity": record.sensitivity,
            "content_mode": record.content_mode,
            "operational_risk": record.operational_risk,
            "child_claims": compact_turn_child_claims(record.child_claims),
            "block_membership": (
                asdict(record.block_membership) if record.block_membership is not None else None
            ),
        },
    )


def index_node_from_json(data: dict[str, Any]) -> MemoryIndexNode:
    payload = dict(data)
    payload["tags"] = tuple(payload.get("tags") or ())
    return MemoryIndexNode(
        **{k: v for k, v in payload.items() if k in MemoryIndexNode.__dataclass_fields__}
    )


def index_node_to_json(node: MemoryIndexNode) -> dict[str, Any]:
    return asdict(node)


def block_from_json(data: dict[str, Any]) -> MemoryBlockRecord:
    if not isinstance(data, dict):
        raise ValueError("canonical memory record must be an object")
    payload = dict(data)
    payload.setdefault("profile_id", "default")
    missing = sorted(_CANONICAL_RECORD_FIELDS.difference(payload))
    unknown = sorted(set(payload).difference(_CANONICAL_RECORD_FIELDS))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError("canonical memory record fields must be exact: " + "; ".join(details))
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != CURRENT_SCHEMA_VERSION
    ):
        raise ValueError(f"canonical memory record schema_version must be {CURRENT_SCHEMA_VERSION}")
    payload["tags"] = tuple(payload.get("tags") or ())
    payload["supersedes_ids"] = tuple(payload.get("supersedes_ids") or ())
    payload["evidence_refs"] = _evidence_tuple(payload.get("evidence_refs") or ())
    if payload.get("block_membership") is not None:
        payload["block_membership"] = RecordBlockMembership.from_mapping(
            payload["block_membership"]
        )
    return MemoryBlockRecord(
        **{k: v for k, v in payload.items() if k in MemoryBlockRecord.__dataclass_fields__}
    )


def block_to_json(record: MemoryBlockRecord) -> dict[str, Any]:
    if type(record.schema_version) is not int or record.schema_version != CURRENT_SCHEMA_VERSION:
        raise ValueError(f"canonical memory record schema_version must be {CURRENT_SCHEMA_VERSION}")
    payload = asdict(record)
    payload["evidence_refs"] = [asdict(ref) for ref in record.evidence_refs]
    return payload
