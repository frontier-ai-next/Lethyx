"""Backend-neutral memory contracts.

These models describe Lethyx memory semantics independently from any physical
storage implementation.  Backends may store the records natively or in a
backend-owned sidecar, but runtime authority is carried by these typed fields.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from lethyx.contracts import EvidenceRef
from lethyx.memory.backends.base import DurableEvent, DurableLesson, MemoryHit

BackendId = Literal["mempalace", "hindsight", "mnemosyne", "composite"]
Lifecycle = Literal[
    "candidate",
    "active",
    "quarantined",
    "superseded",
    "deprecated",
    "retracted",
    "archived",
]
MemorySurface = Literal[
    "directive",
    "current_run_state",
    "supplemental_evidence",
    "archive_hint",
    "reflection_candidate",
]
MemoryScope = Literal[
    "manager",
    "workspace",
    "cross_run_durable",
    "run_scoped",
    "phase_scoped",
    "subtask_scoped",
    "transient",
]
MemoryTier = Literal["always_on", "hot", "warm", "cold", "transient"]

_ALLOWED_BACKENDS = {"mempalace", "hindsight", "mnemosyne", "composite"}
_ALLOWED_LIFECYCLES = {
    "candidate",
    "active",
    "quarantined",
    "superseded",
    "deprecated",
    "retracted",
    "archived",
}
_ALLOWED_SURFACES = {
    "directive",
    "current_run_state",
    "supplemental_evidence",
    "archive_hint",
    "reflection_candidate",
}
_ALLOWED_SCOPES = {
    "manager",
    "workspace",
    "cross_run_durable",
    "run_scoped",
    "phase_scoped",
    "subtask_scoped",
    "transient",
}
_ALLOWED_TIERS = {"always_on", "hot", "warm", "cold", "transient"}
_VERIFIED_TRUST = {
    "workspace_verified",
    "public_verified",
    "supervisor_verified",
    "mutation_verified",
    "hidden_verified",
    "adversarial_verified",
    "human_approved",
}
_DURABLE_KINDS = {"durable", "verification_report", "lesson", "bkb_rule"}
_NON_DIRECTIVE_LIFECYCLES = {
    "candidate",
    "quarantined",
    "superseded",
    "deprecated",
    "retracted",
    "archived",
}
_AUTHORITY_CHANGING_LIFECYCLES = {"active", "superseded", "deprecated", "retracted"}


class EdgeType:
    DERIVED_FROM = "derived_from"
    CITES = "cites"
    TESTS = "tests"
    IMPLEMENTS = "implements"
    SUPERSEDES = "supersedes"
    REFERENCES_FILE = "references_file"
    FROM_PHASE = "from_phase"
    FROM_SUBTASK = "from_subtask"
    TRIGGERED_BY_ERROR = "triggered_by_error"
    FLAGGED_BY = "flagged_by"
    BLOCKS = "blocks"
    APPLIED_REFLECTION = "applied_reflection"


ALLOWED_EDGE_TYPES: frozenset[str] = frozenset(
    {
        EdgeType.DERIVED_FROM,
        EdgeType.CITES,
        EdgeType.TESTS,
        EdgeType.IMPLEMENTS,
        EdgeType.SUPERSEDES,
        EdgeType.REFERENCES_FILE,
        EdgeType.FROM_PHASE,
        EdgeType.FROM_SUBTASK,
        EdgeType.TRIGGERED_BY_ERROR,
        EdgeType.FLAGGED_BY,
        EdgeType.BLOCKS,
        EdgeType.APPLIED_REFLECTION,
    }
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _evidence_from_any(value: Any) -> EvidenceRef:
    if isinstance(value, EvidenceRef):
        return value
    if isinstance(value, dict):
        return EvidenceRef.from_mapping(value)
    return EvidenceRef(ref_type="unknown", ref_id=str(value), produced_by="unknown")


def _evidence_tuple(values: Any) -> tuple[EvidenceRef, ...]:
    if not values:
        return ()
    return tuple(_evidence_from_any(item) for item in values)


def evidence_to_json(refs: tuple[EvidenceRef, ...] | list[EvidenceRef]) -> str:
    return json.dumps([asdict(ref) for ref in refs], ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class BackendCapabilities:
    backend_id: BackendId
    persistent: bool
    semantic_recall: bool = True
    fts_recall: bool = True
    graph_edges: bool = True
    lifecycle_mutation: bool = True
    reflection_candidates: bool = True
    workspace_isolation: bool = True
    run_phase_subtask_isolation: bool = True
    value_tracking: bool = True
    export_import: bool = True
    requires_external_service: bool = False
    supports_embedded: bool = False

    def __post_init__(self) -> None:
        if self.backend_id not in _ALLOWED_BACKENDS:
            raise ValueError(f"backend_id must be one of {sorted(_ALLOWED_BACKENDS)}")


@dataclass(frozen=True)
class MemoryBlockRecord:
    block_id: str
    content: str
    title: str = ""
    kind: str = "observation"
    lifecycle: str = "candidate"
    trust_level: str = "agent_claim"
    scope: str = "run_scoped"
    tier: str = "warm"
    surface: str = "supplemental_evidence"
    workspace_id: str = ""
    run_id: str = ""
    phase_id: str = ""
    subtask_id: str = ""
    producer: str = "agent"
    source_backend: str = ""
    tags: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    external_refs: dict[str, str] = field(default_factory=dict)
    privacy_class: str = "normal"
    risk_level: str = "normal"
    mutation_of: str = ""
    supersedes: tuple[str, ...] = ()
    superseded_by: str = ""
    expires_at: str = ""
    token_budget: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.block_id:
            object.__setattr__(self, "block_id", f"mem_{uuid.uuid4().hex[:16]}")
        if self.lifecycle not in _ALLOWED_LIFECYCLES:
            object.__setattr__(self, "lifecycle", "candidate")
        if self.surface not in _ALLOWED_SURFACES:
            object.__setattr__(self, "surface", "supplemental_evidence")
        if self.scope not in _ALLOWED_SCOPES:
            object.__setattr__(self, "scope", "run_scoped")
        if self.tier not in _ALLOWED_TIERS:
            object.__setattr__(self, "tier", "warm")
        if not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        if not isinstance(self.evidence_refs, tuple):
            object.__setattr__(self, "evidence_refs", _evidence_tuple(self.evidence_refs))
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
    phase_id: str = ""
    subtask_id: str = ""
    reason: str = ""
    weight: float = 1.0
    evidence_refs: tuple[EvidenceRef, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(self, "created_at", _now())
        if not isinstance(self.evidence_refs, tuple):
            object.__setattr__(self, "evidence_refs", _evidence_tuple(self.evidence_refs))


@dataclass(frozen=True)
class MemoryUsageRecord:
    block_id: str
    workspace_id: str = ""
    run_id: str = ""
    phase_id: str = ""
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


def validate_block_record(record: MemoryBlockRecord) -> MemoryBlockRecord:
    issues: list[str] = []
    if record.surface == "directive" and record.lifecycle in _NON_DIRECTIVE_LIFECYCLES:
        issues.append("candidate/quarantined memory cannot use directive surface")
    durable_like = record.kind in _DURABLE_KINDS or record.scope == "cross_run_durable"
    if durable_like and record.trust_level not in _VERIFIED_TRUST:
        issues.append("durable or cross-run memory requires verified trust")
    if durable_like and not record.evidence_refs:
        issues.append("durable memory requires typed evidence_refs")
    if record.surface == "directive" and record.trust_level not in _VERIFIED_TRUST:
        issues.append("directive memory requires verified trust")
    if record.surface == "directive" and record.metadata.get("promotion_gate") not in {
        "tribunal_accept",
        "human_approved",
        "bootstrap_curated",
    }:
        issues.append("directive memory requires accepted promotion gate")
    if issues:
        raise ValueError("; ".join(issues))
    return record


def durable_palace_write_issues(
    *,
    verified: bool,
    trust_level: str,
    evidence_refs_json: str,
) -> list[str]:
    """Gate for direct MemPalace writes/promotions into durable scope.

    Mirrors :func:`validate_block_record` semantics for callers that bypass the
    backend ``retain_block`` path: durable nodes require ``verified=True``,
    verified-or-higher trust, and at least one typed evidence ref.
    """
    issues: list[str] = []
    if not verified:
        issues.append("durable palace writes require verified=True")
    if str(trust_level or "").strip().lower() not in _VERIFIED_TRUST:
        issues.append("durable or cross-run palace writes require verified trust_level")
    refs: list[dict[str, Any]] = []
    raw = str(evidence_refs_json or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            refs = [
                item
                for item in parsed
                if isinstance(item, dict) and str(item.get("ref_id") or "").strip()
            ]
    if not refs:
        issues.append("durable palace writes require typed evidence_refs")
    return issues


def validate_lifecycle_transition(
    current: MemoryBlockRecord,
    lifecycle: str,
    *,
    reason: str = "",
    evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...] | None = None,
    surface: str | None = None,
) -> None:
    if lifecycle not in _ALLOWED_LIFECYCLES:
        raise ValueError(f"unknown lifecycle: {lifecycle}")
    refs = tuple(evidence_refs or ())
    target_surface = surface or current.surface
    if target_surface == "directive" and (
        lifecycle in _NON_DIRECTIVE_LIFECYCLES or current.trust_level not in _VERIFIED_TRUST
    ):
        raise ValueError("directive lifecycle transition requires verified active memory")
    if lifecycle in _AUTHORITY_CHANGING_LIFECYCLES and not reason:
        raise ValueError("authority-changing lifecycle mutation requires reason")
    if lifecycle in {"active", "retracted"} and not (refs or current.evidence_refs):
        raise ValueError("authority-changing lifecycle mutation requires evidence_refs")


def memory_event_to_block_record(event: Any, *, backend_id: str = "") -> MemoryBlockRecord:
    data = event if isinstance(event, dict) else getattr(event, "__dict__", {})
    metadata = dict(data.get("metadata") or {})
    source_path = str(data.get("source_path") or "").strip()
    if source_path and not str(metadata.get("source_path") or "").strip():
        metadata["source_path"] = source_path
    return MemoryBlockRecord(
        block_id=str(
            data.get("block_id") or data.get("canonical_id") or data.get("event_id") or ""
        ),
        content=str(data.get("content") or ""),
        title=str(data.get("title") or ""),
        kind=str(data.get("memory_kind") or data.get("kind") or "observation"),
        lifecycle=str(data.get("lifecycle") or "candidate"),
        trust_level=str(data.get("trust_level") or "agent_claim"),
        scope=str(data.get("scope") or "run_scoped"),
        tier=str(data.get("tier") or "warm"),
        surface=str(data.get("surface") or "supplemental_evidence"),
        workspace_id=str(data.get("workspace_id") or ""),
        run_id=str(data.get("run_id") or ""),
        phase_id=str(data.get("phase_id") or data.get("phase") or ""),
        subtask_id=str(data.get("subtask_id") or ""),
        producer=str(data.get("producer") or data.get("agent_kind") or "agent"),
        source_backend=str(data.get("source_backend") or "") or backend_id,
        tags=tuple(data.get("tags") or ()),
        evidence_refs=_evidence_tuple(data.get("evidence_refs") or ()),
        external_refs=dict(data.get("external_refs") or {}),
        metadata=metadata,
        verified=bool(data.get("verified", False)),
    )


def durable_event_to_block_record(
    event: DurableEvent | Any, *, backend_id: str = ""
) -> MemoryBlockRecord:
    data = event if isinstance(event, dict) else getattr(event, "__dict__", {})
    kind = str(data.get("kind") or "durable")
    return MemoryBlockRecord(
        block_id=str(data.get("event_id") or data.get("block_id") or data.get("id") or ""),
        content=str(data.get("content") or ""),
        title=kind,
        kind=kind,
        lifecycle="active",
        trust_level=str(data.get("trust_level") or "workspace_verified"),
        scope="cross_run_durable"
        if kind in _DURABLE_KINDS or kind in {"run_summary", "architecture_decision"}
        else "run_scoped",
        tier="warm",
        surface="supplemental_evidence",
        workspace_id=str(data.get("workspace_id") or ""),
        run_id=str(data.get("run_id") or ""),
        phase_id=str(data.get("phase_id") or ""),
        subtask_id=str(data.get("subtask_id") or ""),
        producer=str(data.get("agent") or "agent"),
        source_backend=backend_id,
        tags=tuple(data.get("tags") or ()),
        evidence_refs=_evidence_tuple(data.get("evidence_refs") or ()),
        metadata=dict(data.get("metadata") or {}),
        verified=True,
    )


def durable_lesson_to_block_record(
    lesson: DurableLesson | Any, *, backend_id: str = ""
) -> MemoryBlockRecord:
    data = lesson if isinstance(lesson, dict) else getattr(lesson, "__dict__", {})
    return MemoryBlockRecord(
        block_id=str(data.get("lesson_id") or data.get("block_id") or data.get("id") or ""),
        content=str(data.get("content") or ""),
        title=str(data.get("title") or ""),
        kind=str(data.get("kind") or "lesson"),
        lifecycle="active",
        trust_level=str(data.get("trust_level") or "public_verified"),
        scope="cross_run_durable",
        tier="warm",
        surface="supplemental_evidence",
        workspace_id=str(data.get("workspace_id") or ""),
        run_id=str(data.get("run_id") or ""),
        phase_id=str(data.get("phase_id") or ""),
        source_backend=backend_id,
        tags=tuple(data.get("tags") or ()),
        evidence_refs=_evidence_tuple(data.get("evidence_refs") or ()),
        metadata=dict(data.get("metadata") or {}),
        verified=True,
    )


def _memory_hit_score(record: MemoryBlockRecord) -> float:
    metadata = dict(record.metadata or {})
    for key in ("retrieval_score", "memory_block_rrf_score", "score", "native_score"):
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
            "block_id": record.block_id,
            "source_backend": record.source_backend,
            "surface": record.surface,
            "directive": record.surface == "directive",
            "lifecycle": record.lifecycle,
            "trust_level": record.trust_level,
            "workspace_id": record.workspace_id,
            "run_id": record.run_id,
            "phase_id": record.phase_id,
            "privacy_class": record.privacy_class,
            "risk_level": record.risk_level,
            **dict(record.metadata),
        },
    )


def block_from_json(data: dict[str, Any]) -> MemoryBlockRecord:
    payload = dict(data)
    payload["tags"] = tuple(payload.get("tags") or ())
    payload["supersedes"] = tuple(payload.get("supersedes") or ())
    payload["evidence_refs"] = _evidence_tuple(payload.get("evidence_refs") or ())
    return MemoryBlockRecord(
        **{k: v for k, v in payload.items() if k in MemoryBlockRecord.__dataclass_fields__}
    )


def block_to_json(record: MemoryBlockRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["evidence_refs"] = [asdict(ref) for ref in record.evidence_refs]
    return payload


def with_backend(record: MemoryBlockRecord, backend_id: str) -> MemoryBlockRecord:
    return replace(record, source_backend=backend_id or record.source_backend, updated_at=_now())
