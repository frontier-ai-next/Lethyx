"""Canonical MemoryEvent contract for Lethyx memory writes."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from lethyx.contracts import EvidenceRef
from lethyx.contracts.models import TrustLevel, json_ready

MemoryKind = Literal[
    "observation",
    "research_finding",
    "research_lead",
    "lesson",
    "durable",
    "verification_report",
    "bkb_rule",
    "bkb_candidate",
    "phase_plan",
    "phase_plan_candidate",
    "subtask_card",
    "completion_memory",
    "codeptr",
    "skill_index",
    "run_state",
    "error_pattern",
]

Lifecycle = Literal[
    "candidate",
    "active",
    "quarantined",
    "superseded",
    "deprecated",
    "retracted",
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

MemoryTier = Literal[
    "always_on",
    "hot",
    "warm",
    "cold",
    "transient",
]

MemorySurface = Literal[
    "directive",
    "current_run_state",
    "supplemental_evidence",
    "archive_hint",
    "reflection_candidate",
]

_VERIFIED_TRUST: frozenset[TrustLevel] = frozenset(
    {
        "public_verified",
        "mutation_verified",
        "hidden_verified",
        "adversarial_verified",
    }
)

_DURABLE_KINDS: frozenset[MemoryKind] = frozenset(
    {"durable", "verification_report", "lesson", "bkb_rule"}
)
_NON_DIRECTIVE_LIFECYCLES: frozenset[Lifecycle] = frozenset(
    {"candidate", "quarantined", "superseded", "deprecated", "retracted"}
)

_MEMORY_KIND_VALUES: frozenset[str] = frozenset(
    {
        "observation",
        "research_finding",
        "research_lead",
        "lesson",
        "durable",
        "verification_report",
        "bkb_rule",
        "bkb_candidate",
        "phase_plan",
        "phase_plan_candidate",
        "subtask_card",
        "completion_memory",
        "codeptr",
        "skill_index",
        "run_state",
        "error_pattern",
    }
)
_LIFECYCLE_VALUES = frozenset(
    {"candidate", "active", "quarantined", "superseded", "deprecated", "retracted"}
)
_SCOPE_VALUES = frozenset(
    {
        "manager",
        "workspace",
        "cross_run_durable",
        "run_scoped",
        "phase_scoped",
        "subtask_scoped",
        "transient",
    }
)
_TIER_VALUES = frozenset({"always_on", "hot", "warm", "cold", "transient"})
_SURFACE_VALUES = frozenset(
    {
        "directive",
        "current_run_state",
        "supplemental_evidence",
        "archive_hint",
        "reflection_candidate",
    }
)
_TRUST_VALUES: frozenset[str] = frozenset(
    {
        "agent_claim",
        "public_verified",
        "mutation_verified",
        "hidden_verified",
        "adversarial_verified",
    }
)


def _coerce_literal(value: str, allowed: frozenset[str], default: str) -> str:
    clean = str(value or "").strip()
    return clean if clean in allowed else default


@dataclass(frozen=True)
class MemoryEvent:
    content: str
    title: str = ""
    memory_kind: MemoryKind = "observation"
    lifecycle: Lifecycle = "candidate"
    trust_level: TrustLevel = "agent_claim"
    scope: MemoryScope = "run_scoped"
    tier: MemoryTier = "warm"
    surface: MemorySurface = "supplemental_evidence"
    workspace_id: str = ""
    run_id: str = ""
    phase_id: str = ""
    subtask_id: str = ""
    agent_kind: str = ""
    agent_instance_id: str = ""
    producer: str = "agent"
    source_path: str = ""
    source_backend: str = "lethyx"
    tags: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    external_refs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    verified: bool = False


@dataclass(frozen=True)
class MemoryWriteResult:
    saved: bool
    canonical_id: str = ""
    store: str = ""
    backend: str = "mempalace"
    volatile: bool = False
    error: str = ""
    warnings: tuple[str, ...] = ()
    external_refs: dict[str, str] = field(default_factory=dict)
    skipped_duplicate: bool = False
    policy_issues: tuple[str, ...] = ()


def normalize_memory_event(raw: dict[str, Any] | MemoryEvent) -> MemoryEvent:
    if isinstance(raw, MemoryEvent):
        return raw
    data = dict(raw)
    refs_raw = data.get("evidence_refs") or []
    refs: list[EvidenceRef] = []
    if isinstance(refs_raw, (list, tuple)):
        for item in refs_raw:
            if isinstance(item, EvidenceRef):
                refs.append(item)
            elif isinstance(item, dict):
                refs.append(EvidenceRef.from_mapping(item))
    tags_raw = data.get("tags") or ()
    if isinstance(tags_raw, str):
        tags = tuple(t.strip() for t in tags_raw.replace(";", ",").split(",") if t.strip())
    else:
        tags = tuple(str(t) for t in tags_raw if str(t).strip())
    ext = data.get("external_refs")
    external_refs = dict(ext) if isinstance(ext, dict) else {}
    meta = data.get("metadata")
    metadata = dict(meta) if isinstance(meta, dict) else {}
    return MemoryEvent(
        content=str(data.get("content") or ""),
        title=str(data.get("title") or ""),
        memory_kind=cast(
            MemoryKind,
            _coerce_literal(
                str(data.get("memory_kind") or data.get("kind") or ""),
                _MEMORY_KIND_VALUES,
                "observation",
            ),
        ),
        lifecycle=cast(
            Lifecycle,
            _coerce_literal(str(data.get("lifecycle") or ""), _LIFECYCLE_VALUES, "candidate"),
        ),
        trust_level=cast(
            TrustLevel,
            _coerce_literal(
                str(data.get("trust_level") or ""),
                _TRUST_VALUES,
                "agent_claim",
            ),
        ),
        scope=cast(
            MemoryScope,
            _coerce_literal(str(data.get("scope") or ""), _SCOPE_VALUES, "run_scoped"),
        ),
        tier=cast(
            MemoryTier,
            _coerce_literal(str(data.get("tier") or ""), _TIER_VALUES, "warm"),
        ),
        surface=cast(
            MemorySurface,
            _coerce_literal(
                str(data.get("surface") or ""),
                _SURFACE_VALUES,
                "supplemental_evidence",
            ),
        ),
        workspace_id=str(data.get("workspace_id") or ""),
        run_id=str(data.get("run_id") or ""),
        phase_id=str(data.get("phase_id") or data.get("phase") or ""),
        subtask_id=str(data.get("subtask_id") or ""),
        agent_kind=str(data.get("agent_kind") or ""),
        agent_instance_id=str(data.get("agent_instance_id") or ""),
        producer=str(data.get("producer") or "agent"),
        source_path=str(data.get("source_path") or ""),
        source_backend=str(data.get("source_backend") or "lethyx"),
        tags=tags,
        evidence_refs=tuple(refs),
        external_refs=external_refs,
        metadata=metadata,
        verified=bool(data.get("verified", False)),
    )


def _store_for_kind(kind: str) -> str:
    if kind in {"durable", "verification_report"}:
        return "palace.durable"
    if kind in {"lesson", "bkb_rule"}:
        return "palace.lesson"
    return "palace.idea"


def memory_event_from_tool_write(
    *,
    content: str,
    title: str = "",
    memory_kind: str = "observation",
    workspace_id: str = "",
    tags: Iterable[str] = (),
    scope: str = "run_scoped",
    tier: str = "warm",
    phase_id: str = "",
    run_id: str = "",
    subtask_id: str = "",
    source_path: str = "",
    trust_level: str = "agent_claim",
    verified: bool = False,
    evidence_refs: Iterable[Any] = (),
    lifecycle: str = "candidate",
    surface: str = "supplemental_evidence",
    source_backend: str = "lethyx",
    palace_store: str = "",
    metadata: dict[str, Any] | None = None,
) -> MemoryEvent:
    """Build a MemoryEvent from tool-layer write parameters."""
    refs: list[EvidenceRef] = []
    for item in evidence_refs:
        if isinstance(item, EvidenceRef):
            refs.append(item)
        elif isinstance(item, dict):
            refs.append(EvidenceRef.from_mapping(item))
    tag_tuple = tuple(
        str(tag).strip()
        for tag in (tags.replace(";", ",").split(",") if isinstance(tags, str) else tags)
        if str(tag).strip()
    )
    meta = dict(metadata or {})
    if palace_store:
        meta["palace_store"] = palace_store
    return normalize_memory_event(
        {
            "content": content,
            "title": title,
            "memory_kind": memory_kind,
            "workspace_id": workspace_id,
            "tags": tag_tuple,
            "scope": scope,
            "tier": tier,
            "phase_id": phase_id,
            "run_id": run_id,
            "subtask_id": subtask_id,
            "source_path": source_path,
            "trust_level": trust_level,
            "verified": verified,
            "evidence_refs": refs,
            "lifecycle": lifecycle,
            "surface": surface,
            "source_backend": source_backend,
            "metadata": meta,
        }
    )


def memory_event_to_palace_kwargs(event: MemoryEvent) -> dict[str, Any]:
    mem_body = event.content or event.title or ""
    mem_content = f"[{event.title}]\n{mem_body}" if event.title else mem_body
    refs_json = json.dumps(
        [json_ready(r) for r in event.evidence_refs],
        ensure_ascii=False,
        default=str,
    )
    extra: dict[str, Any] = {
        "title": event.title,
        "type": event.memory_kind,
        "kind": event.memory_kind,
        "lifecycle": event.lifecycle,
        "trust_level": event.trust_level,
        "surface": event.surface,
        "source_backend": event.source_backend,
        "evidence_refs_json": refs_json,
        "external_refs_json": json.dumps(event.external_refs, ensure_ascii=False),
        "metadata_json": json.dumps(event.metadata, ensure_ascii=False, default=str),
    }
    store = str(event.metadata.get("palace_store") or "") or _store_for_kind(event.memory_kind)
    return {
        "store": store,
        "content": mem_content,
        "tier": event.tier,
        "scope": event.scope,
        "tags": list(event.tags),
        "phase": event.phase_id or None,
        "subtask_id": event.subtask_id or None,
        "run_id": event.run_id or None,
        "source_path": event.source_path or "memory_event",
        "verified": event.verified,
        "kind": event.memory_kind,
        "extra": extra,
    }


def palace_node_to_memory_event(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_id": node.get("id"),
        "content": node.get("content"),
        "memory_kind": node.get("memory_kind") or node.get("kind") or node.get("type"),
        "store": node.get("store"),
        "lifecycle": node.get("lifecycle"),
        "surface": node.get("surface"),
        "trust_level": node.get("trust_level"),
        "source_backend": node.get("source_backend"),
        "evidence_refs_json": node.get("evidence_refs_json"),
        "external_refs_json": node.get("external_refs_json"),
        "metadata_json": node.get("metadata_json"),
        "producer": node.get("producer"),
        "agent_kind": node.get("agent_kind"),
        "scope": node.get("scope"),
        "tier": node.get("tier"),
        "tags": node.get("tags"),
        "phase_id": node.get("phase"),
        "run_id": node.get("run_id"),
        "verified": node.get("verified"),
    }


def validate_memory_event_for_write(event: MemoryEvent) -> list[str]:
    issues: list[str] = []
    if event.surface == "directive" and event.lifecycle in _NON_DIRECTIVE_LIFECYCLES:
        issues.append("candidate/quarantined memory cannot use directive surface")
    durable_like = event.memory_kind in _DURABLE_KINDS or event.scope == "cross_run_durable"
    if durable_like and event.trust_level not in _VERIFIED_TRUST:
        issues.append(
            "durable or cross-run memory requires verified trust_level "
            "(public_verified/mutation_verified/hidden_verified/adversarial_verified)"
        )
    if durable_like and not event.evidence_refs:
        issues.append("durable memory requires typed evidence_refs")
    return issues
