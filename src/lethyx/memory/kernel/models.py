"""Canonical MemoryEvent contract for Lethyx memory writes."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from lethyx.contracts import (
    ContentMode,
    MemoryBlockRecord,
    MemoryEvent,
    OperationalRisk,
    RecordBlockMembership,
    Sensitivity,
)
from lethyx.contracts.memory import CURRENT_SCHEMA_VERSION


@dataclass(frozen=True)
class MemoryWriteResult:
    saved: bool
    canonical_id: str = ""
    backend: str = "mempalace"
    operation_id: str = ""
    commit_revision: int = 0
    durable_status: str = "unknown"
    projection_status: str = "unknown"
    projection_revision: int = 0
    retryable: bool = False
    source_coverage: float = 0.0
    volatile: bool = False
    error: str = ""
    warnings: tuple[str, ...] = ()
    external_refs: dict[str, str] = field(default_factory=dict)
    skipped_duplicate: bool = False
    policy_issues: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


def block_record_to_event(record: MemoryBlockRecord, *, stage_id: str = "") -> MemoryEvent:
    """Convert one typed canonical record back into the write event contract."""

    metadata = dict(record.metadata)
    return MemoryEvent(
        content=record.content,
        block_membership=record.block_membership,
        block_id=record.block_id,
        canonical_key=record.canonical_key,
        revision_id=record.revision_id,
        duplicate_cluster_id=record.duplicate_cluster_id,
        recorded_at=record.recorded_at,
        title=record.title,
        memory_kind=record.kind,  # type: ignore[arg-type]
        status=record.status,  # type: ignore[arg-type]
        retention=record.retention,  # type: ignore[arg-type]
        delivery_policy=record.delivery_policy,  # type: ignore[arg-type]
        temperature=record.temperature,  # type: ignore[arg-type]
        authority=record.authority,  # type: ignore[arg-type]
        trust_level=record.trust_level,  # type: ignore[arg-type]
        scope=record.scope,  # type: ignore[arg-type]
        verification_status=record.verification_status,  # type: ignore[arg-type]
        verification_methods=record.verification_methods,
        promotion_decision=record.promotion_decision,  # type: ignore[arg-type]
        user_id=record.user_id,
        workspace_id=record.workspace_id,
        profile_id=record.profile_id,
        run_id=record.run_id,
        stage_id=stage_id or record.stage_id,
        subtask_id=record.subtask_id,
        producer=record.producer,
        source_path=str(metadata.get("source_path") or ""),
        source_backend=record.source_backend,
        tags=record.tags,
        evidence_refs=record.evidence_refs,
        external_refs=record.external_refs,
        sensitivity=record.sensitivity,
        content_mode=record.content_mode,
        operational_risk=record.operational_risk,
        revision_of=record.revision_of,
        supersedes_ids=record.supersedes_ids,
        expires_at=record.expires_at,
        child_claims=record.child_claims,
        metadata=metadata,
        session_id=record.session_id,
        task_id=record.task_id,
        parent_session_id=record.parent_session_id,
        artifact_id=record.artifact_id,
        artifact_revision=record.artifact_revision,
        valid_from=record.valid_from,
        valid_to=record.valid_to,
        project_id=record.project_id,
        repository_id=record.repository_id,
        checkout=record.checkout,
        worktree=record.worktree,
        branch=record.branch,
        revision=record.revision,
        dirty_state=record.dirty_state,
        schema_version=record.schema_version,
        hard_policy_version=record.hard_policy_version,
        semantic_catalog_version=record.semantic_catalog_version,
    )


def normalize_memory_event(raw: dict[str, Any] | MemoryEvent) -> MemoryEvent:
    if isinstance(raw, MemoryEvent):
        return raw
    data = dict(raw)
    retired_fields = sorted(
        {"mutation_of", "supersedes", "superseded_by", "token_budget"}.intersection(data)
    )
    if retired_fields:
        raise ValueError(
            "retired memory event fields are unsupported: " + ", ".join(retired_fields)
        )
    tags_raw = data.get("tags") or ()
    if isinstance(tags_raw, str):
        tags = tuple(t.strip() for t in tags_raw.replace(";", ",").split(",") if t.strip())
    else:
        tags = tuple(str(t) for t in tags_raw if str(t).strip())
    return MemoryEvent(
        content=str(data.get("content") or ""),
        block_membership=(
            RecordBlockMembership.from_mapping(data["block_membership"])
            if data.get("block_membership") is not None
            else None
        ),
        block_id=str(data.get("block_id") or ""),
        canonical_key=str(data.get("canonical_key") or ""),
        revision_id=str(data.get("revision_id") or ""),
        duplicate_cluster_id=str(data.get("duplicate_cluster_id") or ""),
        recorded_at=str(data.get("recorded_at") or ""),
        title=str(data.get("title") or ""),
        memory_kind=data.get("memory_kind", data.get("kind", "observation")),
        status=data["status"] if "status" in data else "candidate",
        retention=data["retention"] if "retention" in data else "durable",
        delivery_policy=data["delivery_policy"] if "delivery_policy" in data else "on_demand",
        temperature=data["temperature"] if "temperature" in data else "warm",
        authority=data["authority"] if "authority" in data else "advisory",
        trust_level=data["trust_level"] if "trust_level" in data else "agent_reported",
        scope=data["scope"] if "scope" in data else "workspace",
        verification_status=(
            data["verification_status"] if "verification_status" in data else "unverified"
        ),
        verification_methods=tuple(data.get("verification_methods") or ()),
        promotion_decision=(
            data["promotion_decision"] if "promotion_decision" in data else "not_requested"
        ),
        user_id=str(data.get("user_id") or ""),
        workspace_id=str(data.get("workspace_id") or ""),
        profile_id=str(data.get("profile_id") or "default"),
        run_id=str(data.get("run_id") or ""),
        stage_id=str(data.get("stage_id") or ""),
        subtask_id=str(data.get("subtask_id") or ""),
        agent_kind=str(data.get("agent_kind") or ""),
        agent_instance_id=str(data.get("agent_instance_id") or ""),
        producer=str(data.get("producer") or "agent"),
        source_path=str(data.get("source_path") or ""),
        source_backend=str(data.get("source_backend") or "lethyx"),
        tags=tags,
        evidence_refs=data.get("evidence_refs", ()),
        external_refs=data.get("external_refs", {}),
        sensitivity=data["sensitivity"] if "sensitivity" in data else "internal",
        content_mode=data["content_mode"] if "content_mode" in data else "full",
        operational_risk=(data["operational_risk"] if "operational_risk" in data else "low"),
        revision_of=str(data.get("revision_of") or ""),
        supersedes_ids=tuple(str(item) for item in (data.get("supersedes_ids") or ()) if str(item)),
        expires_at=str(data.get("expires_at") or ""),
        child_claims=data.get("child_claims", ()),
        metadata=data.get("metadata", {}),
        session_id=str(data.get("session_id") or ""),
        task_id=str(data.get("task_id") or ""),
        parent_session_id=str(data.get("parent_session_id") or ""),
        artifact_id=str(data.get("artifact_id") or ""),
        artifact_revision=str(data.get("artifact_revision") or ""),
        valid_from=str(data.get("valid_from") or ""),
        valid_to=str(data.get("valid_to") or ""),
        project_id=str(data.get("project_id") or ""),
        repository_id=str(data.get("repository_id") or ""),
        checkout=str(data.get("checkout") or ""),
        worktree=str(data.get("worktree") or ""),
        branch=str(data.get("branch") or ""),
        revision=str(data.get("revision") or ""),
        dirty_state=str(data.get("dirty_state") or ""),
        schema_version=data.get("schema_version", CURRENT_SCHEMA_VERSION),
        hard_policy_version=data.get("hard_policy_version", 1),
        semantic_catalog_version=str(data.get("semantic_catalog_version") or ""),
    )


def memory_event_from_tool_write(
    *,
    content: str,
    title: str = "",
    memory_kind: str = "observation",
    user_id: str = "",
    workspace_id: str = "",
    tags: Iterable[str] = (),
    scope: str = "workspace",
    retention: str = "durable",
    delivery_policy: str = "on_demand",
    temperature: str = "warm",
    authority: str = "advisory",
    stage_id: str = "",
    run_id: str = "",
    subtask_id: str = "",
    session_id: str = "",
    task_id: str = "",
    project_id: str = "",
    source_path: str = "",
    trust_level: str = "agent_reported",
    verification_status: str = "unverified",
    verification_methods: Iterable[str] = (),
    evidence_refs: Iterable[Any] = (),
    status: str = "candidate",
    promotion_decision: str = "not_requested",
    source_backend: str = "lethyx",
    sensitivity: Sensitivity = "internal",
    content_mode: ContentMode = "full",
    operational_risk: OperationalRisk = "low",
    metadata: dict[str, Any] | None = None,
) -> MemoryEvent:
    """Build a MemoryEvent from tool-layer write parameters."""
    tag_tuple = tuple(
        str(tag).strip()
        for tag in (tags.replace(";", ",").split(",") if isinstance(tags, str) else tags)
        if str(tag).strip()
    )
    return normalize_memory_event(
        {
            "content": content,
            "title": title,
            "memory_kind": memory_kind,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "tags": tag_tuple,
            "scope": scope,
            "retention": retention,
            "delivery_policy": delivery_policy,
            "temperature": temperature,
            "authority": authority,
            "stage_id": stage_id,
            "run_id": run_id,
            "subtask_id": subtask_id,
            "session_id": session_id,
            "task_id": task_id,
            "project_id": project_id,
            "source_path": source_path,
            "trust_level": trust_level,
            "verification_status": verification_status,
            "verification_methods": tuple(verification_methods),
            "evidence_refs": tuple(evidence_refs),
            "status": status,
            "promotion_decision": promotion_decision,
            "source_backend": source_backend,
            "sensitivity": sensitivity,
            "content_mode": content_mode,
            "operational_risk": operational_risk,
            "metadata": dict(metadata or {}),
        }
    )
