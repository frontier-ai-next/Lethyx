"""Public, framework-neutral Lethyx API.

The API is intentionally thin: host frameworks own routing and task control,
while Lethyx owns typed memory retention, recall, governance, and graph export.
"""

import base64
import binascii
import hashlib
import json
import math
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal, ParamSpec, TypeVar, cast

from filelock import FileLock

from lethyx.contracts import (
    AmbiguousEvidenceRefError,
    ConsolidationCommitReceipt,
    CandidateVerificationSubject,
    ContentMode,
    ContextualUtilitySummary,
    EvidenceRef,
    EvidenceResolver,
    HostObservation,
    MemoryBlockRecord,
    MemoryDecisionEvent,
    MemoryDecisionTrace,
    MemoryOutcomeReport,
    MemoryUsageEvent,
    OperationalRisk,
    OutcomeCreditEvent,
    ProcedureReuseDecision,
    ProspectiveLifecycleEvent,
    ProspectiveObligation,
    RecordBlockMembership,
    Sensitivity,
    TransferEnvelope,
    TurnChildClaim,
    VerificationReportRef,
    build_workspace_context,
    json_ready,
    is_protected_memory,
)
from lethyx.contracts.hashing import hash_value
from lethyx.contracts.ledger import append_verification_report_ledger_event
from lethyx.contracts.memory import (
    CURRENT_SCHEMA_VERSION,
    DeliveryPolicy,
    MemoryAuthority,
    MemoryRetention,
    MemoryScope,
    MemoryStatus,
    MemoryTemperature,
    MemoryUsageType,
    PromotionDecision,
    VerificationStatus,
    _closed_value,
    has_decisive_evidence,
    memory_scope_key,
    reserved_memory_metadata_keys,
    validate_memory_safety_axes,
)
from lethyx.contracts.models import APPROVED_TRUST_LEVELS, TrustLevel
from lethyx.llm_budget import estimate_tokens
from lethyx.memory.activation_lease import (
    ActivationLease,
    activation_lease_from_payload,
)
from lethyx.memory.active_mask import (
    ActiveMemoryBlockBudget,
    ActiveMemoryMask,
    ActiveMemoryMaskStatus,
    ActiveMemoryPriority,
    ActiveMemorySelector,
)
from lethyx.memory.authoring.models import MemoryWriteProposal
from lethyx.memory.authoring.router import route_memory_write
from lethyx.memory.backends.base import MemoryHit
from lethyx.memory.backends.contracts import (
    BackendCapabilities,
    EdgeType,
    MemoryEdgeRecord,
    MemoryIndexNode,
    block_record_to_memory_hit,
)
from lethyx.memory.backends.factory import create_durable_backend
from lethyx.memory.blocks.catalog import (
    BlockDescriptor,
    block_descriptor_node_id,
    compile_semantic_catalog,
)
from lethyx.memory.blocks.models import MemoryBlock
from lethyx.memory.blocks.registry import (
    amend_workspace_block,
    archive_workspace_block,
    archive_workspace_session_block,
    create_workspace_block,
    load_memory_blocks,
    load_workspace_topology,
    protect_workspace_block,
    repair_protected_workspace_block_parent,
    validate_leaf_block_contracts,
    validate_memory_block_tree,
)
from lethyx.memory.catalog_store import (
    BlockEdgeType,
    BlockEvolutionTarget,
    BlockGraphEdge,
    BlockMergeSource,
    CatalogRevisionConflict,
    MaintenanceObservationType,
    TopologyBlockState,
    WorkspaceCatalogStore,
)
from lethyx.memory.consolidation import (
    CanonicalConsolidationPipeline,
    ConsolidationScheduler,
    ConsolidationShadowStore,
    ConsolidationSignal,
    ConsolidationTransitionTribunal,
    TransitionSemanticVerification,
)
from lethyx.memory.context_payload import PackedEvidencePayload, pack_evidence_payload
from lethyx.memory.deep_reader import DeepReadResult, deep_read_result
from lethyx.memory.dependencies import DependencyValidation
from lethyx.memory.kernel.models import block_record_to_event
from lethyx.memory.kernel.telemetry import profile_memory_operation
from lethyx.memory.kernel.writer import (
    _write_memory_event,
    erase_memory,
    transition_memory_status,
    write_memory_edge,
)
from lethyx.memory.mutation import (
    AmendmentCommitReceipt,
    CanonicalMutationIdentity,
    MemoryEraseRequest,
    MemoryRetractionReceipt,
    PublicationCommitReceipt,
    block_id_for_canonical_key,
    derive_canonical_key,
    plan_lifecycle_mutation,
    plan_memory_mutation,
    resolve_canonical_revision_head,
)
from lethyx.memory.service_receipts import (
    claim_remember_receipt,
    complete_remember_receipt,
    read_mutation_receipt_status,
    service_receipt_path,
)
from lethyx.memory.paths import normalize_workspace_id
from lethyx.memory.prospective import (
    ProspectiveIntervention,
    ProspectiveTelemetryReport,
    ProspectiveTriggerContext,
    ProspectiveTriggerDecision,
    prospective_intervention,
    prospective_telemetry_report,
    revalidate_prospective_fire,
)
from lethyx.memory.query_analysis import EvidenceRequirement
from lethyx.memory.recall_explanation import RecallExplanation, build_recall_explanation
from lethyx.memory.recall_plan import (
    MEMORY_READ_GUIDANCE,
    MEMORY_WRITE_GUIDANCE,
    RecallPlan,
    RetrievalMode,
    SourceHorizon,
)
from lethyx.memory.recall_strategy import allocate_recall_budget
from lethyx.memory.record_refs import PinnedRecordRef, pin_memory_record
from lethyx.memory.refinement import (
    build_memory_outcome_report,
    verified_procedure_promotion_gate,
)
from lethyx.memory.refinement import (
    qualify_procedure_reuse as _qualify_procedure_reuse,
)
from lethyx.memory.scopes import ScopeLineage, bind_scope_chain
from lethyx.memory.selection import MemorySelectionPolicy, SelectionContext
from lethyx.memory.tribunal.models import TribunalInput
from lethyx.memory.tribunal.runner import run_tribunal
from lethyx.memory.working_set import (
    WorkingRecordSnapshot,
    WorkingSetRead,
    read_working_records,
    snapshot_working_record,
)


@dataclass(frozen=True)
class RememberResult:
    """Result returned by :meth:`MemoryClient.remember`."""

    saved: bool
    memory_id: str = ""
    backend: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSession:
    """Lightweight session binding used by host frameworks."""

    session_id: str
    actor: str = ""
    project: str = ""
    workspace_id: str = ""


@dataclass(frozen=True)
class _ContextBuild:
    plan: RecallPlan
    hits: tuple[MemoryHit, ...]
    payload: PackedEvidencePayload
    decision_id: str


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    data = dict(metadata or {})
    try:
        sensitivity, content_mode, operational_risk = validate_memory_safety_axes(
            data.get("sensitivity"),
            data.get("content_mode"),
            data.get("operational_risk"),
        )
    except ValueError:
        return {"redacted": True}
    if not is_protected_memory(sensitivity, content_mode, operational_risk):
        return data
    return {
        "sensitivity": sensitivity,
        "content_mode": content_mode,
        "operational_risk": operational_risk,
        "redacted": True,
    }


def _safe_hit(hit: MemoryHit) -> MemoryHit:
    raw_metadata = dict(hit.metadata or {})
    if is_protected_memory(
        raw_metadata.get("sensitivity"),
        raw_metadata.get("content_mode"),
        raw_metadata.get("operational_risk"),
    ):
        return MemoryHit(
            text="[redacted non-injectable memory]",
            source="redacted-memory",
            score=None,
            kind="redacted",
            tags=[],
            metadata=_safe_metadata(raw_metadata),
        )
    return MemoryHit(
        text=hit.text,
        source=hit.source,
        score=hit.score,
        kind=hit.kind,
        tags=list(hit.tags),
        metadata=_safe_metadata(raw_metadata),
    )


def _flag_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _string_values(value: Any) -> set[str]:
    candidates: Iterable[object]
    if isinstance(value, str):
        candidates = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        candidates = value
    else:
        candidates = ()
    return {str(item).strip() for item in candidates if str(item).strip()}


_RETRIEVAL_OVERLAY_TRANSPORT_FIELDS = frozenset(
    {
        "backend",
        "backend_id",
        "backend_reference",
        "backend_revision",
        "block_id",
        "canonical_id",
        "commit_revision",
        "idempotency_key",
        "memory_id",
        "mutation_correlation_id",
        "native_id",
        "native_reference",
        "native_revision",
        "record_id",
        "selected_backend",
        "service_receipt_sha256",
    }
)


def _record_from_hit(hit: MemoryHit, plan: RecallPlan) -> MemoryBlockRecord:
    metadata = dict(hit.metadata or {})
    block_membership = (
        RecordBlockMembership.from_mapping(metadata["block_membership"])
        if metadata.get("block_membership") is not None
        else None
    )
    evidence_refs: list[EvidenceRef] = []
    for reference in tuple(metadata.get("evidence_refs") or ()):
        if isinstance(reference, EvidenceRef):
            evidence_refs.append(reference)
        elif isinstance(reference, Mapping):
            evidence_refs.append(EvidenceRef.from_mapping(dict(reference)))
        else:
            raise ValueError("memory hit evidence_refs must contain typed evidence mappings")
    typed_fields = {
        "authority",
        "block_membership",
        "canonical_key",
        "child_claims",
        "content_mode",
        "duplicate_cluster_id",
        "delivery_policy",
        "dirty_state",
        "evidence_refs",
        "expires_at",
        "artifact_id",
        "artifact_revision",
        "branch",
        "checkout",
        "hard_policy_version",
        "operational_risk",
        "parent_session_id",
        "promotion_decision",
        "profile_id",
        "project_id",
        "recorded_at",
        "repository_id",
        "retention",
        "revision_id",
        "revision",
        "run_id",
        "schema_version",
        "semantic_catalog_version",
        "sensitivity",
        "session_id",
        "scope",
        "source_backend",
        "stage_id",
        "status",
        "subtask_id",
        "task_id",
        "temperature",
        "title",
        "trust_level",
        "user_id",
        "verification_methods",
        "verification_status",
        "valid_from",
        "valid_to",
        "worktree",
        "workspace_id",
    }
    return MemoryBlockRecord(
        block_id=str(metadata.get("block_id") or hit.source or ""),
        content=hit.text,
        block_membership=block_membership,
        canonical_key=str(metadata.get("canonical_key") or ""),
        revision_id=str(metadata.get("revision_id") or ""),
        duplicate_cluster_id=str(metadata.get("duplicate_cluster_id") or ""),
        recorded_at=str(metadata.get("recorded_at") or ""),
        title=str(metadata.get("title") or ""),
        kind=hit.kind or str(metadata.get("kind") or "observation"),
        status=cast(
            MemoryStatus,
            _closed_value(
                str(metadata.get("status") or "candidate"), field_name="status", shape=MemoryStatus
            ),
        ),
        retention=cast(
            MemoryRetention,
            _closed_value(
                str(metadata.get("retention") or "durable"),
                field_name="retention",
                shape=MemoryRetention,
            ),
        ),
        delivery_policy=cast(
            DeliveryPolicy,
            _closed_value(
                str(metadata.get("delivery_policy") or "on_demand"),
                field_name="delivery_policy",
                shape=DeliveryPolicy,
            ),
        ),
        temperature=cast(
            MemoryTemperature,
            _closed_value(
                str(metadata.get("temperature") or "warm"),
                field_name="temperature",
                shape=MemoryTemperature,
            ),
        ),
        authority=cast(
            MemoryAuthority,
            _closed_value(
                str(metadata.get("authority") or "advisory"),
                field_name="authority",
                shape=MemoryAuthority,
            ),
        ),
        trust_level=cast(
            TrustLevel,
            _closed_value(
                str(metadata.get("trust_level") or "agent_reported"),
                field_name="trust_level",
                shape=TrustLevel,
            ),
        ),
        scope=cast(
            MemoryScope,
            _closed_value(
                str(metadata.get("scope") or "workspace"), field_name="scope", shape=MemoryScope
            ),
        ),
        verification_status=cast(
            VerificationStatus,
            _closed_value(
                str(metadata.get("verification_status") or "unverified"),
                field_name="verification_status",
                shape=VerificationStatus,
            ),
        ),
        verification_methods=tuple(metadata.get("verification_methods") or ()),
        promotion_decision=cast(
            PromotionDecision,
            _closed_value(
                str(metadata.get("promotion_decision") or "not_requested"),
                field_name="promotion_decision",
                shape=PromotionDecision,
            ),
        ),
        evidence_refs=tuple(evidence_refs),
        user_id=str(metadata.get("user_id") or ""),
        workspace_id=str(metadata.get("workspace_id") or ""),
        profile_id=str(metadata.get("profile_id") or "default"),
        run_id=str(metadata.get("run_id") or ""),
        stage_id=str(metadata.get("stage_id") or ""),
        session_id=str(metadata.get("session_id") or ""),
        task_id=str(metadata.get("task_id") or ""),
        subtask_id=str(metadata.get("subtask_id") or ""),
        parent_session_id=str(metadata.get("parent_session_id") or ""),
        artifact_id=str(metadata.get("artifact_id") or ""),
        artifact_revision=str(metadata.get("artifact_revision") or ""),
        project_id=str(metadata.get("project_id") or ""),
        repository_id=str(metadata.get("repository_id") or ""),
        checkout=str(metadata.get("checkout") or ""),
        worktree=str(metadata.get("worktree") or ""),
        branch=str(metadata.get("branch") or ""),
        revision=str(metadata.get("revision") or ""),
        dirty_state=str(metadata.get("dirty_state") or ""),
        schema_version=int(metadata.get("schema_version") or CURRENT_SCHEMA_VERSION),
        hard_policy_version=int(metadata.get("hard_policy_version") or 1),
        semantic_catalog_version=str(metadata.get("semantic_catalog_version") or ""),
        source_backend=str(metadata.get("source_backend") or ""),
        tags=tuple(hit.tags),
        sensitivity=metadata.get("sensitivity") or "internal",
        content_mode=metadata.get("content_mode") or "full",
        operational_risk=metadata.get("operational_risk") or "low",
        expires_at=str(metadata.get("expires_at") or ""),
        valid_from=str(metadata.get("valid_from") or ""),
        valid_to=str(metadata.get("valid_to") or ""),
        metadata={
            key: value
            for key, value in metadata.items()
            if key not in typed_fields | _RETRIEVAL_OVERLAY_TRANSPORT_FIELDS
        },
    )


def _hit_allowed(hit: MemoryHit, plan: RecallPlan) -> bool:
    try:
        record = _record_from_hit(hit, plan)
    except ValueError:
        return False
    if is_protected_memory(record.sensitivity, record.content_mode, record.operational_risk):
        return False
    if plan.profile_id and record.profile_id != plan.profile_id:
        return False
    policy = MemorySelectionPolicy()
    if not policy.eligible_for_recall(
        record,
        SelectionContext(
            query=plan.query,
            scope_chain=plan.scope_chain,
            run_id=plan.run_id,
            source_horizon=plan.source_horizon,
            include_candidates=plan.include_candidates,
            historical=plan.temporal_mode == "historical" or bool(plan.known_as_of),
            as_of="" if plan.known_as_of else plan.as_of,
        ),
    ):
        return False
    if plan.record_ids and record.block_id not in set(plan.record_ids):
        return False
    if plan.logical_block_ids:
        membership = record.block_membership
        if membership is None or membership.primary_block_id not in set(plan.logical_block_ids):
            return False
    return True


def _current_mutation_identity(
    backend: Any,
    record: MemoryBlockRecord,
    *,
    workspace_id: str,
) -> CanonicalMutationIdentity | None:
    lookup = getattr(backend, "canonical_mutation_identity", None)
    if not callable(lookup):
        return None
    identity = lookup(successor_id=record.block_id, workspace_id=workspace_id)
    if identity is not None and not isinstance(identity, CanonicalMutationIdentity):
        raise RuntimeError(
            "selected memory backend returned an invalid canonical mutation identity"
        )
    return identity


_PROMOTION_SCOPE_ID_FIELDS = {
    "subtask": "subtask_id",
    "task": "task_id",
    "session": "session_id",
    "workspace": "workspace_id",
    "project": "project_id",
    "user_global": "user_id",
}


def _record_scope_identity(record: MemoryBlockRecord) -> str:
    field_name = _PROMOTION_SCOPE_ID_FIELDS[record.scope]
    identity = str(getattr(record, field_name) or "").strip()
    if not identity:
        raise ValueError(f"{record.scope} memory has no bound {field_name}")
    return identity


def _promotion_target_binding(
    lineage: ScopeLineage,
    current: MemoryBlockRecord,
    *,
    target_scope: str | None,
    target_scope_id: str,
) -> tuple[MemoryScope, str]:
    requested_scope = str(target_scope or "").strip()
    requested_id = str(target_scope_id or "").strip()
    if not requested_scope:
        if requested_id:
            raise ValueError("target_scope_id requires target_scope")
        return current.scope, _record_scope_identity(current)
    if requested_scope not in {"workspace", "project", "user_global", "session", "task", "subtask"}:
        raise ValueError("promotion target_scope must name a supported exact scope")
    bound_id = {
        "workspace": lineage.workspace_id,
        "project": lineage.project_id,
        "user_global": lineage.user_id,
        "session": lineage.session_id,
        "task": lineage.task_id,
        "subtask": lineage.subtask_id,
    }[requested_scope]
    if not bound_id:
        raise ValueError(f"promotion target {requested_scope} is not bound to the client lineage")
    if requested_id and requested_id != bound_id:
        raise ValueError("promotion target_scope_id is not bound to the client lineage")
    return cast(MemoryScope, requested_scope), bound_id


def _typed_promotion_evidence(
    values: list[Any] | tuple[Any, ...],
) -> tuple[VerificationReportRef, ...]:
    reports: list[VerificationReportRef] = []
    for value in values:
        if isinstance(value, VerificationReportRef):
            report = value
        elif isinstance(value, dict) and "report_id" in value:
            report = VerificationReportRef.from_mapping(value)
        else:
            raise ValueError("promotion requires passed VerificationReportRef evidence")
        if not (
            report.passed
            and report.report_id.strip()
            and report.verifier_id.strip()
            and report.report_hash.strip()
        ):
            raise ValueError("promotion requires passed verification evidence")
        reports.append(report)
    refs = tuple(report.evidence_ref() for report in reports)
    if not has_decisive_evidence(refs):
        raise ValueError("promotion requires decisive typed verification evidence")
    return tuple(reports)


def _expand_recall_block_ids(
    configured_blocks: Mapping[str, MemoryBlock],
    selected_block_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve selected memory regions to their content-bearing descendants."""

    children_by_parent: dict[str, list[str]] = {}
    for block in configured_blocks.values():
        if block.parent_id:
            children_by_parent.setdefault(block.parent_id, []).append(block.block_id)

    resolved: list[str] = []
    for selected_id in selected_block_ids:
        selected = configured_blocks.get(selected_id)
        if selected is None:
            raise ValueError(f"unknown logical memory block: {selected_id}")
        pending = [selected.block_id]
        while pending:
            block_id = pending.pop()
            block = configured_blocks[block_id]
            if not block.index_only:
                resolved.append(block_id)
                continue
            pending.extend(reversed(children_by_parent.get(block_id, ())))
    return tuple(dict.fromkeys(resolved))


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _serialize_same_creation(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Only identical operation IDs contend; a crashed caller releases the OS lock."""

    @wraps(method)
    def call(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        operation_id = kwargs.get("operation_id", "")
        if not operation_id:
            return method(*args, **kwargs)
        client = cast("MemoryClient", args[0])
        lock_dir = service_receipt_path(client.repo_root).parent / "operation_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        identity = [client.workspace_id, client.profile_id, client._backend_id(), operation_id]
        lock_id = hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()
        with FileLock(lock_dir / f"{lock_id}.lock"):
            return method(*args, **kwargs)

    return call


class MemoryClient:
    """Framework-neutral client for Lethyx durable memory.

    Pass an explicit backend in tests or adapters. In normal use, the client
    creates the selected durable backend from ``backend`` or environment.
    """

    @staticmethod
    def agent_instructions() -> str:
        """Return read and agent-authored maintenance guidance for the host."""
        return MEMORY_READ_GUIDANCE + " " + MEMORY_WRITE_GUIDANCE

    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        workspace_id: str = "default",
        profile_id: str = "default",
        user_id: str = "",
        project_id: str = "",
        session_id: str = "",
        task_id: str = "",
        subtask_id: str = "",
        backend: str | None = None,
        backend_instance: Any | None = None,
        topology_actor: str = "llm",
    ) -> None:
        if topology_actor not in {"llm", "system"}:
            raise ValueError("topology_actor must be llm or system")
        self._topology_actor = topology_actor
        self.repo_root = Path(repo_root)
        self.workspace_id = normalize_workspace_id(workspace_id)
        self.profile_id = str(profile_id or "").strip()
        if not self.profile_id:
            raise ValueError("profile_id is required")
        self.scope_lineage = ScopeLineage(
            workspace_id=self.workspace_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
            subtask_id=subtask_id,
            user_id=user_id,
        )
        self.user_id = self.scope_lineage.user_id
        self.backend_mode = backend
        self._backend = backend_instance
        self._backend_supplied = backend_instance is not None
        self._session: AgentSession | None = None

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = create_durable_backend(
                self.repo_root,
                workspace_id=self.workspace_id,
                mode=self.backend_mode,
            )
        return self._backend

    @property
    def session(self) -> AgentSession | None:
        return self._session

    def capabilities(self) -> BackendCapabilities:
        """Read the selected backend's explicit, current operation contract."""

        read = getattr(self.backend, "capabilities", None)
        if not callable(read):
            raise ValueError("selected backend has no capability contract")
        result = read()
        if not isinstance(result, BackendCapabilities):
            raise ValueError("selected backend returned an untyped capability contract")
        if self.backend_mode is not None and result.backend_id != self.backend_mode.strip().lower():
            raise ValueError("backend capability identity does not match the selected backend")
        return result

    def _require_profile_record(self, memory_id: str) -> MemoryBlockRecord:
        record = self.backend.get_block(str(memory_id or "").strip())
        if not isinstance(record, MemoryBlockRecord):
            raise ValueError(f"memory block {memory_id!r} was not found")
        if record.workspace_id != self.workspace_id:
            raise ValueError("memory block workspace does not match the MemoryClient workspace")
        if record.profile_id != self.profile_id:
            raise ValueError("memory block profile does not match the MemoryClient profile")
        return record

    def validate_dependencies(
        self,
        memory_id: str,
        *,
        as_of: str = "",
        max_nodes: int = 256,
        max_edges: int = 1024,
    ) -> DependencyValidation:
        """Inspect canonical DERIVED_FROM validity within this workspace/profile.

        This host diagnostic is read-only and grants no recall or promotion
        authority. Historical inspection still retains the original records.
        """
        record = self._require_profile_record(memory_id)
        validate = getattr(self.backend, "validate_dependencies", None)
        if not callable(validate):
            raise ValueError("selected backend does not support dependency validation")
        result = validate(
            record.block_id,
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            as_of=as_of,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        if not isinstance(result, DependencyValidation):
            raise ValueError("selected backend returned an untyped dependency validation")
        return result

    def inspect_record(
        self,
        memory_id: str,
        *,
        logical_block_id: str,
        expected_revision_id: str,
    ) -> MemoryBlockRecord:
        """Read an exact revision for host inspection, without granting recall eligibility.

        The host authorizes operator access to this workspace/profile. Unpublished and
        historical statuses remain visible as such; this is not an agent recall operation.
        """
        selected_id = str(memory_id or "").strip()
        selected_block = str(logical_block_id or "").strip()
        selected_revision = str(expected_revision_id or "").strip()
        if not selected_id or not selected_block or not selected_revision:
            raise ValueError("record inspection requires exact record, block and revision IDs")
        if not self.capabilities().exact_record_read:
            raise ValueError("selected backend does not support exact_record_read")
        record = self._require_profile_record(selected_id)
        if (
            record.block_id != selected_id
            or record.revision_id != selected_revision
            or record.block_membership is None
            or record.block_membership.primary_block_id != selected_block
        ):
            raise ValueError("record is unavailable at the requested block and revision")
        return record

    def read_exact_record(
        self,
        *,
        record_id: str = "",
        evidence_ref: str = "",
    ) -> MemoryBlockRecord:
        """Read the canonical lossless payload by id or exact evidence reference."""

        selected_id = str(record_id or "").strip()
        selected_evidence = str(evidence_ref or "").strip()
        if bool(selected_id) == bool(selected_evidence):
            raise ValueError("read_exact_record requires exactly one record_id or evidence_ref")
        if selected_id:
            return self._require_profile_record(selected_id)
        matched: MemoryBlockRecord | None = None
        for candidate_id in self.backend.exact_reference_block_ids(
            selected_evidence, workspace_id=self.workspace_id
        ):
            candidate = self.backend.get_block(candidate_id, workspace_id=self.workspace_id)
            if not isinstance(candidate, MemoryBlockRecord):
                continue
            if (
                candidate.workspace_id != self.workspace_id
                or candidate.profile_id != self.profile_id
            ):
                continue
            if any(
                selected_evidence in {ref.ref_id, ref.hash} for ref in candidate.evidence_refs
            ) or any(
                selected_evidence
                in {
                    str(segment.get("source_id") or ""),
                    str(segment.get("sha256") or ""),
                }
                for segment in candidate.metadata.get("source_segments", ())
                if isinstance(segment, dict)
            ):
                if matched is not None and (
                    matched.block_id != candidate.block_id
                    or matched.revision_id != candidate.revision_id
                ):
                    raise AmbiguousEvidenceRefError("ambiguous_evidence_ref")
                matched = candidate
        if matched is not None:
            return matched
        raise ValueError("exact evidence reference was not found")

    def read_exact_records(
        self, record_ids: list[str] | tuple[str, ...]
    ) -> list[MemoryBlockRecord]:
        """Read an explicit, bounded batch of canonical records in caller order."""

        if (
            not isinstance(record_ids, (list, tuple))
            or not 1 <= len(record_ids) <= 128
            or any(not isinstance(value, str) or not value.strip() for value in record_ids)
            or len(set(record_ids)) != len(record_ids)
        ):
            raise ValueError("read_exact_records requires 1-128 distinct record IDs")
        return [self._require_profile_record(record_id) for record_id in record_ids]

    def list_records(
        self,
        *,
        scope: MemoryScope,
        scope_id: str,
        logical_block_id: str = "",
        limit: int = 32,
        cursor: str = "",
    ) -> dict[str, Any]:
        """Enumerate one explicit scope from a bounded, frozen ID/revision snapshot."""

        if scope not in {"user_global", "workspace", "project", "session", "task", "subtask"}:
            raise ValueError("list_records requires a typed scope")
        if not isinstance(scope_id, str) or not scope_id or scope_id != scope_id.strip():
            raise ValueError("list_records requires an exact scope_id")
        if not isinstance(logical_block_id, str) or logical_block_id != logical_block_id.strip():
            raise ValueError("list_records requires an exact logical_block_id")
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("list_records limit must be 1-128")
        if not isinstance(cursor, str):
            raise ValueError("list_records cursor must be a string")

        if cursor:
            if len(cursor) > 4096:
                raise ValueError("list_records cursor exceeds the bounded snapshot size")
            try:
                decoded = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
                if len(decoded) > 3072:
                    raise ValueError("list_records cursor exceeds the bounded snapshot size")
                snapshot = json.loads(decoded)
            except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("list_records cursor is invalid") from exc
            if (
                not isinstance(snapshot, dict)
                or set(snapshot)
                != {
                    "version",
                    "workspace_id",
                    "profile_id",
                    "scope",
                    "scope_id",
                    "logical_block_id",
                    "position",
                    "snapshot_id",
                }
                or snapshot["version"] != 2
            ):
                raise ValueError("list_records cursor is invalid")
            if (
                snapshot["workspace_id"] != self.workspace_id
                or snapshot["profile_id"] != self.profile_id
                or snapshot["scope"] != scope
                or snapshot["scope_id"] != scope_id
                or snapshot["logical_block_id"] != logical_block_id
            ):
                raise ValueError("list_records cursor scope does not match")
            selected = self._record_list_snapshot(snapshot["snapshot_id"])
            position = snapshot["position"]
            if (
                not isinstance(selected, list)
                or len(selected) > 2048
                or type(position) is not int
                or not 0 <= position < len(selected)
                or any(
                    not isinstance(row, list)
                    or len(row) != 2
                    or any(not isinstance(value, str) or not value for value in row)
                    for row in selected
                )
            ):
                raise ValueError("list_records cursor is invalid")
        else:
            records = sorted(
                (
                    record
                    for candidate_id in self.backend.scoped_record_ids(
                        workspace_id=self.workspace_id,
                        profile_id=self.profile_id,
                        scope_key=f"{scope}:{scope_id}",
                        logical_block_id=logical_block_id,
                    )
                    if (
                        record := self.backend.get_block(
                            candidate_id, workspace_id=self.workspace_id
                        )
                    )
                    is not None
                    and isinstance(record, MemoryBlockRecord)
                    and record.workspace_id == self.workspace_id
                    and record.profile_id == self.profile_id
                    and record.scope == scope
                    and memory_scope_key(record) == f"{scope}:{scope_id}"
                    and record.delivery_policy != "hidden"
                    and (record.block_membership is not None)
                    and (
                        not logical_block_id
                        or record.block_membership.primary_block_id == logical_block_id
                    )
                ),
                key=lambda record: (record.recorded_at, record.block_id),
            )
            if len(records) > 2048:
                raise ValueError("list_records scope exceeds 2048 records; select a narrower scope")
            selected = [[record.block_id, record.revision_id] for record in records]
            position = 0
            snapshot = {
                "version": 2,
                "workspace_id": self.workspace_id,
                "profile_id": self.profile_id,
                "scope": scope,
                "scope_id": scope_id,
                "logical_block_id": logical_block_id,
                "position": 0,
            }

        page: list[dict[str, str]] = []
        for record_id, revision_id in selected[position : position + limit]:
            record = self._require_profile_record(record_id)
            if (
                record.revision_id != revision_id
                or record.scope != scope
                or memory_scope_key(record) != f"{scope}:{scope_id}"
                or record.delivery_policy == "hidden"
                or record.block_membership is None
                or (
                    logical_block_id
                    and record.block_membership.primary_block_id != logical_block_id
                )
            ):
                raise ValueError("list_records snapshot was invalidated")
            page.append(
                {
                    "record_id": record.block_id,
                    "revision_id": record.revision_id,
                    "logical_block_id": record.block_membership.primary_block_id,
                }
            )
        next_position = position + len(page)
        next_cursor = ""
        if next_position < len(selected):
            if not cursor:
                snapshot["snapshot_id"] = self._record_list_snapshot("", selected)
            snapshot["position"] = next_position
            raw = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(raw) > 3072:
                raise ValueError("list_records snapshot exceeds cursor budget")
            next_cursor = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return {
            "records": page,
            "next_cursor": next_cursor,
            "snapshot_count": len(selected),
            "scope": scope,
            "scope_id": scope_id,
            "logical_block_id": logical_block_id,
        }

    def _record_list_snapshot(
        self, snapshot_id: str, records: list[list[str]] | None = None
    ) -> Any:
        """Bounded derived page cache; canonical records stay in their native store."""
        if records is None and (
            not isinstance(snapshot_id, str)
            or len(snapshot_id) != 64
            or any(char not in "0123456789abcdef" for char in snapshot_id)
        ):
            raise ValueError("list_records cursor is invalid")
        profile_key = hashlib.sha256(self.profile_id.encode("utf-8")).hexdigest()
        path = (
            self.repo_root
            / "workspaces"
            / self.workspace_id
            / ".memory"
            / "catalogs"
            / f"{profile_key}.record-pages.sqlite3"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        try:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS snapshots "
                    "(id TEXT PRIMARY KEY, payload TEXT NOT NULL, touched REAL NOT NULL)"
                )
                now = time.time()
                connection.execute("DELETE FROM snapshots WHERE touched < ?", (now - 86400,))
                if records is not None:
                    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
                    if len(payload.encode("utf-8")) > 450_000:
                        raise ValueError("list_records snapshot exceeds cursor budget")
                    snapshot_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                    connection.execute(
                        "INSERT INTO snapshots VALUES (?, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET touched=excluded.touched",
                        (snapshot_id, payload, now),
                    )
                    connection.execute(
                        "DELETE FROM snapshots WHERE id NOT IN "
                        "(SELECT id FROM snapshots ORDER BY touched DESC, id LIMIT 64)"
                    )
                    return snapshot_id
                row = connection.execute(
                    "SELECT payload FROM snapshots WHERE id=?", (snapshot_id,)
                ).fetchone()
                if row is None:
                    raise ValueError("list_records snapshot expired; restart enumeration")
                if hashlib.sha256(row[0].encode("utf-8")).hexdigest() != snapshot_id:
                    raise ValueError("list_records snapshot integrity failure")
                connection.execute("UPDATE snapshots SET touched=? WHERE id=?", (now, snapshot_id))
                return json.loads(row[0])
        finally:
            connection.close()

    def operation_status(self, operation_id: str) -> dict[str, Any]:
        """Read one original creation receipt and its current native projection state."""

        return self._operation_status_for_receipt(
            operation_id, caller_role="memory_client", protocol_version=2
        )

    def _operation_status_for_receipt(
        self, operation_id: str, *, caller_role: str, protocol_version: int
    ) -> dict[str, Any]:
        """Read a receipt in the caller identity already authenticated by its host."""

        if (
            not isinstance(operation_id, str)
            or not operation_id
            or operation_id != operation_id.strip()
            or len(operation_id) > 256
        ):
            raise ValueError("operation_status requires a bounded operation_id")
        entry = read_mutation_receipt_status(
            self.repo_root,
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            backend=self._backend_id(),
            caller_role=caller_role,
            protocol_version=protocol_version,
            mutation_correlation_id=operation_id,
        )
        if (entry is None or entry[1] is None) and caller_role == "memory_client":
            creation_reader = getattr(self.backend, "creation_operation_receipt", None)
            creation = (
                creation_reader(operation_id, workspace_id=self.workspace_id)
                if callable(creation_reader)
                else None
            )
            if creation is not None and creation["profile_id"] == self.profile_id:
                projection_status = str(creation["projection_status"])
                commit_revision = int(creation["commit_revision"])
                return {
                    "ok": True,
                    "operation_id": operation_id,
                    "record_id": str(creation["record_id"]),
                    "durable_status": "committed",
                    "commit_revision": commit_revision,
                    "projection_status": projection_status,
                    "projection_revision": (
                        commit_revision if projection_status == "committed" else 0
                    ),
                    "receipt_recovery": "canonical_creation_receipt",
                    "attention_required": projection_status == "failed",
                    "retryable": projection_status == "pending",
                }
        if entry is None:
            return {
                "ok": False,
                "operation_id": operation_id,
                "reason_code": "operation_not_found",
                "durable_status": "not_found",
            }
        status, stored = entry
        if stored is None:
            return {
                "ok": False,
                "operation_id": operation_id,
                "durable_status": "unknown",
                "projection_status": "unknown",
                "retryable": True,
            }
        original = (
            stored.get("response") if stored.get("receipt_kind") == "mutation_response" else stored
        )
        if not isinstance(original, dict):
            raise RuntimeError("operation receipt response is invalid")
        record_id = str(original.get("memory_id") or original.get("canonical_reference") or "")
        current_status = "unknown"
        commit_revision = 0
        if record_id:
            status_reader = getattr(self.backend, "write_outbox_status", None)
            entry_reader = getattr(self.backend, "latest_write_outbox_entry", None)
            if callable(status_reader):
                current_status = str(status_reader(record_id) or "unknown")
            if callable(entry_reader):
                outbox = entry_reader(block_id=record_id, workspace_id=self.workspace_id)
                commit_revision = int(getattr(outbox, "commit_revision", 0) or 0)
        reason_code = str(original.get("reason_code") or "")
        return {
            "ok": bool(original.get("ok")),
            "operation_id": operation_id,
            "record_id": record_id,
            "durable_status": status,
            "original_receipt": dict(original),
            "commit_revision": commit_revision,
            "projection_status": current_status,
            "projection_revision": commit_revision if current_status == "committed" else 0,
            "reason_code": reason_code,
            "attention_required": current_status == "failed",
            "retryable": (
                reason_code
                in {"native_transport_failure", "transport_timeout", "store_lock_unavailable"}
                if status == "failed"
                else current_status == "pending"
            ),
        }

    def projection_barrier(
        self,
        record_id: str,
        *,
        commit_revision: int = 0,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Observe one durable record's projection through the requested deadline."""

        selected_id = str(record_id or "").strip()
        if not selected_id:
            raise ValueError("projection_barrier requires record_id")
        if type(commit_revision) is not int or commit_revision < 0:
            raise ValueError("projection_barrier requires a non-negative commit_revision")
        if (
            type(timeout_seconds) not in {int, float}
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise ValueError("projection_barrier requires a finite non-negative timeout")
        self._require_profile_record(selected_id)
        deadline = time.monotonic() + timeout_seconds
        status_reader = getattr(self.backend, "write_outbox_status", None)
        entry_reader = getattr(self.backend, "latest_write_outbox_entry", None)
        while True:
            status = status_reader(selected_id) if callable(status_reader) else None
            entry = (
                entry_reader(block_id=selected_id, workspace_id=self.workspace_id)
                if callable(entry_reader)
                else None
            )
            revision = int(getattr(entry, "commit_revision", 0) or 0)
            if commit_revision and revision != commit_revision:
                return {
                    "ok": False,
                    "record_id": selected_id,
                    "projection_status": "unknown",
                    "commit_revision": revision,
                    "projection_revision": revision if status == "committed" else 0,
                    "reason_code": "projection_revision_mismatch",
                    "attention_required": True,
                    "retryable": False,
                }
            if status == "committed" and entry is not None:
                return {
                    "ok": True,
                    "record_id": selected_id,
                    "projection_status": "committed",
                    "commit_revision": revision,
                    "projection_revision": revision or int(commit_revision or 0),
                    "retryable": False,
                }
            if status == "failed" or getattr(entry, "projection_retry_exhausted", False):
                return {
                    "ok": False,
                    "record_id": selected_id,
                    "projection_status": status,
                    "commit_revision": revision,
                    "projection_revision": 0,
                    "reason_code": "native_projection_attention",
                    "attention_required": True,
                    "retryable": False,
                }
            if time.monotonic() >= deadline:
                return {
                    "ok": False,
                    "record_id": selected_id,
                    "projection_status": "pending",
                    "commit_revision": revision,
                    "projection_revision": 0,
                    "reason_code": "projection_deadline_exceeded",
                    "attention_required": False,
                    "retryable": True,
                }
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.05, remaining))

    def start_session(
        self,
        session_id: str | None = None,
        *,
        actor: str = "",
        project: str = "",
    ) -> AgentSession:
        bound_session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        if self.scope_lineage.session_id and self.scope_lineage.session_id != bound_session_id:
            raise ValueError("session_id does not match the MemoryClient scope lineage")
        session = AgentSession(
            session_id=bound_session_id,
            actor=actor,
            project=project,
            workspace_id=self.workspace_id,
        )
        self._session = session
        return session

    def end_session(self, session_id: str | None = None) -> AgentSession | None:
        current = self._session
        if current is None:
            return None
        if session_id and current.session_id != session_id:
            raise ValueError(f"active session is {current.session_id!r}, not {session_id!r}")
        self._session = None
        return current

    def _active_scope_lineage(self) -> ScopeLineage:
        session_id = self.scope_lineage.session_id
        if self._session is not None:
            if session_id and session_id != self._session.session_id:
                raise ValueError("active session does not match the MemoryClient scope lineage")
            session_id = self._session.session_id
        if session_id == self.scope_lineage.session_id:
            return self.scope_lineage
        return replace(self.scope_lineage, session_id=session_id)

    def _bind_write_scope_identity(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        subtask_id: str,
    ) -> ScopeLineage:
        lineage = self._active_scope_lineage()
        for field_name, claim in (
            ("project_id", project_id),
            ("session_id", session_id),
            ("task_id", task_id),
            ("subtask_id", subtask_id),
        ):
            normalized_claim = str(claim or "").strip()
            bound_value = getattr(lineage, field_name)
            if normalized_claim and normalized_claim != bound_value:
                raise ValueError(f"{field_name} is not bound to the MemoryClient scope lineage")
        return lineage

    def _block_catalog_view(
        self,
        block: MemoryBlock,
        descriptor: BlockDescriptor,
        *,
        include_index_state: bool,
        path: tuple[str, ...],
        child_ids: tuple[str, ...] | None = None,
        topology_state: TopologyBlockState | None = None,
        related_block_ids: tuple[str, ...] = (),
        native_partitions: tuple[dict[str, Any], ...] = (),
        block_metrics: tuple[dict[str, Any], ...] = (),
    ) -> dict[str, Any]:
        record_count = 0
        freshness = ""
        synopsis = descriptor.guidance.synopsis
        exact_dimensions: list[str] = []
        if include_index_state:
            canonical_records: list[MemoryBlockRecord] | None = None
            snapshot_reader = getattr(self.backend, "logical_block_descriptor_snapshot", None)
            snapshot = (
                snapshot_reader(
                    block.block_id,
                    workspace_id=self.workspace_id,
                    profile_id=self.profile_id,
                )
                if callable(snapshot_reader)
                else None
            )
            record_ids_for_blocks = getattr(self.backend, "record_ids_for_logical_blocks", None)
            get_block = getattr(self.backend, "get_block", None)
            if snapshot is None and callable(record_ids_for_blocks) and callable(get_block):
                canonical_records = []
                for record_id in record_ids_for_blocks(
                    (block.block_id,),
                    workspace_id=self.workspace_id,
                ):
                    candidate = get_block(record_id, workspace_id=self.workspace_id)
                    if not isinstance(candidate, MemoryBlockRecord):
                        continue
                    if candidate.profile_id != self.profile_id:
                        continue
                    if candidate.status in {
                        "superseded",
                        "deprecated",
                        "retracted",
                        "quarantined",
                        "expired",
                    }:
                        continue
                    canonical_records.append(candidate)
            if snapshot is not None:
                record_count = int(snapshot["record_count"])
                freshness = str(snapshot["latest_updated_at"])
                exact_dimensions = list(snapshot["exact_dimensions"])
                synopsis = (
                    f"{descriptor.guidance.synopsis} Coverage: {record_count} canonical record(s)."
                )
            elif canonical_records is not None:
                record_count = len(canonical_records)
                latest = (
                    max(
                        canonical_records,
                        key=lambda item: (
                            item.updated_at or item.recorded_at or item.created_at,
                            item.block_id,
                        ),
                    )
                    if canonical_records
                    else None
                )
                if latest is not None:
                    freshness = latest.updated_at or latest.recorded_at or latest.created_at
                exact_dimensions = sorted(
                    dimension
                    for dimension in (
                        "paths",
                        "symbols",
                        "entities",
                        "config_keys",
                        "endpoints",
                        "artifact_hashes",
                    )
                    if any(item.metadata.get(dimension) for item in canonical_records)
                )
                synopsis = (
                    f"{descriptor.guidance.synopsis} Coverage: {record_count} canonical record(s)."
                )
            get_index_node = getattr(self.backend, "get_index_node", None)
            if canonical_records is None and callable(get_index_node):
                node = get_index_node(
                    block_descriptor_node_id(self.workspace_id, block.block_id),
                    workspace_id=self.workspace_id,
                )
                if isinstance(node, MemoryIndexNode):
                    coverage = dict(node.metadata.get("coverage") or {})
                    record_count = int(coverage.get("canonical_record_count") or 0)
                    freshness = str(node.updated_at or node.created_at or "")
                    synopsis = (
                        f"{descriptor.guidance.synopsis} Coverage: {record_count} canonical "
                        "record(s)."
                    )
                    exact_dimensions = [
                        str(value) for value in coverage.get("exact_dimensions") or () if str(value)
                    ]
        visible_child_ids = block.child_ids if child_ids is None else child_ids
        writable = (
            block.agent_writable
            and not block.index_only
            and not visible_child_ids
            and descriptor.authority != "directive"
        )
        view = {
            "block_id": block.block_id,
            "name": block.name,
            "path": list(path),
            "tags": list(block.tags),
            "description": descriptor.guidance.synopsis,
            "synopsis": synopsis,
            "purpose": {
                "use_when": list(descriptor.guidance.use_when),
                "avoid_when": list(descriptor.guidance.avoid_when),
                "consumers": list(descriptor.consumers),
            },
            "scope": descriptor.scope,
            "temperature": block.temperature,
            "trust": descriptor.authority,
            "parent_id": block.parent_id,
            "child_ids": list(visible_child_ids),
            "expandable": bool(visible_child_ids),
            "links": {
                "parent_id": block.parent_id,
                "child_ids": list(visible_child_ids),
                "related_block_ids": list(related_block_ids),
            },
            "record_count": record_count,
            "canonical_record_count": record_count,
            "latest_title": "",
            "exact_dimensions": exact_dimensions,
            "freshness": freshness,
            "index_only": block.index_only,
            "agent_writable": block.agent_writable,
            "writable": writable,
            "metadata": dict(block.metadata),
            "workspace_id": self.workspace_id,
            "profile_id": self.profile_id,
            "revision": topology_state.revision if topology_state is not None else 0,
            "status": topology_state.status if topology_state is not None else "active",
            "protected": topology_state.protected if topology_state is not None else False,
            "backend_partition_id": (
                topology_state.backend_partition_id if topology_state is not None else ""
            ),
        }
        if not include_index_state:
            return view
        if writable:
            writable_reason = "agent_writable_leaf"
        elif visible_child_ids:
            writable_reason = "branch_index_content_free"
        elif block.index_only:
            writable_reason = "index_node_content_free"
        elif descriptor.authority == "directive":
            writable_reason = "directive_authority_host_only"
        else:
            writable_reason = "catalog_policy_non_agent_writable"
        allowed_operations = ["recall"]
        if visible_child_ids:
            allowed_operations = ["expand"]
        elif writable:
            allowed_operations.append("remember")
        if not visible_child_ids and descriptor.lifecycle_policy.get("mutable") is True:
            allowed_operations.append("lifecycle_transition")
        if not visible_child_ids and block.promotion_policy:
            allowed_operations.append("promote")
        primary_partition = dict(native_partitions[0]) if native_partitions else {}
        primary_metrics = dict(block_metrics[0]) if block_metrics else {}
        view.update(
            {
                "retention": descriptor.retention,
                "lifecycle": {
                    **dict(descriptor.lifecycle_policy),
                    "delivery_policy": descriptor.delivery_policy,
                },
                "retrieval_mode": descriptor.recall_path,
                "token_budget": descriptor.token_budget,
                "writable_reason": writable_reason,
                "promotion": {
                    "target": str(block.promotion_policy.get("target") or ""),
                    "requirements": [
                        str(value)
                        for value in block.promotion_policy.get("requires") or ()
                        if str(value)
                    ],
                },
                "allowed_operations": allowed_operations,
                "isolation": {
                    **dict(descriptor.isolation_policy),
                    "class": str(primary_partition.get("isolation_class") or "unmaterialized"),
                    "partition_mode": str(
                        primary_partition.get("partition_mode") or "unmaterialized"
                    ),
                },
                "native_partitions": [dict(partition) for partition in native_partitions],
                "metrics": primary_metrics,
                "block_metrics": [dict(metrics) for metrics in block_metrics],
            }
        )
        return view

    def block_map(self) -> list[dict[str, Any]]:
        states = load_workspace_topology(
            self.repo_root,
            self.workspace_id,
            profile_id=self.profile_id,
        )
        blocks = [state.block for state in states]
        states_by_id = {state.block_id: state for state in states}
        child_ids_by_parent = self._catalog_child_ids(blocks)
        catalog = compile_semantic_catalog(blocks)
        root_blocks = [block for block in blocks if not block.parent_id]
        return [
            self._block_catalog_view(
                block,
                catalog.descriptor(block.block_id),
                include_index_state=False,
                path=(block.block_id,),
                child_ids=child_ids_by_parent[block.block_id],
                topology_state=states_by_id[block.block_id],
            )
            for block in root_blocks
        ]

    def catalog_snapshot(self) -> dict[str, Any]:
        """Return one bounded-consumer snapshot without per-block database scans."""

        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            states = store.list_blocks()
            edges = store.list_edges()
            revision = store.catalog_revision()
        finally:
            store.close()
        blocks = [state.block for state in states]
        states_by_id = {state.block_id: state for state in states}
        child_ids_by_parent = self._catalog_child_ids(blocks)
        visible_root_ids = tuple(block.block_id for block in blocks if not block.parent_id)
        catalog = compile_semantic_catalog(blocks)
        descriptors = {descriptor.block_id: descriptor for descriptor in catalog.descriptors}
        paths = self._catalog_paths(
            visible_root_ids=visible_root_ids,
            child_ids_by_parent=child_ids_by_parent,
        )
        health = getattr(self.backend, "health", None)
        health_payload = health() if callable(health) else {}
        partition_rows = [
            dict(row)
            for row in health_payload.get("native_partitions") or ()
            if isinstance(row, dict)
        ]
        metric_rows = [
            dict(row) for row in health_payload.get("block_metrics") or () if isinstance(row, dict)
        ]
        partitions_by_block: dict[str, list[dict[str, object]]] = {}
        metrics_by_block: dict[str, list[dict[str, object]]] = {}
        for row in partition_rows:
            partitions_by_block.setdefault(str(row.get("primary_block_id") or ""), []).append(row)
        for row in metric_rows:
            metrics_by_block.setdefault(str(row.get("primary_block_id") or ""), []).append(row)
        edges_by_block: dict[str, list[BlockGraphEdge]] = {block.block_id: [] for block in blocks}
        for edge in edges:
            if edge.src_id in edges_by_block and edge.dst_id in edges_by_block:
                edges_by_block[edge.src_id].append(edge)
                edges_by_block[edge.dst_id].append(edge)

        def related_for(block_id: str) -> tuple[str, ...]:
            return tuple(
                dict.fromkeys(
                    edge.dst_id if edge.src_id == block_id else edge.src_id
                    for edge in edges_by_block[block_id]
                )
            )

        views = []
        for block in blocks:
            path = paths.get(block.block_id, ())
            if not path:
                raise RuntimeError("catalog block is not reachable from a visible root")
            views.append(
                self._block_catalog_view(
                    block,
                    descriptors[block.block_id],
                    include_index_state=True,
                    path=path,
                    child_ids=child_ids_by_parent[block.block_id],
                    topology_state=states_by_id[block.block_id],
                    related_block_ids=related_for(block.block_id),
                    native_partitions=tuple(partitions_by_block.get(block.block_id, ())),
                    block_metrics=tuple(metrics_by_block.get(block.block_id, ())),
                )
            )
        return {
            "catalog_revision": revision,
            "blocks": views,
            "edges": [edge.to_mapping() for edge in edges],
        }

    def expand_block(self, block_id: str) -> dict[str, Any]:
        selected_id = str(block_id or "").strip()
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            states = store.list_blocks()
            edges_by_block = {state.block_id: store.edges_for(state.block_id) for state in states}
        finally:
            store.close()
        blocks = [state.block for state in states]
        states_by_id = {state.block_id: state for state in states}
        by_id = {block.block_id: block for block in blocks}
        child_ids_by_parent = self._catalog_child_ids(blocks)
        visible_root_ids = tuple(block.block_id for block in blocks if not block.parent_id)
        path_ids = self._catalog_path(
            selected_id,
            visible_root_ids=visible_root_ids,
            child_ids_by_parent=child_ids_by_parent,
        )
        if not path_ids:
            raise ValueError("block_id must identify a visible catalog block")
        catalog = compile_semantic_catalog(blocks)
        selected = by_id[selected_id]
        children = [by_id[child_id] for child_id in child_ids_by_parent[selected_id]]
        health = getattr(self.backend, "health", None)
        health_payload = health() if callable(health) else {}
        partition_rows = [
            dict(row)
            for row in health_payload.get("native_partitions") or ()
            if isinstance(row, dict)
        ]
        metric_rows = [
            dict(row) for row in health_payload.get("block_metrics") or () if isinstance(row, dict)
        ]

        def partitions_for(block_id: str) -> tuple[dict[str, Any], ...]:
            return tuple(
                row for row in partition_rows if str(row.get("primary_block_id") or "") == block_id
            )

        def metrics_for(block_id: str) -> tuple[dict[str, Any], ...]:
            return tuple(
                row for row in metric_rows if str(row.get("primary_block_id") or "") == block_id
            )

        def related_for(block_id: str) -> tuple[str, ...]:
            return tuple(
                dict.fromkeys(
                    edge.dst_id if edge.src_id == block_id else edge.src_id
                    for edge in edges_by_block[block_id]
                )
            )

        return {
            "path_ids": list(path_ids),
            "root": self._block_catalog_view(
                selected,
                catalog.descriptor(selected.block_id),
                include_index_state=True,
                path=path_ids,
                child_ids=child_ids_by_parent[selected.block_id],
                topology_state=states_by_id[selected.block_id],
                related_block_ids=related_for(selected.block_id),
                native_partitions=partitions_for(selected.block_id),
                block_metrics=metrics_for(selected.block_id),
            ),
            "children": [
                self._block_catalog_view(
                    child,
                    catalog.descriptor(child.block_id),
                    include_index_state=True,
                    path=(*path_ids, child.block_id),
                    child_ids=child_ids_by_parent[child.block_id],
                    topology_state=states_by_id[child.block_id],
                    related_block_ids=related_for(child.block_id),
                    native_partitions=partitions_for(child.block_id),
                    block_metrics=metrics_for(child.block_id),
                )
                for child in children
            ],
        }

    def _create_block(
        self,
        *,
        name: str,
        synopsis: str,
        parent_id: str,
        reason: str,
        tags: list[str] | tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        scope: str = "workspace",
        index_only: bool = False,
        token_budget: int = 1000,
        protected: bool = False,
        lifecycle_expires_at: str = "",
    ) -> dict[str, Any]:
        return create_workspace_block(
            self.repo_root,
            self.workspace_id,
            profile_id=self.profile_id,
            actor=self._topology_actor,
            name=name,
            synopsis=synopsis,
            parent_id=parent_id,
            reason=reason,
            tags=tags,
            metadata=metadata,
            scope=scope,
            index_only=index_only,
            token_budget=token_budget,
            protected=protected,
            lifecycle_expires_at=lifecycle_expires_at,
        )

    def create_index_block(
        self,
        *,
        name: str,
        synopsis: str,
        parent_id: str,
        reason: str,
        tags: list[str] | tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        scope: str = "workspace",
        token_budget: int = 0,
        protected: bool = False,
        lifecycle_expires_at: str = "",
    ) -> dict[str, Any]:
        return self._create_block(
            name=name,
            synopsis=synopsis,
            parent_id=parent_id,
            reason=reason,
            tags=tags,
            metadata=metadata,
            scope=scope,
            index_only=True,
            token_budget=token_budget,
            protected=protected,
            lifecycle_expires_at=lifecycle_expires_at,
        )

    def create_leaf_block(
        self,
        *,
        name: str,
        synopsis: str,
        parent_id: str,
        reason: str,
        tags: list[str] | tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        scope: str = "workspace",
        token_budget: int = 1000,
        protected: bool = False,
        lifecycle_expires_at: str = "",
    ) -> dict[str, Any]:
        return self._create_block(
            name=name,
            synopsis=synopsis,
            parent_id=parent_id,
            reason=reason,
            tags=tags,
            metadata=metadata,
            scope=scope,
            index_only=False,
            token_budget=token_budget,
            protected=protected,
            lifecycle_expires_at=lifecycle_expires_at,
        )

    def amend_block(
        self,
        block_id: str,
        *,
        expected_revision: int,
        name: str | None = None,
        synopsis: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        metadata_patch: Mapping[str, Any] | None = None,
        token_budget: int | None = None,
        lifecycle_expires_at: str | None = None,
        reason: str,
    ) -> dict[str, Any]:
        return amend_workspace_block(
            self.repo_root,
            self.workspace_id,
            block_id,
            expected_revision=expected_revision,
            profile_id=self.profile_id,
            actor=self._topology_actor,
            name=name,
            synopsis=synopsis,
            tags=tags,
            metadata_patch=metadata_patch,
            token_budget=token_budget,
            lifecycle_expires_at=lifecycle_expires_at,
            reason=reason,
        )

    def move_block(
        self,
        block_id: str,
        *,
        parent_id: str,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        return amend_workspace_block(
            self.repo_root,
            self.workspace_id,
            block_id,
            expected_revision=expected_revision,
            profile_id=self.profile_id,
            actor=self._topology_actor,
            parent_id=parent_id,
            reason=reason,
        )

    def protect_block(
        self,
        block_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        return protect_workspace_block(
            self.repo_root,
            self.workspace_id,
            block_id,
            expected_revision=expected_revision,
            reason=reason,
            profile_id=self.profile_id,
            actor=self._topology_actor,
        )

    def repair_protected_block_parent(
        self,
        block_id: str,
        *,
        expected_revision: int,
        expected_parent_id: str,
        parent_id: str,
        expected_metadata: Mapping[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        """Repair one host-owned protected block after exact ownership checks."""

        return repair_protected_workspace_block_parent(
            self.repo_root,
            self.workspace_id,
            block_id,
            expected_revision=expected_revision,
            expected_parent_id=expected_parent_id,
            parent_id=parent_id,
            expected_metadata=expected_metadata,
            reason=reason,
            profile_id=self.profile_id,
        )

    def archive_block(
        self,
        block_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        return archive_workspace_block(
            self.repo_root,
            self.workspace_id,
            block_id,
            expected_revision=expected_revision,
            reason=reason,
            profile_id=self.profile_id,
            actor=self._topology_actor,
        )

    def archive_session_block(
        self,
        block_id: str,
        *,
        expected_revision: int,
        session_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Archive a protected scaffold only inside an explicit chat forget."""

        return archive_workspace_session_block(
            self.repo_root,
            self.workspace_id,
            block_id,
            expected_revision=expected_revision,
            session_id=session_id,
            reason=reason,
            profile_id=self.profile_id,
        )

    def revive_block(
        self,
        block_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        return amend_workspace_block(
            self.repo_root,
            self.workspace_id,
            block_id,
            expected_revision=expected_revision,
            profile_id=self.profile_id,
            archived=False,
            reason=reason,
        )

    def revive_session_block(
        self,
        block_id: str,
        *,
        expected_revision: int,
        session_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Rollback a protected session scaffold in a failed forget transaction."""

        return amend_workspace_block(
            self.repo_root,
            self.workspace_id,
            block_id,
            expected_revision=expected_revision,
            profile_id=self.profile_id,
            archived=False,
            reason=reason,
            actor="system",
            allow_protected_session=True,
            expected_session_id=session_id,
        )

    def set_session_block_expiry(
        self,
        block_id: str,
        *,
        expected_revision: int,
        session_id: str,
        expires_at: str,
        reason: str,
    ) -> dict[str, Any]:
        """Set an explicit TTL on a block owned by one chat session."""

        return amend_workspace_block(
            self.repo_root,
            self.workspace_id,
            block_id,
            expected_revision=expected_revision,
            profile_id=self.profile_id,
            lifecycle_expires_at=expires_at,
            reason=reason,
            actor="system",
            allow_protected_session=True,
            expected_session_id=session_id,
        )

    def expire_due_blocks(
        self,
        *,
        as_of: str = "",
        reason: str = "memory block lifecycle TTL elapsed",
    ) -> dict[str, Any]:
        """Archive explicitly expired, otherwise unreferenced topology blocks.

        This is an opt-in maintenance pass.  It never guesses that a block is
        stale from emptiness alone and reports blockers instead of bypassing
        graph, mask, record, or protection guards.
        """

        raw_as_of = str(as_of or "").strip()
        try:
            now = (
                datetime.fromisoformat(raw_as_of.replace("Z", "+00:00"))
                if raw_as_of
                else datetime.now(timezone.utc)
            )
        except ValueError as exc:
            raise ValueError("as_of must be ISO-8601") from exc
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now = now.astimezone(timezone.utc)

        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            states = store.list_blocks()
        finally:
            store.close()

        def parse_timestamp(value: object, *, field_name: str) -> datetime:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{field_name} must be ISO-8601") from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        def expiry_for(state: TopologyBlockState) -> datetime | None:
            value = str(state.block.lifecycle_policy.get("expires_at") or "").strip()
            if not value:
                return None
            return parse_timestamp(value, field_name=f"memory block {state.block_id!r} expiry")

        due = []
        for state in states:
            expiry = expiry_for(state)
            if expiry is not None and expiry <= now:
                due.append(state)
        by_id = {state.block_id: state for state in states}

        def depth(state: TopologyBlockState) -> int:
            value = 0
            parent_id = state.block.parent_id
            while parent_id in by_id:
                value += 1
                parent_id = by_id[parent_id].block.parent_id
            return value

        archived: list[str] = []
        blockers: dict[str, list[str]] = {}
        for state in sorted(due, key=depth, reverse=True):
            current_store = WorkspaceCatalogStore(
                self.repo_root,
                self.workspace_id,
                self.profile_id,
            )
            try:
                current = current_store.get_block(state.block_id)
                if current is None or current.status != "active":
                    continue
                current_expiry = expiry_for(current)
                if current_expiry is None or current_expiry > now:
                    continue
                children = tuple(
                    item.block_id
                    for item in current_store.list_blocks()
                    if item.block.parent_id == current.block_id
                )
                reasons: list[str] = []
                if children:
                    reasons.append("block_has_children")
                if current_store.edges_for(current.block_id):
                    reasons.append("block_has_graph_edges")
                if current_store.block_has_active_mask_reference(current.block_id):
                    reasons.append("block_has_active_mask_references")
            finally:
                current_store.close()
            if reasons:
                blockers[state.block_id] = reasons
                continue
            try:
                if self._current_records_in_block(state.block_id):
                    blockers[state.block_id] = ["block_has_current_records"]
                    continue
            except ValueError as exc:
                blockers[state.block_id] = [f"record_coverage_unavailable:{exc}"]
                continue

            session_id = str(current.block.metadata.get("umbrella_memory_session_id") or "").strip()
            if current.protected:
                if current.block.scope != "session" or not session_id:
                    blockers[state.block_id] = ["block_is_protected"]
                    continue
                archive_workspace_session_block(
                    self.repo_root,
                    self.workspace_id,
                    state.block_id,
                    expected_revision=current.revision,
                    session_id=session_id,
                    reason=reason,
                    profile_id=self.profile_id,
                    actor="system",
                )
            else:
                archive_workspace_block(
                    self.repo_root,
                    self.workspace_id,
                    state.block_id,
                    expected_revision=current.revision,
                    reason=reason,
                    profile_id=self.profile_id,
                    actor="system",
                )
            archived.append(state.block_id)

        return {
            "ok": True,
            "archived_block_ids": archived,
            "blockers": blockers,
            "as_of": now.isoformat(),
        }

    def delete_block(
        self,
        block_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            state = store.get_block(block_id)
            if state is None:
                raise ValueError("memory block was not found in this profile")
            if state.status != "archived":
                raise ValueError("hard delete requires an archived block")
            if any(
                item.block.parent_id == block_id
                for item in store.list_blocks(include_archived=True)
            ):
                raise ValueError("hard delete requires no children")
            if store.edges_for(block_id):
                raise ValueError("hard delete requires no graph edges")
            if store.block_has_active_mask_reference(block_id):
                raise ValueError("hard delete requires no active mask references")
            record_ids_for_blocks = getattr(self.backend, "record_ids_for_logical_blocks", None)
            get_block = getattr(self.backend, "get_block", None)
            if not callable(record_ids_for_blocks) or not callable(get_block):
                raise ValueError("hard delete cannot prove that the block has no current records")
            current_records = [
                record
                for record_id in record_ids_for_blocks(
                    (block_id,),
                    workspace_id=self.workspace_id,
                )
                if isinstance(
                    (record := get_block(record_id, workspace_id=self.workspace_id)),
                    MemoryBlockRecord,
                )
                and record.profile_id == self.profile_id
                and record.status not in {"superseded", "deprecated", "retracted", "expired"}
            ]
            if current_records:
                raise ValueError("hard delete requires no current records")
            tombstone = store.delete_block(
                state,
                expected_revision=expected_revision,
                actor=self._topology_actor,
                reason=reason,
            )
            return {"ok": True, "deleted": True, "tombstone": tombstone}
        finally:
            store.close()

    def _current_records_in_block(self, block_id: str) -> tuple[MemoryBlockRecord, ...]:
        record_ids_for_blocks = getattr(self.backend, "record_ids_for_logical_blocks", None)
        get_block = getattr(self.backend, "get_block", None)
        if not callable(record_ids_for_blocks) or not callable(get_block):
            raise ValueError("selected backend cannot prove block record coverage")
        records = tuple(
            record
            for record_id in record_ids_for_blocks(
                (block_id,),
                workspace_id=self.workspace_id,
                profile_id=self.profile_id,
            )
            if isinstance(
                (record := get_block(record_id, workspace_id=self.workspace_id)),
                MemoryBlockRecord,
            )
            and record.profile_id == self.profile_id
            and record.status
            not in {"superseded", "deprecated", "quarantined", "retracted", "expired"}
        )
        return records

    def _catalog_revision_id(self) -> str:
        return compile_semantic_catalog(
            load_memory_blocks(
                self.repo_root,
                self.workspace_id,
                profile_id=self.profile_id,
                strict=True,
            )
        ).revision_id

    def _backend_partition_id_for_block(self, block_id: str) -> str:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            state = store.get_block(block_id)
        finally:
            store.close()
        if state is not None:
            if not state.backend_partition_id:
                raise ValueError("content-bearing block has no backend_partition_id")
            return state.backend_partition_id
        material = json.dumps([self.profile_id, block_id], separators=(",", ":")).encode("utf-8")
        return "static:" + hashlib.sha256(material).hexdigest()

    def _topology_evolution_receipt(self, operation_id: str) -> dict[str, Any]:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            evolution = store.topology_evolution(operation_id)
            if evolution is None or evolution["state"] != "published":
                raise ValueError("topology evolution is not published")
            payload = evolution["payload"]
            source_ids = tuple(payload["sources"])
            target_ids = tuple(item["block"]["block_id"] for item in payload["targets"])
            sources = [store.get_block(source_id) for source_id in source_ids]
            targets = [store.get_block(target_id) for target_id in target_ids]
            if any(state is None for state in (*sources, *targets)):
                raise RuntimeError("published topology evolution has a missing block")
            source_states = cast(list[TopologyBlockState], sources)
            target_states = cast(list[TopologyBlockState], targets)
            edges = [
                edge.to_mapping()
                for target_id in target_ids
                for edge in store.edges_for(target_id)
                if edge.src_id == target_id
                and edge.dst_id in source_ids
                and edge.edge_type == "derived_from"
            ]
            result: dict[str, Any] = {
                "ok": True,
                "operation": payload["operation"],
                "operation_id": operation_id,
                "generation_status": "published",
                "record_assignments": {
                    record_id: item["target_block_id"]
                    for record_id, item in payload["assignments"].items()
                },
                "provenance_edges": edges,
                "catalog_revision": store.catalog_revision(),
            }
            if payload["operation"] == "split":
                result["source"] = source_states[0].to_mapping()
                result["targets"] = [state.to_mapping() for state in target_states]
            else:
                result["sources"] = [state.to_mapping() for state in source_states]
                result["target"] = target_states[0].to_mapping()
            return result
        finally:
            store.close()

    def _evolution_target_block(
        self,
        template: MemoryBlock,
        target: BlockEvolutionTarget,
        *,
        parent_id: str,
        operation_id: str,
        ordinal: int,
    ) -> tuple[MemoryBlock, str]:
        namespace = f"lethyx:{self.workspace_id}:{self.profile_id}:{operation_id}:{ordinal}"
        block = replace(
            template,
            block_id=str(uuid.uuid5(uuid.NAMESPACE_URL, namespace + ":block")),
            parent_id=parent_id,
            name=target.name,
            guidance={"synopsis": target.synopsis, "use_when": (), "avoid_when": ()},
            tags=target.tags,
            metadata=target.metadata or {},
            token_budget=target.token_budget,
        )
        partition_id = str(uuid.uuid5(uuid.NAMESPACE_URL, namespace + ":partition"))
        return block, partition_id

    def _execute_topology_evolution(
        self, operation_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        stage = getattr(self.backend, "stage_topology_projection", None)
        canonical_gate = getattr(self.backend, "canonical_write_gate", None)
        if (
            not self.capabilities().staged_topology_evolution
            or not callable(stage)
            or not callable(canonical_gate)
        ):
            raise ValueError("selected backend does not support staged_topology_evolution")
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            evolution = store.prepare_topology_evolution(operation_id, payload)
            if evolution["state"] == "published":
                return self._topology_evolution_receipt(operation_id)
            if evolution["state"] == "aborted":
                raise ValueError(
                    "topology evolution was aborted: " + str(evolution["abort_reason"])
                )
            if evolution["state"] == "prepared":
                targets = {item["block"]["block_id"]: item for item in payload["targets"]}
                projection_groups: dict[str, list[MemoryBlockRecord]] = {
                    target_id: [] for target_id in targets
                }
                for record_id, assignment in payload["assignments"].items():
                    record = self._require_profile_record(record_id)
                    if record.revision_id != assignment["record_revision_id"]:
                        store.abort_topology_evolution(
                            operation_id, reason="source_record_revision_changed"
                        )
                        raise ValueError(
                            "topology source record revision changed during projection"
                        )
                    membership = record.block_membership
                    if membership is None or membership.primary_block_id not in payload["sources"]:
                        store.abort_topology_evolution(
                            operation_id, reason="source_record_membership_changed"
                        )
                        raise ValueError("topology source record has no primary membership")
                    target = targets[assignment["target_block_id"]]
                    projected = replace(
                        record,
                        block_membership=replace(
                            membership,
                            primary_block_id=assignment["target_block_id"],
                            backend_partition_id=target["backend_partition_id"],
                            catalog_revision_id=f"topology:{operation_id}",
                            secondary_block_ids=tuple(
                                block_id
                                for block_id in membership.secondary_block_ids
                                if block_id != assignment["target_block_id"]
                            ),
                            membership_revision_id="",
                        ),
                    )
                    projection_groups[assignment["target_block_id"]].append(projected)

                def project_group(records: list[MemoryBlockRecord]) -> None:
                    try:
                        for projected in records:
                            stage(projected)
                    finally:
                        index = getattr(self.backend, "_index", None)
                        close_connection = getattr(index, "close_thread_connection", None)
                        if callable(close_connection):
                            close_connection()

                with ThreadPoolExecutor(max_workers=min(4, len(projection_groups))) as workers:
                    futures = [
                        workers.submit(project_group, records)
                        for records in projection_groups.values()
                    ]
                    for future in futures:
                        future.result()
                store.mark_topology_projections_ready(operation_id)
            with canonical_gate():
                latest = store.topology_evolution(operation_id)
                if latest is not None and latest["state"] == "published":
                    return self._topology_evolution_receipt(operation_id)
                current = {
                    record.block_id: record.revision_id
                    for source_id in payload["sources"]
                    for record in self._current_records_in_block(source_id)
                }
                expected = {
                    record_id: item["record_revision_id"]
                    for record_id, item in payload["assignments"].items()
                }
                if current != expected:
                    store.abort_topology_evolution(operation_id, reason="source_record_set_changed")
                    raise ValueError("topology source record set changed before publication")
                try:
                    store.publish_topology_evolution(operation_id)
                except CatalogRevisionConflict:
                    store.abort_topology_evolution(operation_id, reason="catalog_revision_conflict")
                    raise
        finally:
            store.close()
        return self._topology_evolution_receipt(operation_id)

    def pending_topology_evolutions(self) -> tuple[dict[str, Any], ...]:
        """List exact pending generation IDs without record text or topology semantics."""

        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            return store.pending_topology_evolutions()
        finally:
            store.close()

    def resume_topology_evolution(self, operation_id: str) -> dict[str, Any]:
        """Resume only the previously authored and durably prepared exact mapping."""

        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            evolution = store.topology_evolution(operation_id)
        finally:
            store.close()
        if evolution is None:
            raise ValueError("topology evolution operation was not found")
        return self._execute_topology_evolution(operation_id, evolution["payload"])

    def abort_topology_evolution(self, operation_id: str, *, reason: str) -> dict[str, Any]:
        """Explicitly abandon an unpublished mapping while keeping its audit journal."""

        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            store.abort_topology_evolution(operation_id, reason=reason)
            evolution = store.topology_evolution(operation_id)
            if evolution is None:
                raise RuntimeError("aborted topology evolution is missing")
            return {
                "ok": True,
                "operation_id": operation_id,
                "state": evolution["state"],
                "reason": evolution["abort_reason"],
            }
        finally:
            store.close()

    def split_block(
        self,
        block_id: str,
        *,
        expected_revision: int,
        targets: tuple[BlockEvolutionTarget, ...],
        reason: str,
        operation_id: str = "",
    ) -> dict[str, Any]:
        if not targets:
            raise ValueError("split requires at least one typed target")
        if any(not isinstance(target, BlockEvolutionTarget) for target in targets):
            raise TypeError("split targets must contain BlockEvolutionTarget values")
        if (
            not isinstance(operation_id, str)
            or operation_id != operation_id.strip()
            or len(operation_id) > 256
        ):
            raise ValueError("split operation_id must be a bounded opaque string")
        if not self.capabilities().staged_topology_evolution:
            raise ValueError("selected backend does not support staged_topology_evolution")
        selected_operation = str(operation_id or uuid.uuid4()).strip()
        request_hash = hash_value(
            {
                "operation": "split",
                "source": block_id,
                "expected_revision": expected_revision,
                "targets": [asdict(target) for target in targets],
                "reason": reason,
            }
        )
        if operation_id:
            store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
            try:
                existing = store.topology_evolution(selected_operation)
            finally:
                store.close()
            if existing is not None:
                if existing["payload"].get("request_hash") != request_hash:
                    raise ValueError("operation_id already binds a different topology request")
                return self._execute_topology_evolution(selected_operation, existing["payload"])
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            source = store.get_block(block_id)
            if source is None or source.status != "active":
                raise ValueError("split source must be an active block in this profile")
            if source.protected or source.block.index_only:
                raise ValueError("split source must be an agent-writable leaf")
            if source.revision != expected_revision:
                raise ValueError("expected_revision does not match the split source")
            if any(item.block.parent_id == block_id for item in store.list_blocks()):
                raise ValueError("split source must not have active children")
            if any(
                edge.src_id != block_id or edge.edge_type != "derived_from"
                for edge in store.edges_for(block_id)
            ):
                raise ValueError("split source graph edges must be explicitly relocated first")
            if store.block_has_active_mask_reference(block_id):
                raise ValueError("split source has active mask references")
            parent_id = source.block.parent_id
        finally:
            store.close()
        records = self._current_records_in_block(block_id)
        actual_ids = {record.block_id for record in records}
        assigned_list = [record_id for target in targets for record_id in target.record_ids]
        if len(assigned_list) != len(set(assigned_list)):
            raise ValueError("split must assign every current record exactly once")
        if set(assigned_list) != actual_ids:
            raise ValueError("split record mapping must exactly cover current records")
        target_rows = [
            self._evolution_target_block(
                source.block,
                target,
                parent_id=parent_id,
                operation_id=selected_operation,
                ordinal=ordinal,
            )
            for ordinal, target in enumerate(targets)
        ]
        candidate = [
            *load_memory_blocks(self.repo_root, self.workspace_id, profile_id=self.profile_id),
            *(block for block, _ in target_rows),
        ]
        validate_memory_block_tree(candidate)
        validate_leaf_block_contracts(candidate)
        payload = {
            "operation": "split",
            "sources": {block_id: expected_revision},
            "targets": [
                {"block": asdict(block), "backend_partition_id": partition_id}
                for block, partition_id in target_rows
            ],
            "assignments": {
                record_id: {
                    "record_revision_id": record.revision_id,
                    "target_block_id": block.block_id,
                }
                for (block, _), target in zip(target_rows, targets, strict=True)
                for record_id in target.record_ids
                for record in records
                if record.block_id == record_id
            },
            "reason": reason,
            "actor": self._topology_actor,
            "request_hash": request_hash,
        }
        return self._execute_topology_evolution(selected_operation, payload)

    def merge_blocks(
        self,
        sources: tuple[BlockMergeSource, ...],
        *,
        target: BlockEvolutionTarget,
        parent_id: str,
        reason: str,
        operation_id: str = "",
    ) -> dict[str, Any]:
        if len(sources) < 2 or any(not isinstance(source, BlockMergeSource) for source in sources):
            raise ValueError("merge requires at least two typed sources")
        if not isinstance(target, BlockEvolutionTarget):
            raise TypeError("merge target must be a BlockEvolutionTarget")
        if (
            not isinstance(operation_id, str)
            or operation_id != operation_id.strip()
            or len(operation_id) > 256
        ):
            raise ValueError("merge operation_id must be a bounded opaque string")
        if not self.capabilities().staged_topology_evolution:
            raise ValueError("selected backend does not support staged_topology_evolution")
        source_ids = tuple(source.block_id for source in sources)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("merge source blocks must be unique")
        selected_operation = str(operation_id or uuid.uuid4()).strip()
        request_hash = hash_value(
            {
                "operation": "merge",
                "sources": [asdict(source) for source in sources],
                "target": asdict(target),
                "parent_id": parent_id,
                "reason": reason,
            }
        )
        if operation_id:
            store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
            try:
                existing = store.topology_evolution(selected_operation)
            finally:
                store.close()
            if existing is not None:
                if existing["payload"].get("request_hash") != request_hash:
                    raise ValueError("operation_id already binds a different topology request")
                return self._execute_topology_evolution(selected_operation, existing["payload"])
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            parent = store.get_block(parent_id)
            if parent is None or parent.status != "active" or not parent.block.index_only:
                raise ValueError("merge target parent must be an active index block")
            states = []
            for source in sources:
                state = store.get_block(source.block_id)
                if state is None or state.status != "active" or state.block.index_only:
                    raise ValueError("merge sources must be active leaves in this profile")
                if state.revision != source.expected_revision:
                    raise ValueError("expected_revision does not match a merge source")
                if any(edge.dst_id == state.block_id for edge in store.edges_for(state.block_id)):
                    raise ValueError(
                        "merge source incoming edges must be explicitly relocated first"
                    )
                if store.block_has_active_mask_reference(state.block_id):
                    raise ValueError("merge source has active mask references")
                states.append(state)
            authority = {state.block.authority for state in states}
            scope = {state.block.scope for state in states}
            retention = {state.block.retention for state in states}
            if len(authority) != 1 or len(scope) != 1 or len(retention) != 1:
                raise ValueError("merge sources have incompatible governance")
        finally:
            store.close()
        original_assignment: dict[str, str] = {}
        for source_id in source_ids:
            for record in self._current_records_in_block(source_id):
                if record.block_id in original_assignment:
                    raise ValueError("merge source record coverage overlaps")
                original_assignment[record.block_id] = source_id
        if set(target.record_ids) != set(original_assignment) or len(target.record_ids) != len(
            original_assignment
        ):
            raise ValueError("merge target record mapping must exactly cover all current records")
        block, partition_id = self._evolution_target_block(
            states[0].block,
            target,
            parent_id=parent_id,
            operation_id=selected_operation,
            ordinal=0,
        )
        candidate = [
            *load_memory_blocks(self.repo_root, self.workspace_id, profile_id=self.profile_id),
            block,
        ]
        validate_memory_block_tree(candidate)
        validate_leaf_block_contracts(candidate)
        records = {
            record.block_id: record
            for source_id in source_ids
            for record in self._current_records_in_block(source_id)
        }
        payload = {
            "operation": "merge",
            "sources": {source.block_id: source.expected_revision for source in sources},
            "targets": [{"block": asdict(block), "backend_partition_id": partition_id}],
            "assignments": {
                record_id: {
                    "record_revision_id": records[record_id].revision_id,
                    "target_block_id": block.block_id,
                }
                for record_id in target.record_ids
            },
            "reason": reason,
            "actor": self._topology_actor,
            "request_hash": request_hash,
        }
        return self._execute_topology_evolution(selected_operation, payload)

    def add_block_edge(
        self,
        src_id: str,
        dst_id: str,
        *,
        edge_type: BlockEdgeType,
        reason: str,
    ) -> dict[str, Any]:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            return store.add_edge(
                src_id,
                dst_id,
                edge_type=edge_type,
                reason=reason,
                actor=self._topology_actor,
            ).to_mapping()
        finally:
            store.close()

    def remove_block_edge(
        self,
        edge_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            edge = store.remove_edge(
                edge_id,
                expected_revision=expected_revision,
                reason=reason,
                actor=self._topology_actor,
            )
            return {"ok": True, "removed": edge.to_mapping()}
        finally:
            store.close()

    def related_blocks(self, block_id: str) -> list[dict[str, Any]]:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            related: list[dict[str, Any]] = []
            for edge in store.edges_for(block_id):
                other_id = edge.dst_id if edge.src_id == block_id else edge.src_id
                state = store.get_block(other_id)
                if state is not None:
                    related.append({**state.to_mapping(), "edge": edge.to_mapping()})
            return related
        finally:
            store.close()

    def block_path(self, block_id: str) -> list[str]:
        return list(self.expand_block(block_id)["path_ids"])

    def children(self, block_id: str) -> list[dict[str, Any]]:
        return list(self.expand_block(block_id)["children"])

    def catalog_revision(self) -> int:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            return store.catalog_revision()
        finally:
            store.close()

    def observe_maintenance(
        self,
        block_id: str,
        *,
        observation_type: MaintenanceObservationType,
        values: dict[str, int | float],
        threshold_revision: str,
        evidence_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            return store.add_observation(
                block_id=block_id,
                observation_type=observation_type,
                values=values,
                threshold_revision=threshold_revision,
                evidence_ids=evidence_ids,
            ).to_mapping()
        finally:
            store.close()

    def maintenance_observations(self, block_id: str) -> list[dict[str, Any]]:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            return [item.to_mapping() for item in store.observations(block_id)]
        finally:
            store.close()

    def navigation_envelope(self, block_id: str) -> dict[str, Any]:
        """Return content-free navigation, health, and advisory maintenance facts."""

        expanded = self.expand_block(block_id)
        block = dict(expanded["root"])
        observations = self.maintenance_observations(block_id)
        proposal_by_observation = {
            "capacity_pressure": ("propose_split", "propose_budget_change"),
            "low_usage": ("propose_merge", "propose_archive"),
            "high_revision_count": ("propose_consolidation",),
            "many_conflicts": ("propose_split", "propose_edge_review"),
            "active_reference_present": ("propose_mask_review",),
            "consolidation_available": ("propose_consolidation",),
        }
        proposal_options = list(
            dict.fromkeys(
                proposal
                for observation in observations
                for proposal in proposal_by_observation.get(
                    str(observation.get("observation_type") or ""),
                    (),
                )
            )
        )
        return {
            "workspace_id": self.workspace_id,
            "profile_id": self.profile_id,
            "catalog_revision": self.catalog_revision(),
            "tree_path": list(expanded["path_ids"]),
            "block": block,
            "children": list(expanded["children"]),
            "related_blocks": self.related_blocks(block_id),
            "health_hints": {
                "native_partitions": list(block.get("native_partitions") or ()),
                "block_metrics": list(block.get("block_metrics") or ()),
                "freshness": str(block.get("freshness") or ""),
            },
            "maintenance_observations": observations,
            "proposal_options": proposal_options,
            "mutation_performed": False,
        }

    def get_active_mask(self, mask_id: str = "") -> ActiveMemoryMask | None:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            if mask_id:
                return store.get_mask(mask_id)
            lineage = self._active_scope_lineage()
            return store.current_mask(
                session_id=lineage.session_id,
                task_id=lineage.task_id,
            )
        finally:
            store.close()

    def set_active_mask(
        self,
        *,
        selected_block_ids: tuple[str, ...],
        token_budget: int,
        activation_reason: str,
        context_refs: tuple[str, ...] = (),
        pinned_record_ids: tuple[str, ...] = (),
        selectors: tuple[ActiveMemorySelector, ...] = (),
        priorities: tuple[ActiveMemoryPriority, ...] = (),
        block_token_budgets: tuple[ActiveMemoryBlockBudget, ...] = (),
        expires_at: str = "",
    ) -> ActiveMemoryMask:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            selected = tuple(dict.fromkeys(str(value).strip() for value in selected_block_ids))
            if not selected:
                raise ValueError("active memory mask requires selected_block_ids")
            selected_states = {block_id: store.get_block(block_id) for block_id in selected}
            if any(state is None for state in selected_states.values()):
                raise ValueError("active memory mask references blocks outside this profile")
            if any(state.status != "active" for state in selected_states.values() if state):
                raise ValueError("active memory mask requires active selected blocks")
            lineage = self._active_scope_lineage()
            if store.has_live_mask_for_scope(
                session_id=lineage.session_id,
                task_id=lineage.task_id,
            ):
                raise ValueError("an active memory mask already exists for this exact scope")
            now = datetime.now(timezone.utc).isoformat()
            mask = ActiveMemoryMask(
                workspace_id=self.workspace_id,
                profile_id=self.profile_id,
                session_id=lineage.session_id,
                task_id=lineage.task_id,
                context_refs=context_refs,
                revision=1,
                status="active",
                selected_block_ids=selected,
                pinned_record_ids=pinned_record_ids,
                selectors=selectors,
                priorities=priorities,
                token_budget=token_budget,
                block_token_budgets=block_token_budgets,
                lifecycle="durable",
                activation_reason=activation_reason,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                catalog_revision=store.catalog_revision(),
            )
            return store.put_mask(mask)
        finally:
            store.close()

    def amend_active_mask(
        self,
        mask_id: str,
        *,
        expected_revision: int,
        activation_reason: str,
        selected_block_ids: tuple[str, ...] | None = None,
        context_refs: tuple[str, ...] | None = None,
        pinned_record_ids: tuple[str, ...] | None = None,
        selectors: tuple[ActiveMemorySelector, ...] | None = None,
        priorities: tuple[ActiveMemoryPriority, ...] | None = None,
        token_budget: int | None = None,
        block_token_budgets: tuple[ActiveMemoryBlockBudget, ...] | None = None,
        expires_at: str | None = None,
    ) -> ActiveMemoryMask:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            current = store.get_mask(mask_id)
            if current is None:
                raise ValueError("active memory mask was not found")
            selected = (
                current.selected_block_ids
                if selected_block_ids is None
                else tuple(selected_block_ids)
            )
            selected_states = {block_id: store.get_block(block_id) for block_id in selected}
            if any(state is None for state in selected_states.values()):
                raise ValueError("active memory mask references blocks outside this profile")
            if any(state.status != "active" for state in selected_states.values() if state):
                raise ValueError("active memory mask requires active selected blocks")
            updated = replace(
                current,
                revision=current.revision + 1,
                selected_block_ids=selected,
                context_refs=current.context_refs if context_refs is None else context_refs,
                pinned_record_ids=(
                    current.pinned_record_ids if pinned_record_ids is None else pinned_record_ids
                ),
                selectors=current.selectors if selectors is None else selectors,
                priorities=current.priorities if priorities is None else priorities,
                token_budget=current.token_budget if token_budget is None else token_budget,
                block_token_budgets=(
                    current.block_token_budgets
                    if block_token_budgets is None
                    else block_token_budgets
                ),
                activation_reason=activation_reason,
                updated_at=datetime.now(timezone.utc).isoformat(),
                expires_at=current.expires_at if expires_at is None else expires_at,
                catalog_revision=store.catalog_revision(),
            )
            return store.put_mask(updated, expected_revision=expected_revision)
        finally:
            store.close()

    def _transition_active_mask(
        self,
        mask_id: str,
        *,
        expected_revision: int,
        status: ActiveMemoryMaskStatus,
        reason: str,
    ) -> ActiveMemoryMask:
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            current = store.get_mask(mask_id)
            if current is None:
                raise ValueError("active memory mask was not found")
            allowed = {
                ("active", "suspended"),
                ("suspended", "active"),
                ("active", "expired"),
                ("suspended", "expired"),
            }
            if (current.status, status) not in allowed:
                raise ValueError("active memory mask lifecycle transition is invalid")
            if status == "active":
                if current.is_expired():
                    raise ValueError("an elapsed active memory mask cannot be resumed")
                if store.has_live_mask_for_scope(
                    session_id=current.session_id,
                    task_id=current.task_id,
                    exclude_mask_id=current.mask_id,
                ):
                    raise ValueError(
                        "another active memory mask already exists for this exact scope"
                    )
            updated = replace(
                current,
                revision=current.revision + 1,
                status=status,
                activation_reason=str(reason or "").strip(),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            return store.put_mask(updated, expected_revision=expected_revision)
        finally:
            store.close()

    def suspend_active_mask(
        self,
        mask_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> ActiveMemoryMask:
        return self._transition_active_mask(
            mask_id,
            expected_revision=expected_revision,
            status="suspended",
            reason=reason,
        )

    def resume_active_mask(
        self,
        mask_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> ActiveMemoryMask:
        return self._transition_active_mask(
            mask_id,
            expected_revision=expected_revision,
            status="active",
            reason=reason,
        )

    def deactivate_active_mask(
        self,
        mask_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> ActiveMemoryMask:
        return self._transition_active_mask(
            mask_id,
            expected_revision=expected_revision,
            status="expired",
            reason=reason,
        )

    def _active_lease(
        self,
        mask: ActiveMemoryMask | None,
        *,
        records: tuple[MemoryBlockRecord, ...] = (),
        resolution_status: str,
        scope_validation_result: str,
        refresh_reason: str = "",
        max_tokens: int | None = None,
        semantic_recall_performed: bool = False,
    ) -> ActivationLease:
        logical_block_ids = mask.selected_block_ids if mask is not None else ()
        record_ids = tuple(record.block_id for record in records)
        token_budget = mask.token_budget if mask is not None else 0
        if max_tokens is not None:
            token_budget = min(token_budget, max_tokens)
        plan = RecallPlan(
            query="",
            query_type="exact_lookup",
            workspace_id=self.workspace_id,
            run_id=self._active_scope_lineage().session_id,
            logical_block_ids=logical_block_ids,
            record_ids=record_ids,
            include_candidates=True,
            max_tokens=token_budget,
            limit=len(record_ids),
            reason="active_mask_materialization",
        )
        projected: list[dict[str, Any]] = []
        for record in records:
            hit = block_record_to_memory_hit(record)
            metadata = dict(hit.metadata)
            membership = record.block_membership
            metadata["logical_block_id"] = (
                membership.primary_block_id if membership is not None else ""
            )
            projected.append(
                {
                    "text": hit.text,
                    "source": hit.source,
                    "score": hit.score,
                    "kind": hit.kind,
                    "tags": list(hit.tags),
                    "metadata": metadata,
                    "authority": "advisory",
                }
            )
        payload = pack_evidence_payload(
            projected,
            max_tokens=token_budget,
            query="",
            requirements=(),
        )
        lease = activation_lease_from_payload(
            plan=plan,
            payload=payload,
            backend_id=self._backend_id(),
            mode="active_mask",
        )
        health = getattr(self.backend, "health", None)
        health_payload = health() if callable(health) else {}
        trace = replace(
            lease.trace,
            event_type="active_resolve",
            mask_id=mask.mask_id if mask is not None else "",
            mask_revision=mask.revision if mask is not None else 0,
            catalog_revision=self.catalog_revision(),
            backend_revision=str(
                health_payload.get("canonical_revision") or health_payload.get("revision") or ""
            ),
            refresh_reason=refresh_reason,
            excluded_lifecycle_statuses=(
                "superseded",
                "deprecated",
                "quarantined",
                "retracted",
                "expired",
            ),
            scope_validation_result=scope_validation_result,
            resolution_status=resolution_status,
            semantic_recall_performed=semantic_recall_performed,
        )
        return replace(lease, trace=trace)

    def resolve_active_mask(
        self,
        *,
        mask_id: str = "",
        expected_revision: int | None = None,
        max_tokens: int | None = None,
    ) -> ActivationLease:
        if expected_revision is not None and (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 1
        ):
            raise ValueError("active mask expected revision must be positive")
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0
        ):
            raise ValueError("active mask token ceiling must be non-negative")
        mask = self.get_active_mask(mask_id)
        if mask is None:
            return self._active_lease(
                None,
                resolution_status="not_found",
                scope_validation_result="matched",
            )
        lineage = self._active_scope_lineage()
        if (
            mask.workspace_id != self.workspace_id
            or mask.profile_id != self.profile_id
            or mask.session_id != lineage.session_id
            or mask.task_id != lineage.task_id
        ):
            return self._active_lease(
                mask,
                resolution_status="scope_mismatch",
                scope_validation_result="scope_mismatch",
            )
        if mask.status != "active":
            return self._active_lease(
                mask,
                resolution_status=mask.status,
                scope_validation_result="matched",
            )
        if expected_revision is not None and mask.revision != expected_revision:
            return self._active_lease(
                mask,
                resolution_status="revision_mismatch",
                scope_validation_result="matched",
                max_tokens=max_tokens,
            )
        if mask.is_expired():
            return self._active_lease(
                mask,
                resolution_status="expired",
                scope_validation_result="matched",
            )
        if max_tokens == 0:
            return self._active_lease(
                mask,
                resolution_status="resolved",
                scope_validation_result="matched",
                max_tokens=0,
            )
        store = WorkspaceCatalogStore(self.repo_root, self.workspace_id, self.profile_id)
        try:
            selected_states = {
                block_id: store.get_block(block_id) for block_id in mask.selected_block_ids
            }
            if any(state is None for state in selected_states.values()):
                return self._active_lease(
                    mask,
                    resolution_status="catalog_mismatch",
                    scope_validation_result="matched",
                    refresh_reason="selected_block_missing",
                )
            if any(state.status != "active" for state in selected_states.values() if state):
                return self._active_lease(
                    mask,
                    resolution_status="catalog_mismatch",
                    scope_validation_result="matched",
                    refresh_reason="selected_block_inactive",
                )
        finally:
            store.close()
        get_block = getattr(self.backend, "get_block", None)
        record_ids_for_blocks = getattr(self.backend, "record_ids_for_logical_blocks", None)
        validate_dependencies = getattr(self.backend, "validate_dependencies", None)
        if (
            not callable(get_block)
            or not callable(record_ids_for_blocks)
            or not callable(validate_dependencies)
        ):
            return self._active_lease(
                mask,
                resolution_status="backend_unavailable",
                scope_validation_result="matched",
            )
        excluded = {"superseded", "deprecated", "quarantined", "retracted", "expired"}
        record_ids: list[str] = list(mask.pinned_record_ids)
        selectors = mask.selectors or tuple(
            ActiveMemorySelector(block_id=block_id, selector_type="canonical_heads")
            for block_id in mask.selected_block_ids
        )
        for selector in selectors:
            if selector.selector_type == "pinned_records":
                record_ids.extend(selector.record_ids)
                continue
            if selector.selector_type == "backend_query":
                hits = self.recall(
                    selector.query,
                    limit=selector.limit,
                    include_candidates=True,
                    logical_block_ids=(selector.block_id,),
                )
                record_ids.extend(
                    str(hit.metadata.get("block_id") or "")
                    for hit in hits
                    if str(hit.metadata.get("block_id") or "")
                )
                continue
            candidates = list(
                record_ids_for_blocks(
                    (selector.block_id,),
                    workspace_id=self.workspace_id,
                )
            )
            if selector.selector_type == "working_state":
                candidates = [
                    record_id
                    for record_id in candidates
                    if isinstance(
                        (record := get_block(record_id, workspace_id=self.workspace_id)),
                        MemoryBlockRecord,
                    )
                    and record.profile_id == self.profile_id
                    and record.delivery_policy == "working_set"
                    and (
                        not selector.working_state_id
                        or record.metadata.get("working_state_id") == selector.working_state_id
                    )
                ]
            if selector.selector_type in {"latest_records", "working_state"}:
                typed = [
                    record
                    for record_id in candidates
                    if isinstance(
                        (record := get_block(record_id, workspace_id=self.workspace_id)),
                        MemoryBlockRecord,
                    )
                    and record.profile_id == self.profile_id
                ]
                typed.sort(
                    key=lambda record: (
                        record.updated_at or record.recorded_at or record.created_at,
                        record.block_id,
                    ),
                    reverse=True,
                )
                candidates = [record.block_id for record in typed[: selector.limit]]
            record_ids.extend(candidates)
        selected_set = set(mask.selected_block_ids)
        records: list[MemoryBlockRecord] = []
        primary_blocks: dict[str, str] = {}
        edges_for = getattr(self.backend, "edges_for", None)

        def scoped_get_block(record_id: str) -> MemoryBlockRecord | None:
            candidate = get_block(record_id, workspace_id=self.workspace_id)
            return (
                candidate
                if isinstance(candidate, MemoryBlockRecord)
                and candidate.profile_id == self.profile_id
                else None
            )

        for record_id in dict.fromkeys(record_ids):
            record = scoped_get_block(record_id)
            if record is not None:
                record = resolve_canonical_revision_head(
                    record,
                    canonical_key=record.canonical_key,
                    workspace_id=self.workspace_id,
                    get_block=scoped_get_block,
                    edges_for=edges_for if callable(edges_for) else None,
                )
            if not isinstance(record, MemoryBlockRecord) or record.status in excluded:
                continue
            dependency_check = validate_dependencies(
                record.block_id, workspace_id=self.workspace_id, profile_id=self.profile_id
            )
            if (
                not isinstance(dependency_check, DependencyValidation)
                or dependency_check.status != "valid"
            ):
                continue
            membership = record.block_membership
            if membership is None or membership.primary_block_id not in selected_set:
                continue
            records.append(record)
            primary_blocks[record.block_id] = membership.primary_block_id
        priority = {value.block_id: value.priority for value in mask.priorities}
        records.sort(
            key=lambda record: (
                priority.get(primary_blocks[record.block_id], 0),
                record.updated_at or record.recorded_at or record.created_at,
                record.block_id,
            ),
            reverse=True,
        )
        block_budgets = {value.block_id: value.token_budget for value in mask.block_token_budgets}
        block_usage: dict[str, int] = {}
        budgeted: list[MemoryBlockRecord] = []
        for record in records:
            block_id = primary_blocks[record.block_id]
            cost = estimate_tokens(record.content)
            block_budget = block_budgets.get(block_id)
            if block_budget is not None and block_usage.get(block_id, 0) + cost > block_budget:
                continue
            block_usage[block_id] = block_usage.get(block_id, 0) + cost
            budgeted.append(record)
        current_revision = self.catalog_revision()
        return self._active_lease(
            mask,
            records=tuple(budgeted),
            resolution_status="resolved",
            scope_validation_result="matched",
            refresh_reason=(
                "catalog_revision_changed"
                if current_revision != mask.catalog_revision
                else "canonical_heads_refreshed"
            ),
            max_tokens=max_tokens,
            semantic_recall_performed=any(
                selector.selector_type == "backend_query" for selector in selectors
            ),
        )

    @staticmethod
    def _catalog_child_ids(blocks: list[MemoryBlock]) -> dict[str, tuple[str, ...]]:
        """Materialize configured child links plus parent-owned workspace additions."""

        child_ids: dict[str, list[str]] = {
            block.block_id: list(block.child_ids) for block in blocks
        }
        known_children = {parent: set(children) for parent, children in child_ids.items()}
        for block in blocks:
            if block.parent_id and block.block_id not in known_children[block.parent_id]:
                child_ids[block.parent_id].append(block.block_id)
                known_children[block.parent_id].add(block.block_id)
        return {block_id: tuple(items) for block_id, items in child_ids.items()}

    @staticmethod
    def _visible_catalog_root_ids(
        blocks: list[MemoryBlock],
        child_ids_by_parent: dict[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        top_level = [block for block in blocks if not block.parent_id]
        if (
            len(top_level) == 1
            and top_level[0].index_only
            and child_ids_by_parent[top_level[0].block_id]
        ):
            return child_ids_by_parent[top_level[0].block_id]
        return tuple(block.block_id for block in top_level)

    @staticmethod
    def _catalog_paths(
        *,
        visible_root_ids: tuple[str, ...],
        child_ids_by_parent: dict[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        """Walk the snapshot once; work is proportional to links and emitted path lengths."""
        pending: list[tuple[str, tuple[str, ...]]] = [
            (root_id, (root_id,)) for root_id in reversed(visible_root_ids)
        ]
        paths: dict[str, tuple[str, ...]] = {}
        while pending:
            block_id, path = pending.pop()
            if block_id in paths:
                continue
            paths[block_id] = path
            pending.extend(
                (child_id, (*path, child_id))
                for child_id in reversed(child_ids_by_parent.get(block_id, ()))
            )
        return paths

    @staticmethod
    def _catalog_path(
        selected_id: str,
        *,
        visible_root_ids: tuple[str, ...],
        child_ids_by_parent: dict[str, tuple[str, ...]],
    ) -> tuple[str, ...]:
        pending: list[tuple[str, tuple[str, ...]]] = [
            (root_id, (root_id,)) for root_id in reversed(visible_root_ids)
        ]
        visited: set[str] = set()
        while pending:
            block_id, path = pending.pop()
            if block_id in visited:
                continue
            visited.add(block_id)
            if block_id == selected_id:
                return path
            pending.extend(
                (child_id, (*path, child_id))
                for child_id in reversed(child_ids_by_parent.get(block_id, ()))
            )
        return ()

    def create_prospective_obligation(
        self,
        obligation: ProspectiveObligation,
        *,
        event_id: str,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        """Persist one typed obligation in the selected backend and its trigger index."""
        if not isinstance(obligation, ProspectiveObligation):
            raise ValueError("obligation must use ProspectiveObligation")
        return self.backend.create_prospective_obligation(
            obligation,
            event_id=event_id,
            idempotency_key=idempotency_key,
        )

    def get_prospective_obligation(self, obligation_id: str) -> ProspectiveObligation | None:
        """Read one current obligation from the selected backend's canonical sidecar."""
        return self.backend.get_prospective_obligation(obligation_id)

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
        """Apply one evidence-backed lifecycle CAS without executing a host action."""
        if status == "fired":
            raise ValueError(
                "firing requires apply_prospective_lifecycle_event with trigger revalidation"
            )
        if any(not isinstance(ref, EvidenceRef) for ref in evidence_refs):
            raise ValueError("evidence_refs must contain typed EvidenceRef values")
        if any(not isinstance(ref, EvidenceRef) for ref in satisfaction_evidence_refs):
            raise ValueError("satisfaction_evidence_refs must contain typed EvidenceRef values")
        return self.backend.transition_prospective_obligation(
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

    def apply_prospective_lifecycle_event(
        self,
        obligation_id: str,
        event: ProspectiveLifecycleEvent,
        *,
        trigger_context: ProspectiveTriggerContext | None = None,
    ) -> dict[str, Any]:
        """Apply one typed, evidence-backed, idempotent obligation transition."""

        if not isinstance(event, ProspectiveLifecycleEvent):
            raise ValueError("event must use ProspectiveLifecycleEvent")
        if event.event_type == "fire":
            current = self.backend.get_prospective_obligation(obligation_id)
            if current is not None and current.status == "armed":
                if not isinstance(trigger_context, ProspectiveTriggerContext):
                    raise ValueError(
                        "new prospective fire requires ProspectiveTriggerContext revalidation"
                    )
                revalidation = revalidate_prospective_fire(current, trigger_context)
                if not revalidation.allowed:
                    raise ValueError(revalidation.reason)
        return self.backend.transition_prospective_obligation(
            obligation_id,
            expected_version=event.expected_version,
            expected_status=event.expected_status,
            status=event.status,
            evidence_refs=event.evidence_refs,
            satisfaction_evidence_refs=event.satisfaction_evidence_refs,
            satisfaction_verifier=event.satisfaction_verifier,
            satisfaction_verifier_result=event.satisfaction_verifier_result,
            event_id=event.event_id,
            idempotency_key=event.idempotency_key,
        )

    def prospective_trigger_candidates(
        self,
        context: ProspectiveTriggerContext,
        *,
        limit: int = 50,
    ) -> list[ProspectiveTriggerDecision]:
        """Evaluate the selected backend's bounded trigger index as advisory candidates."""
        if not isinstance(context, ProspectiveTriggerContext):
            raise ValueError("context must use ProspectiveTriggerContext")
        return self.backend.prospective_trigger_candidates(
            context,
            workspace_id=self.workspace_id,
            limit=limit,
        )

    @staticmethod
    def prospective_intervention_from_obligation(
        obligation: ProspectiveObligation,
    ) -> ProspectiveIntervention:
        return prospective_intervention(obligation)

    def prospective_intervention(self, obligation_id: str) -> ProspectiveIntervention:
        obligation = self.backend.get_prospective_obligation(obligation_id)
        if obligation is None:
            raise ValueError("prospective obligation is missing")
        return prospective_intervention(obligation)

    def prospective_telemetry_report(self, *, limit: int = 256) -> ProspectiveTelemetryReport:
        """Publish bounded obligation telemetry outside the ordinary recall channel."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 256:
            raise ValueError("prospective telemetry limit must be an integer from 1 to 256")
        nodes = self.backend.select_index_nodes(
            workspace_id=self.workspace_id,
            node_type="prospective_trigger",
            limit=limit,
        )
        obligations: list[ProspectiveObligation] = []
        for node in nodes:
            obligation_id = str(node.metadata.get("obligation_id", ""))
            if (
                node.workspace_id != self.workspace_id
                or not obligation_id
                or node.metadata.get("host_action_authorized") is not False
            ):
                raise ValueError("prospective telemetry index linkage is invalid")
            obligation = self.backend.get_prospective_obligation(obligation_id)
            if obligation is None or node.metadata.get("status") != obligation.status:
                raise ValueError("prospective telemetry obligation state is invalid")
            obligations.append(obligation)
        return prospective_telemetry_report(tuple(obligations))

    def record_memory_decision(self, event: MemoryDecisionEvent) -> dict[str, Any]:
        """Append one immutable decision snapshot to the selected backend journal."""
        if not isinstance(event, MemoryDecisionEvent):
            raise ValueError("event must use MemoryDecisionEvent")
        if event.workspace_id != self.workspace_id:
            raise ValueError("memory decision workspace mismatch")
        return self.backend.record_memory_decision(event)

    def record_memory_usage_event(self, event: MemoryUsageEvent) -> dict[str, Any]:
        """Append one typed candidate-to-action usage observation."""
        if not isinstance(event, MemoryUsageEvent):
            raise ValueError("event must use MemoryUsageEvent")
        if event.workspace_id != self.workspace_id:
            raise ValueError("memory usage workspace mismatch")
        return self.backend.record_memory_usage_event(event)

    def memory_decision_trace(self, decision_id: str) -> MemoryDecisionTrace:
        """Read a decision and its ordered usage events outside ordinary recall."""
        trace = self.backend.memory_decision_trace(decision_id)
        if not isinstance(trace, MemoryDecisionTrace):
            raise ValueError("selected backend returned an invalid memory decision trace")
        return trace

    def record_outcome_credit(self, event: OutcomeCreditEvent) -> dict[str, Any]:
        """Append verified or explicitly neutral outcome attribution."""
        if not isinstance(event, OutcomeCreditEvent):
            raise ValueError("event must use OutcomeCreditEvent")
        if event.workspace_id != self.workspace_id:
            raise ValueError("outcome credit workspace mismatch")
        return self.backend.record_outcome_credit(event)

    def outcome_credits(self, decision_id: str) -> tuple[OutcomeCreditEvent, ...]:
        """Read immutable outcome credits for one decision outside recall."""

        credits = self.backend.outcome_credits(decision_id)
        if not isinstance(credits, tuple) or any(
            not isinstance(credit, OutcomeCreditEvent) for credit in credits
        ):
            raise ValueError("selected backend returned invalid outcome credits")
        return credits

    def memory_operator_report(self, decision_id: str) -> MemoryOutcomeReport:
        """Explain outcomes and unknown observation coverage for one decision."""

        trace = self.memory_decision_trace(decision_id)
        return build_memory_outcome_report(
            report_id=f"operator-report:{trace.decision.decision_id}",
            trace=trace,
            outcome_credits=self.outcome_credits(decision_id),
        )

    def qualify_procedure_reuse(
        self,
        *,
        memory_id: str,
        source_task_signature: str,
        requested_task_signature: str,
        transfer_envelope: TransferEnvelope | None = None,
    ) -> ProcedureReuseDecision:
        """Return a non-authorizing same-task or qualified-transfer reuse decision."""

        return _qualify_procedure_reuse(
            memory_id=memory_id,
            source_task_signature=source_task_signature,
            requested_task_signature=requested_task_signature,
            transfer_envelope=transfer_envelope,
        )

    def contextual_utility_summary(
        self,
        *,
        memory_id: str,
        task_family: str,
        stage: str,
        project_id: str,
        host_id: str,
        profile_id: str,
        minimum_support: int = 3,
        minimum_independent_outcomes: int = 2,
    ) -> ContextualUtilitySummary:
        """Build exact-context utility for shadow evaluation only."""
        summary = self.backend.contextual_utility_summary(
            memory_id=memory_id,
            task_family=task_family,
            stage=stage,
            project_id=project_id,
            host_id=host_id,
            profile_id=profile_id,
            minimum_support=minimum_support,
            minimum_independent_outcomes=minimum_independent_outcomes,
        )
        if not isinstance(summary, ContextualUtilitySummary):
            raise ValueError("selected backend returned an invalid contextual utility summary")
        return summary

    @_serialize_same_creation
    def remember(
        self,
        content: str,
        *,
        title: str = "",
        kind: str = "observation",
        tags: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
        write_mode: Literal["standard", "lossless_atomic"] = "standard",
        evidence_refs: list[Any] | tuple[Any, ...] | None = None,
        scope: str = "workspace",
        status: str = "candidate",
        retention: str = "durable",
        delivery_policy: str = "on_demand",
        temperature: str = "warm",
        authority: str = "advisory",
        trust_level: str = "agent_reported",
        verification_status: str = "unverified",
        verification_methods: tuple[str, ...] = (),
        promotion_decision: str = "not_requested",
        sensitivity: Sensitivity = "internal",
        content_mode: ContentMode = "full",
        operational_risk: OperationalRisk = "low",
        child_claims: tuple[TurnChildClaim, ...] = (),
        requested_block_id: str = "",
        source_id: str = "",
        operation_id: str = "",
        evidence_kind: str = "",
        memory_id: str = "",
        run_id: str = "",
        stage_id: str = "",
        subtask_id: str = "",
        session_id: str = "",
        task_id: str = "",
        parent_session_id: str = "",
        artifact_id: str = "",
        artifact_revision: str = "",
        valid_from: str = "",
        valid_to: str = "",
        project_id: str = "",
        repository_id: str = "",
        checkout: str = "",
        worktree: str = "",
        branch: str = "",
        revision: str = "",
        dirty_state: str = "",
        schema_version: int = CURRENT_SCHEMA_VERSION,
        hard_policy_version: int = 1,
        semantic_catalog_version: str = "",
        _recorded_at: str = "",
    ) -> RememberResult:
        """Retain one typed memory block.

        Agent-authored writes begin as advisory candidates. Promotion is a
        separate typed action and never comes from flexible metadata.
        """

        if memory_id:
            raise ValueError("agent-authored remember does not accept caller memory_id")
        if write_mode not in {"standard", "lossless_atomic"}:
            raise ValueError("write_mode must be standard or lossless_atomic")
        if (
            not isinstance(operation_id, str)
            or operation_id != operation_id.strip()
            or len(operation_id) > 256
        ):
            raise ValueError("operation_id must be a bounded opaque string")
        request_parameters = dict(locals())
        operation_arguments: dict[str, object] = {
            "tool": "remember",
            "request": json_ready(
                {
                    key: value
                    for key, value in request_parameters.items()
                    if key not in {"self", "operation_id", "_recorded_at"}
                }
            ),
            "scope_lineage": json_ready(self.scope_lineage),
        }
        receipt_recorded_at = ""

        def rejected_creation(result: RememberResult) -> RememberResult:
            if operation_id:
                complete_remember_receipt(
                    self.repo_root,
                    workspace_id=self.workspace_id,
                    profile_id=self.profile_id,
                    backend=self._backend_id(),
                    caller_role="memory_client",
                    protocol_version=2,
                    mutation_correlation_id=operation_id,
                    arguments=operation_arguments,
                    response={
                        "receipt_kind": "mutation_response",
                        "mutation_tool": "remember",
                        "response": {
                            "ok": False,
                            "saved": False,
                            "memory_id": result.memory_id,
                            "backend": result.backend,
                            "raw": result.raw,
                        },
                    },
                )
            return result

        if operation_id:
            receipt_status, receipt, receipt_recorded_at = claim_remember_receipt(
                self.repo_root,
                workspace_id=self.workspace_id,
                profile_id=self.profile_id,
                backend=self._backend_id(),
                caller_role="memory_client",
                protocol_version=2,
                mutation_correlation_id=operation_id,
                arguments=operation_arguments,
            )
            if receipt_status == "committed":
                if receipt is None or not isinstance(receipt.get("response"), dict):
                    raise ValueError("creation receipt response is invalid")
                response = receipt.get("response")
                if not isinstance(response, dict):
                    raise ValueError("creation receipt response is invalid")
                if (
                    type(response.get("saved")) is not bool
                    or not isinstance(response.get("memory_id"), str)
                    or not isinstance(response.get("backend"), str)
                    or not isinstance(response.get("raw"), dict)
                ):
                    raise ValueError("creation receipt response is invalid")
                replay_raw = {**response["raw"], "replayed": True}
                if "operation_id" in replay_raw:
                    replay_raw["operation_id"] = operation_id
                if response["saved"] and "source_segments" in replay_raw:
                    stored = self.backend.get_block(
                        response["memory_id"], workspace_id=self.workspace_id
                    )
                    if stored is None:
                        raise ValueError("creation receipt canonical record is unavailable")
                    replay_raw["source_segments"] = list(
                        stored.metadata.get("source_segments") or ()
                    )
                return RememberResult(
                    saved=response["saved"],
                    memory_id=response["memory_id"],
                    backend=response["backend"],
                    raw=replay_raw,
                )
        requested_block_id = str(requested_block_id or "").strip()
        if not requested_block_id:
            raise ValueError("agent-authored remember requires requested_block_id")
        if kind == "prospective_trigger":
            return rejected_creation(
                RememberResult(
                    saved=False,
                    raw={
                        "ok": False,
                        "reason": "prospective_obligation_requires_typed_api",
                        "policy_issues": ["prospective_obligation_not_semantic_memory"],
                    },
                )
            )
        if (
            scope != "workspace"
            or status != "candidate"
            or retention != "durable"
            or delivery_policy != "on_demand"
            or temperature != "warm"
            or authority != "advisory"
            or trust_level != "agent_reported"
            or verification_status != "unverified"
            or verification_methods
            or promotion_decision != "not_requested"
        ):
            raise ValueError("agent-authored remember governance is selected by the routed writer")
        lineage = self._bind_write_scope_identity(
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
            subtask_id=subtask_id,
        )
        project_id = lineage.project_id
        session_id = lineage.session_id
        task_id = lineage.task_id
        subtask_id = lineage.subtask_id
        session = self._session
        if not isinstance(child_claims, tuple) or any(
            not isinstance(claim, TurnChildClaim) for claim in child_claims
        ):
            raise ValueError("child_claims must use canonical TurnChildClaim values")
        meta = dict(metadata or {})
        if write_mode == "lossless_atomic":
            raw_segments = meta.get("source_segments")
            if raw_segments is None:
                raw_segments = [
                    {
                        "source_id": source_id or f"record:{operation_id or uuid.uuid4().hex}",
                        "text": content,
                        "speaker": str(meta.get("speaker") or ""),
                        "occurred_at": str(meta.get("occurred_at") or _recorded_at or ""),
                    }
                ]
            if not isinstance(raw_segments, (list, tuple)) or not raw_segments:
                raise ValueError("lossless_atomic source_segments must be a non-empty sequence")
            normalized_segments: list[dict[str, Any]] = []
            covered_bytes = 0
            registered_sizes: dict[str, int] = {}
            registered_ranges: dict[str, list[tuple[int, int]]] = {}
            registered_witnesses: dict[str, MemoryBlockRecord] = {}
            bound_segment_count = 0
            for ordinal, raw_segment in enumerate(raw_segments):
                if not isinstance(raw_segment, dict):
                    raise ValueError("lossless_atomic source_segments must contain mappings")
                unknown_segment_fields = sorted(
                    set(raw_segment).difference(
                        {
                            "source_id",
                            "text",
                            "sha256",
                            "speaker",
                            "occurred_at",
                            "source_sha256",
                            "byte_start",
                            "byte_end",
                        }
                    )
                )
                if unknown_segment_fields:
                    raise ValueError(
                        "lossless_atomic source segment has unsupported fields: "
                        + ", ".join(unknown_segment_fields)
                    )
                if any(
                    not isinstance(value, str)
                    for key, value in raw_segment.items()
                    if key not in {"byte_start", "byte_end"}
                ):
                    raise ValueError("lossless_atomic source segment text fields must be strings")
                segment_text = raw_segment.get("text", "")
                if not segment_text:
                    raise ValueError("lossless_atomic source segment text is required")
                source_segment_id = str(
                    raw_segment.get("source_id")
                    or source_id
                    or f"record:{operation_id or uuid.uuid4().hex}:{ordinal}"
                ).strip()
                if not source_segment_id:
                    raise ValueError("lossless_atomic source segment source_id is required")
                expected_hash = hashlib.sha256(segment_text.encode("utf-8")).hexdigest()
                provided_hash = str(raw_segment.get("sha256") or expected_hash).strip().lower()
                if provided_hash != expected_hash:
                    raise ValueError("lossless_atomic source segment sha256 does not match text")
                normalized_segment: dict[str, Any] = {
                    "source_id": source_segment_id,
                    "text": segment_text,
                    "sha256": expected_hash,
                    "speaker": str(raw_segment.get("speaker") or ""),
                    "occurred_at": str(raw_segment.get("occurred_at") or ""),
                }
                if any(key in raw_segment for key in ("source_sha256", "byte_start", "byte_end")):
                    bound_segment_count += 1
                    if not all(
                        key in raw_segment for key in ("source_sha256", "byte_start", "byte_end")
                    ):
                        raise ValueError("source segment binding requires hash and byte offsets")
                    byte_start = raw_segment["byte_start"]
                    byte_end = raw_segment["byte_end"]
                    if (
                        type(byte_start) is not int
                        or type(byte_end) is not int
                        or byte_start < 0
                        or byte_end <= byte_start
                    ):
                        raise ValueError("source segment byte range is invalid")
                    witness = self._registered_source_record(source_segment_id)
                    if witness is None or witness.session_id != lineage.session_id:
                        raise ValueError("source segment registration is unavailable in this scope")
                    source_bytes = witness.content.encode("utf-8")
                    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
                    if raw_segment["source_sha256"] != source_sha256:
                        raise ValueError("source segment source_sha256 does not match registration")
                    if byte_end > len(source_bytes) or source_bytes[
                        byte_start:byte_end
                    ] != segment_text.encode("utf-8"):
                        raise ValueError("source segment byte range does not match registration")
                    registered_speaker = str(witness.metadata.get("source_speaker") or "")
                    registered_time = str(witness.metadata.get("source_occurred_at") or "")
                    if normalized_segment["speaker"] not in {
                        "",
                        registered_speaker,
                    } or normalized_segment["occurred_at"] not in {"", registered_time}:
                        raise ValueError("source segment provenance does not match registration")
                    normalized_segment.update(
                        source_sha256=source_sha256,
                        byte_start=byte_start,
                        byte_end=byte_end,
                        witness_id=witness.block_id,
                        speaker=registered_speaker,
                        occurred_at=registered_time,
                    )
                    registered_sizes[source_segment_id] = len(source_bytes)
                    registered_witnesses[source_segment_id] = witness
                    registered_ranges.setdefault(source_segment_id, []).append(
                        (byte_start, byte_end)
                    )
                normalized_segments.append(normalized_segment)
                covered_bytes += len(segment_text.encode("utf-8"))
            if content != "".join(segment["text"] for segment in normalized_segments):
                raise ValueError(
                    "lossless_atomic content must be the exact concatenation of source_segments"
                )
            if registered_ranges and bound_segment_count != len(normalized_segments):
                raise ValueError("source segment bindings cannot mix with unregistered segments")
            if not registered_ranges and len(
                {segment["source_id"] for segment in normalized_segments}
            ) != len(normalized_segments):
                raise ValueError(
                    "lossless_atomic source segment IDs must be unique within a record"
                )
            meta["memory_mode"] = "lossless_atomic"
            meta["summary_count"] = 0
            meta["source_segments"] = normalized_segments
            meta["source_dates"] = "; ".join(
                dict.fromkeys(
                    segment["occurred_at"]
                    for segment in normalized_segments
                    if segment["occurred_at"]
                )
            )
            meta["source_speakers"] = "; ".join(
                dict.fromkeys(
                    segment["speaker"] for segment in normalized_segments if segment["speaker"]
                )
            )
            meta["source_bytes"] = (
                sum(registered_sizes.values())
                if registered_ranges
                else len(content.encode("utf-8"))
            )
            meta["stored_bytes"] = covered_bytes
            if registered_ranges:
                union_bytes = 0
                for ranges in registered_ranges.values():
                    end = 0
                    for start, stop in sorted(ranges):
                        union_bytes += max(0, stop - max(start, end))
                        end = max(end, stop)
                meta["exact_source_coverage"] = union_bytes / sum(registered_sizes.values())
                meta["source_coverage_status"] = "registered"
            else:
                meta["exact_source_coverage"] = 0.0
                meta["source_coverage_status"] = "unregistered"
            meta["exact_segment_fidelity"] = 1.0
        if session is not None:
            if session.actor:
                meta.setdefault("actor", session.actor)
            if session.project:
                meta.setdefault("project", session.project)
        tag_tuple = tuple(tags or ())
        record_title = title or content.strip().splitlines()[0][:96]
        evidence_values = list(evidence_refs or ())
        if write_mode == "lossless_atomic":
            for source_id, witness in registered_witnesses.items():
                if any(
                    isinstance(ref, EvidenceRef) and ref.ref_id == witness.block_id
                    for ref in evidence_values
                ):
                    continue
                producer_id = next(
                    (ref.producer_id for ref in witness.evidence_refs if ref.producer_id),
                    "",
                )
                evidence_values.append(
                    EvidenceRef(
                        ref_type="artifact",
                        ref_id=witness.block_id,
                        hash="sha256:"
                        + hashlib.sha256(witness.content.encode("utf-8")).hexdigest(),
                        produced_by="watcher",
                        producer_id=producer_id,
                    )
                )
        proposal = MemoryWriteProposal(
            title=record_title,
            content=content,
            kind=kind,
            tags=tag_tuple,
            workspace_id=self.workspace_id,
            run_id=run_id,
            stage_id=stage_id,
            subtask_id=subtask_id,
            source_id=source_id,
            evidence_kind=evidence_kind,
            evidence_refs=tuple(
                dict(value) if isinstance(value, dict) else asdict(value)
                for value in evidence_values
            ),
            child_claims=child_claims,
            write_mode=write_mode,
            requested_block_id=requested_block_id,
            metadata=meta,
        )
        route = route_memory_write(
            self.repo_root,
            proposal,
            profile_id=self.profile_id,
        )
        scope_identity = {
            "workspace": self.workspace_id,
            "session": lineage.session_id,
            "task": lineage.task_id,
            "subtask": lineage.subtask_id,
            "project": lineage.project_id,
            "user_global": lineage.user_id,
        }.get(route.scope, "")
        if route.accepted and not scope_identity:
            catalog = load_memory_blocks(
                self.repo_root,
                self.workspace_id,
                profile_id=self.profile_id,
            )
            parent_ids = {block.parent_id for block in catalog if block.parent_id}
            workspace_leaf_ids = sorted(
                block.block_id
                for block in catalog
                if block.scope == "workspace"
                and block.agent_writable
                and not block.index_only
                and block.authority == "advisory"
                and not block.child_ids
                and block.block_id not in parent_ids
            )
            suggested_block_id = (
                "decisions_and_contracts"
                if "decisions_and_contracts" in workspace_leaf_ids
                else (workspace_leaf_ids[0] if workspace_leaf_ids else "")
            )
            return rejected_creation(
                RememberResult(
                    saved=False,
                    raw={
                        "ok": False,
                        "reason": "requested_block_scope_not_bound",
                        "route_decision": asdict(route),
                        "policy_issues": [f"missing_scope_identity:{route.scope}"],
                        **(
                            {
                                "suggested_route": {
                                    "action": "retry_remember",
                                    "requested_block_id": suggested_block_id,
                                }
                            }
                            if suggested_block_id
                            else {}
                        ),
                    },
                )
            )
        route_payload = asdict(route)
        if not route.accepted:
            suggested_route = (
                {
                    "action": route.suggested_action,
                    "requested_block_id": route.suggested_block_id,
                }
                if route.suggested_block_id and route.suggested_action
                else None
            )
            return rejected_creation(
                RememberResult(
                    saved=False,
                    raw={
                        "ok": False,
                        "reason": route.reason,
                        "route_decision": route_payload,
                        "policy_issues": list(route.policy_issues),
                        **(
                            {"suggested_route": suggested_route}
                            if suggested_route is not None
                            else {}
                        ),
                    },
                )
            )
        routed_metadata = {**meta, "route_decision_reason": route.reason, "write_mode": write_mode}
        if operation_id:
            routed_metadata["operation_id"] = operation_id
        if source_id:
            routed_metadata["source_id"] = source_id
        if evidence_kind:
            routed_metadata["evidence_kind"] = evidence_kind
        record = MemoryBlockRecord(
            block_id="",
            title=record_title,
            content=content,
            block_membership=RecordBlockMembership(
                primary_block_id=route.block_id,
                catalog_revision_id=route.catalog_revision_id,
                backend_partition_id=self._backend_partition_id_for_block(route.block_id),
            ),
            kind=kind,
            status=route.status,
            retention=route.retention,
            delivery_policy=route.delivery_policy,
            temperature=route.temperature,
            authority=route.authority,
            trust_level=route.trust_level,
            scope=route.scope,
            verification_status=route.verification_status,
            verification_methods=route.verification_methods,
            promotion_decision=route.promotion_decision,
            user_id=lineage.user_id,
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            run_id=run_id,
            stage_id=stage_id,
            subtask_id=subtask_id,
            source_backend="lethyx",
            tags=route.tags,
            evidence_refs=tuple(evidence_values),
            sensitivity=sensitivity,
            content_mode=content_mode,
            operational_risk=operational_risk,
            child_claims=child_claims,
            metadata=routed_metadata,
            session_id=session_id,
            task_id=task_id,
            parent_session_id=parent_session_id,
            artifact_id=artifact_id,
            artifact_revision=artifact_revision,
            valid_from=valid_from,
            valid_to=valid_to,
            project_id=project_id,
            repository_id=repository_id,
            checkout=checkout,
            worktree=worktree,
            branch=branch,
            revision=revision,
            dirty_state=dirty_state,
            schema_version=schema_version,
            hard_policy_version=hard_policy_version,
            semantic_catalog_version=route.catalog_revision_id,
            recorded_at=_recorded_at,
        )
        backend = self.backend
        if operation_id:
            record = replace(
                record,
                block_id="operation_" + hashlib.sha256(operation_id.encode()).hexdigest(),
                recorded_at=receipt_recorded_at.replace("+00:00", "Z"),
            )
        initial = plan_memory_mutation(record, existing=None)
        canonical_key = initial.record.canonical_key
        existing: MemoryBlockRecord | None = None
        current_identity: CanonicalMutationIdentity | None = None
        if hasattr(backend, "get_block"):
            existing = backend.get_block(block_id_for_canonical_key(canonical_key))
            if isinstance(existing, MemoryBlockRecord):
                creation_identity = _current_mutation_identity(
                    backend, existing, workspace_id=self.workspace_id
                )
                if operation_id and creation_identity is not None:
                    if (
                        existing.revision_id != initial.record.revision_id
                        or creation_identity.action != "insert"
                    ):
                        raise ValueError("creation operation receipt conflict")
                    raw: dict[str, Any] = {
                        "canonical_key": existing.canonical_key,
                        "revision_id": existing.revision_id,
                        "recorded_at": existing.recorded_at,
                        "scope": existing.scope,
                        "scope_id": memory_scope_key(existing).split(":", 1)[1],
                        "mutation_action": "insert",
                    }
                    self._link_raw_witness_evidence(
                        existing.block_id,
                        existing.evidence_refs,
                        reason="agent_authored_from_raw_witness",
                    )
                    response = {
                        "ok": True,
                        "saved": True,
                        "memory_id": existing.block_id,
                        "backend": self._backend_id(),
                        "raw": raw,
                    }
                    complete_remember_receipt(
                        self.repo_root,
                        workspace_id=self.workspace_id,
                        profile_id=self.profile_id,
                        backend=self._backend_id(),
                        caller_role="memory_client",
                        protocol_version=2,
                        mutation_correlation_id=operation_id,
                        arguments=operation_arguments,
                        response={
                            "receipt_kind": "mutation_response",
                            "mutation_tool": "remember",
                            "response": response,
                        },
                    )
                    return RememberResult(
                        saved=True,
                        memory_id=existing.block_id,
                        backend=self._backend_id(),
                        raw={**raw, "replayed": True},
                    )
                raw_edges_for = getattr(backend, "edges_for", None)
                edges_for = raw_edges_for if callable(raw_edges_for) else None
                existing = resolve_canonical_revision_head(
                    existing,
                    canonical_key=canonical_key,
                    workspace_id=self.workspace_id,
                    get_block=backend.get_block,
                    edges_for=edges_for,
                )
                current_identity = _current_mutation_identity(
                    backend,
                    existing,
                    workspace_id=self.workspace_id,
                )
        decision = plan_memory_mutation(
            initial.record,
            existing=existing,
            current_identity=current_identity,
        )
        record = decision.record
        if operation_id and existing is not None and decision.action != "retry":
            raise ValueError("creation operation receipt conflict")
        mutation_action = decision.action
        superseded_block_id = decision.superseded_block_id
        write_result = _write_memory_event(
            self.repo_root,
            block_record_to_event(record),
            workspace_id=self.workspace_id,
            backend_instance=backend,
            mutation=decision,
        )
        raw = dict(write_result.raw)
        if write_result.policy_issues:
            raw.setdefault("policy_issues", list(write_result.policy_issues))
        if write_result.error:
            raw.setdefault("error", write_result.error)
        raw.setdefault("canonical_key", record.canonical_key)
        raw.setdefault("revision_id", record.revision_id)
        raw.setdefault("recorded_at", record.recorded_at)
        raw["scope"] = record.scope
        raw["scope_id"] = memory_scope_key(record).split(":", 1)[1]
        raw["project_id"] = record.project_id
        raw["user_id"] = record.user_id
        raw.setdefault("route_decision", route_payload)
        raw.setdefault("mutation_action", mutation_action)
        if superseded_block_id:
            raw.setdefault("superseded_block_id", superseded_block_id)
        saved = write_result.saved
        saved_id = str(write_result.canonical_id or raw.get("block_id") or record.block_id)
        if saved:
            raw["provenance_edges"] = self._link_raw_witness_evidence(
                saved_id,
                record.evidence_refs,
                reason="agent_authored_from_raw_witness",
            )
        result = RememberResult(
            saved=saved,
            memory_id=saved_id,
            backend=str(
                write_result.backend
                or raw.get("backend")
                or raw.get("source_backend")
                or self.backend_mode
                or ""
            ),
            raw=raw,
        )
        if operation_id:
            complete_remember_receipt(
                self.repo_root,
                workspace_id=self.workspace_id,
                profile_id=self.profile_id,
                backend=self._backend_id(),
                caller_role="memory_client",
                protocol_version=2,
                mutation_correlation_id=operation_id,
                arguments=operation_arguments,
                response={
                    "receipt_kind": "mutation_response",
                    "mutation_tool": "remember",
                    "response": {
                        "ok": result.saved,
                        "saved": result.saved,
                        "memory_id": result.memory_id,
                        "backend": result.backend,
                        "raw": {
                            key: value
                            for key, value in raw.items()
                            if key not in {"operation_id", "source_segments"}
                        }
                        | {"operation_id": "", "source_segments": []},
                    },
                },
            )
        return result

    def _registered_source_record(self, source_id: str) -> MemoryBlockRecord | None:
        canonical_key = derive_canonical_key(
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            kind="raw_host_witness",
            record_id=f"host:{source_id}:raw_host_witness",
        )
        record = self.backend.get_block(block_id_for_canonical_key(canonical_key))
        if (
            isinstance(record, MemoryBlockRecord)
            and record.workspace_id == self.workspace_id
            and record.profile_id == self.profile_id
            and record.metadata.get("record_kind") == "raw_host_witness"
            and record.metadata.get("host_event_id") == source_id
            and record.metadata.get("host_event_kind") == "conversation_turn"
        ):
            return record
        return None

    def register_source(
        self,
        *,
        source_id: str,
        text: str,
        producer_id: str,
        speaker: str = "",
        occurred_at: str = "",
    ) -> dict[str, Any]:
        """Register exact host source bytes as a hidden witness, without semantic placement."""

        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id != source_id.strip()
            or len(source_id) > 256
            or "\x00" in source_id
        ):
            raise ValueError("registered source requires a bounded source_id")
        if not isinstance(text, str) or not text:
            raise ValueError("registered source requires nonempty text")
        if (
            not isinstance(producer_id, str)
            or not producer_id
            or producer_id != producer_id.strip()
        ):
            raise ValueError("registered source requires producer_id")
        if not isinstance(speaker, str) or not isinstance(occurred_at, str):
            raise TypeError("registered source provenance must be text")
        try:
            source_bytes = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("registered source must be valid UTF-8") from exc
        lineage = self._active_scope_lineage()
        if not lineage.session_id:
            raise ValueError("registered source requires a bound session_id")
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        existing = self._registered_source_record(source_id)
        if existing is None:
            canonical_key = derive_canonical_key(
                workspace_id=self.workspace_id,
                profile_id=self.profile_id,
                kind="raw_host_witness",
                record_id=f"host:{source_id}:raw_host_witness",
            )
            if self.backend.get_block(block_id_for_canonical_key(canonical_key)) is not None:
                raise ValueError("registered_source_conflict")
        if existing is not None:
            if (
                not isinstance(existing, MemoryBlockRecord)
                or existing.workspace_id != self.workspace_id
                or existing.profile_id != self.profile_id
                or existing.session_id != lineage.session_id
                or existing.task_id != lineage.task_id
                or existing.content != text
                or existing.metadata.get("host_event_id") != source_id
                or existing.metadata.get("host_event_kind") != "conversation_turn"
                or existing.metadata.get("source_speaker", "") != speaker
                or existing.metadata.get("source_occurred_at", "") != occurred_at
                or not any(ref.producer_id == producer_id for ref in existing.evidence_refs)
            ):
                raise ValueError("registered_source_conflict")
            witness_id = existing.block_id
        else:
            observed = self.observe(
                HostObservation(
                    event_id=source_id,
                    event_kind="conversation_turn",
                    raw_evidence=text,
                    observed_at=occurred_at or datetime.now(timezone.utc).isoformat(),
                    producer_id=producer_id,
                    speaker=speaker,
                    source_occurred_at=occurred_at,
                )
            )
            if observed.get("ok") is not True:
                raise RuntimeError("registered source witness write failed")
            witness_id = str(observed["witness_id"])
        return {
            "source_id": source_id,
            "source_sha256": source_hash,
            "source_bytes": len(source_bytes),
            "witness_id": witness_id,
            "workspace_id": self.workspace_id,
            "profile_id": self.profile_id,
            "session_id": lineage.session_id,
            "task_id": lineage.task_id,
        }

    def source_coverage(self, expected_source_ids: list[str] | tuple[str, ...]) -> dict[str, Any]:
        """Measure exact semantic byte coverage for a host-supplied source ID set."""

        if (
            not isinstance(expected_source_ids, (list, tuple))
            or not expected_source_ids
            or any(not isinstance(value, str) or not value for value in expected_source_ids)
            or len(set(expected_source_ids)) != len(expected_source_ids)
        ):
            raise ValueError("source coverage requires distinct expected source IDs")
        registered_bytes = 0
        covered_bytes = 0
        fully_covered = 0
        unknown_ids: list[str] = []
        covered_records: set[str] = set()
        per_source: dict[str, dict[str, Any]] = {}
        for source_id in expected_source_ids:
            witness = self._registered_source_record(source_id)
            if witness is None:
                unknown_ids.append(source_id)
                per_source[source_id] = {"registered": False, "covered_bytes": 0}
                continue
            source_bytes = witness.content.encode("utf-8")
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            ranges: list[tuple[int, int]] = []
            for candidate_id in self.backend.exact_reference_block_ids(
                source_id, workspace_id=self.workspace_id
            ):
                record = self.backend.get_block(candidate_id, workspace_id=self.workspace_id)
                if (
                    not isinstance(record, MemoryBlockRecord)
                    or record.profile_id != self.profile_id
                    or record.status not in {"candidate", "accepted"}
                    or record.metadata.get("memory_mode") != "lossless_atomic"
                ):
                    continue
                for segment in record.metadata.get("source_segments", ()):
                    if (
                        isinstance(segment, dict)
                        and segment.get("source_id") == source_id
                        and segment.get("source_sha256") == source_hash
                        and segment.get("witness_id") == witness.block_id
                        and type(segment.get("byte_start")) is int
                        and type(segment.get("byte_end")) is int
                        and 0 <= segment["byte_start"] < segment["byte_end"] <= len(source_bytes)
                        and source_bytes[segment["byte_start"] : segment["byte_end"]]
                        == str(segment.get("text") or "").encode("utf-8")
                    ):
                        ranges.append((segment["byte_start"], segment["byte_end"]))
                        covered_records.add(record.block_id)
            source_covered = 0
            end = 0
            for start, stop in sorted(ranges):
                source_covered += max(0, stop - max(start, end))
                end = max(end, stop)
            registered_bytes += len(source_bytes)
            covered_bytes += source_covered
            fully_covered += int(source_covered == len(source_bytes))
            per_source[source_id] = {
                "registered": True,
                "source_bytes": len(source_bytes),
                "covered_bytes": source_covered,
            }
        return {
            "expected_source_ids": list(expected_source_ids),
            "unknown_source_ids": unknown_ids,
            "source_bytes": registered_bytes if not unknown_ids else None,
            "covered_bytes": covered_bytes,
            "source_byte_fraction": (
                covered_bytes / registered_bytes if not unknown_ids and registered_bytes else None
            ),
            "source_id_fraction": fully_covered / len(expected_source_ids),
            "record_ids": sorted(covered_records),
            "sources": per_source,
        }

    def observe(self, observation: HostObservation) -> dict[str, Any]:
        """Retain one host-authenticated fact without choosing its semantics.

        The raw witness is hidden evidence. Every semantic placement remains an
        explicit agent-authored write into the model-visible evolving topology.
        """

        if not isinstance(observation, HostObservation):
            raise TypeError("observe requires a HostObservation")
        lineage = self._active_scope_lineage()
        if not lineage.session_id:
            raise ValueError("host observation requires a bound session_id")
        blocks = load_memory_blocks(
            self.repo_root,
            self.workspace_id,
            profile_id=self.profile_id,
            strict=True,
        )
        catalog = compile_semantic_catalog(blocks)
        by_id = {block.block_id: block for block in blocks}

        def write_record(record: MemoryBlockRecord) -> RememberResult:
            initial = plan_memory_mutation(record, existing=None)
            existing: MemoryBlockRecord | None = None
            current_identity: CanonicalMutationIdentity | None = None
            getter = getattr(self.backend, "get_block", None)
            if callable(getter):
                candidate = getter(block_id_for_canonical_key(initial.record.canonical_key))
                if isinstance(candidate, MemoryBlockRecord):
                    raw_edges_for = getattr(self.backend, "edges_for", None)
                    candidate = resolve_canonical_revision_head(
                        candidate,
                        canonical_key=initial.record.canonical_key,
                        workspace_id=self.workspace_id,
                        get_block=getter,
                        edges_for=raw_edges_for if callable(raw_edges_for) else None,
                    )
                    existing = candidate
                    current_identity = _current_mutation_identity(
                        self.backend,
                        existing,
                        workspace_id=self.workspace_id,
                    )
            if (
                existing is not None
                and record.metadata.get("host_event_kind") == "conversation_turn"
            ):
                if (
                    existing.workspace_id != record.workspace_id
                    or existing.profile_id != record.profile_id
                    or existing.session_id != record.session_id
                    or existing.task_id != record.task_id
                    or existing.content != record.content
                    or existing.metadata.get("host_event_kind") != "conversation_turn"
                    or existing.metadata.get("source_speaker", "")
                    != record.metadata.get("source_speaker", "")
                    or existing.metadata.get("source_occurred_at", "")
                    != record.metadata.get("source_occurred_at", "")
                    or existing.evidence_refs != record.evidence_refs
                ):
                    raise ValueError("registered_source_conflict")
                return RememberResult(
                    saved=True,
                    memory_id=existing.block_id,
                    backend=self._backend_id(),
                    raw={"replayed": True},
                )
            mutation = plan_memory_mutation(
                initial.record,
                existing=existing,
                current_identity=current_identity,
            )
            result = _write_memory_event(
                self.repo_root,
                block_record_to_event(mutation.record),
                workspace_id=self.workspace_id,
                backend_instance=self.backend,
                mutation=mutation,
            )
            return RememberResult(
                saved=result.saved,
                memory_id=str(result.canonical_id or mutation.record.block_id),
                backend=str(result.backend or self.backend_mode or ""),
                raw={
                    **result.raw,
                    "mutation_action": mutation.action,
                    "canonical_key": mutation.record.canonical_key,
                    "revision_id": mutation.record.revision_id,
                },
            )

        witness_block = by_id.get("transient_logs")
        if witness_block is None:
            raise RuntimeError("memory catalog does not define transient_logs")
        raw_hash = "sha256:" + hashlib.sha256(observation.raw_evidence.encode("utf-8")).hexdigest()
        event_ref = EvidenceRef(
            ref_type="session_event",
            ref_id=observation.event_id,
            hash=raw_hash,
            produced_by="watcher",
            producer_id=observation.producer_id,
        )
        witness = MemoryBlockRecord(
            block_id="",
            title=f"Raw host witness {observation.event_id}"[:96],
            content=observation.raw_evidence,
            block_membership=RecordBlockMembership(
                primary_block_id=witness_block.block_id,
                catalog_revision_id=catalog.revision_id,
                backend_partition_id=self._backend_partition_id_for_block(witness_block.block_id),
            ),
            kind="raw_host_witness",
            status="candidate",
            retention=witness_block.retention,
            delivery_policy=witness_block.delivery_policy,
            temperature=witness_block.temperature,
            authority="advisory",
            trust_level="source_observed",
            scope=witness_block.scope,
            verification_status="observed",
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            run_id=lineage.session_id,
            producer="host_observer",
            source_backend="lethyx",
            tags=tuple(dict.fromkeys((*witness_block.tags, "raw_witness", observation.event_kind))),
            evidence_refs=(event_ref,),
            metadata={
                "record_kind": "raw_host_witness",
                "host_event_id": observation.event_id,
                "host_event_kind": observation.event_kind,
                "observed_at": observation.observed_at,
                "raw_sha256": raw_hash,
                "tool_name": observation.tool_name,
                **({"source_speaker": observation.speaker} if observation.speaker else {}),
                **(
                    {"source_occurred_at": observation.source_occurred_at}
                    if observation.source_occurred_at
                    else {}
                ),
            },
            session_id=lineage.session_id,
            task_id=lineage.task_id,
            subtask_id=lineage.subtask_id,
            project_id=lineage.project_id,
        )
        witness_result = write_record(witness)
        if not witness_result.saved:
            return {
                "ok": False,
                "witness_id": witness_result.memory_id,
                "memory_id": "",
                "requested_block_id": "",
                "actual_block_id": "",
                "route_outcome": "witness_rejected",
                "reason": witness_result.raw.get("error")
                or witness_result.raw.get("policy_issues")
                or "witness_write_failed",
            }

        witness_ref = EvidenceRef(
            ref_type="artifact",
            ref_id=witness_result.memory_id,
            hash=raw_hash,
            produced_by="watcher",
            producer_id=observation.producer_id,
        )
        verification_report: VerificationReportRef | None = None
        if observation.event_kind in {"test_passed", "test_failed"} and observation.workspace_root:
            workspace_root = Path(observation.workspace_root).expanduser().resolve()
            if not workspace_root.is_dir():
                raise ValueError("test observation workspace_root must be an existing directory")
            verification_report = append_verification_report_ledger_event(
                repo_root=self.repo_root,
                workspace_id=self.workspace_id,
                verifier_id=observation.producer_id,
                report_hash=observation.artifact_hash or raw_hash,
                passed=observation.event_kind == "test_passed",
                stage_id=f"host_observation:{observation.event_id}",
                workspace_root=workspace_root,
                changed_files=observation.changed_files,
            )
        visible_blocks = load_workspace_topology(
            self.repo_root,
            self.workspace_id,
            profile_id=self.profile_id,
        )
        visible_parent_ids = {
            state.block.parent_id for state in visible_blocks if state.block.parent_id
        }
        visible_parent_ids.update(
            state.block_id for state in visible_blocks if state.block.child_ids
        )
        writable = [
            {
                "block_id": state.block_id,
                "use_when": list(state.block.guidance.get("use_when") or ()),
                "avoid_when": list(state.block.guidance.get("avoid_when") or ()),
            }
            for state in visible_blocks
            if state.block_id not in visible_parent_ids
            and state.status == "active"
            and state.block.agent_writable
            and not state.block.index_only
            and state.block.retention == "durable"
        ]
        outcome = (
            "raw_witness_only"
            if observation.event_kind == "conversation_turn"
            else "needs_agent_choice"
        )
        receipt = {
            "ok": True,
            "witness_id": witness_result.memory_id,
            "memory_id": "",
            "requested_block_id": "",
            "actual_block_id": "",
            "route_outcome": outcome,
            "reason": (
                "conversation_turn_has_no_automatic_semantic_projection"
                if outcome == "raw_witness_only"
                else "semantic_placement_requires_explicit_agent_choice"
            ),
            "evidence_refs": [asdict(witness_ref)],
            "placement_manifest": {
                "requires_one_writable_leaf": outcome == "needs_agent_choice",
                "witness_id": witness_result.memory_id,
                "candidate_blocks": writable if outcome == "needs_agent_choice" else [],
            },
            "status": "observed",
            "authority": "advisory",
            "verification_status": "observed",
        }
        if verification_report is not None:
            receipt["verification_report"] = asdict(verification_report)
        return receipt

    def consolidate_observations(
        self,
        source_ids: list[str] | tuple[str, ...],
        *,
        content: str,
        title: str,
        reason: str,
        verification_reports: list[Any] | tuple[Any, ...],
        semantic_verification_report: Any,
        requested_block_id: str,
    ) -> dict[str, Any]:
        """Consolidate canonical host-derived observations through the existing tribunal."""

        source_keys = tuple(sorted(dict.fromkeys(str(value).strip() for value in source_ids)))
        if len(source_keys) < 2 or any(not value for value in source_keys):
            raise ValueError("consolidation requires at least two distinct source_ids")
        compact_content = str(content or "").strip()
        compact_title = str(title or "").strip()
        compact_reason = str(reason or "").strip()
        if not compact_content or len(compact_content) > 1024:
            raise ValueError("consolidation content must contain at most 1024 characters")
        if not compact_title or len(compact_title) > 96:
            raise ValueError("consolidation title must contain at most 96 characters")
        if not compact_reason:
            raise ValueError("consolidation requires a reason")
        try:
            semantic_report = _typed_promotion_evidence((semantic_verification_report,))[0]
        except ValueError as exc:
            raise ValueError(
                "consolidation requires a verifier-owned semantic verification report"
            ) from exc
        selected_block_id = str(requested_block_id or "").strip()
        target_state = next(
            (
                state
                for state in load_workspace_topology(
                    self.repo_root,
                    self.workspace_id,
                    profile_id=self.profile_id,
                )
                if state.block_id == selected_block_id
            ),
            None,
        )
        if target_state is None or target_state.status != "active":
            raise ValueError("consolidation target must be an active block in this profile")
        target = target_state.block
        if target.index_only or target.child_ids:
            raise ValueError("consolidation target must be a writable leaf")

        canonical_key = (
            f"{self.workspace_id}:{self.profile_id}:consolidation:"
            + hashlib.sha256(
                json.dumps(
                    {
                        "source_ids": source_keys,
                        "title": compact_title,
                        "content": compact_content,
                        "reason": compact_reason,
                        "requested_block_id": selected_block_id,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24]
        )
        getter = getattr(self.backend, "get_block", None)
        edges_for = getattr(self.backend, "edges_for", None)
        if not callable(getter) or not callable(edges_for):
            raise RuntimeError(
                "selected memory backend does not support consolidation source lookup"
            )
        existing = getter(block_id_for_canonical_key(canonical_key))
        if isinstance(existing, MemoryBlockRecord):
            current = resolve_canonical_revision_head(
                existing,
                canonical_key=canonical_key,
                workspace_id=self.workspace_id,
                get_block=getter,
                edges_for=edges_for,
            )
            if (
                current.content != compact_content
                or current.title != compact_title
                or tuple(current.metadata.get("source_record_ids") or ()) != source_keys
                or current.block_membership is None
                or current.block_membership.primary_block_id != selected_block_id
            ):
                raise ValueError("consolidation idempotency key conflicts with existing record")
            return {
                "ok": True,
                "saved": True,
                "replayed": True,
                "memory_id": current.block_id,
                "requested_block_id": selected_block_id,
                "actual_block_id": selected_block_id,
                "route_outcome": "idempotent_replay",
                "status": current.status,
                "authority": current.authority,
                "verification_status": current.verification_status,
                "source_record_ids": list(source_keys),
                "raw_witness_ids": list(current.metadata.get("raw_witness_ids") or ()),
            }

        reports = _typed_promotion_evidence(verification_reports)
        resolver = EvidenceResolver(
            build_workspace_context(
                repo_root=self.repo_root,
                workspace_root=self.repo_root,
                workspace_id=self.workspace_id,
            )
        )
        issues = [
            issue
            for report in (*reports, semantic_report)
            for issue in resolver.validate_verification_report_ref(report)
        ]
        if issues:
            codes = ", ".join(sorted({issue.code for issue in issues}))
            raise ValueError(f"consolidation verification evidence rejected: {codes}")
        verification_refs = tuple(
            dict.fromkeys(report.evidence_ref() for report in (*reports, semantic_report))
        )

        sources: list[MemoryBlockRecord] = []
        for source_id in source_keys:
            source = getter(source_id)
            if not isinstance(source, MemoryBlockRecord):
                raise ValueError(f"consolidation source {source_id!r} was not found")
            if source.workspace_id != self.workspace_id:
                raise ValueError("consolidation source is outside the client workspace")
            if source.profile_id != self.profile_id:
                raise ValueError("consolidation source is outside the client profile")
            current = resolve_canonical_revision_head(
                source,
                canonical_key=source.canonical_key,
                workspace_id=self.workspace_id,
                get_block=getter,
                edges_for=edges_for,
            )
            if current.block_id != source.block_id:
                raise ValueError("consolidation source is not the current canonical head")
            membership = current.block_membership
            if (
                membership is None
                or current.retention != "durable"
                or current.delivery_policy == "hidden"
                or current.scope != "workspace"
            ):
                raise ValueError("consolidation accepts only durable workspace semantic sources")
            raw_witness_id = str(current.metadata.get("raw_witness_id") or "").strip()
            if not raw_witness_id or not any(
                edge.edge_type == EdgeType.DERIVED_FROM
                and edge.src_id == current.block_id
                and edge.dst_id == raw_witness_id
                for edge in edges_for(current.block_id)
            ):
                raise ValueError(
                    "consolidation source requires typed provenance to a raw host witness"
                )
            sources.append(current)
        source_records = tuple(sorted(sources, key=lambda record: record.block_id))
        canonical_source_ids = tuple(record.block_id for record in source_records)

        semantic_row = resolver.ledger_row(semantic_report.report_id)
        semantic_result = semantic_row.get("result") if semantic_row is not None else None
        if (
            semantic_row is None
            or semantic_row.get("actor") != "verifier"
            or semantic_row.get("tool") != "consolidation_semantic_verification"
            or not isinstance(semantic_result, dict)
        ):
            raise ValueError("consolidation requires a dedicated verifier-owned semantic report")
        semantic_subject_hash = hash_value(
            {
                "source_ids": canonical_source_ids,
                "content": compact_content,
                "title": compact_title,
                "requested_block_id": selected_block_id,
            }
        )
        verifier_id = str(semantic_result.get("verifier_id") or "").strip()
        verifier_model_version = str(semantic_result.get("verifier_model_version") or "").strip()
        proposer_model_version = "deterministic-host-proposer-v1"
        reasons_value = semantic_result.get("reasons")
        if not isinstance(reasons_value, (list, tuple)):
            raise ValueError("semantic verification report reasons must be typed")
        semantic_reasons = tuple(str(value or "").strip() for value in reasons_value)
        semantic_outcomes: dict[str, bool] = {}
        for field_name in ("coverage", "preservation", "faithfulness"):
            outcome = semantic_result.get(field_name)
            if not isinstance(outcome, bool):
                raise ValueError("semantic verification report outcomes must be bool values")
            semantic_outcomes[field_name] = outcome
        expected_semantic_report_hash = hash_value(
            {
                "semantic_subject_hash": semantic_subject_hash,
                "verifier_id": verifier_id,
                "verifier_model_version": verifier_model_version,
                "proposer_model_version": proposer_model_version,
                **semantic_outcomes,
                "reasons": semantic_reasons,
            }
        )
        if (
            semantic_result.get("semantic_subject_hash") != semantic_subject_hash
            or semantic_result.get("proposer_model_version") != proposer_model_version
            or verifier_id != semantic_report.verifier_id
            or not verifier_model_version
            or verifier_model_version == proposer_model_version
            or not semantic_reasons
            or any(not value for value in semantic_reasons)
            or semantic_report.report_hash != expected_semantic_report_hash
            or semantic_result.get("passed") is not True
            or not all(semantic_outcomes.values())
        ):
            raise ValueError("semantic verification report is not bound to this accepted proposal")

        signal = ConsolidationSignal(
            workspace_id=self.workspace_id,
            source_episode_ids=canonical_source_ids,
            grouping_key="typed_host_observations",
            task_family="host_lifecycle",
            trigger="session_boundary",
            proposed_action="consolidate",
            expected_applicability="same_workspace_host_observations",
            risk="low",
            cost_budget_tokens=1024,
            rationale=compact_reason,
            evidence_refs=verification_refs,
            expected_value=1.0,
            estimated_cost_tokens=max(1, estimate_tokens(compact_content)),
        )
        schedule = ConsolidationScheduler().build_schedule((signal,), window="controlled")
        entry = schedule.entries[0]
        if entry.disposition != "scheduled":
            raise ValueError(f"consolidation was deferred: {entry.reason}")
        correction_ids = tuple(
            sorted(
                {
                    str(value).strip()
                    for record in source_records
                    for value in record.metadata.get("correction_of", ())
                    if str(value).strip()
                }
            )
        )
        contradiction_ids = tuple(
            sorted(
                {
                    str(value).strip()
                    for record in source_records
                    for value in record.metadata.get("contradiction_of", ())
                    if str(value).strip()
                }
            )
        )
        candidate = {
            "workspace_id": self.workspace_id,
            "authority": "advisory",
            "status": "candidate",
            "scope": "workspace",
            "source_record_ids": list(canonical_source_ids),
            "claims": [
                {
                    "content": compact_content,
                    "authority": "advisory",
                    "source_record_ids": list(canonical_source_ids),
                }
            ],
            "materialized_views": [],
            "continuation_applicability": signal.expected_applicability,
            "correction_ids": list(correction_ids),
            "contradiction_ids": list(contradiction_ids),
        }
        pipeline = CanonicalConsolidationPipeline()
        transition = pipeline.propose(
            entry.decision,
            source_records=source_records,
            new_evidence_refs=verification_refs,
            proposed_operations=("consolidate_host_observations",),
            touched_view_ids=(),
            post_state_candidate=candidate,
            source_coverage_after=canonical_source_ids,
            policy_version="host-lifecycle-v1",
            model_version="deterministic-host-proposer-v1",
            prompt_version="none",
        )
        semantic_verification = TransitionSemanticVerification(
            verifier_id=verifier_id,
            verifier_model_version=verifier_model_version,
            coverage=semantic_outcomes["coverage"],
            preservation=semantic_outcomes["preservation"],
            faithfulness=semantic_outcomes["faithfulness"],
            independent_from_proposer=True,
            reasons=semantic_reasons,
            evidence_refs=(semantic_report.evidence_ref(),),
        )
        tribunal = ConsolidationTransitionTribunal().assess(
            transition,
            decision=entry.decision,
            source_records=source_records,
            semantic_verification=semantic_verification,
            assessment_id=f"assessment-{transition.transition_id}",
        )
        shadow = ConsolidationShadowStore()
        snapshot = shadow.stage(workspace_id=self.workspace_id, transition=transition)
        resolution = shadow.resolve(snapshot.snapshot_id, report=tribunal)
        if resolution.disposition != "committed" or resolution.commit_receipt is None:
            return {
                "ok": False,
                "saved": False,
                "route_outcome": "tribunal_rejected",
                "schedule": asdict(schedule),
                "transition": asdict(transition),
                "tribunal": asdict(tribunal),
                "shadow_resolution": asdict(resolution),
            }

        blocks = load_memory_blocks(
            self.repo_root,
            self.workspace_id,
            profile_id=self.profile_id,
            strict=True,
        )
        catalog = compile_semantic_catalog(blocks)
        record = MemoryBlockRecord(
            block_id="",
            canonical_key=canonical_key,
            title=compact_title,
            content=compact_content,
            block_membership=RecordBlockMembership(
                primary_block_id=target.block_id,
                catalog_revision_id=catalog.revision_id,
                backend_partition_id=self._backend_partition_id_for_block(target.block_id),
            ),
            kind="observation",
            status="candidate",
            retention=target.retention,
            delivery_policy=target.delivery_policy,
            temperature=target.temperature,
            authority="advisory",
            trust_level="source_observed",
            scope="workspace",
            verification_status="observed",
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            run_id=self.scope_lineage.session_id,
            producer="supervisor_tribunal",
            source_backend="lethyx",
            tags=tuple(dict.fromkeys((*target.tags, "consolidated", "host_observed"))),
            evidence_refs=verification_refs,
            metadata={
                "record_kind": "consolidated_host_observation",
                "source_record_ids": list(canonical_source_ids),
                "raw_witness_ids": [
                    str(source.metadata["raw_witness_id"]) for source in source_records
                ],
                "transition_id": transition.transition_id,
                "tribunal_report_id": tribunal.report_id,
                "shadow_commit_receipt_id": resolution.commit_receipt.receipt_id,
            },
            session_id=self.scope_lineage.session_id,
            task_id=self.scope_lineage.task_id,
            subtask_id=self.scope_lineage.subtask_id,
            project_id=self.scope_lineage.project_id,
        )
        initial = plan_memory_mutation(record, existing=None)
        commit_receipt = ConsolidationCommitReceipt(
            receipt_id=resolution.commit_receipt.receipt_id,
            evidence_refs=resolution.commit_receipt.evidence_refs,
            committed_ids=(initial.record.block_id,),
        )
        mutation = plan_memory_mutation(
            record,
            existing=None,
            consolidation_receipt=commit_receipt,
        )
        result = _write_memory_event(
            self.repo_root,
            block_record_to_event(mutation.record),
            workspace_id=self.workspace_id,
            backend_instance=self.backend,
            mutation=mutation,
        )
        if not result.saved:
            return {
                **result.raw,
                "ok": False,
                "saved": False,
                "route_outcome": "commit_rejected",
            }
        consolidated_id = str(result.canonical_id or mutation.record.block_id)
        provenance_receipts = [
            self.add_edge(
                MemoryEdgeRecord(
                    src_id=consolidated_id,
                    dst_id=source.block_id,
                    edge_type=EdgeType.DERIVED_FROM,
                    workspace_id=self.workspace_id,
                    run_id=self.scope_lineage.session_id,
                    subtask_id=self.scope_lineage.subtask_id,
                    reason="accepted_consolidation_transition",
                    evidence_refs=verification_refs,
                    metadata={
                        "transition_id": transition.transition_id,
                        "tribunal_report_id": tribunal.report_id,
                    },
                )
            )
            for source in source_records
        ]
        return {
            **result.raw,
            "ok": True,
            "saved": True,
            "replayed": False,
            "memory_id": consolidated_id,
            "requested_block_id": target.block_id,
            "actual_block_id": target.block_id,
            "route_outcome": "consolidated",
            "status": mutation.record.status,
            "authority": mutation.record.authority,
            "verification_status": mutation.record.verification_status,
            "source_record_ids": list(canonical_source_ids),
            "raw_witness_ids": [
                str(record.metadata["raw_witness_id"]) for record in source_records
            ],
            "schedule": asdict(schedule),
            "transition": asdict(transition),
            "tribunal": asdict(tribunal),
            "shadow_resolution": asdict(resolution),
            "consolidation_receipt": asdict(commit_receipt),
            "provenance_edges": provenance_receipts,
        }

    def _recall_backend_hits(self, plan: RecallPlan) -> list[MemoryHit]:
        if plan.retrieval_mode == "hybrid" and self._backend_id() not in {"mempalace", "zvec"}:
            raise ValueError("hybrid retrieval is supported only by MemPalace and Zvec")
        payload = plan.to_backend_payload()
        lineage = self._active_scope_lineage()
        with profile_memory_operation(
            self.repo_root,
            operation="backend_recall",
            component="memory_client.backend",
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            user_id=self.user_id,
            chat_id=lineage.session_id,
            run_id=plan.run_id,
            backend=self._backend_id(),
            input_value=payload,
        ) as profile:
            if hasattr(self.backend, "recall_evidence"):
                hits = self.backend.recall_evidence(payload)
            else:
                hits = self.backend.recall(payload)
            profile.set_output(
                {
                    "hit_count": len(hits),
                    "hit_types": sorted({type(hit).__name__ for hit in hits}),
                }
            )
            profile.set_result(
                outcome="returned" if hits else "empty",
                reason="backend_recall",
            )
        normalized: list[MemoryHit] = []
        for hit in hits:
            normalized_hit: MemoryHit | None = None
            if isinstance(hit, MemoryHit):
                normalized_hit = hit
            elif isinstance(hit, MemoryBlockRecord):
                converted = block_record_to_memory_hit(hit)
                normalized_hit = MemoryHit(
                    text=converted.text,
                    source=converted.source,
                    score=converted.score,
                    kind=converted.kind,
                    tags=list(converted.tags),
                    metadata=dict(converted.metadata),
                )
            elif isinstance(hit, dict):
                metadata = dict(hit.get("metadata") or {})
                for key in (
                    "block_id",
                    "workspace_id",
                    "profile_id",
                    "run_id",
                    "scope",
                    "status",
                    "retention",
                    "delivery_policy",
                    "temperature",
                    "authority",
                    "trust_level",
                    "verification_status",
                    "verification_methods",
                    "promotion_decision",
                    "sensitivity",
                    "content_mode",
                    "duplicate_cluster_id",
                    "operational_risk",
                    "block_membership",
                    "recorded_at",
                    "valid_from",
                    "valid_to",
                    "expires_at",
                ):
                    if key in hit:
                        metadata[key] = hit[key]
                normalized_hit = MemoryHit(
                    text=str(hit.get("text") or hit.get("content") or ""),
                    source=str(hit.get("source") or hit.get("block_id") or ""),
                    score=hit.get("score") if isinstance(hit.get("score"), (int, float)) else None,
                    kind=str(hit.get("kind") or "memory"),
                    tags=list(str(tag) for tag in (hit.get("tags") or ())),
                    metadata=metadata,
                )
            if normalized_hit is None or not _hit_allowed(normalized_hit, plan):
                continue
            safe = _safe_hit(normalized_hit)
            if plan.temporal_mode == "historical" or plan.known_as_of:
                safe = MemoryHit(
                    text=safe.text,
                    source=safe.source,
                    score=safe.score,
                    kind=safe.kind,
                    tags=list(safe.tags),
                    metadata={
                        **safe.metadata,
                        "authority": "advisory",
                        "historical": True,
                        "as_of": plan.as_of,
                        "known_as_of": plan.known_as_of,
                    },
                )
            normalized.append(safe)
        deduplicated: list[MemoryHit] = []
        seen_ids: set[tuple[str, str]] = set()
        for hit in normalized:
            identity = (
                str(hit.metadata.get("workspace_id") or ""),
                str(hit.metadata.get("block_id") or hit.source or hit.text),
            )
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            deduplicated.append(hit)
            if len(deduplicated) >= plan.limit:
                break
        return deduplicated

    def _record_context_decision(
        self,
        *,
        plan: RecallPlan,
        hits: tuple[MemoryHit, ...],
        payload: PackedEvidencePayload,
        latency_ms: int,
        catalog_revision: str,
    ) -> str:
        record_decision = getattr(self.backend, "record_memory_decision", None)
        record_usage = getattr(self.backend, "record_memory_usage_event", None)
        if not callable(record_decision) or not callable(record_usage) or not hits:
            return ""
        first_metadata = dict(hits[0].metadata or {})
        candidate_ids = tuple(
            dict.fromkeys(
                str(value).strip()
                for hit in hits
                for value in hit.metadata.get("candidate_pool_ids") or ()
                if str(value).strip()
            )
        )
        selected_ids = tuple(
            dict.fromkeys(
                str(hit.metadata.get("block_id") or hit.source or "").strip()
                for hit in hits
                if str(hit.metadata.get("block_id") or hit.source or "").strip()
            )
        )
        candidate_ids = tuple(dict.fromkeys((*candidate_ids, *selected_ids)))
        rendered_ids = tuple(
            dict.fromkeys(
                str(
                    (item.get("metadata") or {}).get("block_id") or item.get("source") or ""
                ).strip()
                for item in payload.items
                if str(
                    (item.get("metadata") or {}).get("block_id") or item.get("source") or ""
                ).strip()
            )
        )
        selection_decision_id = str(first_metadata.get("decision_id") or "").strip()
        # Context decisions are events, not content-addressed cache entries. A
        # repeated recall must retain a distinct ledger event instead of
        # colliding with the first event's independently assigned timestamp.
        material = json.dumps(
            [
                uuid.uuid4().hex,
                selection_decision_id,
                plan.normalized_query,
                plan.workspace_id,
                plan.run_id,
                candidate_ids,
                selected_ids,
                payload.rendered_context,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        decision_id = "decision_" + hashlib.sha256(material.encode("utf-8")).hexdigest()

        def snapshot(value: object) -> str:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return "sha256:" + hashlib.sha256(encoded).hexdigest()

        evidence = (
            EvidenceRef(
                ref_type="ledger_event",
                ref_id=f"context-decision:{decision_id}",
                produced_by="harness",
                producer_id="lethyx:context_builder",
            ),
        )
        run_id = plan.run_id or "unbound"
        task_signature = (
            "recall:" + hashlib.sha256(plan.normalized_query.encode("utf-8")).hexdigest()[:24]
        )
        action_trace_id = f"context:{decision_id}"
        record_decision(
            MemoryDecisionEvent(
                decision_id=decision_id,
                workspace_id=self.workspace_id,
                run_id=run_id,
                task_signature=task_signature,
                action_trace_id=action_trace_id,
                action="recall",
                canonical_snapshot_sha256=snapshot(candidate_ids),
                context_snapshot_sha256=snapshot(payload.rendered_context),
                live_state_snapshot_sha256=snapshot(
                    [self.workspace_id, run_id, list(plan.scope_chain)]
                ),
                live_trigger="context_build",
                candidate_ids=candidate_ids,
                selected_candidate_ids=selected_ids,
                policy_revision="evidence-requirements-specificity-v2",
                catalog_revision=catalog_revision,
                cost_tokens=payload.estimated_tokens,
                cost_latency_ms=latency_ms,
                downstream_outcome="neutral_pending_verifier",
                evidence_refs=evidence,
            )
        )
        stages: tuple[tuple[MemoryUsageType, tuple[str, ...]], ...] = (
            ("candidate", candidate_ids),
            ("selected", selected_ids),
            ("rendered", rendered_ids),
        )
        sequence = 0
        for usage_type, memory_ids in stages:
            for memory_id in memory_ids:
                event_material = f"{decision_id}|{usage_type}|{memory_id}|{sequence}"
                event_id = "usage_" + hashlib.sha256(event_material.encode("utf-8")).hexdigest()
                record_usage(
                    MemoryUsageEvent(
                        event_id=event_id,
                        decision_id=decision_id,
                        memory_id=memory_id,
                        workspace_id=self.workspace_id,
                        task_signature=task_signature,
                        action_trace_id=action_trace_id,
                        usage_type=usage_type,
                        sequence=sequence,
                        evidence_refs=evidence,
                    )
                )
                sequence += 1
        record_rendered_tokens = getattr(
            self.backend,
            "record_rendered_tokens",
            None,
        )
        if callable(record_rendered_tokens) and rendered_ids:
            weights: dict[str, int] = {}
            for item in payload.items:
                memory_id = str(
                    (item.get("metadata") or {}).get("block_id") or item.get("source") or ""
                ).strip()
                if not memory_id:
                    continue
                weights[memory_id] = weights.get(memory_id, 0) + max(
                    1,
                    estimate_tokens(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                )
            weight_total = sum(weights.values())
            allocations = {
                memory_id: (payload.estimated_tokens * weight) // weight_total
                for memory_id, weight in weights.items()
            }
            remainder = payload.estimated_tokens - sum(allocations.values())
            for memory_id in weights:
                if remainder <= 0:
                    break
                allocations[memory_id] += 1
                remainder -= 1
            record_rendered_tokens(allocations)
        from lethyx.memory.local_service import build_local_service_identity
        from lethyx.memory.operational_slo import CostSample, OperationalSloLedger

        ledger = OperationalSloLedger(
            build_local_service_identity(
                repo_root=self.repo_root,
                workspace_id=self.workspace_id,
                profile_id=self.profile_id,
                backend=self.backend_mode,
                caller_role="operator",
            )
        )
        ledger.record_cost(
            CostSample(
                "online_recall",
                latency_ms,
                None,
                None,
                None,
                decision_id,
            )
        )
        return decision_id

    def observe_runtime_cost(
        self,
        *,
        observation_id: str,
        input_tokens: int | None,
        total_tokens: int | None,
        cached_input_tokens: int | None = None,
    ) -> dict[str, object]:
        """Retain actual provider totals without inferring latency, segment cache or savings."""
        from lethyx.memory.local_service import build_local_service_identity
        from lethyx.memory.operational_slo import (
            CostSample,
            OperationalSloLedger,
            PrefixCacheSample,
        )

        if not observation_id.strip():
            raise ValueError("runtime cost requires a stable observation identity")
        identity = build_local_service_identity(
            repo_root=self.repo_root,
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            backend=self.backend_mode,
            caller_role="operator",
        )
        ledger = OperationalSloLedger(identity)
        ledger.record_cost(CostSample("model", None, None, total_tokens, None, observation_id))
        ledger.record_prefix_cache(
            PrefixCacheSample(
                source="request",
                eligible_tokens=input_tokens,
                reused_tokens=cached_input_tokens,
                invalidated_tokens=None,
                input_tokens=input_tokens,
                baseline_input_tokens=None,
                total_compute_ms=None,
                baseline_total_compute_ms=None,
                observation_id=observation_id,
            )
        )
        return {
            "observation_id": observation_id,
            "cost": ledger.headline_cost_report(),
            "prefix_cache": ledger.prefix_cache_report(),
            "basis": "provider_reported",
            "memory_segment_cache": "unavailable",
        }

    def pin_record(
        self,
        record_id: str,
        *,
        owner_id: str = "",
        generation_id: str = "",
        scope_chain: list[str] | tuple[str, ...] = (),
    ) -> PinnedRecordRef:
        """Pin exact source bytes without exposing content or broadening recall.

        Accepted records must be eligible in the client's current scope.
        Unpublished records additionally require the trusted host's exact
        working owner and generation. A pin is identity, not proof of quality.
        """
        reference = pin_memory_record(
            self.backend.get_block(record_id),
            record_id=record_id,
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            scope_chain=bind_scope_chain(self._active_scope_lineage(), tuple(scope_chain)),
            owner_id=owner_id,
            generation_id=generation_id,
        )
        if (
            reference.status == "accepted"
            and self.validate_dependencies(record_id).status != "valid"
        ):
            raise ValueError(
                "memory record is unavailable because its dependencies are not current"
            )
        return reference

    def snapshot_working_record(
        self,
        record_id: str,
        *,
        owner_id: str,
        generation_id: str,
        expected_revision_id: str,
        expected_content_hash: str,
        max_tokens: int = 2048,
        max_bytes: int = 32768,
        scope_chain: list[str] | tuple[str, ...] = (),
    ) -> WorkingRecordSnapshot:
        """Read complete candidate content for an exact authorized review target.

        Oversized records are rejected rather than silently excerpted. This is
        read-only, excludes private metadata, and does not make candidates
        eligible for accepted recall or certify their evidence references.
        """
        return snapshot_working_record(
            self.backend.get_block(record_id),
            record_id=record_id,
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            scope_chain=bind_scope_chain(self._active_scope_lineage(), tuple(scope_chain)),
            owner_id=owner_id,
            generation_id=generation_id,
            expected_revision_id=expected_revision_id,
            expected_content_hash=expected_content_hash,
            max_tokens=max_tokens,
            max_bytes=max_bytes,
        )

    def read_working_set(
        self,
        *,
        owner_id: str,
        generation_id: str,
        record_ids: list[str] | tuple[str, ...],
        max_tokens: int = 2048,
        max_bytes: int = 32768,
        scope_chain: list[str] | tuple[str, ...] = (),
    ) -> WorkingSetRead:
        """Read exact unpublished records owned by one task generation.

        Hosts bind ``working_set_owner`` at write admission. This explicit read
        never broadens accepted recall or changes a candidate's lifecycle.
        Budgets apply to rendered context; metadata and exact refs are separate.
        """
        return read_working_records(
            self.backend.get_block,
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            scope_chain=bind_scope_chain(self._active_scope_lineage(), tuple(scope_chain)),
            owner_id=owner_id,
            generation_id=generation_id,
            record_ids=record_ids,
            max_tokens=max_tokens,
            max_bytes=max_bytes,
        )

    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        tags: list[str] | tuple[str, ...] | None = None,
        run_id: str = "",
        max_tokens: int = 2048,
        source_horizon: SourceHorizon = "durable",
        include_candidates: bool = False,
        scope_chain: list[str] | tuple[str, ...] | None = None,
        temporal_mode: str = "current",
        as_of: str = "",
        known_as_of: str = "",
        revision_id: str = "",
        logical_block_ids: list[str] | tuple[str, ...] | None = None,
        search_all: bool = False,
        retrieval_mode: RetrievalMode = "semantic",
        record_ids: list[str] | tuple[str, ...] | None = None,
        source_speakers: list[str] | tuple[str, ...] | None = None,
        recall_plan: RecallPlan | dict[str, Any] | None = None,
    ) -> list[MemoryHit]:
        """Recall memory as non-directive evidence for a host agent."""

        plan = self._coerce_recall_plan(
            query,
            limit=limit,
            tags=tags,
            run_id=run_id,
            max_tokens=max_tokens,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            scope_chain=scope_chain,
            temporal_mode=temporal_mode,
            as_of=as_of,
            known_as_of=known_as_of,
            revision_id=revision_id,
            logical_block_ids=logical_block_ids,
            search_all=search_all,
            retrieval_mode=retrieval_mode,
            record_ids=record_ids,
            recall_plan=recall_plan,
        )
        effective_plan = plan
        if not effective_plan.logical_block_ids and not effective_plan.search_all:
            raise ValueError("recall requires logical_block_ids or search_all=True")
        configured_blocks = {
            block.block_id: block
            for block in load_memory_blocks(
                self.repo_root,
                self.workspace_id,
                profile_id=self.profile_id,
            )
        }
        recall_block_ids = _expand_recall_block_ids(
            configured_blocks,
            effective_plan.logical_block_ids,
        )
        if recall_block_ids != effective_plan.logical_block_ids:
            effective_plan = replace(
                effective_plan,
                logical_block_ids=recall_block_ids,
            )
        if not effective_plan.logical_block_ids and not effective_plan.search_all:
            return []
        if source_speakers is not None and (
            not isinstance(source_speakers, (list, tuple))
            or not 1 <= len(source_speakers) <= 8
            or any(not isinstance(value, str) or not value.strip() for value in source_speakers)
        ):
            raise ValueError("source_speakers must be 1-8 nonempty names")
        if effective_plan.limit <= 0 or effective_plan.max_tokens <= 0:
            return []
        if source_speakers is None:
            return self._recall_backend_hits(effective_plan)
        selected = {speaker.strip().casefold() for speaker in source_speakers}
        candidates = self._recall_backend_hits(
            replace(effective_plan, limit=max(effective_plan.limit, 64))
        )
        matching: list[MemoryHit] = []
        for hit in candidates:
            segments = hit.metadata.get("source_segments")
            if not isinstance(segments, (list, tuple)):
                continue
            if any(
                isinstance(segment, dict)
                and str(segment.get("speaker") or "").strip().casefold() in selected
                for segment in segments
            ):
                matching.append(hit)
                if len(matching) >= effective_plan.limit:
                    break
        return matching

    def _coerce_recall_plan(
        self,
        query: str,
        *,
        limit: int,
        tags: list[str] | tuple[str, ...] | None,
        run_id: str,
        max_tokens: int,
        source_horizon: SourceHorizon,
        include_candidates: bool,
        scope_chain: list[str] | tuple[str, ...] | None,
        temporal_mode: str,
        as_of: str,
        known_as_of: str,
        revision_id: str,
        logical_block_ids: list[str] | tuple[str, ...] | None,
        search_all: bool,
        retrieval_mode: RetrievalMode = "semantic",
        record_ids: list[str] | tuple[str, ...] | None,
        recall_plan: RecallPlan | dict[str, Any] | None,
    ) -> RecallPlan:
        if isinstance(recall_plan, RecallPlan):
            return self._bind_recall_plan(recall_plan, requested_run_id=run_id)
        if isinstance(recall_plan, dict):
            return self._bind_recall_plan(
                RecallPlan.from_mapping(recall_plan),
                requested_run_id=run_id,
            )
        return self._bind_recall_plan(
            RecallPlan.compile(
                query=query,
                workspace_id=self.workspace_id,
                run_id=run_id,
                logical_block_ids=tuple(logical_block_ids or ()),
                search_all=search_all,
                retrieval_mode=retrieval_mode,
                record_ids=tuple(record_ids or ()),
                tags=tuple(tags or ()),
                source_horizon=source_horizon,
                include_candidates=include_candidates,
                scope_chain=tuple(scope_chain or ()),
                temporal_mode=temporal_mode,
                as_of=as_of,
                known_as_of=known_as_of,
                revision_id=revision_id,
                max_tokens=max_tokens,
                limit=limit,
                reason=("explicit_search_all" if search_all else "explicit_block_first_recall"),
            ),
            requested_run_id=run_id,
        )

    def _bind_recall_plan(
        self,
        plan: RecallPlan,
        *,
        requested_run_id: str,
    ) -> RecallPlan:
        plan_workspace_id = normalize_workspace_id(plan.workspace_id)
        if plan_workspace_id and plan_workspace_id != self.workspace_id:
            raise ValueError(
                "recall_plan workspace_id does not match the MemoryClient workspace_id"
            )
        if requested_run_id and plan.run_id and requested_run_id != plan.run_id:
            raise ValueError("recall_plan run_id does not match the requested run_id")
        if plan.profile_id and plan.profile_id != self.profile_id:
            raise ValueError("recall_plan profile_id does not match the MemoryClient profile_id")
        bound_run_id = requested_run_id or plan.run_id
        bound_scope_chain = bind_scope_chain(self._active_scope_lineage(), plan.scope_chain)
        if (
            plan.workspace_id == self.workspace_id
            and plan.profile_id == self.profile_id
            and plan.run_id == bound_run_id
            and plan.scope_chain == bound_scope_chain
        ):
            return plan
        return plan.with_overrides(
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            run_id=bound_run_id,
            scope_chain=bound_scope_chain,
        )

    def compile_recall_plan(
        self,
        query: str,
        *,
        limit: int = 10,
        tags: list[str] | tuple[str, ...] | None = None,
        run_id: str = "",
        max_tokens: int = 2048,
        source_horizon: SourceHorizon = "durable",
        include_candidates: bool = False,
        scope_chain: list[str] | tuple[str, ...] | None = None,
        temporal_mode: str = "current",
        as_of: str = "",
        known_as_of: str = "",
        revision_id: str = "",
        logical_block_ids: list[str] | tuple[str, ...] | None = None,
        search_all: bool = False,
        retrieval_mode: RetrievalMode = "semantic",
        record_ids: list[str] | tuple[str, ...] | None = None,
        recall_plan: RecallPlan | dict[str, Any] | None = None,
    ) -> RecallPlan:
        """Compile the backend-neutral block-first plan used for context recall."""

        return self._coerce_recall_plan(
            query,
            limit=limit,
            tags=tags,
            run_id=run_id,
            max_tokens=max_tokens,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            scope_chain=scope_chain,
            temporal_mode=temporal_mode,
            as_of=as_of,
            known_as_of=known_as_of,
            revision_id=revision_id,
            logical_block_ids=logical_block_ids,
            search_all=search_all,
            retrieval_mode=retrieval_mode,
            record_ids=record_ids,
            recall_plan=recall_plan,
        )

    def forget(
        self,
        memory_id: str,
        *,
        reason: str = "user_requested",
        evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...] = (),
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Retract a memory block without deleting its audit trail."""

        self._require_profile_record(memory_id)

        return transition_memory_status(
            self.repo_root,
            memory_id,
            "retracted",
            workspace_id=self.workspace_id,
            reason=reason,
            evidence_refs=evidence_refs,
            retention="archive",
            delivery_policy="hidden",
            backend_instance=self.backend,
            expected_revision=expected_revision,
        )

    def compensate_promotion(
        self,
        memory_id: str,
        *,
        expected_revision: str,
        operation_id: str,
        reason: str,
    ) -> MemoryRetractionReceipt:
        """Recover a pinned candidate or its direct, operation-owned publication."""

        if not operation_id.strip() or not expected_revision.strip() or not reason.strip():
            raise ValueError("compensation requires an operation, source revision, and reason")
        source = self._require_profile_record(memory_id)
        if source.revision_id != expected_revision:
            raise ValueError("compensation source revision does not match its pinned identity")
        raw_edges_for = getattr(self.backend, "edges_for", None)
        current = resolve_canonical_revision_head(
            source,
            canonical_key=source.canonical_key,
            workspace_id=self.workspace_id,
            get_block=self.backend.get_block,
            edges_for=raw_edges_for if callable(raw_edges_for) else None,
        )
        if current.profile_id != self.profile_id:
            raise ValueError("compensation current revision is outside the selected profile")
        if source.metadata.get("correction_target_id"):
            preserved = self._compensate_correction_publication(source, operation_id, reason)
            if preserved is not None:
                return preserved
        if current.status == "retracted":
            return MemoryRetractionReceipt(
                memory_id, current.block_id, current.revision_id, "already_retracted"
            )
        owns_candidate = current.block_id == memory_id and current.status == "candidate"
        owns_publication = (
            current.status == "accepted"
            and current.revision_of == memory_id
            and current.metadata.get("promotion_operation_id") == operation_id
        )
        if not (owns_candidate or owns_publication):
            return MemoryRetractionReceipt(
                memory_id, current.block_id, current.revision_id, "superseded"
            )
        return self._retract_current_revision(memory_id, current, reason)

    def _compensate_correction_publication(
        self,
        source: MemoryBlockRecord,
        operation_id: str,
        reason: str,
    ) -> MemoryRetractionReceipt | None:
        """Restore only this operation's exact target before hiding its proposal."""

        target = self._require_profile_record(str(source.metadata["correction_target_id"]))
        expected_target_revision = str(source.metadata.get("correction_target_revision_id") or "")
        if not expected_target_revision or target.revision_id != expected_target_revision:
            raise ValueError("compensation correction target does not match its pinned revision")
        raw_edges_for = getattr(self.backend, "edges_for", None)
        target_head = resolve_canonical_revision_head(
            target,
            canonical_key=target.canonical_key,
            workspace_id=self.workspace_id,
            get_block=self.backend.get_block,
            edges_for=raw_edges_for if callable(raw_edges_for) else None,
        )
        if target_head.profile_id != self.profile_id:
            raise ValueError("compensation target revision is outside the selected profile")
        if target_head.block_id == target.block_id and target.status == "accepted":
            return None
        identity = _current_mutation_identity(
            self.backend, target_head, workspace_id=self.workspace_id
        )
        restored = (
            identity is not None
            and identity.action == "rollback"
            and identity.rollback_target_id == target.block_id
            and target_head.status == "accepted"
        )
        published = (
            self._require_profile_record(identity.predecessor_id)
            if restored and identity is not None
            else target_head
        )
        published_identity = _current_mutation_identity(
            self.backend, published, workspace_id=self.workspace_id
        )
        owns_publication = (
            published.status == "accepted"
            and published.revision_of == target.block_id
            and published.metadata.get("promotion_operation_id") == operation_id
            and published.metadata.get("promotion_source_id") == source.block_id
            and published.metadata.get("promotion_source_revision_id") == source.revision_id
            and published.metadata.get("approved_correction_candidate_id") == source.block_id
            and published_identity is not None
            and published_identity.action == "correction"
            and published_identity.predecessor_id == target.block_id
        )
        if not owns_publication:
            return MemoryRetractionReceipt(
                source.block_id, target_head.block_id, target_head.revision_id, "superseded"
            )
        if restored:
            return None
        decision = plan_lifecycle_mutation(
            target_head,
            action="rollback",
            status=target.status,
            reason=reason,
            rollback_target=target,
            current_identity=identity,
        )
        result = _write_memory_event(
            self.repo_root,
            block_record_to_event(decision.record),
            workspace_id=self.workspace_id,
            backend_instance=self.backend,
            mutation=decision,
        )
        if not result.saved:
            raise RuntimeError(result.error or "correction predecessor restoration failed")
        return None

    def retract_working_record(
        self,
        memory_id: str,
        *,
        owner_id: str,
        generation_id: str,
        reason: str,
    ) -> MemoryRetractionReceipt:
        """Discard owned working revisions while preserving any accepted successor."""

        if not owner_id.strip() or not generation_id.strip() or not reason.strip():
            raise ValueError("working retraction requires an owner, generation, and reason")
        owner = {"owner_id": owner_id, "generation_id": generation_id}
        source = self._require_profile_record(memory_id)
        if source.metadata.get("working_set_owner") != owner:
            raise ValueError("working record belongs to another owner or generation")
        raw_edges_for = getattr(self.backend, "edges_for", None)
        current = resolve_canonical_revision_head(
            source,
            canonical_key=source.canonical_key,
            workspace_id=self.workspace_id,
            get_block=self.backend.get_block,
            edges_for=raw_edges_for if callable(raw_edges_for) else None,
        )
        if current.profile_id != self.profile_id:
            raise ValueError("working revision is outside the selected profile")
        if current.status == "retracted":
            return MemoryRetractionReceipt(
                memory_id, current.block_id, current.revision_id, "already_retracted"
            )
        if current.status != "candidate" or current.metadata.get("working_set_owner") != owner:
            return MemoryRetractionReceipt(
                memory_id, current.block_id, current.revision_id, "superseded"
            )
        return self._retract_current_revision(memory_id, current, reason)

    def _retract_current_revision(
        self,
        memory_id: str,
        current: MemoryBlockRecord,
        reason: str,
    ) -> MemoryRetractionReceipt:
        result = self.forget(current.block_id, expected_revision=current.revision_id, reason=reason)
        if result.get("ok") is not True or result.get("saved") is not True:
            raise RuntimeError(str(result.get("error") or "memory retraction failed"))
        retracted = self._require_profile_record(str(result.get("block_id") or ""))
        if retracted.status != "retracted" or retracted.revision_of != current.block_id:
            raise RuntimeError("memory retraction returned an invalid successor")
        return MemoryRetractionReceipt(
            memory_id, retracted.block_id, retracted.revision_id, "retracted"
        )

    def amend(
        self,
        memory_id: str,
        *,
        expected_revision: str,
        content: str | None = None,
        title: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        metadata_patch: Mapping[str, Any] | None = None,
        retention: str | None = None,
        expires_at: str | None = None,
        reason: str,
        evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...] = (),
        requested_block_id: str = "",
        operation_id: str = "",
    ) -> dict[str, Any]:
        """Append one CAS-guarded correction without changing memory governance."""

        if (
            not isinstance(operation_id, str)
            or operation_id != operation_id.strip()
            or len(operation_id) > 256
        ):
            raise ValueError("amend operation_id must be a bounded opaque string")
        request_parameters = dict(locals())
        request_hash = hash_value(
            {
                "tool": "amend",
                "request": json_ready(
                    {
                        key: value
                        for key, value in request_parameters.items()
                        if key not in {"self", "operation_id"}
                    }
                ),
                "scope_lineage": json_ready(self.scope_lineage),
            }
        )
        receipt_id = "amendment:" + hash_value([self.workspace_id, self.profile_id, operation_id])
        if operation_id:
            receipt_reader = getattr(self.backend, "amendment_receipt", None)
            if not callable(receipt_reader):
                raise RuntimeError("selected backend cannot acknowledge canonical amendments")
            receipt = receipt_reader(receipt_id)
            if receipt is not None:
                if not isinstance(receipt, AmendmentCommitReceipt) or (
                    receipt.request_hash != request_hash
                    or receipt.workspace_id != self.workspace_id
                    or receipt.profile_id != self.profile_id
                ):
                    raise ValueError("amend operation_id conflicts with its committed request")
                record = self._require_profile_record(receipt.record_id)
                if record.revision_id != receipt.revision_id:
                    raise ValueError("committed amendment revision is unavailable")
                self._link_raw_witness_evidence(
                    record.block_id, tuple(evidence_refs), reason="amend_from_raw_witness"
                )
                return {**receipt.response(self._backend_id()), "replayed": True}
        target_id = str(memory_id or "").strip()
        expected = str(expected_revision or "").strip()
        amendment_reason = str(reason or "").strip()
        if not target_id or not expected or not amendment_reason:
            raise ValueError("amend requires memory_id, expected_revision, and reason")
        if any(not isinstance(ref, EvidenceRef) for ref in evidence_refs):
            raise ValueError("amend evidence_refs must contain typed EvidenceRef values")
        root = self._require_profile_record(target_id)
        raw_edges_for = getattr(self.backend, "edges_for", None)
        current = resolve_canonical_revision_head(
            root,
            canonical_key=root.canonical_key,
            workspace_id=self.workspace_id,
            get_block=self.backend.get_block,
            edges_for=raw_edges_for if callable(raw_edges_for) else None,
        )
        if current.revision_id != expected:
            raise ValueError("amend expected_revision does not match the current revision")
        if (
            current.status == "accepted"
            or current.authority == "directive"
            or current.trust_level in APPROVED_TRUST_LEVELS
            or current.verification_status == "verified"
        ):
            raise ValueError(
                "trusted memory amend requires the existing correction proposal and tribunal path"
            )
        forbidden_metadata = {
            "authority",
            "delivery_policy",
            "promotion_decision",
            "scope",
            "status",
            "trust_level",
            "verification_methods",
            "verification_status",
            "workspace_id",
        }
        patch = {str(key): value for key, value in dict(metadata_patch or {}).items()}
        forbidden = sorted(forbidden_metadata.intersection(patch))
        if forbidden:
            raise ValueError(
                "amend metadata patch cannot change governance fields: " + ", ".join(forbidden)
            )
        if "importance" in patch:
            importance = patch["importance"]
            if (
                isinstance(importance, bool)
                or not isinstance(importance, (int, float))
                or not 0 <= float(importance) <= 1
            ):
                raise ValueError("amend importance must be between 0 and 1")
        updated_metadata = dict(current.metadata)
        for key, value in patch.items():
            if value is None:
                updated_metadata.pop(key, None)
            else:
                updated_metadata[key] = value
        updated_metadata["status_reason"] = amendment_reason
        json.dumps(updated_metadata, ensure_ascii=True, sort_keys=True)

        catalog_blocks = load_memory_blocks(
            self.repo_root,
            self.workspace_id,
            profile_id=self.profile_id,
            strict=True,
        )
        catalog = compile_semantic_catalog(catalog_blocks)
        by_id = {block.block_id: block for block in catalog_blocks}
        current_membership = current.block_membership
        selected_block_id = str(requested_block_id or "").strip() or (
            current_membership.primary_block_id if current_membership is not None else ""
        )
        selected_block = by_id.get(selected_block_id)
        parent_ids = {block.parent_id for block in catalog_blocks if block.parent_id}
        if (
            selected_block is None
            or selected_block.index_only
            or selected_block.block_id in parent_ids
            or not selected_block.agent_writable
            or selected_block.authority != "advisory"
        ):
            raise ValueError("amend requires an active agent-writable advisory leaf block")
        if selected_block.scope != current.scope:
            raise ValueError("amend cannot move a record across scope dimensions")

        selected_retention = current.retention if retention is None else str(retention).strip()
        allowed_retention = {
            "ephemeral": {"ephemeral"},
            "session": {"ephemeral", "session"},
            "durable": {"ephemeral", "session", "durable"},
            "archive": {"archive"},
        }[current.retention]
        if selected_retention not in allowed_retention:
            raise ValueError("amend cannot increase or bypass the current retention policy")
        selected_content = current.content if content is None else str(content).strip()
        if not selected_content:
            raise ValueError("amend content cannot be empty")
        selected_title = current.title if title is None else str(title).strip()
        if not selected_title:
            raise ValueError("amend title cannot be empty")
        replacement = replace(
            current,
            content=selected_content,
            title=selected_title,
            tags=current.tags if tags is None else tuple(tags),
            metadata=updated_metadata,
            evidence_refs=tuple(dict.fromkeys([*current.evidence_refs, *evidence_refs])),
            retention=cast(MemoryRetention, selected_retention),
            expires_at=current.expires_at if expires_at is None else str(expires_at).strip(),
            block_membership=RecordBlockMembership(
                primary_block_id=selected_block_id,
                catalog_revision_id=catalog.revision_id,
                backend_partition_id=self._backend_partition_id_for_block(selected_block_id),
            ),
            duplicate_cluster_id="",
        )
        decision = plan_memory_mutation(
            replacement,
            existing=current,
            action="correction",
            current_identity=_current_mutation_identity(
                self.backend,
                current,
                workspace_id=self.workspace_id,
            ),
        )
        if operation_id:
            decision = replace(
                decision,
                amendment_receipt=AmendmentCommitReceipt(
                    receipt_id=receipt_id,
                    request_hash=request_hash,
                    workspace_id=self.workspace_id,
                    profile_id=self.profile_id,
                    predecessor_id=current.block_id,
                    predecessor_revision_id=current.revision_id,
                    record_id=decision.record.block_id,
                    revision_id=decision.record.revision_id,
                ),
            )
        result = _write_memory_event(
            self.repo_root,
            block_record_to_event(decision.record),
            workspace_id=self.workspace_id,
            backend_instance=self.backend,
            mutation=decision,
        )
        new_id = str(result.canonical_id or decision.record.block_id)
        if result.saved:
            self._link_raw_witness_evidence(
                new_id,
                tuple(evidence_refs),
                reason="amend_from_raw_witness",
            )
        return {
            **result.raw,
            "ok": result.saved,
            "saved": result.saved,
            "backend": str(result.backend or self.backend_mode or ""),
            "workspace_id": self.workspace_id,
            "old_id": current.block_id,
            "new_id": new_id,
            "memory_id": new_id,
            "old_revision": current.revision_id,
            "new_revision": decision.record.revision_id,
            "requested_block_id": selected_block_id,
            "lifecycle": {
                "old_status": "superseded" if result.saved else current.status,
                "new_status": decision.record.status,
            },
            "graph_edge": asdict(decision.edge) if decision.edge is not None else {},
            "active_frame_invalidation": {
                "mode": "canonical_revision",
                "superseded_record_ids": [current.block_id] if result.saved else [],
            },
        }

    def correction(
        self,
        memory_id: str,
        content: str,
        *,
        title: str = "",
        reason: str = "correction",
        evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...] = (),
        requested_block_id: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        correction_metadata = dict(metadata or {})
        if reserved_memory_metadata_keys(correction_metadata):
            raise ValueError("correction metadata cannot override memory governance")
        owner = correction_metadata.get("working_set_owner")
        if owner is not None and (
            not isinstance(owner, dict)
            or set(owner) != {"owner_id", "generation_id"}
            or any(not isinstance(value, str) or not value.strip() for value in owner.values())
        ):
            raise ValueError("working_set_owner requires an exact owner and generation")
        root = self._require_profile_record(memory_id)
        raw_edges_for = getattr(self.backend, "edges_for", None)
        current = resolve_canonical_revision_head(
            root,
            canonical_key=root.canonical_key,
            workspace_id=self.workspace_id,
            get_block=self.backend.get_block,
            edges_for=raw_edges_for if callable(raw_edges_for) else None,
        )
        selected_block_id = str(requested_block_id or "").strip()
        current_membership = current.block_membership
        if not selected_block_id:
            raise ValueError("agent-authored correction requires requested_block_id")
        if current_membership is None or current_membership.primary_block_id != selected_block_id:
            raise ValueError("correction target is outside the selected logical block")
        if (
            current.status == "accepted"
            or current.authority == "directive"
            or current.trust_level in APPROVED_TRUST_LEVELS
        ):
            lineage = self._active_scope_lineage()
            proposal_key = (
                f"{current.canonical_key}:correction-proposal:"
                + hash_value(
                    {
                        "target_id": current.block_id,
                        "owner": owner,
                        "session_id": lineage.session_id,
                    }
                )[:16]
            )
            proposal = replace(
                current,
                block_id="",
                canonical_key=proposal_key,
                revision_id="",
                duplicate_cluster_id="",
                revision_of="",
                supersedes_ids=(),
                recorded_at="",
                run_id=lineage.session_id,
                session_id=lineage.session_id,
                task_id=lineage.task_id,
                subtask_id=lineage.subtask_id,
                stage_id=lineage.subtask_id,
                content=content,
                title=title or current.title,
                status="candidate",
                authority="advisory",
                trust_level="agent_reported",
                verification_status="unverified",
                verification_methods=(),
                promotion_decision="not_requested",
                evidence_refs=tuple(dict.fromkeys([*current.evidence_refs, *evidence_refs])),
                metadata={
                    **{
                        key: value
                        for key, value in current.metadata.items()
                        if key not in {"working_set_owner", "promotion_operation_id"}
                    },
                    **correction_metadata,
                    "status_reason": reason,
                    "correction_target_id": current.block_id,
                    "correction_target_revision_id": current.revision_id,
                    "pending_host_approval": True,
                },
            )
            proposal_root = self.backend.get_block(block_id_for_canonical_key(proposal_key))
            existing_proposal = (
                resolve_canonical_revision_head(
                    proposal_root,
                    canonical_key=proposal_key,
                    workspace_id=self.workspace_id,
                    get_block=self.backend.get_block,
                    edges_for=raw_edges_for if callable(raw_edges_for) else None,
                )
                if isinstance(proposal_root, MemoryBlockRecord)
                else None
            )
            proposal_decision = plan_memory_mutation(
                proposal,
                existing=existing_proposal,
                action="correction",
                current_identity=(
                    _current_mutation_identity(
                        self.backend,
                        existing_proposal,
                        workspace_id=self.workspace_id,
                    )
                    if existing_proposal is not None
                    else None
                ),
            )
            contradiction_edge = (
                MemoryEdgeRecord(
                    src_id=proposal_decision.record.block_id,
                    dst_id=current.block_id,
                    edge_type="contradicts",
                    workspace_id=self.workspace_id,
                    run_id=proposal_decision.record.run_id,
                    subtask_id=proposal_decision.record.subtask_id,
                    reason="pending_host_correction",
                    evidence_refs=proposal_decision.record.evidence_refs,
                    metadata={"correction_target_revision_id": current.revision_id},
                )
                if existing_proposal is None
                else None
            )
            pending_result = _write_memory_event(
                self.repo_root,
                block_record_to_event(proposal_decision.record),
                workspace_id=self.workspace_id,
                backend_instance=self.backend,
                mutation=proposal_decision,
            )
            if pending_result.saved and contradiction_edge is not None:
                add_edge = getattr(self.backend, "add_edge", None)
                if not callable(add_edge):
                    raise RuntimeError("selected memory backend does not support correction links")
                add_edge(contradiction_edge)
            if pending_result.saved:
                pending_id = str(pending_result.canonical_id or proposal_decision.record.block_id)
                self._link_raw_witness_evidence(
                    pending_id,
                    tuple(dict.fromkeys(evidence_refs)),
                    reason="correction_proposal_from_raw_witness",
                )
            return {
                **pending_result.raw,
                "ok": pending_result.saved,
                "saved": pending_result.saved,
                "block_id": pending_result.canonical_id,
                "pending_host_approval": True,
                "correction_target_id": current.block_id,
            }
        if owner is not None and current.metadata.get("working_set_owner") not in (None, owner):
            raise ValueError("correction candidate belongs to another owner or generation")
        replacement = replace(
            current,
            content=content,
            duplicate_cluster_id="",
            title=title or current.title,
            evidence_refs=tuple(dict.fromkeys([*current.evidence_refs, *evidence_refs])),
            metadata={**current.metadata, **correction_metadata, "status_reason": reason},
        )
        decision = plan_memory_mutation(
            replacement,
            existing=current,
            action="correction",
            current_identity=_current_mutation_identity(
                self.backend,
                current,
                workspace_id=self.workspace_id,
            ),
        )
        result = _write_memory_event(
            self.repo_root,
            block_record_to_event(decision.record),
            workspace_id=self.workspace_id,
            backend_instance=self.backend,
            mutation=decision,
        )
        if result.saved:
            self._link_raw_witness_evidence(
                str(result.canonical_id or decision.record.block_id),
                tuple(dict.fromkeys(evidence_refs)),
                reason="correction_from_raw_witness",
            )
        return {
            **result.raw,
            "ok": result.saved,
            "saved": result.saved,
            "block_id": result.canonical_id,
        }

    def deprecate(
        self,
        memory_id: str,
        *,
        reason: str,
        evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...] = (),
    ) -> dict[str, Any]:
        self._require_profile_record(memory_id)
        return transition_memory_status(
            self.repo_root,
            memory_id,
            "deprecated",
            workspace_id=self.workspace_id,
            reason=reason,
            evidence_refs=evidence_refs,
            retention="archive",
            delivery_policy="hidden",
            backend_instance=self.backend,
        )

    def rollback(
        self,
        memory_id: str,
        *,
        revision_id: str,
        reason: str,
        evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...] = (),
    ) -> dict[str, Any]:
        root = self._require_profile_record(memory_id)
        raw_edges_for = getattr(self.backend, "edges_for", None)
        current = resolve_canonical_revision_head(
            root,
            canonical_key=root.canonical_key,
            workspace_id=self.workspace_id,
            get_block=self.backend.get_block,
            edges_for=raw_edges_for if callable(raw_edges_for) else None,
        )
        history = getattr(self.backend, "canonical_history", None)
        if not callable(history):
            raise RuntimeError("selected memory backend does not support canonical history")
        target = next(
            (
                record
                for record in history(
                    canonical_key=current.canonical_key, workspace_id=self.workspace_id
                )
                if record.revision_id == revision_id
            ),
            None,
        )
        if not isinstance(target, MemoryBlockRecord):
            raise ValueError("rollback revision was not found in the canonical history")
        decision = plan_lifecycle_mutation(
            current,
            action="rollback",
            status=target.status,
            reason=reason,
            evidence_refs=tuple(evidence_refs),
            rollback_target=target,
            current_identity=_current_mutation_identity(
                self.backend,
                current,
                workspace_id=self.workspace_id,
            ),
        )
        result = _write_memory_event(
            self.repo_root,
            block_record_to_event(decision.record),
            workspace_id=self.workspace_id,
            backend_instance=self.backend,
            mutation=decision,
        )
        return {
            **result.raw,
            "ok": result.saved,
            "saved": result.saved,
            "block_id": result.canonical_id,
        }

    def erase(
        self,
        memory_id: str,
        *,
        request_id: str,
        reason: str,
        authorization: EvidenceRef,
        evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...] = (),
    ) -> dict[str, Any]:
        self._require_profile_record(memory_id)
        return erase_memory(
            self.repo_root,
            MemoryEraseRequest(
                request_id=request_id,
                memory_id=memory_id,
                workspace_id=self.workspace_id,
                reason=reason,
                authorization=authorization,
                evidence_refs=tuple(evidence_refs),
            ),
            workspace_id=self.workspace_id,
            backend_instance=self.backend,
            backend_id=str(self.backend_mode or ""),
        )

    def add_edge(self, edge: MemoryEdgeRecord) -> dict[str, Any]:
        return write_memory_edge(
            self.repo_root,
            edge,
            workspace_id=self.workspace_id,
            backend_instance=self.backend,
        )

    def _link_raw_witness_evidence(
        self,
        memory_id: str,
        evidence_refs: tuple[EvidenceRef, ...],
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        if not evidence_refs:
            return []
        getter = getattr(self.backend, "get_block", None)
        if not callable(getter):
            raise RuntimeError("selected memory backend does not support provenance lookup")
        receipts: list[dict[str, Any]] = []
        for evidence_ref in evidence_refs:
            witness = getter(evidence_ref.ref_id)
            if (
                not isinstance(witness, MemoryBlockRecord)
                or witness.workspace_id != self.workspace_id
                or witness.profile_id != self.profile_id
                or witness.metadata.get("record_kind") != "raw_host_witness"
            ):
                continue
            receipts.append(
                self.add_edge(
                    MemoryEdgeRecord(
                        src_id=memory_id,
                        dst_id=witness.block_id,
                        edge_type=EdgeType.DERIVED_FROM,
                        workspace_id=self.workspace_id,
                        run_id=self.scope_lineage.session_id,
                        subtask_id=self.scope_lineage.subtask_id,
                        reason=reason,
                        evidence_refs=(evidence_ref,),
                        metadata={"source_kind": "raw_host_witness"},
                    )
                )
            )
        return receipts

    def verification_subject(
        self,
        memory_id: str,
        *,
        expected_source_revision_id: str,
        authority: str = "advisory",
        target_block_id: str = "",
        target_scope: str | None = None,
        target_scope_id: str = "",
        source_evidence_refs: tuple[EvidenceRef, ...] = (),
    ) -> CandidateVerificationSubject:
        source = self._require_profile_record(memory_id)
        if not expected_source_revision_id or source.revision_id != expected_source_revision_id:
            raise ValueError("verification subject requires the exact source revision")
        if target_block_id:
            target = next(
                (
                    block
                    for block in load_memory_blocks(
                        self.repo_root, self.workspace_id, profile_id=self.profile_id, strict=True
                    )
                    if block.block_id == target_block_id
                ),
                None,
            )
            if target is None or target.index_only or target.child_ids:
                raise ValueError("publication target requires a materialized leaf")
            if target_scope is not None and target_scope != target.scope:
                raise ValueError("publication target scope does not match its leaf")
            target_scope = target.scope
        scope, scope_id = _promotion_target_binding(
            self.scope_lineage,
            source,
            target_scope=target_scope,
            target_scope_id=target_scope_id,
        )
        if any(not isinstance(ref, EvidenceRef) for ref in source_evidence_refs):
            raise TypeError("verification subject evidence requires EvidenceRef values")
        membership = source.block_membership
        if membership is None:
            raise ValueError("verification subject requires an exact leaf membership")
        return CandidateVerificationSubject(
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            memory_id=source.block_id,
            revision_id=source.revision_id,
            payload_hash=hash_value(asdict(source)),
            source_evidence_hash=hash_value([asdict(ref) for ref in source_evidence_refs]),
            target_block_id=target_block_id or membership.primary_block_id,
            authority=authority,
            scope=scope,
            scope_id=scope_id,
        )

    def promote(
        self,
        memory_id: str,
        *,
        reason: str,
        evidence_refs: list[Any] | tuple[Any, ...],
        authority: str = "advisory",
        source_evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...] = (),
        target_block_id: str = "",
        target_scope: str | None = None,
        target_scope_id: str = "",
        expected_source_revision_id: str = "",
        operation_id: str = "",
        workspace_root: str | Path | None = None,
        changed_files: tuple[str, ...] = (),
        allow_workspace_hash_mismatch_if_diff_matches: bool = False,
    ) -> dict[str, Any]:
        """Publish with the host's exact review scope, without inferring semantic assurance."""

        if not str(reason or "").strip():
            raise ValueError("promotion requires a reason")
        if (
            not isinstance(operation_id, str)
            or operation_id != operation_id.strip()
            or len(operation_id) > 256
        ):
            raise ValueError("promotion operation_id must be an exact identifier")
        if operation_id and not expected_source_revision_id:
            raise ValueError("promotion semantic verification requires expected_source_revision_id")
        reports = _typed_promotion_evidence(evidence_refs)
        source_refs = tuple(source_evidence_refs)
        if any(not isinstance(ref, EvidenceRef) for ref in source_refs):
            raise TypeError("promotion source evidence must contain EvidenceRef values")
        operation_arguments: dict[str, object] = {
            "tool": "promote",
            "memory_id": memory_id,
            "reason": reason,
            "evidence_refs": [asdict(report) for report in reports],
            "source_evidence_refs": [asdict(ref) for ref in source_refs],
            "authority": authority,
            "target_block_id": target_block_id,
            "target_scope": target_scope,
            "target_scope_id": target_scope_id,
            "expected_source_revision_id": expected_source_revision_id,
            "workspace_root": str(Path(workspace_root or self.repo_root).resolve()),
            "changed_files": list(changed_files),
            "allow_workspace_hash_mismatch_if_diff_matches": allow_workspace_hash_mismatch_if_diff_matches,
        }
        publication_id = "publication_" + hash_value(
            [self.workspace_id, self.profile_id, operation_id]
        )

        def complete_publication(receipt: PublicationCommitReceipt) -> dict[str, Any]:
            response = receipt.response(self._backend_id())
            complete_remember_receipt(
                self.repo_root,
                workspace_id=self.workspace_id,
                profile_id=self.profile_id,
                backend=self._backend_id(),
                caller_role="memory_client",
                protocol_version=2,
                mutation_correlation_id=operation_id,
                arguments=operation_arguments,
                response={
                    "receipt_kind": "mutation_response",
                    "mutation_tool": "promote",
                    "response": response,
                },
            )
            return response

        if operation_id:
            receipt_status, stored, _ = claim_remember_receipt(
                self.repo_root,
                workspace_id=self.workspace_id,
                profile_id=self.profile_id,
                backend=self._backend_id(),
                caller_role="memory_client",
                protocol_version=2,
                mutation_correlation_id=operation_id,
                arguments=operation_arguments,
            )
            if receipt_status == "committed":
                response = stored.get("response") if stored is not None else None
                if not isinstance(response, dict):
                    raise ValueError("publication receipt response is invalid")
                return {**response, "replayed": True}
            receipt_reader = getattr(self.backend, "publication_receipt", None)
            if not callable(receipt_reader):
                raise RuntimeError("selected backend does not support atomic publication receipts")
            committed = receipt_reader(publication_id)
            if committed is not None:
                if not isinstance(committed, PublicationCommitReceipt) or (
                    committed.request_hash != hash_value(operation_arguments)
                    or committed.workspace_id != self.workspace_id
                    or committed.profile_id != self.profile_id
                    or committed.source_id != memory_id
                    or committed.source_revision_id != expected_source_revision_id
                ):
                    raise ValueError("canonical publication receipt conflict")
                if committed.materialized_copy:
                    original_source = self._require_profile_record(committed.source_id)
                    edge = self.add_edge(
                        MemoryEdgeRecord(
                            src_id=committed.record_id,
                            dst_id=committed.source_id,
                            edge_type=EdgeType.DERIVED_FROM,
                            workspace_id=self.workspace_id,
                            run_id=original_source.run_id,
                            subtask_id=original_source.subtask_id,
                            reason="tribunal_verified_lesson_promotion",
                            evidence_refs=tuple(
                                dict.fromkeys(
                                    [*source_refs, *(report.evidence_ref() for report in reports)]
                                )
                            ),
                            metadata={
                                "tribunal_decision": "accept",
                                "source_revision_id": committed.source_revision_id,
                                "target_block_id": committed.target_block_id,
                            },
                        )
                    )
                    if edge.get("ok") is not True:
                        raise RuntimeError("publication provenance recovery failed")
                return {**complete_publication(committed), "replayed": True}
        resolver = EvidenceResolver(
            build_workspace_context(
                repo_root=self.repo_root,
                workspace_root=workspace_root or self.repo_root,
                workspace_id=self.workspace_id,
                changed_files=changed_files,
                compute_workspace_hash=(not allow_workspace_hash_mismatch_if_diff_matches),
            )
        )
        issues = [
            issue
            for report in reports
            for issue in resolver.validate_verification_report_ref(
                report,
                allow_workspace_hash_mismatch_if_diff_matches=(
                    allow_workspace_hash_mismatch_if_diff_matches
                ),
            )
        ]
        if issues:
            codes = ", ".join(sorted({issue.code for issue in issues}))
            raise ValueError(f"promotion verification evidence rejected: {codes}")
        source_refs = tuple(source_evidence_refs)
        if any(not isinstance(ref, EvidenceRef) for ref in source_refs):
            raise TypeError("promotion source evidence must contain EvidenceRef values")
        source_issues = [issue for ref in source_refs for issue in resolver.validate_ref(ref)]
        if source_issues:
            codes = ", ".join(sorted({issue.code for issue in source_issues}))
            raise ValueError(f"promotion source evidence rejected: {codes}")
        refs = tuple(report.evidence_ref() for report in reports)
        promotion_refs = tuple(dict.fromkeys([*source_refs, *refs]))
        backend = self.backend
        getter = getattr(backend, "get_block", None)
        if not callable(getter):
            raise RuntimeError("selected memory backend does not support promotion lookup")
        source = self._require_profile_record(memory_id)
        raw_edges_for = getattr(backend, "edges_for", None)
        current = resolve_canonical_revision_head(
            source,
            canonical_key=source.canonical_key,
            workspace_id=self.workspace_id,
            get_block=getter,
            edges_for=raw_edges_for if callable(raw_edges_for) else None,
        )
        expected_revision = str(expected_source_revision_id or "").strip()
        if not expected_revision:
            raise ValueError("promotion semantic verification requires expected_source_revision_id")
        if expected_revision and (
            current.block_id != source.block_id or source.revision_id != expected_revision
        ):
            raise ValueError("promotion source is no longer the exact canonical head")
        if current.status in {"quarantined", "superseded", "deprecated", "retracted"}:
            raise ValueError(f"memory in status {current.status!r} cannot be promoted")
        if current.kind == "skill_index" or _flag_enabled(
            (current.metadata or {}).get("index_only")
        ):
            raise ValueError("index-only memory cannot be promoted as evidence")

        subject = self.verification_subject(
            source.block_id,
            expected_source_revision_id=expected_revision,
            authority=authority,
            target_block_id=target_block_id,
            target_scope=target_scope,
            target_scope_id=target_scope_id,
            source_evidence_refs=source_refs,
        )
        semantic_fields = {
            "subject",
            "semantic_subject_hash",
            "verifier_id",
            "verifier_model_version",
            "proposer_model_version",
            "verifier_prompt_version",
            "coverage",
            "preservation",
            "faithfulness",
            "instruction_safety",
            "reasons",
        }
        semantic_receipt = None
        for report in reports:
            row = resolver.ledger_row(report.report_id)
            if row is None or row.get("tool") != "candidate_semantic_verification":
                continue
            result = row.get("result")
            if not isinstance(result, dict):
                raise ValueError("candidate semantic verification result is malformed")
            semantic_result = {key: result.get(key) for key in semantic_fields}
            outcomes = [
                result.get(key)
                for key in ("coverage", "preservation", "faithfulness", "instruction_safety")
            ]
            reasons = result.get("reasons")
            if (
                row.get("actor") != "verifier"
                or result.get("subject") != asdict(subject)
                or result.get("semantic_subject_hash") != hash_value(asdict(subject))
                or report.report_hash != hash_value(semantic_result)
                or result.get("verifier_id") != report.verifier_id
                or not result.get("verifier_model_version")
                or not result.get("proposer_model_version")
                or result.get("verifier_model_version") == result.get("proposer_model_version")
                or any(value is not True for value in outcomes)
                or not isinstance(reasons, list)
                or not reasons
                or any(not isinstance(value, str) or not value.strip() for value in reasons)
                or result.get("passed") is not True
            ):
                raise ValueError("candidate semantic verification is not bound to this publication")
            semantic_receipt = {"result": semantic_result, "report": asdict(report)}
        if semantic_receipt is None:
            raise ValueError("promotion requires dedicated candidate semantic verification")

        procedure_gate_receipt: dict[str, Any] = {}
        if current.kind == "procedure":
            applicability = current.metadata.get("procedure_applicability")
            signature = current.metadata.get("source_task_signature")
            episode_ids = current.metadata.get("source_episode_ids")
            if (
                not isinstance(applicability, str)
                or not applicability.strip()
                or not isinstance(signature, str)
                or not signature.strip()
                or not isinstance(episode_ids, (list, tuple))
                or not episode_ids
                or any(not isinstance(item, str) or not item.strip() for item in episode_ids)
                or len(set(episode_ids)) != len(episode_ids)
                or current.block_id in episode_ids
            ):
                raise ValueError(
                    "procedure requires applicability, task signature and source episodes"
                )
            for episode_id in episode_ids:
                episode = self._require_profile_record(episode_id)
                episode_head = resolve_canonical_revision_head(
                    episode,
                    canonical_key=episode.canonical_key,
                    workspace_id=self.workspace_id,
                    get_block=getter,
                    edges_for=raw_edges_for if callable(raw_edges_for) else None,
                )
                if episode_head.block_id != episode.block_id or episode.status in {
                    "quarantined",
                    "superseded",
                    "deprecated",
                    "retracted",
                }:
                    raise ValueError("procedure source episode is no longer current")
            credits = backend.outcome_credits_for_memory(current.block_id)
            gate = verified_procedure_promotion_gate(
                memory_id=current.block_id,
                profile_id=self.profile_id,
                task_signature=signature,
                outcome_credits=credits,
                resolver=resolver,
            )
            if not gate.allowed:
                raise ValueError("procedure requires two independent verified applications")
            procedure_gate_receipt = {
                "procedure_promotion_gate": {
                    "verified_outcome_ids": list(gate.verified_outcome_ids),
                    "independence_domains": list(gate.independence_domains),
                    "minimum_verified_outcomes": gate.minimum_verified_outcomes,
                    "allowed": gate.allowed,
                    "reason": gate.reason,
                }
            }

        target_authority = str(authority or "advisory").strip()
        if target_authority not in {"advisory", "directive"}:
            raise ValueError("promotion authority must be advisory or directive")
        bound_scope, bound_scope_id = _promotion_target_binding(
            self.scope_lineage,
            current,
            target_scope=target_scope,
            target_scope_id=target_scope_id,
        )

        correction_target_id = str(current.metadata.get("correction_target_id") or "").strip()
        if operation_id and current.status != "candidate":
            raise ValueError("recoverable promotion requires an unpublished candidate")
        tribunal_edges = tuple(backend.edges_for(current.block_id))
        if correction_target_id:
            tribunal_edges = tuple(
                edge
                for edge in tribunal_edges
                if not (
                    edge.edge_type == "contradicts"
                    and edge.src_id == current.block_id
                    and edge.dst_id == correction_target_id
                )
            )
        decision = run_tribunal(
            TribunalInput(
                candidate_id=current.block_id,
                rules=(
                    {
                        "id": current.block_id,
                        "title": current.title,
                        "rule": current.content,
                        "status": current.status,
                    },
                ),
                source_evidence=tuple(asdict(ref) for ref in promotion_refs),
                actor="verifier",
                source_backend=current.source_backend,
                workspace_id=current.workspace_id,
                target_workspace_id=self.workspace_id,
                graph_edges=tribunal_edges,
            )
        )
        if decision.decision != "accept":
            raise ValueError(f"promotion tribunal {decision.decision}: {decision.reason}")

        requested_target_block = str(target_block_id or "").strip()
        if requested_target_block:
            if target_authority != "advisory":
                raise ValueError("publication into another leaf remains advisory")
            blocks = load_memory_blocks(
                self.repo_root,
                self.workspace_id,
                profile_id=self.profile_id,
                strict=True,
            )
            catalog = compile_semantic_catalog(blocks)
            target_block = next(
                (block for block in blocks if block.block_id == requested_target_block),
                None,
            )
            if target_block is None or target_block.index_only or target_block.child_ids:
                raise ValueError("verified lesson target is not a storable leaf")
            if target_block.scope != subject.scope:
                raise ValueError("publication leaf scope changed after verification")
            promoted_insert = replace(
                current,
                block_id="",
                canonical_key=(
                    f"{self.workspace_id}:{self.profile_id}:publication:"
                    + hashlib.sha256(
                        f"{current.canonical_key}|{current.revision_id}|{target_block.block_id}|{subject.scope}|{subject.scope_id}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:24]
                ),
                revision_id="",
                duplicate_cluster_id="",
                revision_of="",
                supersedes_ids=(),
                block_membership=RecordBlockMembership(
                    primary_block_id=target_block.block_id,
                    catalog_revision_id=catalog.revision_id,
                    backend_partition_id=self._backend_partition_id_for_block(
                        target_block.block_id
                    ),
                ),
                status="accepted",
                retention=target_block.retention,
                delivery_policy=target_block.delivery_policy,
                temperature=target_block.temperature,
                authority="advisory",
                trust_level="workspace_verified",
                scope=target_block.scope,
                project_id=subject.scope_id if subject.scope == "project" else current.project_id,
                user_id=subject.scope_id if subject.scope == "user_global" else current.user_id,
                session_id=subject.scope_id if subject.scope == "session" else current.session_id,
                task_id=subject.scope_id if subject.scope == "task" else current.task_id,
                subtask_id=subject.scope_id if subject.scope == "subtask" else current.subtask_id,
                evidence_refs=tuple(dict.fromkeys([*current.evidence_refs, *promotion_refs])),
                verification_status="verified",
                verification_methods=("verification_report",),
                promotion_decision="accepted",
                metadata={
                    **current.metadata,
                    "promotion_source_id": current.block_id,
                    "promotion_source_revision_id": current.revision_id,
                    "promoted_to_block_id": target_block.block_id,
                    "status_reason": reason,
                    "candidate_semantic_verification": semantic_receipt,
                    **procedure_gate_receipt,
                },
            )
            existing_promoted = getter(block_id_for_canonical_key(promoted_insert.canonical_key))
            if isinstance(existing_promoted, MemoryBlockRecord):
                existing_membership = existing_promoted.block_membership
                if (
                    existing_promoted.metadata.get("promotion_source_id") != current.block_id
                    or existing_promoted.metadata.get("promotion_source_revision_id")
                    != current.revision_id
                    or existing_promoted.content != current.content
                    or existing_promoted.profile_id != self.profile_id
                    or existing_membership is None
                    or existing_membership.primary_block_id != target_block.block_id
                    or existing_promoted.status != "accepted"
                    or existing_promoted.verification_status != "verified"
                ):
                    raise ValueError(
                        "verified lesson promotion conflicts with an existing canonical record"
                    )
                if operation_id:
                    receipt = PublicationCommitReceipt(
                        receipt_id=publication_id,
                        request_hash=hash_value(operation_arguments),
                        workspace_id=self.workspace_id,
                        profile_id=self.profile_id,
                        source_id=current.block_id,
                        source_revision_id=current.revision_id,
                        record_id=existing_promoted.block_id,
                        revision_id=existing_promoted.revision_id,
                        target_block_id=target_block.block_id,
                        target_scope=subject.scope,
                        target_scope_id=subject.scope_id,
                        authority=target_authority,
                        materialized_copy=True,
                    )
                    receipt_writer = getattr(backend, "record_publication_receipt", None)
                    if not callable(receipt_writer):
                        raise RuntimeError(
                            "selected backend cannot acknowledge the existing publication"
                        )
                    receipt_writer(receipt)
                    return {**complete_publication(receipt), "replayed": True}
                return {
                    "ok": True,
                    "saved": True,
                    "replayed": True,
                    "block_id": existing_promoted.block_id,
                    "source_memory_id": current.block_id,
                    "result_memory_id": existing_promoted.block_id,
                    "promotion_receipt_id": existing_promoted.block_id,
                    "requested_block_id": target_block.block_id,
                    "actual_block_id": target_block.block_id,
                    "status": existing_promoted.status,
                    "authority": existing_promoted.authority,
                    "verification_status": existing_promoted.verification_status,
                    "promotion_decision": existing_promoted.promotion_decision,
                    "tribunal_decision": decision.decision,
                    "evidence_refs": [asdict(ref) for ref in existing_promoted.evidence_refs],
                    "source_status": current.status,
                }
            insertion = plan_memory_mutation(promoted_insert, existing=None)
            if operation_id:
                insertion = replace(
                    insertion,
                    publication_receipt=PublicationCommitReceipt(
                        receipt_id=publication_id,
                        request_hash=hash_value(operation_arguments),
                        workspace_id=self.workspace_id,
                        profile_id=self.profile_id,
                        source_id=current.block_id,
                        source_revision_id=current.revision_id,
                        record_id=insertion.record.block_id,
                        revision_id=insertion.record.revision_id,
                        target_block_id=target_block.block_id,
                        target_scope=subject.scope,
                        target_scope_id=subject.scope_id,
                        authority=target_authority,
                        materialized_copy=True,
                    ),
                )
            result = _write_memory_event(
                self.repo_root,
                block_record_to_event(insertion.record),
                workspace_id=self.workspace_id,
                backend_instance=backend,
                mutation=insertion,
            )
            if not result.saved:
                payload = dict(result.raw)
                payload.setdefault("ok", False)
                payload.setdefault("error", result.error or "verified lesson write failed")
                return payload
            provenance = MemoryEdgeRecord(
                src_id=str(result.canonical_id or insertion.record.block_id),
                dst_id=current.block_id,
                edge_type=EdgeType.DERIVED_FROM,
                workspace_id=self.workspace_id,
                run_id=current.run_id,
                subtask_id=current.subtask_id,
                reason="tribunal_verified_lesson_promotion",
                evidence_refs=promotion_refs,
                metadata={
                    "tribunal_decision": decision.decision,
                    "source_revision_id": current.revision_id,
                    "target_block_id": target_block.block_id,
                },
            )
            edge_receipt = self.add_edge(provenance)
            if edge_receipt.get("ok") is not True:
                raise RuntimeError("publication provenance write failed")
            promoted_id = str(result.canonical_id or insertion.record.block_id)
            if insertion.publication_receipt is not None:
                return complete_publication(insertion.publication_receipt)
            return {
                **result.raw,
                "ok": True,
                "saved": True,
                "block_id": promoted_id,
                "source_memory_id": current.block_id,
                "result_memory_id": promoted_id,
                "promotion_receipt_id": promoted_id,
                "requested_block_id": target_block.block_id,
                "actual_block_id": target_block.block_id,
                "status": "accepted",
                "authority": "advisory",
                "verification_status": "verified",
                "promotion_decision": "accepted",
                "tribunal_decision": decision.decision,
                "evidence_refs": [asdict(ref) for ref in promotion_refs],
                "provenance_edge": edge_receipt,
                "source_status": current.status,
            }

        mutation_base = current
        mutation_action = "promote"
        approved_correction_candidate_id = ""
        if correction_target_id:
            target_root = self._require_profile_record(correction_target_id)
            target = resolve_canonical_revision_head(
                target_root,
                canonical_key=target_root.canonical_key,
                workspace_id=self.workspace_id,
                get_block=getter,
                edges_for=raw_edges_for if callable(raw_edges_for) else None,
            )
            expected_target_revision = str(
                current.metadata.get("correction_target_revision_id") or ""
            ).strip()
            if not expected_target_revision or target.revision_id != expected_target_revision:
                raise ValueError("correction target changed before host approval")
            if (
                current.block_membership is None
                or target.block_membership is None
                or current.block_membership.primary_block_id
                != target.block_membership.primary_block_id
            ):
                raise ValueError("correction candidate moved outside its target logical block")
            approved_metadata = {
                **target.metadata,
                **current.metadata,
                "approved_correction_candidate_id": current.block_id,
                "status_reason": reason,
            }
            approved_metadata.pop("pending_host_approval", None)
            mutation_base = replace(
                target,
                title=current.title,
                content=current.content,
                tags=current.tags,
                child_claims=current.child_claims,
                metadata=approved_metadata,
            )
            mutation_action = "correction"
            approved_correction_candidate_id = current.block_id
        membership = mutation_base.block_membership
        if membership is not None:
            mutation_base = replace(
                mutation_base,
                block_membership=RecordBlockMembership(
                    primary_block_id=membership.primary_block_id,
                    secondary_block_ids=membership.secondary_block_ids,
                    catalog_revision_id=self._catalog_revision_id(),
                    backend_partition_id=membership.backend_partition_id,
                ),
            )
        combined_refs = tuple(
            dict.fromkeys([*mutation_base.evidence_refs, *current.evidence_refs, *promotion_refs])
        )
        promoted = replace(
            mutation_base,
            status="accepted",
            authority=cast(MemoryAuthority, target_authority),
            trust_level="workspace_verified",
            evidence_refs=combined_refs,
            verification_status="verified",
            verification_methods=("verification_report",),
            promotion_decision="accepted",
            metadata={
                **mutation_base.metadata,
                "promotion_operation_id": operation_id,
                "promotion_source_id": current.block_id,
                "promotion_source_revision_id": current.revision_id,
                "candidate_semantic_verification": semantic_receipt,
                **procedure_gate_receipt,
            },
            scope=bound_scope,
            project_id=bound_scope_id if bound_scope == "project" else current.project_id,
            user_id=bound_scope_id if bound_scope == "user_global" else current.user_id,
            session_id=bound_scope_id if bound_scope == "session" else current.session_id,
            task_id=bound_scope_id if bound_scope == "task" else current.task_id,
            subtask_id=bound_scope_id if bound_scope == "subtask" else current.subtask_id,
        )
        mutation = plan_memory_mutation(
            promoted,
            existing=mutation_base,
            action=mutation_action,
            current_identity=_current_mutation_identity(
                backend,
                mutation_base,
                workspace_id=self.workspace_id,
            ),
        )
        if operation_id:
            publication_membership = mutation.record.block_membership
            if publication_membership is None:
                raise ValueError("publication requires exact leaf membership")
            mutation = replace(
                mutation,
                publication_receipt=PublicationCommitReceipt(
                    receipt_id=publication_id,
                    request_hash=hash_value(operation_arguments),
                    workspace_id=self.workspace_id,
                    profile_id=self.profile_id,
                    source_id=current.block_id,
                    source_revision_id=current.revision_id,
                    record_id=mutation.record.block_id,
                    revision_id=mutation.record.revision_id,
                    target_block_id=publication_membership.primary_block_id,
                    target_scope=bound_scope,
                    target_scope_id=bound_scope_id,
                    authority=target_authority,
                ),
            )
        result = _write_memory_event(
            self.repo_root,
            block_record_to_event(mutation.record),
            workspace_id=self.workspace_id,
            backend_instance=backend,
            mutation=mutation,
        )
        if result.saved and mutation.publication_receipt is not None:
            return complete_publication(mutation.publication_receipt)
        payload = dict(result.raw)
        payload.setdefault("ok", result.saved)
        payload.setdefault("block_id", result.canonical_id or memory_id)
        payload.setdefault("backend", result.backend)
        if result.policy_issues:
            payload.setdefault("policy_issues", list(result.policy_issues))
        if not result.saved:
            error = result.error or "; ".join(result.policy_issues)
            if error:
                payload.setdefault("error", error)
        mutation_identity = _current_mutation_identity(
            backend,
            mutation.record,
            workspace_id=self.workspace_id,
        )
        payload.setdefault("promotion_receipt_id", result.canonical_id or memory_id)
        payload.setdefault("source_memory_id", memory_id)
        payload.setdefault("result_memory_id", result.canonical_id or memory_id)
        payload.setdefault("target_scope", bound_scope)
        payload.setdefault("target_scope_id", bound_scope_id)
        if approved_correction_candidate_id:
            payload.setdefault("approved_correction_candidate_id", approved_correction_candidate_id)
        if mutation_identity is not None:
            payload.setdefault("promotion_mutation", asdict(mutation_identity))
        return payload

    def _backend_id(self) -> str:
        if self.backend_mode:
            return str(self.backend_mode)
        name = type(self.backend).__name__
        return name.removesuffix("Backend").casefold() or "selected_backend"

    def _build_packed_context(
        self,
        query: str,
        *,
        limit: int,
        max_tokens: int,
        tags: list[str] | tuple[str, ...] | None,
        run_id: str,
        source_horizon: SourceHorizon,
        include_candidates: bool,
        scope_chain: list[str] | tuple[str, ...] | None,
        logical_block_ids: list[str] | tuple[str, ...] | None,
        search_all: bool,
        retrieval_mode: RetrievalMode = "semantic",
        record_ids: list[str] | tuple[str, ...] | None,
        recall_plan: RecallPlan | dict[str, Any] | None,
        source_speakers: list[str] | tuple[str, ...] | None = None,
    ) -> _ContextBuild:
        started_ns = perf_counter_ns()
        catalog_revision = self._catalog_revision_id()
        plan = self._coerce_recall_plan(
            query,
            limit=limit,
            tags=tags,
            run_id=run_id,
            max_tokens=max_tokens,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            scope_chain=scope_chain,
            temporal_mode="current",
            as_of="",
            known_as_of="",
            revision_id="",
            logical_block_ids=logical_block_ids,
            search_all=search_all,
            retrieval_mode=retrieval_mode,
            record_ids=record_ids,
            recall_plan=recall_plan,
        )
        hits = self.recall(plan.query, recall_plan=plan, source_speakers=source_speakers)
        selection_context = SelectionContext(
            query=plan.query,
            scope_chain=plan.scope_chain,
            run_id=plan.run_id,
            source_horizon=plan.source_horizon,
            include_candidates=plan.include_candidates,
            historical=plan.temporal_mode == "historical" or bool(plan.known_as_of),
            as_of="" if plan.known_as_of else plan.as_of,
        )
        projected: list[dict[str, Any]] = []
        for hit in hits:
            try:
                record = _record_from_hit(hit, plan)
            except ValueError:
                continue
            if not MemorySelectionPolicy().eligible_for_overlay(record, selection_context):
                continue
            projected.append(
                {
                    "text": hit.text,
                    "source": hit.source,
                    "score": hit.score,
                    "kind": hit.kind,
                    "tags": list(hit.tags),
                    "metadata": dict(hit.metadata),
                    "authority": "advisory",
                }
            )
        allocation = allocate_recall_budget(plan)
        effective_requirements = plan.requirements
        if hits and "effective_evidence_requirements" in hits[0].metadata:
            raw_requirements = hits[0].metadata["effective_evidence_requirements"]
            if not isinstance(raw_requirements, (list, tuple)):
                raise ValueError("effective evidence requirements must be a sequence")
            effective_requirements = tuple(
                EvidenceRequirement.from_mapping(requirement) for requirement in raw_requirements
            )
        lineage = self._active_scope_lineage()
        with profile_memory_operation(
            self.repo_root,
            operation="pack_context",
            component="memory_client.packer",
            workspace_id=self.workspace_id,
            profile_id=self.profile_id,
            user_id=self.user_id,
            chat_id=lineage.session_id,
            run_id=plan.run_id,
            backend=self._backend_id(),
            input_value={
                "record_count": len(projected),
                "max_tokens": allocation.allocated_tokens,
                "query": plan.query,
            },
        ) as profile:
            payload = pack_evidence_payload(
                projected,
                max_tokens=allocation.allocated_tokens,
                query=plan.query,
                requirements=effective_requirements,
            )
            profile.set_output(
                {
                    "item_count": len(payload.items),
                    "estimated_tokens": payload.estimated_tokens,
                    "rendered_chars": len(payload.rendered_context),
                }
            )
            profile.set_result(outcome="packed", reason="evidence_budget")
        latency_ms = (perf_counter_ns() - started_ns) // 1_000_000
        decision_id = self._record_context_decision(
            plan=plan,
            hits=tuple(hits),
            payload=payload,
            latency_ms=latency_ms,
            catalog_revision=catalog_revision,
        )
        return _ContextBuild(
            plan=plan,
            hits=tuple(hits),
            payload=payload,
            decision_id=decision_id,
        )

    def build_context(
        self,
        query: str,
        *,
        limit: int = 8,
        max_tokens: int = 2048,
        tags: list[str] | tuple[str, ...] | None = None,
        run_id: str = "",
        source_horizon: SourceHorizon = "durable",
        include_candidates: bool = False,
        scope_chain: list[str] | tuple[str, ...] | None = None,
        logical_block_ids: list[str] | tuple[str, ...] | None = None,
        search_all: bool = False,
        retrieval_mode: RetrievalMode = "semantic",
        record_ids: list[str] | tuple[str, ...] | None = None,
        source_speakers: list[str] | tuple[str, ...] | None = None,
        recall_plan: RecallPlan | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build fast one-hop prompt evidence through the one final packer."""

        built = self._build_packed_context(
            query,
            limit=limit,
            max_tokens=max_tokens,
            tags=tags,
            run_id=run_id,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            scope_chain=scope_chain,
            logical_block_ids=logical_block_ids,
            search_all=search_all,
            retrieval_mode=retrieval_mode,
            record_ids=record_ids,
            source_speakers=source_speakers,
            recall_plan=recall_plan,
        )
        return [
            {
                **item,
                "metadata": {
                    **dict(item.get("metadata") or {}),
                    **({"context_decision_id": built.decision_id} if built.decision_id else {}),
                },
            }
            for item in built.payload.items
        ]

    def build_activation_lease(
        self,
        query: str,
        *,
        limit: int = 8,
        max_tokens: int = 2048,
        tags: list[str] | tuple[str, ...] | None = None,
        run_id: str = "",
        source_horizon: SourceHorizon = "durable",
        include_candidates: bool = False,
        scope_chain: list[str] | tuple[str, ...] | None = None,
        logical_block_ids: list[str] | tuple[str, ...] | None = None,
        search_all: bool = False,
        retrieval_mode: RetrievalMode = "semantic",
        record_ids: list[str] | tuple[str, ...] | None = None,
        recall_plan: RecallPlan | dict[str, Any] | None = None,
    ) -> ActivationLease:
        """Activate the final fast payload without changing source-record authority."""

        built = self._build_packed_context(
            query,
            limit=limit,
            max_tokens=max_tokens,
            tags=tags,
            run_id=run_id,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            scope_chain=scope_chain,
            logical_block_ids=logical_block_ids,
            search_all=search_all,
            retrieval_mode=retrieval_mode,
            record_ids=record_ids,
            recall_plan=recall_plan,
        )
        return activation_lease_from_payload(
            plan=built.plan,
            payload=built.payload,
            backend_id=self._backend_id(),
        )

    def deep_read(
        self,
        query: str,
        *,
        logical_block_id: str,
        limit: int = 16,
        max_tokens: int = 2048,
        tags: list[str] | tuple[str, ...] | None = None,
        run_id: str = "",
        source_horizon: SourceHorizon = "durable",
        include_candidates: bool = False,
        record_ids: list[str] | tuple[str, ...] | None = None,
        recall_plan: RecallPlan | dict[str, Any] | None = None,
    ) -> DeepReadResult:
        """Run explicit bounded reconstruction inside one logical block and backend."""

        block_id = str(logical_block_id or "").strip()
        if not block_id:
            raise ValueError("deep_read requires one logical_block_id")
        bounded_limit = min(32, max(1, int(limit)))
        bounded_tokens = min(4096, max(0, int(max_tokens)))
        plan = self._coerce_recall_plan(
            query,
            limit=bounded_limit,
            tags=tags,
            run_id=run_id,
            max_tokens=bounded_tokens,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            scope_chain=(f"workspace:{self.workspace_id}",),
            temporal_mode="current",
            as_of="",
            known_as_of="",
            revision_id="",
            logical_block_ids=(block_id,),
            search_all=False,
            record_ids=record_ids,
            recall_plan=recall_plan,
        )
        if plan.logical_block_ids and block_id not in plan.logical_block_ids:
            raise ValueError("deep_read recall_plan does not select the requested logical block")
        if plan.includes_scope("user_global"):
            raise ValueError("deep_read cannot include user-global scope")
        raw_descriptors = plan.metadata.get("block_descriptors")
        if isinstance(raw_descriptors, (list, tuple)):
            descriptors = list(raw_descriptors)
        else:
            catalog = compile_semantic_catalog(
                load_memory_blocks(
                    self.repo_root,
                    self.workspace_id,
                    profile_id=self.profile_id,
                )
            )
            descriptors = [catalog.descriptor(block_id).to_mapping()]
        plan = plan.with_overrides(
            logical_block_ids=(block_id,),
            scope_chain=(f"workspace:{self.workspace_id}",),
            limit=bounded_limit,
            max_tokens=bounded_tokens,
            metadata={
                **plan.metadata,
                "block_descriptors": descriptors,
                "deep_read_trace": {
                    "mode": "explicit_deep_block_local",
                    "primary_logical_block_ids": [block_id],
                    "widen_on_empty": False,
                    "maximum_widening_attempts": 0,
                    "outcome": "strict_block_local_reader",
                },
            },
        )
        built = self._build_packed_context(
            query,
            limit=bounded_limit,
            max_tokens=bounded_tokens,
            tags=tags,
            run_id=run_id,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            scope_chain=(f"workspace:{self.workspace_id}",),
            logical_block_ids=(block_id,),
            search_all=False,
            record_ids=record_ids,
            recall_plan=plan,
        )
        backend_id = self._backend_id()
        lease = activation_lease_from_payload(
            plan=built.plan,
            payload=built.payload,
            backend_id=backend_id,
            mode="deep_block_local",
        )
        return deep_read_result(
            lease=lease,
            plan=built.plan,
            backend_id=backend_id,
            logical_block_id=block_id,
            limit=bounded_limit,
        )

    def explain_recall(
        self,
        query: str,
        *,
        limit: int = 8,
        max_tokens: int = 2048,
        tags: list[str] | tuple[str, ...] | None = None,
        run_id: str = "",
        source_horizon: SourceHorizon = "durable",
        include_candidates: bool = False,
        scope_chain: list[str] | tuple[str, ...] | None = None,
        logical_block_ids: list[str] | tuple[str, ...] | None = None,
        search_all: bool = False,
        retrieval_mode: RetrievalMode = "semantic",
        record_ids: list[str] | tuple[str, ...] | None = None,
        recall_plan: RecallPlan | dict[str, Any] | None = None,
        gold_block_ids: list[str] | tuple[str, ...] = (),
        gold_literals: list[str] | tuple[str, ...] = (),
    ) -> RecallExplanation:
        """Explain candidate and rendered delivery without placing trace in the prompt."""

        built = self._build_packed_context(
            query,
            limit=limit,
            max_tokens=max_tokens,
            tags=tags,
            run_id=run_id,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            scope_chain=scope_chain,
            logical_block_ids=logical_block_ids,
            search_all=search_all,
            retrieval_mode=retrieval_mode,
            record_ids=record_ids,
            recall_plan=recall_plan,
        )
        return build_recall_explanation(
            plan=built.plan,
            hits=list(built.hits),
            payload=built.payload,
            backend_id=self._backend_id(),
            gold_block_ids=tuple(str(value) for value in gold_block_ids if str(value)),
            gold_literals=tuple(str(value) for value in gold_literals if str(value)),
        )

    def export_graph(self) -> dict[str, Any]:
        """Export the scoped graph with machine-readable provenance integrity."""

        blocks = list(self.backend.export_blocks() or [])
        export_edges = getattr(self.backend, "export_edges", None)
        if not callable(export_edges):
            raise RuntimeError("selected memory backend does not support graph export")
        edges = list(export_edges() or [])
        records = {
            record.block_id: record
            for record in blocks
            if isinstance(record, MemoryBlockRecord) and record.workspace_id == self.workspace_id
        }
        get_index_node = getattr(self.backend, "get_index_node", None)
        index_node_ids = {
            endpoint_id
            for edge in edges
            if isinstance(edge, MemoryEdgeRecord)
            for endpoint_id in (edge.src_id, edge.dst_id)
            if endpoint_id not in records
            and callable(get_index_node)
            and get_index_node(endpoint_id, workspace_id=self.workspace_id) is not None
        }
        endpoint_ids = set(records) | index_node_ids
        dangling_edges = [
            {
                "src_id": edge.src_id,
                "dst_id": edge.dst_id,
                "edge_type": edge.edge_type,
            }
            for edge in edges
            if isinstance(edge, MemoryEdgeRecord)
            and (edge.src_id not in endpoint_ids or edge.dst_id not in endpoint_ids)
        ]
        cross_workspace_edges = [
            {
                "src_id": edge.src_id,
                "dst_id": edge.dst_id,
                "edge_type": edge.edge_type,
                "workspace_id": edge.workspace_id,
            }
            for edge in edges
            if isinstance(edge, MemoryEdgeRecord) and edge.workspace_id != self.workspace_id
        ]
        missing_edge_provenance = [
            {
                "src_id": edge.src_id,
                "dst_id": edge.dst_id,
                "edge_type": edge.edge_type,
            }
            for edge in edges
            if isinstance(edge, MemoryEdgeRecord) and not edge.evidence_refs
        ]
        revision_edge_types = {EdgeType.SUPERSEDES, EdgeType.REVISES}
        cross_block_revision_edges: list[dict[str, str]] = []
        for edge in edges:
            if not isinstance(edge, MemoryEdgeRecord) or edge.edge_type not in revision_edge_types:
                continue
            source = records.get(edge.src_id)
            target = records.get(edge.dst_id)
            source_membership = source.block_membership if source is not None else None
            target_membership = target.block_membership if target is not None else None
            if (
                source is None
                or target is None
                or source.canonical_key != target.canonical_key
                or source_membership is None
                or target_membership is None
                or source_membership.primary_block_id != target_membership.primary_block_id
            ):
                cross_block_revision_edges.append(
                    {
                        "src_id": edge.src_id,
                        "dst_id": edge.dst_id,
                        "edge_type": edge.edge_type,
                    }
                )

        acyclic_edge_types = {
            EdgeType.DERIVED_FROM,
            EdgeType.SUPERSEDES,
            EdgeType.REVISES,
        }
        adjacency: dict[str, set[str]] = {}
        for edge in edges:
            if isinstance(edge, MemoryEdgeRecord) and edge.edge_type in acyclic_edge_types:
                adjacency.setdefault(edge.src_id, set()).add(edge.dst_id)
        cycle_paths: list[list[str]] = []
        visited: set[str] = set()
        active: list[str] = []
        active_set: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in active_set:
                start = active.index(node_id)
                cycle_paths.append([*active[start:], node_id])
                return
            if node_id in visited:
                return
            active.append(node_id)
            active_set.add(node_id)
            for child_id in sorted(adjacency.get(node_id, ())):
                visit(child_id)
            active.pop()
            active_set.remove(node_id)
            visited.add(node_id)

        for node_id in sorted(adjacency):
            visit(node_id)

        provenance_edge_types = {
            EdgeType.DERIVED_FROM,
            EdgeType.OBSERVED_IN,
            EdgeType.PRODUCED_BY,
            EdgeType.PRODUCED_IN,
            EdgeType.VERIFIED_BY,
            EdgeType.VERIFIES,
            EdgeType.REFUTES,
            EdgeType.REVISES,
            EdgeType.SUPERSEDES,
            EdgeType.SUPPORTS,
            EdgeType.CITES,
            EdgeType.REFERENCES_FILE,
            EdgeType.FROM_SUBTASK,
        }
        provenance_routes: dict[str, list[dict[str, Any]]] = {}
        for record_id in records:
            routes = [
                {
                    "edge_type": edge.edge_type,
                    "source_id": edge.src_id,
                    "target_id": edge.dst_id,
                    "reason": edge.reason,
                    "evidence_refs": [asdict(ref) for ref in edge.evidence_refs],
                    "metadata": dict(edge.metadata),
                }
                for edge in edges
                if isinstance(edge, MemoryEdgeRecord)
                and edge.src_id == record_id
                and edge.edge_type in provenance_edge_types
            ]
            if routes:
                provenance_routes[record_id] = routes
        durable_semantic_ids = sorted(
            record.block_id
            for record in records.values()
            if record.retention == "durable"
            and record.delivery_policy != "hidden"
            and record.status
            not in {
                "quarantined",
                "superseded",
                "deprecated",
                "retracted",
            }
        )
        isolated_durable_semantic_ids = [
            record_id for record_id in durable_semantic_ids if record_id not in provenance_routes
        ]
        unreachable_derived_witness_ids = sorted(
            record.block_id
            for record in records.values()
            if record.metadata.get("record_kind") == "raw_host_witness"
            and any(
                str(candidate.metadata.get("raw_witness_id") or "") == record.block_id
                for candidate in records.values()
                if candidate.retention == "durable"
            )
            and not any(
                edge.dst_id == record.block_id and edge.edge_type in provenance_edge_types
                for edge in edges
                if isinstance(edge, MemoryEdgeRecord)
            )
        )
        audit = {
            "valid": not any(
                (
                    dangling_edges,
                    cross_workspace_edges,
                    missing_edge_provenance,
                    cross_block_revision_edges,
                    cycle_paths,
                    isolated_durable_semantic_ids,
                    unreachable_derived_witness_ids,
                )
            ),
            "record_count": len(records),
            "edge_count": len(edges),
            "durable_semantic_record_count": len(durable_semantic_ids),
            "isolated_durable_semantic_record_ids": isolated_durable_semantic_ids,
            "dangling_edges": dangling_edges,
            "cross_workspace_edges": cross_workspace_edges,
            "cross_block_revision_edges": cross_block_revision_edges,
            "cycle_paths": cycle_paths,
            "missing_edge_provenance": missing_edge_provenance,
            "unreachable_derived_witness_ids": unreachable_derived_witness_ids,
            "semantic_similarity_edges_created": 0,
        }
        return {
            "blocks": blocks,
            "edges": edges,
            "index_node_ids": sorted(index_node_ids),
            "provenance_routes": provenance_routes,
            "audit": audit,
        }

    def rebuild_indexes(self) -> dict[str, Any]:
        """Rebuild selected-backend materialized indexes from canonical records."""

        rebuild = getattr(self.backend, "rebuild_memory_indexes", None)
        if not callable(rebuild):
            raise RuntimeError("selected memory backend does not support index rebuild")
        catalog = compile_semantic_catalog(
            load_memory_blocks(
                self.repo_root,
                self.workspace_id,
                profile_id=self.profile_id,
            )
        )
        return dict(rebuild(catalog=catalog) or {})


def remember(content: str, **kwargs: Any) -> RememberResult:
    """Retain memory through a one-off default client."""

    client_kwargs = _client_kwargs(kwargs)
    return MemoryClient(**client_kwargs).remember(content, **kwargs)


def recall(query: str, **kwargs: Any) -> list[MemoryHit]:
    """Recall memory through a one-off default client."""

    client_kwargs = _client_kwargs(kwargs)
    return MemoryClient(**client_kwargs).recall(query, **kwargs)


def _client_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    client_keys = {
        "repo_root",
        "workspace_id",
        "user_id",
        "project_id",
        "session_id",
        "task_id",
        "subtask_id",
        "backend",
        "backend_instance",
    }
    client_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in client_keys}
    return client_kwargs
