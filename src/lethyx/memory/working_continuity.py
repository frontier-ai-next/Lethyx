"""Task-local working projections over exact immutable host events."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from collections.abc import Callable

from lethyx.contracts.hashing import hash_value

from lethyx.contracts import (
    ContinuationPacket,
    CandidateVerificationSubject,
    EvidenceResolver,
    VerificationReportRef,
    build_workspace_context,
    ContinuationSufficiencyAssessment,
    FoldedBranch,
    MemoryBlockRecord,
    RecordBlockMembership,
    WorkingStateCheckpoint,
    json_ready,
)
from lethyx.memory.backends.factory import create_durable_backend
from lethyx.memory.blocks.catalog import compile_semantic_catalog
from lethyx.memory.blocks.registry import load_memory_blocks
from lethyx.memory.catalog_store import WorkspaceCatalogStore
from lethyx.memory.kernel.models import MemoryWriteResult, block_record_to_event
from lethyx.memory.kernel.writer import _write_memory_event
from lethyx.memory.mutation import (
    block_id_for_canonical_key,
    plan_memory_mutation,
    resolve_canonical_revision_head,
)


PageBackTrigger = Literal[
    "uncertainty",
    "failure",
    "plan_transition",
    "source_transition",
    "state_transition",
    "missing_literal",
    "verifier_coverage_gap",
]
ContinuityStrategy = Literal[
    "no_compaction",
    "threshold_summary",
    "structured_checkpoint",
    "branch_fold",
    "adaptive_manager",
]
BranchReopenReason = Literal["unchanged", "new_evidence", "state_change"]
ContinuationRecoveryMode = Literal["last_known_good", "exact_recent_raw"]
_PAGE_BACK_TRIGGERS = frozenset(
    {
        "uncertainty",
        "failure",
        "plan_transition",
        "source_transition",
        "state_transition",
        "missing_literal",
        "verifier_coverage_gap",
    }
)
_CONTINUITY_STRATEGIES: tuple[ContinuityStrategy, ...] = (
    "no_compaction",
    "threshold_summary",
    "structured_checkpoint",
    "branch_fold",
    "adaptive_manager",
)


@dataclass(frozen=True)
class PageBackView:
    trigger: PageBackTrigger
    workspace_id: str
    session_id: str
    task_id: str
    source_event_sha256: str
    raw_refs: tuple[str, ...]
    records: tuple[MemoryBlockRecord, ...]


@dataclass(frozen=True)
class ContinuityTrial:
    strategy: ContinuityStrategy
    task_id: str
    budget_tokens: int
    tool_turn_count: int
    action_succeeded: bool
    repeated_work_turns: int
    reacquisition_turns: int
    expected_exact_literals: int
    survived_exact_literals: int
    tokens_used: int
    latency_samples_ms: tuple[float, ...]
    prefix_cache_eligible_tokens: int
    prefix_cache_reused_tokens: int
    page_back_count: int
    folded_branch_count: int
    folded_branches_with_raw_refs: int
    working_projection_advisory: bool
    semantic_truth_created: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", _continuity_strategy(self.strategy))
        object.__setattr__(self, "task_id", _required_identity(self.task_id, field_name="task_id"))
        for field_name in ("budget_tokens", "tool_turn_count", "expected_exact_literals"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in (
            "repeated_work_turns",
            "reacquisition_turns",
            "survived_exact_literals",
            "tokens_used",
            "prefix_cache_eligible_tokens",
            "prefix_cache_reused_tokens",
            "page_back_count",
            "folded_branch_count",
            "folded_branches_with_raw_refs",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in (
            "action_succeeded",
            "working_projection_advisory",
            "semantic_truth_created",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a bool")
        if self.tokens_used > self.budget_tokens:
            raise ValueError("tokens_used cannot exceed budget_tokens")
        if self.repeated_work_turns > self.tool_turn_count:
            raise ValueError("repeated_work_turns cannot exceed tool_turn_count")
        if self.reacquisition_turns > self.tool_turn_count:
            raise ValueError("reacquisition_turns cannot exceed tool_turn_count")
        if self.survived_exact_literals > self.expected_exact_literals:
            raise ValueError("survived_exact_literals cannot exceed expected_exact_literals")
        if self.prefix_cache_reused_tokens > self.prefix_cache_eligible_tokens:
            raise ValueError(
                "prefix_cache_reused_tokens cannot exceed prefix_cache_eligible_tokens"
            )
        if self.folded_branches_with_raw_refs > self.folded_branch_count:
            raise ValueError("folded raw-reference count cannot exceed folded branch count")
        if not isinstance(self.latency_samples_ms, (tuple, list)) or not self.latency_samples_ms:
            raise ValueError("latency_samples_ms must contain observed samples")
        samples: list[float] = []
        for sample in self.latency_samples_ms:
            if (
                not isinstance(sample, (int, float))
                or isinstance(sample, bool)
                or not math.isfinite(sample)
                or sample < 0
            ):
                raise ValueError("latency_samples_ms must contain finite non-negative values")
            samples.append(float(sample))
        object.__setattr__(self, "latency_samples_ms", tuple(samples))


@dataclass(frozen=True)
class ContinuityStrategyMetrics:
    strategy: ContinuityStrategy
    trial_count: int
    continuation_success_rate: float
    repeated_work_rate: float
    reacquisition_rate: float
    exact_literal_survival_rate: float
    tokens_used: int
    latency_p50_ms: float
    latency_p95_ms: float
    prefix_cache_reuse_rate: float
    page_back_frequency: float


@dataclass(frozen=True)
class ContinuityComparison:
    task_ids: tuple[str, ...]
    budget_tokens: tuple[int, ...]
    metrics: tuple[ContinuityStrategyMetrics, ...]
    adaptive_pareto_improves_threshold: bool
    safety_and_literal_gates_pass: bool

    def metric_for(self, strategy: str) -> ContinuityStrategyMetrics:
        selected = _continuity_strategy(strategy)
        return next(metric for metric in self.metrics if metric.strategy == selected)


@dataclass(frozen=True)
class ForcedRestartContinuationProof:
    tool_turn_count: int
    transcript_required: bool
    constraints_preserved: bool
    evidence_preserved: bool
    packet: ContinuationPacket


@dataclass(frozen=True)
class FoldedBranchReopenDecision:
    reopen: bool
    reason: BranchReopenReason
    previous_source_event_sha256: str
    observed_source_event_sha256: str
    previous_state_sha256: str
    observed_state_sha256: str


@dataclass(frozen=True)
class ContinuationRecovery:
    mode: ContinuationRecoveryMode
    packet: ContinuationPacket | None
    raw_view: PageBackView | None


def _required_identity(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not (text := value.strip()):
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _continuity_strategy(value: object) -> ContinuityStrategy:
    if value not in _CONTINUITY_STRATEGIES:
        raise ValueError("strategy must be one of the five continuity strategies")
    return cast(ContinuityStrategy, value)


def _sha256_reference(value: object, *, field_name: str) -> str:
    text = _required_identity(value, field_name=field_name)
    digest = text.removeprefix("sha256:")
    if not text.startswith("sha256:") or len(digest) != 64:
        raise ValueError(f"{field_name} must be a sha256 reference")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a sha256 reference") from exc
    return text


def working_state_canonical_key(
    *, workspace_id: str, session_id: str, task_id: str, profile_id: str, owner_generation: str
) -> str:
    """Return the one stable canonical key for a task's working projection."""

    scope = (
        _required_identity(workspace_id, field_name="workspace_id"),
        _required_identity(profile_id, field_name="profile_id"),
        _required_identity(owner_generation, field_name="owner_generation"),
        _required_identity(session_id, field_name="session_id"),
        _required_identity(task_id, field_name="task_id"),
    )
    digest = hashlib.sha256(
        json.dumps(scope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"{scope[0]}:task_state:working_continuity:{digest}"


def _referenced_ids(checkpoint: WorkingStateCheckpoint) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *checkpoint.source_event_refs,
                *checkpoint.task_anchor_refs,
                *checkpoint.system_anchor_refs,
                *checkpoint.exact_recent_tail_refs,
                *(ref for page in checkpoint.evidence_page_refs for ref in page),
                *checkpoint.transitive_raw_refs,
            )
        )
    )


def _load_raw_event(
    backend: Any,
    raw_id: str,
    *,
    checkpoint: WorkingStateCheckpoint,
    source_resolver: Callable[[str], MemoryBlockRecord] | None = None,
) -> MemoryBlockRecord:
    getter = source_resolver if source_resolver is not None else getattr(backend, "get_block", None)
    if not callable(getter):
        raise RuntimeError("selected backend cannot resolve exact raw references")
    record = getter(raw_id)
    if not isinstance(record, MemoryBlockRecord):
        raise ValueError(f"raw reference {raw_id!r} is missing")
    if record.block_id != raw_id:
        raise ValueError(f"raw reference {raw_id!r} did not resolve exactly")
    if record.workspace_id != checkpoint.workspace_id or record.profile_id != checkpoint.profile_id:
        raise ValueError(f"raw reference {raw_id!r} is outside the workspace")
    if record.session_id != checkpoint.session_id:
        raise ValueError(f"raw reference {raw_id!r} is outside the session")
    if record.task_id != checkpoint.task_id:
        raise ValueError(f"raw reference {raw_id!r} is outside the task")
    if not record.canonical_key or not record.revision_id:
        raise ValueError(f"raw reference {raw_id!r} lacks immutable canonical identity")
    record_kind = str(record.metadata.get("record_kind") or "").strip()
    host_event_id = str(
        record.metadata.get("host_event_id") or record.external_refs.get("host_event_id") or ""
    ).strip()
    if (
        record.producer != "host_observer"
        or not record_kind
        or not host_event_id
        or record_kind == "working_state_checkpoint"
        or record.revision_of
        or record.supersedes_ids
    ):
        raise ValueError(f"raw reference {raw_id!r} is not an immutable host event")
    return record


def continuation_window_hash(records: tuple[MemoryBlockRecord, ...]) -> str:
    manifest = [
        {
            "block_id": record.block_id,
            "canonical_key": record.canonical_key,
            "revision_id": record.revision_id,
            "content_sha256": hashlib.sha256(record.content.encode("utf-8")).hexdigest(),
        }
        for record in records
    ]
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def continuation_source_hash(record: MemoryBlockRecord) -> str:
    return "sha256:" + hash_value(
        {
            name: getattr(record, name)
            for name in (
                "block_id",
                "canonical_key",
                "revision_id",
                "workspace_id",
                "profile_id",
                "session_id",
                "task_id",
                "content",
            )
        }
    )


def compile_continuation_packet(
    checkpoint: WorkingStateCheckpoint,
    *,
    backend: Any,
    source_resolver: Callable[[str], MemoryBlockRecord] | None = None,
) -> ContinuationPacket:
    """Compile one structured packet after exact, task-local raw-reference checks."""

    if not isinstance(checkpoint, WorkingStateCheckpoint):
        raise TypeError("checkpoint must be a WorkingStateCheckpoint")
    records = {
        raw_id: _load_raw_event(
            backend, raw_id, checkpoint=checkpoint, source_resolver=source_resolver
        )
        for raw_id in _referenced_ids(checkpoint)
    }
    for raw_id, digest in checkpoint.source_ref_sha256:
        if digest != continuation_source_hash(records[raw_id]):
            raise ValueError("exact continuation source hash mismatch")
    source_event_sha256 = continuation_window_hash(
        tuple(records[ref] for ref in checkpoint.source_event_refs)
    )
    if source_event_sha256 != checkpoint.source_event_sha256:
        raise ValueError("source_event_sha256 mismatch for immutable host events")
    return _continuation_packet(checkpoint)


def checkpoint_verification_subject(
    checkpoint: WorkingStateCheckpoint,
) -> CandidateVerificationSubject:
    return CandidateVerificationSubject(
        workspace_id=checkpoint.workspace_id,
        profile_id=checkpoint.profile_id,
        memory_id="host-checkpoint:" + checkpoint.host_event_id,
        revision_id=checkpoint.artifact_revision or checkpoint.source_event_sha256,
        payload_hash=hash_value(json_ready(checkpoint)),
        source_evidence_hash=hash_value(checkpoint.source_ref_sha256),
        target_block_id=checkpoint.requested_block_id,
        authority="advisory",
        scope="task",
        scope_id=checkpoint.task_id,
    )


def _continuation_packet(checkpoint: WorkingStateCheckpoint) -> ContinuationPacket:
    return ContinuationPacket(
        workspace_id=checkpoint.workspace_id,
        profile_id=checkpoint.profile_id,
        owner_generation=checkpoint.owner_generation,
        session_id=checkpoint.session_id,
        task_id=checkpoint.task_id,
        checkpoint=checkpoint,
        source_event_range=checkpoint.source_event_refs,
        source_event_sha256=checkpoint.source_event_sha256,
        task_anchor_refs=checkpoint.task_anchor_refs,
        system_anchor_refs=checkpoint.system_anchor_refs,
        exact_recent_tail_refs=checkpoint.exact_recent_tail_refs,
        evidence_page_refs=checkpoint.evidence_page_refs,
        schema_version=checkpoint.schema_version,
        hard_policy_version=checkpoint.hard_policy_version,
        semantic_catalog_version=checkpoint.semantic_catalog_version,
        host_version=checkpoint.host_version,
        model_version=checkpoint.model_version,
        prompt_version=checkpoint.prompt_version,
        exact_literal_manifest=checkpoint.exact_literal_manifest,
        continuation_sufficiency=checkpoint.continuation_sufficiency,
        expires_at=checkpoint.valid_to,
        rollback_or_page_back_ref=checkpoint.rollback_or_page_back_ref,
    )


def page_back_continuation(
    checkpoint: WorkingStateCheckpoint,
    *,
    trigger: str,
    backend: Any,
    source_resolver: Callable[[str], MemoryBlockRecord] | None = None,
) -> PageBackView:
    """Return the exact task-local raw view for a closed set of recovery triggers."""

    if not isinstance(checkpoint, WorkingStateCheckpoint):
        raise TypeError("checkpoint must be a WorkingStateCheckpoint")
    if trigger not in _PAGE_BACK_TRIGGERS:
        raise ValueError("page-back trigger must identify a recovery or transition condition")
    packet = compile_continuation_packet(
        checkpoint, backend=backend, source_resolver=source_resolver
    )
    raw_refs = _referenced_ids(checkpoint)
    return PageBackView(
        trigger=cast(PageBackTrigger, trigger),
        workspace_id=checkpoint.workspace_id,
        session_id=checkpoint.session_id,
        task_id=checkpoint.task_id,
        source_event_sha256=packet.source_event_sha256,
        raw_refs=raw_refs,
        records=tuple(
            _load_raw_event(backend, raw_id, checkpoint=checkpoint, source_resolver=source_resolver)
            for raw_id in raw_refs
        ),
    )


def assess_continuation_sufficiency(
    checkpoint: WorkingStateCheckpoint,
    *,
    branch_id: str,
    backend: Any,
) -> ContinuationSufficiencyAssessment:
    """Verify exact continuation coverage before a branch may be folded."""

    if not isinstance(checkpoint, WorkingStateCheckpoint):
        raise TypeError("checkpoint must be a WorkingStateCheckpoint")
    branch_id = _required_identity(branch_id, field_name="branch_id")
    packet = compile_continuation_packet(checkpoint, backend=backend)
    return ContinuationSufficiencyAssessment(
        workspace_id=checkpoint.workspace_id,
        session_id=checkpoint.session_id,
        task_id=checkpoint.task_id,
        branch_id=branch_id,
        branch_completed=(
            branch_id in checkpoint.completed_branch_ids
            and branch_id not in checkpoint.open_branch_ids
        ),
        branch_isolated=branch_id in checkpoint.isolated_branch_ids,
        raw_page_back_verified=True,
        projection_sufficiency=checkpoint.continuation_sufficiency,
        source_event_sha256=packet.source_event_sha256,
        exact_literal_manifest=checkpoint.exact_literal_manifest,
        evidence_refs=checkpoint.evidence_refs,
    )


def fold_completed_branch(
    checkpoint: WorkingStateCheckpoint,
    *,
    branch_id: str,
    backend: Any,
    result: str,
    remaining_risks: tuple[str, ...] = (),
    raw_span_refs: tuple[str, ...] | None = None,
) -> FoldedBranch:
    """Create a folded projection only after deterministic sufficiency checks."""

    assessment = assess_continuation_sufficiency(
        checkpoint,
        branch_id=branch_id,
        backend=backend,
    )
    if not assessment.sufficient:
        raise ValueError("branch must be completed and isolated before fold")
    spans = checkpoint.source_event_refs if raw_span_refs is None else tuple(raw_span_refs)
    if not spans or any(ref not in checkpoint.source_event_refs for ref in spans):
        raise ValueError("raw_span_refs must be a non-empty subset of source_event_refs")
    return FoldedBranch(
        workspace_id=checkpoint.workspace_id,
        session_id=checkpoint.session_id,
        task_id=checkpoint.task_id,
        branch_id=assessment.branch_id,
        source_event_refs=checkpoint.source_event_refs,
        source_event_sha256=checkpoint.source_event_sha256,
        schema_version=checkpoint.schema_version,
        hard_policy_version=checkpoint.hard_policy_version,
        semantic_catalog_version=checkpoint.semantic_catalog_version,
        host_version=checkpoint.host_version,
        model_version=checkpoint.model_version,
        prompt_version=checkpoint.prompt_version,
        exact_literal_manifest=checkpoint.exact_literal_manifest,
        sufficiency_assessment=assessment,
        continuation_sufficiency="sufficient",
        expires_at=checkpoint.valid_to,
        rollback_or_page_back_ref=checkpoint.rollback_or_page_back_ref,
        result=result,
        remaining_risks=remaining_risks,
        raw_span_refs=spans,
        evidence_refs=checkpoint.evidence_refs,
    )


def _current_projection(
    backend: Any,
    *,
    canonical_key: str,
    workspace_id: str,
) -> MemoryBlockRecord | None:
    getter = getattr(backend, "get_block", None)
    if not callable(getter):
        raise RuntimeError("selected backend cannot resolve canonical working state")
    root = getter(block_id_for_canonical_key(canonical_key))
    if root is None:
        return None
    if not isinstance(root, MemoryBlockRecord):
        raise RuntimeError("selected backend returned an invalid working-state record")
    if root.workspace_id != workspace_id or root.canonical_key != canonical_key:
        raise ValueError("canonical working-state root is outside the requested scope")
    raw_edges_for = getattr(backend, "edges_for", None)
    return resolve_canonical_revision_head(
        root,
        canonical_key=canonical_key,
        workspace_id=workspace_id,
        get_block=getter,
        edges_for=raw_edges_for if callable(raw_edges_for) else None,
    )


def _established_anchor_refs(
    current: MemoryBlockRecord,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if (
        current.kind != "task_state"
        or current.status != "candidate"
        or current.authority != "advisory"
        or current.scope != "task"
        or current.metadata.get("record_kind") != "working_state_checkpoint"
        or current.metadata.get("task_local") is not True
    ):
        raise ValueError("stored working-state payload is not trustworthy")
    try:
        payload = json.loads(current.content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("stored working-state payload is not trustworthy") from exc
    checkpoint_payload = payload.get("checkpoint") if isinstance(payload, dict) else None
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("stored working-state payload is not trustworthy")
    expected_scope = {
        "workspace_id": current.workspace_id,
        "session_id": current.session_id,
        "task_id": current.task_id,
        "schema_version": current.schema_version,
    }
    if any(
        payload.get(field_name) != expected or checkpoint_payload.get(field_name) != expected
        for field_name, expected in expected_scope.items()
    ):
        raise ValueError("stored working-state payload is not trustworthy")
    established: list[tuple[str, ...]] = []
    for field_name in ("task_anchor_refs", "system_anchor_refs"):
        raw_refs = payload.get(field_name)
        checkpoint_refs = checkpoint_payload.get(field_name)
        if (
            not isinstance(raw_refs, list)
            or not raw_refs
            or raw_refs != checkpoint_refs
            or any(
                not isinstance(raw_ref, str) or not raw_ref.strip() or raw_ref != raw_ref.strip()
                for raw_ref in raw_refs
            )
            or len(set(raw_refs)) != len(raw_refs)
        ):
            raise ValueError("stored working-state payload is not trustworthy")
        established.append(tuple(raw_refs))
    return established[0], established[1]


def _reject_anchor_drift(
    current: MemoryBlockRecord | None,
    checkpoint: WorkingStateCheckpoint,
) -> None:
    if current is None:
        return
    task_anchor_refs, system_anchor_refs = _established_anchor_refs(current)
    if (
        task_anchor_refs != checkpoint.task_anchor_refs
        or system_anchor_refs != checkpoint.system_anchor_refs
    ):
        raise ValueError("working-state anchor drift is not allowed")


def commit_working_state_checkpoint(
    repo_root: str | Path,
    checkpoint: WorkingStateCheckpoint,
    *,
    backend: Any | None = None,
    source_resolver: Callable[[str], MemoryBlockRecord] | None = None,
    verification_report: VerificationReportRef | None = None,
) -> MemoryWriteResult:
    """Append one immutable candidate revision to the task's working projection."""

    if not isinstance(checkpoint, WorkingStateCheckpoint):
        raise TypeError("checkpoint must be a WorkingStateCheckpoint")
    selected = (
        backend
        if backend is not None
        else create_durable_backend(Path(repo_root), workspace_id=checkpoint.workspace_id)
    )
    if verification_report is None:
        packet = compile_continuation_packet(
            checkpoint, backend=selected, source_resolver=source_resolver
        )
    else:
        resolver = EvidenceResolver(
            build_workspace_context(
                repo_root=repo_root, workspace_root=repo_root, workspace_id=checkpoint.workspace_id
            )
        )
        issues = resolver.validate_verification_report_ref(verification_report)
        row = resolver.ledger_row(verification_report.report_id)
        result = row.get("result", {}) if row else {}
        semantic = {
            key: result.get(key)
            for key in (
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
            )
        }
        if (
            issues
            or not row
            or row.get("tool") != "candidate_semantic_verification"
            or row.get("actor") != "verifier"
            or result.get("subject") != json_ready(checkpoint_verification_subject(checkpoint))
            or result.get("semantic_subject_hash")
            != hash_value(json_ready(checkpoint_verification_subject(checkpoint)))
            or verification_report.report_hash != hash_value(semantic)
            or not isinstance(result.get("reasons"), list)
            or not result.get("reasons")
            or any(
                not isinstance(reason, str) or not reason.strip()
                for reason in result.get("reasons", ())
            )
            or result.get("proposer_model_version") != checkpoint.model_version
            or not result.get("verifier_model_version")
            or result.get("verifier_model_version") == checkpoint.model_version
            or any(
                result.get(key) is not True
                for key in (
                    "passed",
                    "coverage",
                    "preservation",
                    "faithfulness",
                    "instruction_safety",
                )
            )
        ):
            raise ValueError(
                "working checkpoint verification is not bound to the exact host projection"
            )
        packet = _continuation_packet(checkpoint)
    catalog_revision_id = compile_semantic_catalog(
        load_memory_blocks(
            Path(repo_root),
            workspace_id=checkpoint.workspace_id,
            profile_id=checkpoint.profile_id,
            strict=True,
        )
    ).revision_id
    target = next(
        (
            block
            for block in load_memory_blocks(
                Path(repo_root),
                workspace_id=checkpoint.workspace_id,
                profile_id=checkpoint.profile_id,
                strict=True,
            )
            if block.block_id == checkpoint.requested_block_id
        ),
        None,
    )
    if (
        target is None
        or target.child_ids
        or target.index_only
        or not target.agent_writable
        or target.scope != "task"
        or target.authority != "advisory"
    ):
        raise ValueError("working checkpoint requires a model-selected writable task leaf")
    store = WorkspaceCatalogStore(Path(repo_root), checkpoint.workspace_id, checkpoint.profile_id)
    try:
        target_state = store.get_block(checkpoint.requested_block_id)
        if target_state is None or not target_state.backend_partition_id:
            raise ValueError("working checkpoint requires a materialized native leaf partition")
        partition_id = target_state.backend_partition_id
    finally:
        store.close()
    canonical_key = working_state_canonical_key(
        workspace_id=checkpoint.workspace_id,
        session_id=checkpoint.session_id,
        task_id=checkpoint.task_id,
        profile_id=checkpoint.profile_id,
        owner_generation=checkpoint.owner_generation,
    )
    current = _current_projection(
        selected,
        canonical_key=canonical_key,
        workspace_id=checkpoint.workspace_id,
    )
    _reject_anchor_drift(current, checkpoint)
    packet_payload = json_ready(packet)
    source_payload: list[dict[str, object]] = [
        {
            "block_id": raw_id,
            "role": role,
        }
        for role, refs in (
            ("source_event", checkpoint.source_event_refs),
            ("task_anchor", checkpoint.task_anchor_refs),
            ("system_anchor", checkpoint.system_anchor_refs),
            ("recent_tail", checkpoint.exact_recent_tail_refs),
        )
        for raw_id in refs
    ]
    for page_number, page in enumerate(checkpoint.evidence_page_refs, start=1):
        source_payload.extend(
            {"block_id": raw_id, "role": "evidence_card", "page": page_number} for raw_id in page
        )
    record = MemoryBlockRecord(
        block_id="",
        canonical_key=canonical_key,
        block_membership=RecordBlockMembership(
            primary_block_id=checkpoint.requested_block_id,
            catalog_revision_id=catalog_revision_id,
            backend_partition_id=partition_id,
        ),
        content=json.dumps(
            packet_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        title=f"task-local working state: {checkpoint.task_id}",
        kind="task_state",
        status="candidate",
        retention="durable",
        delivery_policy="working_set",
        temperature="hot",
        authority="advisory",
        trust_level="agent_reported",
        scope="task",
        verification_status="unverified",
        promotion_decision="not_requested",
        workspace_id=checkpoint.workspace_id,
        profile_id=checkpoint.profile_id,
        run_id=checkpoint.task_id.split(":", 1)[0],
        stage_id="working_state",
        producer="agent",
        source_backend="lethyx",
        tags=("working-state-checkpoint", "task-local", "structured-projection"),
        evidence_refs=checkpoint.evidence_refs,
        external_refs={
            "host_event_id": checkpoint.host_event_id,
            "session_id": checkpoint.session_id,
            "task_id": checkpoint.task_id,
            "artifact_id": checkpoint.artifact_id,
            "artifact_revision": checkpoint.artifact_revision,
        },
        metadata={
            "record_kind": "working_state_checkpoint",
            "task_local": True,
            "working_set_owner": {
                "profile_id": checkpoint.profile_id,
                "session_id": checkpoint.session_id,
                "task_id": checkpoint.task_id,
                "generation": checkpoint.owner_generation,
            },
            "checkpoint_payload": json_ready(checkpoint),
            "source_event_range": list(packet.source_event_range),
            "source_event_sha256": packet.source_event_sha256,
            "raw_reference_manifest": source_payload,
        },
        session_id=checkpoint.session_id,
        task_id=checkpoint.task_id,
        parent_session_id=checkpoint.parent_session_id,
        artifact_id=checkpoint.artifact_id,
        artifact_revision=checkpoint.artifact_revision,
        valid_from=checkpoint.valid_from,
        valid_to=checkpoint.valid_to,
        project_id=checkpoint.project_id,
        repository_id=checkpoint.repository_id,
        checkout=checkpoint.checkout,
        worktree=checkpoint.worktree,
        branch=checkpoint.branch,
        revision=checkpoint.revision,
        dirty_state=checkpoint.dirty_state,
        schema_version=checkpoint.schema_version,
        hard_policy_version=checkpoint.hard_policy_version,
        semantic_catalog_version=catalog_revision_id,
    )
    initial = plan_memory_mutation(record, existing=None)
    decision = plan_memory_mutation(initial.record, existing=current)
    return _write_memory_event(
        repo_root,
        block_record_to_event(decision.record),
        workspace_id=checkpoint.workspace_id,
        backend_instance=selected,
        mutation=decision,
    )


def commit_host_compaction_projection(
    repo_root: str | Path,
    checkpoint: WorkingStateCheckpoint,
    *,
    backend: Any | None = None,
) -> MemoryWriteResult:
    """Commit a recursively safe host summary as an advisory working projection."""

    if not isinstance(checkpoint, WorkingStateCheckpoint):
        raise TypeError("checkpoint must be a WorkingStateCheckpoint")
    selected = (
        backend
        if backend is not None
        else create_durable_backend(Path(repo_root), workspace_id=checkpoint.workspace_id)
    )
    getter = getattr(selected, "get_block", None)
    if checkpoint.summary_source_refs and not callable(getter):
        raise RuntimeError("selected backend cannot resolve prior working projections")
    for projection_ref in checkpoint.summary_source_refs:
        if not callable(getter):
            raise RuntimeError("selected backend cannot resolve prior working projections")
        record = getter(projection_ref)
        if (
            not isinstance(record, MemoryBlockRecord)
            or record.block_id != projection_ref
            or record.workspace_id != checkpoint.workspace_id
            or record.profile_id != checkpoint.profile_id
            or record.metadata.get("working_set_owner", {}).get("generation")
            != checkpoint.owner_generation
            or record.session_id != checkpoint.session_id
            or record.task_id != checkpoint.task_id
            or record.kind != "task_state"
            or record.status != "candidate"
            or record.authority != "advisory"
            or record.scope != "task"
            or record.metadata.get("record_kind") != "working_state_checkpoint"
        ):
            raise ValueError("summary_source_refs must resolve to a prior working projection")
    return commit_working_state_checkpoint(repo_root, checkpoint, backend=selected)


def load_working_state_history(
    backend: Any,
    *,
    workspace_id: str,
    session_id: str,
    task_id: str,
    profile_id: str,
    owner_generation: str,
) -> tuple[MemoryBlockRecord, ...]:
    """Load only the canonical immutable revisions for one task projection."""

    canonical_key = working_state_canonical_key(
        workspace_id=workspace_id,
        session_id=session_id,
        task_id=task_id,
        profile_id=profile_id,
        owner_generation=owner_generation,
    )
    history_loader = getattr(backend, "canonical_history", None)
    if not callable(history_loader):
        raise RuntimeError("selected backend cannot load canonical working-state history")
    records = tuple(history_loader(canonical_key=canonical_key, workspace_id=workspace_id) or ())
    for record in records:
        if not isinstance(record, MemoryBlockRecord):
            raise RuntimeError("selected backend returned invalid working-state history")
        if (
            record.canonical_key != canonical_key
            or record.workspace_id != workspace_id
            or record.profile_id != profile_id
            or record.metadata.get("working_set_owner", {}).get("generation") != owner_generation
            or record.session_id != session_id
            or record.task_id != task_id
        ):
            raise ValueError("canonical working-state history escaped task scope")
        if (
            record.kind != "task_state"
            or record.status != "candidate"
            or record.authority != "advisory"
            or record.scope != "task"
        ):
            raise ValueError("canonical working-state history changed authority")
    return records


def _nearest_rank(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _strategy_metrics(
    strategy: ContinuityStrategy,
    trials: tuple[ContinuityTrial, ...],
) -> ContinuityStrategyMetrics:
    turns = sum(trial.tool_turn_count for trial in trials)
    expected_literals = sum(trial.expected_exact_literals for trial in trials)
    cache_eligible = sum(trial.prefix_cache_eligible_tokens for trial in trials)
    latencies = [sample for trial in trials for sample in trial.latency_samples_ms]
    return ContinuityStrategyMetrics(
        strategy=strategy,
        trial_count=len(trials),
        continuation_success_rate=sum(trial.action_succeeded for trial in trials) / len(trials),
        repeated_work_rate=sum(trial.repeated_work_turns for trial in trials) / turns,
        reacquisition_rate=sum(trial.reacquisition_turns for trial in trials) / turns,
        exact_literal_survival_rate=(
            sum(trial.survived_exact_literals for trial in trials) / expected_literals
        ),
        tokens_used=sum(trial.tokens_used for trial in trials),
        latency_p50_ms=_nearest_rank(latencies, 0.50),
        latency_p95_ms=_nearest_rank(latencies, 0.95),
        prefix_cache_reuse_rate=(
            sum(trial.prefix_cache_reused_tokens for trial in trials) / cache_eligible
            if cache_eligible
            else 0.0
        ),
        page_back_frequency=sum(trial.page_back_count for trial in trials) / turns,
    )


def compare_continuity_strategies(
    trials: tuple[ContinuityTrial, ...],
) -> ContinuityComparison:
    """Compare the closed strategy matrix over identical tasks and token budgets."""

    if not isinstance(trials, (tuple, list)) or not trials:
        raise ValueError("comparison requires observed continuity trials")
    by_task: dict[str, dict[ContinuityStrategy, ContinuityTrial]] = {}
    for trial in trials:
        if not isinstance(trial, ContinuityTrial):
            raise TypeError("trials must contain ContinuityTrial values")
        task_trials = by_task.setdefault(trial.task_id, {})
        if trial.strategy in task_trials:
            raise ValueError("comparison contains a duplicate task/strategy trial")
        task_trials[trial.strategy] = trial
    expected = set(_CONTINUITY_STRATEGIES)
    for task_id, task_trials in by_task.items():
        if set(task_trials) != expected:
            raise ValueError(f"task {task_id!r} must include all five continuity strategies")
        if len({trial.budget_tokens for trial in task_trials.values()}) != 1:
            raise ValueError(f"task {task_id!r} strategies must use the same token budget")

    metrics = tuple(
        _strategy_metrics(
            strategy,
            tuple(by_task[task_id][strategy] for task_id in sorted(by_task)),
        )
        for strategy in _CONTINUITY_STRATEGIES
    )
    metric_by_strategy = {metric.strategy: metric for metric in metrics}
    safety_and_literals = all(
        trial.working_projection_advisory
        and not trial.semantic_truth_created
        and trial.folded_branches_with_raw_refs == trial.folded_branch_count
        and trial.survived_exact_literals == trial.expected_exact_literals
        for trial in trials
    )
    adaptive = metric_by_strategy["adaptive_manager"]
    threshold = metric_by_strategy["threshold_summary"]
    pareto = (
        safety_and_literals
        and adaptive.continuation_success_rate >= threshold.continuation_success_rate
        and adaptive.tokens_used <= threshold.tokens_used
        and (
            adaptive.continuation_success_rate > threshold.continuation_success_rate
            or adaptive.tokens_used < threshold.tokens_used
        )
        and adaptive.exact_literal_survival_rate >= threshold.exact_literal_survival_rate
    )
    task_ids = tuple(sorted(by_task))
    return ContinuityComparison(
        task_ids=task_ids,
        budget_tokens=tuple(
            by_task[task_id][_CONTINUITY_STRATEGIES[0]].budget_tokens for task_id in task_ids
        ),
        metrics=metrics,
        adaptive_pareto_improves_threshold=pareto,
        safety_and_literal_gates_pass=safety_and_literals,
    )


def verify_forced_restart_continuation(
    checkpoint: WorkingStateCheckpoint,
    *,
    backend: Any,
    tool_turn_count: int,
    required_literals: tuple[str, ...],
    required_evidence_ref_ids: tuple[str, ...],
) -> ForcedRestartContinuationProof:
    """Verify transcript-free restart coverage for the Phase 3.5 turn window."""

    if (
        not isinstance(tool_turn_count, int)
        or isinstance(tool_turn_count, bool)
        or not 100 <= tool_turn_count <= 500
    ):
        raise ValueError("tool_turn_count must be between 100 and 500")
    packet = compile_continuation_packet(checkpoint, backend=backend)
    literals = tuple(
        _required_identity(value, field_name="required_literals") for value in required_literals
    )
    evidence_ids = tuple(
        _required_identity(value, field_name="required_evidence_ref_ids")
        for value in required_evidence_ref_ids
    )
    if not literals:
        raise ValueError("required_literals must declare mandatory continuation constraints")
    if not evidence_ids:
        raise ValueError("required_evidence_ref_ids must declare mandatory continuation evidence")
    available_literals = {
        literal
        for literal_set in checkpoint.exact_literal_manifest
        for literal in literal_set.literals
    }
    missing_literals = tuple(value for value in literals if value not in available_literals)
    if missing_literals:
        raise ValueError(f"required continuation literals are missing: {missing_literals!r}")
    available_evidence = {reference.ref_id for reference in checkpoint.evidence_refs}
    missing_evidence = tuple(value for value in evidence_ids if value not in available_evidence)
    if missing_evidence:
        raise ValueError(f"required continuation evidence is missing: {missing_evidence!r}")
    return ForcedRestartContinuationProof(
        tool_turn_count=tool_turn_count,
        transcript_required=False,
        constraints_preserved=True,
        evidence_preserved=True,
        packet=packet,
    )


def evaluate_folded_branch_reopen(
    folded: FoldedBranch,
    *,
    observed_source_event_sha256: str,
    previous_state_sha256: str,
    observed_state_sha256: str,
) -> FoldedBranchReopenDecision:
    """Trace whether a completed branch may reopen for new evidence or state."""

    if not isinstance(folded, FoldedBranch):
        raise TypeError("folded must be a FoldedBranch")
    observed_source = _sha256_reference(
        observed_source_event_sha256,
        field_name="observed_source_event_sha256",
    )
    previous_state = _sha256_reference(previous_state_sha256, field_name="previous_state_sha256")
    observed_state = _sha256_reference(observed_state_sha256, field_name="observed_state_sha256")
    if observed_source != folded.source_event_sha256:
        reason: BranchReopenReason = "new_evidence"
    elif observed_state != previous_state:
        reason = "state_change"
    else:
        reason = "unchanged"
    return FoldedBranchReopenDecision(
        reopen=reason != "unchanged",
        reason=reason,
        previous_source_event_sha256=folded.source_event_sha256,
        observed_source_event_sha256=observed_source,
        previous_state_sha256=previous_state,
        observed_state_sha256=observed_state,
    )


def recover_continuation_after_failure(
    checkpoint: WorkingStateCheckpoint,
    *,
    backend: Any,
    last_known_good: WorkingStateCheckpoint | None,
) -> ContinuationRecovery:
    """Return a verified checkpoint or exact raw view after verifier failure."""

    if last_known_good is not None:
        if not isinstance(last_known_good, WorkingStateCheckpoint):
            raise TypeError("last_known_good must be a WorkingStateCheckpoint")
        if (
            last_known_good.workspace_id,
            last_known_good.session_id,
            last_known_good.task_id,
        ) != (checkpoint.workspace_id, checkpoint.session_id, checkpoint.task_id):
            raise ValueError("last_known_good checkpoint is outside the failed task scope")
        return ContinuationRecovery(
            mode="last_known_good",
            packet=compile_continuation_packet(last_known_good, backend=backend),
            raw_view=None,
        )
    return ContinuationRecovery(
        mode="exact_recent_raw",
        packet=None,
        raw_view=page_back_continuation(checkpoint, trigger="failure", backend=backend),
    )
