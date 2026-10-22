"""Deterministic reconstructive-lite planning over canonical indexes and edges."""

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

from um_mem.contracts import EvidenceRef, RecallFrame
from um_mem.memory.backends.contracts import EdgeType, MemoryEdgeRecord, MemoryIndexNode
from um_mem.memory.operational_slo import (
    DEFAULT_RECONSTRUCTIVE_RECALL_BUDGET,
    RecallBudget,
)
from um_mem.memory.query_analysis import matches_evidence_literal
from um_mem.memory.recall_plan import RecallPlan
from um_mem.memory.reconstruction_qualification import ReconstructionProductQualification

ReconstructionOperator = Literal["cue", "entity", "path", "time", "decision", "procedure"]
_OPERATOR_ORDER: tuple[ReconstructionOperator, ...] = (
    "cue",
    "entity",
    "path",
    "time",
    "decision",
    "procedure",
)
_DECISION_EDGE_TYPES = frozenset(
    {EdgeType.IMPLEMENTS, EdgeType.SUPPORTS, EdgeType.VERIFIED_BY, EdgeType.SUPERSEDES}
)
_PROCEDURE_EDGE_TYPES = frozenset(
    {EdgeType.APPLIED_REFLECTION, EdgeType.DERIVED_FROM, EdgeType.TESTS}
)


@dataclass(frozen=True)
class PinnedRecallFrame:
    frame: RecallFrame
    canonical_snapshot_revision: str
    catalog_revision: str
    policy_revision: str
    frame_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame, RecallFrame):
            raise TypeError("frame must be a RecallFrame")
        for name in (
            "canonical_snapshot_revision",
            "catalog_revision",
            "policy_revision",
        ):
            _required(getattr(self, name), field_name=name)
        expected = _frame_revision(
            self.frame,
            canonical_snapshot_revision=self.canonical_snapshot_revision,
            catalog_revision=self.catalog_revision,
            policy_revision=self.policy_revision,
        )
        if self.frame_revision != expected:
            raise ValueError("pinned recall frame revision does not match its snapshot")

    def reinstated_context(self) -> dict[str, object]:
        """Return the complete typed context restored for every operator step."""

        return {
            "episode_id": self.frame.episode_id,
            "task_id": self.frame.task_id,
            "subgoal_id": self.frame.subgoal_id,
            "subtask_id": self.frame.subtask_id,
            "repository": {
                "repository_id": self.frame.repository_id,
                "worktree": self.frame.worktree,
                "revision_id": self.frame.revision_id,
                "dirty_state": self.frame.dirty_state,
            },
            "actor": {
                "actor_id": self.frame.actor_id,
                "speaker_id": self.frame.speaker_id,
                "belief_holder_id": self.frame.belief_holder_id,
            },
            "tool_state": {
                "tool_name": self.frame.tool_name,
                "tool_state": self.frame.tool_state,
                "environment_state": self.frame.environment_state,
            },
            "temporal": {
                "event_time": self.frame.event_time,
                "mention_time": self.frame.mention_time,
                "observation_time": self.frame.observation_time,
                "query_time": self.frame.query_time,
                "mode": self.frame.desired_temporal_mode,
            },
        }


def pin_recall_frame(
    frame: RecallFrame,
    *,
    canonical_snapshot_revision: str,
    catalog_revision: str,
    policy_revision: str,
) -> PinnedRecallFrame:
    if not isinstance(frame, RecallFrame):
        raise TypeError("frame must be a RecallFrame")
    return PinnedRecallFrame(
        frame=frame,
        canonical_snapshot_revision=_required(
            canonical_snapshot_revision,
            field_name="canonical_snapshot_revision",
        ),
        catalog_revision=_required(catalog_revision, field_name="catalog_revision"),
        policy_revision=_required(policy_revision, field_name="policy_revision"),
        frame_revision=_frame_revision(
            frame,
            canonical_snapshot_revision=canonical_snapshot_revision,
            catalog_revision=catalog_revision,
            policy_revision=policy_revision,
        ),
    )


@dataclass(frozen=True)
class ReconstructionStep:
    ordinal: int
    operator: ReconstructionOperator
    plan: RecallPlan
    candidate_ids: tuple[str, ...]
    reinstated_context_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 1:
            raise ValueError("reconstruction step ordinal must be positive")
        if self.operator not in _OPERATOR_ORDER:
            raise ValueError("unknown reconstruction operator")
        if not isinstance(self.plan, RecallPlan):
            raise TypeError("reconstruction step plan must be a RecallPlan")
        object.__setattr__(self, "candidate_ids", _identifiers(self.candidate_ids))
        object.__setattr__(
            self,
            "reinstated_context_fields",
            _identifiers(self.reinstated_context_fields),
        )


@dataclass(frozen=True)
class ReconstructiveLitePlan:
    frame: PinnedRecallFrame
    base_plan: RecallPlan
    steps: tuple[ReconstructionStep, ...]
    strategy: Literal["reconstructive_lite"] = "reconstructive_lite"
    backend_path: Literal["selected_backend"] = "selected_backend"

    def __post_init__(self) -> None:
        if not isinstance(self.frame, PinnedRecallFrame):
            raise TypeError("frame must be a PinnedRecallFrame")
        if not isinstance(self.base_plan, RecallPlan):
            raise TypeError("base_plan must be a RecallPlan")
        if not self.steps or len(self.steps) > len(_OPERATOR_ORDER):
            raise ValueError("reconstructive-lite plan must contain one to six steps")
        if tuple(step.ordinal for step in self.steps) != tuple(range(1, len(self.steps) + 1)):
            raise ValueError("reconstruction step ordinals must be contiguous")
        operator_positions = [_OPERATOR_ORDER.index(step.operator) for step in self.steps]
        if operator_positions != sorted(set(operator_positions)):
            raise ValueError("reconstruction operators must be unique and ordered")
        for step in self.steps:
            if (
                step.plan.workspace_id != self.base_plan.workspace_id
                or step.plan.run_id != self.base_plan.run_id
                or step.plan.scope_chain != self.base_plan.scope_chain
                or step.plan.scope_chain != self.base_plan.scope_chain
            ):
                raise ValueError("reconstruction operator changed the pinned recall scope")

    @property
    def operators(self) -> tuple[ReconstructionOperator, ...]:
        return tuple(step.operator for step in self.steps)


def compile_reconstructive_lite_plan(
    base_plan: RecallPlan,
    *,
    frame: PinnedRecallFrame,
    index_nodes: tuple[MemoryIndexNode, ...] = (),
    edges: tuple[MemoryEdgeRecord, ...] = (),
) -> ReconstructiveLitePlan:
    """Compile bounded operator plans without creating or selecting another backend."""

    if not isinstance(base_plan, RecallPlan):
        raise TypeError("base_plan must be a RecallPlan")
    if not isinstance(frame, PinnedRecallFrame):
        raise TypeError("frame must be a PinnedRecallFrame")
    _validate_frame_plan(frame.frame, base_plan)
    nodes = tuple(
        sorted(
            (
                node
                for node in index_nodes
                if isinstance(node, MemoryIndexNode)
                and node.workspace_id in {"", base_plan.workspace_id}
            ),
            key=lambda item: item.node_id,
        )
    )
    if len(nodes) != len(index_nodes) and any(
        not isinstance(node, MemoryIndexNode) for node in index_nodes
    ):
        raise TypeError("index_nodes must contain MemoryIndexNode values")
    typed_edges = tuple(
        sorted(
            (
                edge
                for edge in edges
                if isinstance(edge, MemoryEdgeRecord)
                and edge.workspace_id in {"", base_plan.workspace_id}
            ),
            key=lambda item: (item.edge_type, item.src_id, item.dst_id),
        )
    )
    if any(not isinstance(edge, MemoryEdgeRecord) for edge in edges):
        raise TypeError("edges must contain MemoryEdgeRecord values")
    nodes = _qualify_contextual_nodes(nodes, frame.frame, typed_edges)
    known_ids = frozenset(record_id for node in nodes if (record_id := _node_record_id(node)))
    entities = frame.frame.relevant_entities
    paths = frame.frame.relevant_paths
    symbols = frame.frame.relevant_symbols
    context_fields = tuple(frame.reinstated_context())
    steps: list[ReconstructionStep] = []

    def add(
        operator: ReconstructionOperator,
        plan: RecallPlan,
        candidates: tuple[str, ...],
    ) -> None:
        steps.append(
            ReconstructionStep(
                ordinal=len(steps) + 1,
                operator=operator,
                plan=plan,
                candidate_ids=candidates,
                reinstated_context_fields=context_fields,
            )
        )

    # The cue step deliberately delegates ordinary semantic search to the
    # selected backend. Deep-read planning must not recreate a lexical router.
    add("cue", base_plan, ())

    if entities:
        entity_ids = _matching_exact_node_ids(nodes, entities)
        add(
            "entity",
            _candidate_plan(base_plan.with_overrides(entities=entities), entity_ids),
            entity_ids,
        )
    if paths or symbols:
        path_ids = _matching_exact_node_ids(nodes, (*paths, *symbols))
        add(
            "path",
            _candidate_plan(
                base_plan.with_overrides(paths=paths, symbols=symbols),
                path_ids,
            ),
            path_ids,
        )
    temporal_value = next(
        (
            value
            for value in (
                frame.frame.query_time,
                frame.frame.observation_time,
                frame.frame.event_time,
                frame.frame.mention_time,
            )
            if value
        ),
        "",
    )
    if temporal_value or frame.frame.desired_temporal_mode != "current":
        temporal_ids = _node_kind_ids(nodes, {"temporal", "episode", "history"})
        add(
            "time",
            _candidate_plan(
                base_plan.with_overrides(
                    temporal_mode=frame.frame.desired_temporal_mode,
                    as_of=temporal_value,
                ),
                temporal_ids,
            ),
            temporal_ids,
        )
    decision_requested = (
        base_plan.query_type in {"synthesis", "task_action"}
        or "decision" in frame.frame.required_evidence_roles
    )
    if decision_requested:
        decision_ids = _typed_candidate_ids(
            nodes,
            typed_edges,
            known_ids,
            node_kinds={"decision"},
            edge_types=_DECISION_EDGE_TYPES,
        )
        roles = tuple(dict.fromkeys((*base_plan.evidence_roles, "decision")))
        add(
            "decision",
            _candidate_plan(base_plan.with_overrides(evidence_roles=roles), decision_ids),
            decision_ids,
        )
    procedure_requested = "procedure" in frame.frame.expected_source_types
    if procedure_requested:
        procedure_ids = _typed_candidate_ids(
            nodes,
            typed_edges,
            known_ids,
            node_kinds={"procedure"},
            edge_types=_PROCEDURE_EDGE_TYPES,
        )
        add(
            "procedure",
            _candidate_plan(
                base_plan.with_overrides(tags=tuple(dict.fromkeys((*base_plan.tags, "procedure")))),
                procedure_ids,
            ),
            procedure_ids,
        )
    return ReconstructiveLitePlan(frame=frame, base_plan=base_plan, steps=tuple(steps))


def _candidate_plan(plan: RecallPlan, candidates: tuple[str, ...]) -> RecallPlan:
    return plan.with_overrides(
        record_ids=candidates or plan.record_ids,
        limit=min(plan.limit, max(1, len(candidates))) if candidates else plan.limit,
    )


def _typed_candidate_ids(
    nodes: tuple[MemoryIndexNode, ...],
    edges: tuple[MemoryEdgeRecord, ...],
    known_ids: frozenset[str],
    *,
    node_kinds: set[str],
    edge_types: frozenset[str],
) -> tuple[str, ...]:
    selected = set(_node_kind_ids(nodes, node_kinds))
    for edge in edges:
        if edge.edge_type in edge_types:
            selected.update({edge.src_id, edge.dst_id}.intersection(known_ids))
    return tuple(sorted(selected))


def _matching_exact_node_ids(
    nodes: tuple[MemoryIndexNode, ...],
    values: tuple[str, ...],
) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(
            str(value).strip().replace("\\", "/").casefold()
            for value in values
            if str(value).strip()
        )
    )
    if not normalized:
        return ()

    def matches(node: MemoryIndexNode) -> bool:
        metadata_values = (
            node.metadata.get("term_value"),
            node.metadata.get("structure_value"),
        )
        searchable = tuple(
            str(value).replace("\\", "/").casefold()
            for value in (node.title, *metadata_values)
            if value
        )
        return any(
            matches_evidence_literal(candidate, value)
            for value in normalized
            for candidate in searchable
        )

    return tuple(
        sorted(
            {record_id for node in nodes if matches(node) and (record_id := _node_record_id(node))}
        )
    )


def _node_kind_ids(
    nodes: tuple[MemoryIndexNode, ...],
    kinds: set[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                record_id
                for node in nodes
                if (str(node.metadata.get("kind") or node.node_type).strip().casefold() in kinds)
                and (record_id := _node_record_id(node))
            }
        )
    )


def _node_record_id(node: MemoryIndexNode) -> str:
    for field_name in ("record_id", "block_id"):
        value = str(node.metadata.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _qualify_contextual_nodes(
    nodes: tuple[MemoryIndexNode, ...],
    frame: RecallFrame,
    edges: tuple[MemoryEdgeRecord, ...],
) -> tuple[MemoryIndexNode, ...]:
    """Prevent a context-labelled distractor from replacing available current evidence."""

    axes = {
        "episode_id": frame.episode_id,
        "revision_id": frame.revision_id,
        "tool_state": frame.tool_state,
        "environment_state": frame.environment_state,
    }
    active_axes = {key: value for key, value in axes.items() if value}
    if not active_axes:
        return nodes

    def context_state(node: MemoryIndexNode) -> str:
        labelled = False
        for key, expected in active_axes.items():
            observed = str(node.metadata.get(key) or "").strip()
            if not observed:
                continue
            labelled = True
            if observed != expected:
                return "conflict"
        return "current" if labelled else "unlabelled"

    current_ids = {_node_record_id(node) for node in nodes if context_state(node) == "current"}
    current_ids.discard("")
    if not current_ids:
        return nodes
    supported_conflicts: set[str] = set()
    contextual_edge_types = {
        EdgeType.DERIVED_FROM,
        EdgeType.SUPPORTS,
        EdgeType.VERIFIED_BY,
    }
    for edge in edges:
        if edge.edge_type not in contextual_edge_types or not edge.evidence_refs:
            continue
        if edge.src_id in current_ids:
            supported_conflicts.add(edge.dst_id)
        if edge.dst_id in current_ids:
            supported_conflicts.add(edge.src_id)
    return tuple(
        node
        for node in nodes
        if context_state(node) != "conflict" or _node_record_id(node) in supported_conflicts
    )


def _validate_frame_plan(frame: RecallFrame, plan: RecallPlan) -> None:
    if frame.query.casefold().strip() != plan.query.casefold().strip():
        raise ValueError("pinned RecallFrame query does not match the recall plan")
    if frame.workspace_id and frame.workspace_id != plan.workspace_id:
        raise ValueError("pinned RecallFrame workspace does not match the recall plan")
    if frame.run_id and frame.run_id != plan.run_id:
        raise ValueError("pinned RecallFrame run does not match the recall plan")
    if frame.scope_chain and tuple(frame.scope_chain) != plan.scope_chain:
        raise ValueError("pinned RecallFrame scope chain does not match the recall plan")


def _frame_revision(
    frame: RecallFrame,
    *,
    canonical_snapshot_revision: str,
    catalog_revision: str,
    policy_revision: str,
) -> str:
    payload = json.dumps(
        {
            "frame": asdict(frame),
            "canonical_snapshot_revision": _required(
                canonical_snapshot_revision,
                field_name="canonical_snapshot_revision",
            ),
            "catalog_revision": _required(catalog_revision, field_name="catalog_revision"),
            "policy_revision": _required(policy_revision, field_name="policy_revision"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identifiers(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_required(value, field_name="identifier") for value in values))


def _required(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 1_024:
        raise ValueError(f"{field_name} must be a bounded non-empty identifier")
    return text


@dataclass(frozen=True)
class RestrictedOperatorPolicyRequest:
    """Minimal, non-authoritative input exposed to an optional LLM policy."""

    frame_revision: str
    allowed_operators: tuple[ReconstructionOperator, ...]
    attempted_operators: tuple[ReconstructionOperator, ...]
    trigger_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _required(self.frame_revision, field_name="frame_revision")
        if not self.allowed_operators or any(
            value not in _OPERATOR_ORDER for value in self.allowed_operators
        ):
            raise ValueError("allowed_operators must contain declared reconstruction operators")
        if any(value not in self.allowed_operators for value in self.attempted_operators):
            raise ValueError("attempted operators must be part of the allowed plan")
        object.__setattr__(self, "trigger_reasons", _identifiers(self.trigger_reasons))


@dataclass(frozen=True)
class RestrictedOperatorSelection:
    step: ReconstructionStep | None
    source: Literal["deterministic", "llm_policy", "exhausted"]
    rejected_policy_value: str = ""
    authority_change_allowed: Literal[False] = False
    scope_change_allowed: Literal[False] = False
    promotion_allowed: Literal[False] = False

    def __post_init__(self) -> None:
        if self.step is not None and not isinstance(self.step, ReconstructionStep):
            raise TypeError("selected operator step must be typed")
        if self.source not in {"deterministic", "llm_policy", "exhausted"}:
            raise ValueError("operator selection source is invalid")
        if self.source == "exhausted" and self.step is not None:
            raise ValueError("exhausted operator selection cannot contain a step")
        if (
            self.authority_change_allowed is not False
            or self.scope_change_allowed is not False
            or self.promotion_allowed is not False
        ):
            raise ValueError("operator policy cannot change authority, scope or promotion")


RestrictedOperatorPolicy = Callable[[RestrictedOperatorPolicyRequest], object]


def select_reconstruction_operator(
    plan: ReconstructiveLitePlan,
    *,
    trigger_reasons: tuple[str, ...],
    attempted_operators: tuple[ReconstructionOperator, ...] = (),
    policy: RestrictedOperatorPolicy | None = None,
) -> RestrictedOperatorSelection:
    """Let an optional untrusted policy select only an existing operator step."""

    if not isinstance(plan, ReconstructiveLitePlan):
        raise TypeError("plan must be a ReconstructiveLitePlan")
    attempted = tuple(attempted_operators)
    if any(value not in plan.operators for value in attempted):
        raise ValueError("attempted operators must be part of the compiled plan")
    available = tuple(step for step in plan.steps if step.operator not in attempted)
    if not available:
        return RestrictedOperatorSelection(None, "exhausted")
    if policy is None:
        return RestrictedOperatorSelection(available[0], "deterministic")
    if not callable(policy):
        raise TypeError("policy must be callable")
    request = RestrictedOperatorPolicyRequest(
        frame_revision=plan.frame.frame_revision,
        allowed_operators=plan.operators,
        attempted_operators=attempted,
        trigger_reasons=trigger_reasons,
    )
    raw = policy(request)
    selected = str(raw or "").strip().casefold()
    for step in available:
        if step.operator == selected:
            return RestrictedOperatorSelection(step, "llm_policy")
    return RestrictedOperatorSelection(
        available[0],
        "deterministic",
        rejected_policy_value=selected[:128],
    )


ReconstructionRelationKind = Literal[
    "reported_dependency",
    "temporal_precondition",
    "semantic_association",
    "verified_dependency",
]
_RELATION_KINDS = frozenset(
    {
        "reported_dependency",
        "temporal_precondition",
        "semantic_association",
        "verified_dependency",
    }
)


@dataclass(frozen=True)
class ReconstructionRelation:
    src_id: str
    dst_id: str
    relation: ReconstructionRelationKind
    source_refs: tuple[EvidenceRef, ...]
    verifier_refs: tuple[EvidenceRef, ...] = ()
    origin: Literal["observed", "deterministic", "llm_extracted"] = "observed"
    surface: Literal["supplemental_evidence"] = "supplemental_evidence"

    def __post_init__(self) -> None:
        _required(self.src_id, field_name="src_id")
        _required(self.dst_id, field_name="dst_id")
        if self.relation not in _RELATION_KINDS:
            raise ValueError("reconstruction relation kind is invalid")
        if self.origin not in {"observed", "deterministic", "llm_extracted"}:
            raise ValueError("reconstruction relation origin is invalid")
        if not self.source_refs or any(
            not isinstance(ref, EvidenceRef) for ref in self.source_refs
        ):
            raise ValueError("reconstruction relations require typed source refs")
        if any(not isinstance(ref, EvidenceRef) for ref in self.verifier_refs):
            raise ValueError("verifier refs must be typed evidence refs")
        if self.relation == "verified_dependency" and not self.verifier_refs:
            raise ValueError("verified_dependency requires verifier refs")
        if self.origin == "llm_extracted" and self.relation == "verified_dependency":
            raise ValueError("LLM extraction alone cannot create a verified dependency")
        if self.surface != "supplemental_evidence":
            raise ValueError("reconstruction relations remain supplemental evidence")


def reconstruction_relation(
    *,
    src_id: str,
    dst_id: str,
    requested_relation: str,
    source_refs: tuple[EvidenceRef, ...],
    verifier_refs: tuple[EvidenceRef, ...] = (),
    origin: Literal["observed", "deterministic", "llm_extracted"] = "observed",
) -> ReconstructionRelation:
    """Classify a relation without promoting an LLM association to causality."""

    normalized = str(requested_relation or "").strip().casefold()
    if origin == "llm_extracted" and normalized not in {
        "reported_dependency",
        "temporal_precondition",
        "semantic_association",
    }:
        normalized = "semantic_association"
    if normalized not in _RELATION_KINDS:
        raise ValueError("requested reconstruction relation is invalid")
    return ReconstructionRelation(
        src_id=src_id,
        dst_id=dst_id,
        relation=cast(ReconstructionRelationKind, normalized),
        source_refs=source_refs,
        verifier_refs=verifier_refs,
        origin=origin,
    )


@dataclass(frozen=True)
class ReconstructionEscalationDecision:
    escalate: bool
    reasons: tuple[str, ...]
    one_hop_fallback_required: bool = True


def decide_reconstruction_escalation(
    plan: RecallPlan,
    *,
    frame: PinnedRecallFrame,
    temporal_ambiguity: bool = False,
    conflict_count: int = 0,
    minimum_coverage: float = 0.70,
    product_qualification: ReconstructionProductQualification | None = None,
) -> ReconstructionEscalationDecision:
    """Allow reconstruction only for the four declared deterministic triggers."""

    if not isinstance(plan, RecallPlan) or not isinstance(frame, PinnedRecallFrame):
        raise TypeError("plan and frame must be typed recall values")
    _validate_frame_plan(frame.frame, plan)
    if not isinstance(temporal_ambiguity, bool):
        raise ValueError("temporal_ambiguity must be a bool")
    if (
        isinstance(conflict_count, bool)
        or not isinstance(conflict_count, int)
        or conflict_count < 0
    ):
        raise ValueError("conflict_count must be a non-negative integer")
    if (
        isinstance(minimum_coverage, bool)
        or not isinstance(minimum_coverage, (int, float))
        or not 0.0 <= float(minimum_coverage) <= 1.0
    ):
        raise ValueError("minimum_coverage must be between zero and one")
    if plan.query_type == "exact_lookup":
        return ReconstructionEscalationDecision(False, ("exact_lookup_one_hop",))
    if not frame.frame.widening_allowed:
        return ReconstructionEscalationDecision(False, ("widening_not_allowed",))
    reasons: list[str] = []
    if plan.query_type == "synthesis":
        reasons.append("synthesis")
    if plan.query_type == "temporal" and temporal_ambiguity:
        reasons.append("temporal_ambiguity")
    if conflict_count:
        reasons.append("conflict")
    if frame.frame.coverage < float(minimum_coverage):
        reasons.append("low_coverage")
    if not reasons:
        return ReconstructionEscalationDecision(False, ("one_hop_sufficient",))
    if (
        not isinstance(product_qualification, ReconstructionProductQualification)
        or not product_qualification.qualified
        or not product_qualification.safe_one_hop_fallback
    ):
        return ReconstructionEscalationDecision(False, ("product_not_qualified",))
    return ReconstructionEscalationDecision(True, tuple(reasons))


@dataclass(frozen=True)
class ReconstructionObservation:
    evidence_ids: tuple[str, ...]
    tokens_used: int
    latency_ms: float
    coverage_after: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", _identifiers(self.evidence_ids))
        if isinstance(self.tokens_used, bool) or not isinstance(self.tokens_used, int):
            raise ValueError("tokens_used must be a non-negative integer")
        if self.tokens_used < 0:
            raise ValueError("tokens_used must be a non-negative integer")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be finite and non-negative")
        if (
            isinstance(self.coverage_after, bool)
            or not isinstance(self.coverage_after, (int, float))
            or not 0.0 <= float(self.coverage_after) <= 1.0
        ):
            raise ValueError("coverage_after must be between zero and one")
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        object.__setattr__(self, "coverage_after", float(self.coverage_after))


@dataclass(frozen=True)
class ReconstructionTraceStep:
    ordinal: int
    operator: ReconstructionOperator
    candidate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    tokens_used: int
    latency_ms: float
    coverage_before: float
    coverage_after: float
    useful: bool
    outcome: str

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionTrace:
    trace_id: str
    frame_revision: str
    trigger_reasons: tuple[str, ...]
    steps: tuple[ReconstructionTraceStep, ...]
    disposition: Literal["complete", "partial", "abstained"]
    reason: str
    total_candidates: int
    total_tokens: int
    total_latency_ms: float
    final_coverage: float
    prompt_visible: Literal[False] = False
    host_actions_allowed: Literal[False] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger_reasons", _identifiers(self.trigger_reasons))
        steps = tuple(self.steps)
        if any(not isinstance(step, ReconstructionTraceStep) for step in steps):
            raise TypeError("reconstruction trace steps must be typed")
        object.__setattr__(self, "steps", steps)
        if self.disposition not in {"complete", "partial", "abstained"}:
            raise ValueError("reconstruction trace disposition is invalid")
        object.__setattr__(self, "reason", _required(self.reason, field_name="trace reason"))
        for name in ("total_candidates", "total_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("total_latency_ms", "final_coverage"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.final_coverage > 1:
            raise ValueError("final_coverage must not exceed one")
        if self.prompt_visible is not False or self.host_actions_allowed is not False:
            raise ValueError("reconstruction trace cannot enter prompt or authorize host actions")
        if self.trace_id != _trace_id(self._payload_without_id()):
            raise ValueError("reconstruction trace id does not match its content")

    def to_mapping(self) -> dict[str, object]:
        return {"trace_id": self.trace_id, **self._payload_without_id()}

    def _payload_without_id(self) -> dict[str, object]:
        return {
            "frame_revision": self.frame_revision,
            "trigger_reasons": list(self.trigger_reasons),
            "steps": [step.to_mapping() for step in self.steps],
            "disposition": self.disposition,
            "reason": self.reason,
            "total_candidates": self.total_candidates,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "final_coverage": self.final_coverage,
            "prompt_visible": self.prompt_visible,
            "host_actions_allowed": self.host_actions_allowed,
        }


@dataclass(frozen=True)
class ReconstructionRun:
    evidence_ids: tuple[str, ...]
    trace: ReconstructionTrace


ReconstructionExecutor = Callable[
    [ReconstructionStep, RecallBudget],
    ReconstructionObservation,
]


def run_reconstructive_lite(
    plan: ReconstructiveLitePlan,
    *,
    escalation: ReconstructionEscalationDecision,
    executor: ReconstructionExecutor,
    budget: RecallBudget = DEFAULT_RECONSTRUCTIVE_RECALL_BUDGET,
    trace_store: "ReconstructionTraceStore | None" = None,
    target_coverage: float = 0.90,
    operator_policy: RestrictedOperatorPolicy | None = None,
) -> ReconstructionRun:
    """Run bounded operator calls and persist a complete out-of-prompt trace."""

    if not isinstance(plan, ReconstructiveLitePlan):
        raise TypeError("plan must be a ReconstructiveLitePlan")
    if not isinstance(escalation, ReconstructionEscalationDecision):
        raise TypeError("escalation must be a ReconstructionEscalationDecision")
    if not callable(executor):
        raise TypeError("executor must be callable")
    if not isinstance(budget, RecallBudget):
        raise TypeError("budget must be a RecallBudget")
    if (
        isinstance(target_coverage, bool)
        or not isinstance(target_coverage, (int, float))
        or not 0.0 <= float(target_coverage) <= 1.0
    ):
        raise ValueError("target_coverage must be between zero and one")
    if not escalation.escalate:
        trace = _make_trace(
            frame_revision=plan.frame.frame_revision,
            trigger_reasons=escalation.reasons,
            steps=(),
            disposition="abstained",
            reason=escalation.reasons[0],
            total_candidates=0,
            total_tokens=0,
            total_latency_ms=0.0,
            final_coverage=plan.frame.frame.coverage,
        )
        if trace_store is not None:
            trace_store.append(trace)
        return ReconstructionRun((), trace)

    observations: list[ReconstructionTraceStep] = []
    evidence_ids: list[str] = []
    total_candidates = 0
    total_tokens = 0
    total_latency = 0.0
    coverage = plan.frame.frame.coverage
    attempted_operators: tuple[ReconstructionOperator, ...] = ()
    disposition: Literal["complete", "partial", "abstained"] = "complete"
    reason = "operator_plan_complete"
    while len(attempted_operators) < len(plan.steps):
        remaining_steps = budget.max_steps - len(observations)
        remaining_candidates = budget.max_candidates - total_candidates
        remaining_tokens = budget.max_tokens - total_tokens
        remaining_latency = budget.max_latency_ms - total_latency
        if (
            min(remaining_steps, remaining_candidates, remaining_tokens) <= 0
            or remaining_latency <= 0
        ):
            disposition = "partial" if observations else "abstained"
            reason = "hard_budget_exhausted"
            break
        selection = select_reconstruction_operator(
            plan,
            trigger_reasons=escalation.reasons,
            attempted_operators=attempted_operators,
            policy=operator_policy,
        )
        step = selection.step
        if step is None:
            break
        attempted_operators = (*attempted_operators, step.operator)
        bounded_candidates = step.candidate_ids[:remaining_candidates]
        bounded_plan = _candidate_plan(step.plan, bounded_candidates)
        bounded_step = replace(step, plan=bounded_plan, candidate_ids=bounded_candidates)
        step_budget = RecallBudget(
            max_latency_ms=max(1, math.floor(remaining_latency)),
            max_steps=remaining_steps,
            max_tokens=remaining_tokens,
            max_candidates=remaining_candidates,
        )
        try:
            observation = executor(bounded_step, step_budget)
        except TimeoutError:
            observations.append(
                ReconstructionTraceStep(
                    ordinal=step.ordinal,
                    operator=step.operator,
                    candidate_ids=bounded_candidates,
                    evidence_ids=(),
                    tokens_used=0,
                    latency_ms=float(step_budget.max_latency_ms),
                    coverage_before=coverage,
                    coverage_after=coverage,
                    useful=False,
                    outcome="operator_timeout",
                )
            )
            total_candidates += len(bounded_candidates)
            total_latency += float(step_budget.max_latency_ms)
            disposition = "partial" if evidence_ids else "abstained"
            reason = "operator_timeout"
            break
        if not isinstance(observation, ReconstructionObservation):
            raise TypeError("reconstruction executor must return ReconstructionObservation")
        budget_violation = (
            observation.tokens_used > remaining_tokens
            or observation.latency_ms > remaining_latency
            or len(bounded_candidates) > remaining_candidates
        )
        new_evidence = (
            ()
            if budget_violation
            else tuple(item for item in observation.evidence_ids if item not in evidence_ids)
        )
        accepted_coverage = coverage if budget_violation else observation.coverage_after
        useful = accepted_coverage > coverage or bool(new_evidence)
        observations.append(
            ReconstructionTraceStep(
                ordinal=step.ordinal,
                operator=step.operator,
                candidate_ids=bounded_candidates,
                evidence_ids=new_evidence,
                tokens_used=observation.tokens_used,
                latency_ms=observation.latency_ms,
                coverage_before=coverage,
                coverage_after=accepted_coverage,
                useful=useful,
                outcome="operator_budget_violation" if budget_violation else "completed",
            )
        )
        total_candidates += len(bounded_candidates)
        total_tokens += min(observation.tokens_used, remaining_tokens)
        total_latency += min(observation.latency_ms, remaining_latency)
        coverage = max(coverage, accepted_coverage)
        evidence_ids.extend(new_evidence)
        if budget_violation:
            disposition = "partial" if len(observations) > 1 else "abstained"
            reason = "operator_budget_violation"
            break
        if coverage >= float(target_coverage):
            reason = "target_coverage_reached"
            break
    trace = _make_trace(
        frame_revision=plan.frame.frame_revision,
        trigger_reasons=escalation.reasons,
        steps=tuple(observations),
        disposition=disposition,
        reason=reason,
        total_candidates=total_candidates,
        total_tokens=total_tokens,
        total_latency_ms=total_latency,
        final_coverage=coverage,
    )
    if trace_store is not None:
        trace_store.append(trace)
    return ReconstructionRun(tuple(evidence_ids), trace)


class ReconstructionTraceStore:
    """Append full content-free reconstruction traces outside prompt state."""

    def __init__(self, repo_root: str | Path, *, workspace_id: str) -> None:
        root = Path(repo_root).expanduser().resolve()
        workspace = _required(workspace_id, field_name="workspace_id")
        workspace_hash = hashlib.sha256(workspace.encode("utf-8")).hexdigest()[:20]
        self.path = (
            root / ".UM-MEM" / "memory" / "reconstruction_traces" / f"{workspace_hash}.jsonl"
        )

    def append(self, trace: ReconstructionTrace) -> None:
        if not isinstance(trace, ReconstructionTrace):
            raise TypeError("trace must be a ReconstructionTrace")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(trace.to_mapping(), ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def read(self) -> tuple[dict[str, object], ...]:
        if not self.path.exists():
            return ()
        rows: list[dict[str, object]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("reconstruction trace store is invalid") from exc
            if not isinstance(value, dict):
                raise RuntimeError("reconstruction trace store is invalid")
            trace_id = value.get("trace_id")
            payload = {key: item for key, item in value.items() if key != "trace_id"}
            if trace_id != _trace_id(payload):
                raise RuntimeError("reconstruction trace store is invalid")
            rows.append(value)
        return tuple(rows)


def _make_trace(
    *,
    frame_revision: str,
    trigger_reasons: tuple[str, ...],
    steps: tuple[ReconstructionTraceStep, ...],
    disposition: Literal["complete", "partial", "abstained"],
    reason: str,
    total_candidates: int,
    total_tokens: int,
    total_latency_ms: float,
    final_coverage: float,
) -> ReconstructionTrace:
    payload = {
        "frame_revision": frame_revision,
        "trigger_reasons": list(trigger_reasons),
        "steps": [step.to_mapping() for step in steps],
        "disposition": disposition,
        "reason": reason,
        "total_candidates": total_candidates,
        "total_tokens": total_tokens,
        "total_latency_ms": total_latency_ms,
        "final_coverage": final_coverage,
        "prompt_visible": False,
        "host_actions_allowed": False,
    }
    return ReconstructionTrace(
        trace_id=_trace_id(payload),
        frame_revision=frame_revision,
        trigger_reasons=trigger_reasons,
        steps=steps,
        disposition=disposition,
        reason=reason,
        total_candidates=total_candidates,
        total_tokens=total_tokens,
        total_latency_ms=total_latency_ms,
        final_coverage=final_coverage,
    )


def _trace_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
