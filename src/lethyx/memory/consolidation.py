"""Canonical, proposal-only consolidation pipeline and trigger scheduler."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Literal, Mapping, cast, get_args

from lethyx.contracts import (
    ConsolidationAction,
    ConsolidationDecision,
    ConsolidationTransition,
    ConsolidationTrigger,
    EvidenceRef,
    MemoryBlockRecord,
    MemoryTransitionAssessment,
    OperationalRisk,
    SourceCoverageDiff,
)


CONSOLIDATION_TRIGGERS = frozenset(get_args(ConsolidationTrigger))


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _text_ids(value: object, *, field_name: str, required: bool = True) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    normalized = tuple(_required_text(item, field_name=field_name) for item in value)
    if required and not normalized:
        raise ValueError(f"{field_name} are required")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must be unique")
    return tuple(sorted(normalized))


def _evidence_refs(value: object, *, field_name: str) -> tuple[EvidenceRef, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    refs: list[EvidenceRef] = []
    for ref in value:
        if not isinstance(ref, EvidenceRef):
            raise ValueError(f"{field_name} must contain EvidenceRef values")
        if ref not in refs:
            refs.append(ref)
    if not refs:
        raise ValueError(f"{field_name} are required")
    return tuple(refs)


def _canonical_json(value: object, *, field_name: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class ConsolidationSignal:
    """One independent typed reason to consider a bounded consolidation proposal."""

    workspace_id: str
    source_episode_ids: tuple[str, ...]
    grouping_key: str
    task_family: str
    trigger: ConsolidationTrigger
    proposed_action: ConsolidationAction
    expected_applicability: str
    risk: OperationalRisk
    cost_budget_tokens: int
    rationale: str
    evidence_refs: tuple[EvidenceRef, ...]
    cluster_key: str = ""
    rare_critical: bool = False
    expected_value: float = 0.0
    age_seconds: int = 0
    estimated_cost_tokens: int = 1
    dependency_ids: tuple[str, ...] = ()
    dependencies_closed: bool = True
    recurrence_count: int = 2

    def __post_init__(self) -> None:
        for field_name in (
            "workspace_id",
            "grouping_key",
            "task_family",
            "expected_applicability",
            "rationale",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "source_episode_ids",
            _text_ids(self.source_episode_ids, field_name="source_episode_ids"),
        )
        object.__setattr__(
            self,
            "cluster_key",
            _required_text(
                self.cluster_key or self.grouping_key,
                field_name="cluster_key",
            ),
        )
        if self.trigger not in CONSOLIDATION_TRIGGERS:
            raise ValueError("consolidation trigger is invalid")
        if self.proposed_action not in get_args(ConsolidationAction):
            raise ValueError("consolidation action is invalid")
        if self.risk not in get_args(OperationalRisk):
            raise ValueError("consolidation risk is invalid")
        if (
            isinstance(self.cost_budget_tokens, bool)
            or not isinstance(self.cost_budget_tokens, int)
            or self.cost_budget_tokens < 1
        ):
            raise ValueError("cost_budget_tokens must be a positive integer")
        if not isinstance(self.rare_critical, bool):
            raise ValueError("rare_critical must be a bool")
        if (
            isinstance(self.expected_value, bool)
            or not isinstance(self.expected_value, (int, float))
            or not 0.0 <= float(self.expected_value) <= 1.0
        ):
            raise ValueError("expected_value must be between 0 and 1")
        object.__setattr__(self, "expected_value", float(self.expected_value))
        if (
            isinstance(self.age_seconds, bool)
            or not isinstance(self.age_seconds, int)
            or self.age_seconds < 0
        ):
            raise ValueError("age_seconds must be a non-negative integer")
        if (
            isinstance(self.estimated_cost_tokens, bool)
            or not isinstance(self.estimated_cost_tokens, int)
            or self.estimated_cost_tokens < 1
            or self.estimated_cost_tokens > self.cost_budget_tokens
        ):
            raise ValueError("estimated_cost_tokens must fit the positive cost budget")
        object.__setattr__(
            self,
            "dependency_ids",
            _text_ids(
                self.dependency_ids,
                field_name="dependency_ids",
                required=False,
            ),
        )
        if not isinstance(self.dependencies_closed, bool):
            raise ValueError("dependencies_closed must be a bool")
        if (
            isinstance(self.recurrence_count, bool)
            or not isinstance(self.recurrence_count, int)
            or self.recurrence_count < 1
        ):
            raise ValueError("recurrence_count must be a positive integer")
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence_refs(self.evidence_refs, field_name="signal evidence_refs"),
        )


@dataclass(frozen=True)
class ConsolidationScheduleEntry:
    signal: ConsolidationSignal
    decision: ConsolidationDecision
    order: int
    priority_score: float
    disposition: Literal["scheduled", "deferred"]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.signal, ConsolidationSignal):
            raise ValueError("schedule entry requires ConsolidationSignal")
        if not isinstance(self.decision, ConsolidationDecision):
            raise ValueError("schedule entry requires ConsolidationDecision")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 1:
            raise ValueError("schedule entry order must be positive")
        if self.disposition not in {"scheduled", "deferred"}:
            raise ValueError("schedule entry disposition is invalid")
        object.__setattr__(self, "reason", _required_text(self.reason, field_name="reason"))

    @property
    def rare_critical(self) -> bool:
        return self.signal.rare_critical


@dataclass(frozen=True)
class ConsolidationSchedule:
    schedule_id: str
    window: str
    entries: tuple[ConsolidationScheduleEntry, ...]
    grouping_keys: tuple[str, ...]
    cluster_keys: tuple[str, ...]
    input_set_fingerprint_sha256: str
    canonical_mutation_applied: bool = False
    host_action_authorized: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schedule_id",
            _required_text(self.schedule_id, field_name="schedule_id"),
        )
        object.__setattr__(self, "window", _required_text(self.window, field_name="window"))
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, ConsolidationScheduleEntry) for entry in self.entries
        ):
            raise ValueError("schedule requires typed entries")
        if tuple(entry.order for entry in self.entries) != tuple(range(1, len(self.entries) + 1)):
            raise ValueError("schedule entry order must be contiguous")
        object.__setattr__(
            self,
            "grouping_keys",
            _text_ids(self.grouping_keys, field_name="grouping_keys", required=False),
        )
        object.__setattr__(
            self,
            "cluster_keys",
            _text_ids(self.cluster_keys, field_name="cluster_keys", required=False),
        )
        if not self.input_set_fingerprint_sha256.startswith("sha256:"):
            raise ValueError("input_set_fingerprint_sha256 must be a sha256 reference")
        if self.canonical_mutation_applied or self.host_action_authorized:
            raise ValueError("consolidation schedule must remain non-authorizing")


@dataclass(frozen=True)
class AtomicMemoryClaim:
    claim_id: str
    workspace_id: str
    content: str
    source_record_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    authority: Literal["advisory"] = "advisory"
    status: Literal["candidate"] = "candidate"
    host_action_authorized: bool = False

    def __post_init__(self) -> None:
        for field_name in ("claim_id", "workspace_id", "content"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "source_record_ids",
            _text_ids(self.source_record_ids, field_name="source_record_ids"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence_refs(self.evidence_refs, field_name="claim evidence_refs"),
        )
        if self.authority != "advisory" or self.status != "candidate":
            raise ValueError("atomic memory claims must remain advisory candidates")
        if self.host_action_authorized:
            raise ValueError("atomic memory claims cannot authorize host action")


@dataclass(frozen=True)
class MaterializedMemoryView:
    view_id: str
    workspace_id: str
    content: str
    claim_ids: tuple[str, ...]
    transitive_raw_record_ids: tuple[str, ...]
    authority: Literal["advisory"] = "advisory"
    status: Literal["candidate"] = "candidate"
    host_action_authorized: bool = False

    def __post_init__(self) -> None:
        for field_name in ("view_id", "workspace_id", "content"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "claim_ids",
            _text_ids(self.claim_ids, field_name="claim_ids"),
        )
        object.__setattr__(
            self,
            "transitive_raw_record_ids",
            _text_ids(
                self.transitive_raw_record_ids,
                field_name="transitive_raw_record_ids",
            ),
        )
        if self.authority != "advisory" or self.status != "candidate":
            raise ValueError("materialized memory views must remain advisory candidates")
        if self.host_action_authorized:
            raise ValueError("materialized memory views cannot authorize host action")


@dataclass(frozen=True)
class ConsolidationProvenanceDag:
    dag_id: str
    workspace_id: str
    raw_record_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    atomic_claims: tuple[AtomicMemoryClaim, ...]
    materialized_views: tuple[MaterializedMemoryView, ...]
    edges: tuple[tuple[str, str, str], ...]
    input_set_fingerprint_sha256: str
    canonical_mutation_applied: bool = False
    host_action_authorized: bool = False

    def __post_init__(self) -> None:
        for field_name in ("dag_id", "workspace_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("raw_record_ids", "claim_ids", "view_ids"):
            object.__setattr__(
                self,
                field_name,
                _text_ids(getattr(self, field_name), field_name=field_name),
            )
        if not self.input_set_fingerprint_sha256.startswith("sha256:"):
            raise ValueError("DAG input_set_fingerprint_sha256 must be a sha256 reference")
        if self.canonical_mutation_applied or self.host_action_authorized:
            raise ValueError("consolidation provenance DAG must remain non-authorizing")


@dataclass(frozen=True)
class TransitionInvariantCheck:
    """One deterministic tribunal check and its audit-safe reason."""

    check_id: str
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "check_id",
            _required_text(self.check_id, field_name="check_id"),
        )
        if not isinstance(self.passed, bool):
            raise ValueError("transition invariant result must be a bool")
        object.__setattr__(self, "reason", _required_text(self.reason, field_name="reason"))


@dataclass(frozen=True)
class TransitionSemanticVerification:
    """Independent semantic coverage/preservation/faithfulness result."""

    verifier_id: str
    verifier_model_version: str
    coverage: bool
    preservation: bool
    faithfulness: bool
    independent_from_proposer: bool
    reasons: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        for field_name in ("verifier_id", "verifier_model_version"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "coverage",
            "preservation",
            "faithfulness",
            "independent_from_proposer",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a bool")
        object.__setattr__(
            self,
            "reasons",
            _text_ids(self.reasons, field_name="semantic verification reasons"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence_refs(
                self.evidence_refs,
                field_name="semantic verification evidence_refs",
            ),
        )


@dataclass(frozen=True)
class ConsolidationTribunalReport:
    """Bounded, proposal-only tribunal result produced before any commit."""

    report_id: str
    transition_id: str
    verdict: Literal["accept", "reject", "defer"]
    deterministic_checks: tuple[TransitionInvariantCheck, ...]
    semantic_verification: TransitionSemanticVerification
    protected_literals: tuple[str, ...]
    assessment: MemoryTransitionAssessment
    canonical_mutation_applied: bool = False
    host_action_authorized: bool = False

    def __post_init__(self) -> None:
        for field_name in ("report_id", "transition_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if self.verdict not in {"accept", "reject", "defer"}:
            raise ValueError("tribunal report verdict is invalid")
        if not isinstance(self.deterministic_checks, tuple) or any(
            not isinstance(check, TransitionInvariantCheck) for check in self.deterministic_checks
        ):
            raise ValueError("tribunal report requires deterministic checks")
        if not self.deterministic_checks:
            raise ValueError("tribunal report requires deterministic checks")
        if not isinstance(self.semantic_verification, TransitionSemanticVerification):
            raise ValueError("tribunal report requires semantic verification")
        object.__setattr__(
            self,
            "protected_literals",
            _text_ids(
                self.protected_literals,
                field_name="protected_literals",
                required=False,
            ),
        )
        if not isinstance(self.assessment, MemoryTransitionAssessment):
            raise ValueError("tribunal report requires MemoryTransitionAssessment")
        if self.assessment.verdict != self.verdict:
            raise ValueError("tribunal report and assessment verdicts must match")
        if self.canonical_mutation_applied or self.host_action_authorized:
            raise ValueError("tribunal report must remain non-authorizing")

    @property
    def failed_check_ids(self) -> tuple[str, ...]:
        return tuple(check.check_id for check in self.deterministic_checks if not check.passed)


def _candidate_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(text for item in value.values() for text in _candidate_strings(item))
    if isinstance(value, (list, tuple)):
        return tuple(text for item in value for text in _candidate_strings(item))
    return ()


def _candidate_text_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _record_metadata_texts(record: MemoryBlockRecord, key: str) -> tuple[str, ...]:
    value = record.metadata.get(key, ())
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


class ConsolidationTransitionTribunal:
    """Fail-closed deterministic and independent-semantic pre-commit tribunal."""

    _SCOPE_RANK = {
        "subtask": 0,
        "task": 1,
        "session": 2,
        "workspace": 3,
        "project": 4,
        "user_global": 5,
    }

    def assess(
        self,
        transition: ConsolidationTransition,
        *,
        decision: ConsolidationDecision,
        source_records: tuple[MemoryBlockRecord, ...],
        semantic_verification: TransitionSemanticVerification,
        assessment_id: str,
    ) -> ConsolidationTribunalReport:
        if not isinstance(transition, ConsolidationTransition):
            raise ValueError("transition must be a ConsolidationTransition")
        if not isinstance(decision, ConsolidationDecision):
            raise ValueError("decision must be a ConsolidationDecision")
        if transition.decision_id != decision.decision_id:
            raise ValueError("transition decision identity mismatch")
        if not isinstance(source_records, tuple) or any(
            not isinstance(record, MemoryBlockRecord) for record in source_records
        ):
            raise ValueError("source_records must contain MemoryBlockRecord values")
        if not source_records:
            raise ValueError("source_records are required")
        records = {record.block_id: record for record in source_records}
        if len(records) != len(source_records):
            raise ValueError("source record IDs must be unique")
        if tuple(sorted(records)) != decision.source_episode_ids:
            raise ValueError("tribunal sources must match the decision")
        if not isinstance(semantic_verification, TransitionSemanticVerification):
            raise ValueError("semantic_verification must be typed")

        candidate = json.loads(transition.post_state_candidate)
        if not isinstance(candidate, dict):
            raise ValueError("post-state candidate must be an object")
        candidate_strings = "\n".join(_candidate_strings(candidate))
        source_ids = set(records)
        erased_ids = {
            record.block_id
            for record in source_records
            if record.metadata.get("privacy_erased") is True
        }
        active_source_ids = source_ids - erased_ids
        candidate_source_ids = _candidate_text_set(candidate.get("source_record_ids"))
        coverage_diff = transition.source_coverage_diff
        if coverage_diff is None:
            raise ValueError("consolidation transition requires source coverage")
        after_source_ids = set(coverage_diff.after_source_ids)

        checks: list[TransitionInvariantCheck] = []

        def add(check_id: str, passed: bool, ok: str, failure: str) -> None:
            checks.append(
                TransitionInvariantCheck(
                    check_id=check_id,
                    passed=passed,
                    reason=ok if passed else failure,
                )
            )

        origin_ok = (
            candidate.get("workspace_id") == decision.workspace_id
            and {record.workspace_id for record in source_records} == {decision.workspace_id}
            and set(transition.touched_record_ids) == source_ids
            and candidate_source_ids <= source_ids
        )
        add(
            "origin_integrity",
            origin_ok,
            "workspace and canonical source identities close",
            "workspace or canonical source identity does not close",
        )

        authority_values = [candidate.get("authority")]
        for collection in (candidate.get("claims", []), candidate.get("materialized_views", [])):
            if isinstance(collection, list):
                authority_values.extend(
                    item.get("authority", "advisory")
                    for item in collection
                    if isinstance(item, dict)
                )
        authority_ok = all(value == "advisory" for value in authority_values)
        add(
            "no_authority_uplift",
            authority_ok,
            "candidate remains advisory",
            "candidate attempts authority uplift",
        )

        candidate_scope = str(candidate.get("scope", ""))
        source_scope_rank = max(self._SCOPE_RANK.get(record.scope, 99) for record in source_records)
        candidate_scope_rank = self._SCOPE_RANK.get(candidate_scope, 99)
        lifecycle_ok = (
            candidate_scope_rank <= source_scope_rank
            and candidate.get("status") == "candidate"
            and all(record.status not in {"quarantined", "retracted"} for record in source_records)
            and all(
                not record.valid_from
                or not candidate.get("valid_from")
                or candidate.get("valid_from") == record.valid_from
                for record in source_records
            )
            and all(
                not record.valid_to
                or not candidate.get("valid_to")
                or candidate.get("valid_to") == record.valid_to
                for record in source_records
            )
        )
        add(
            "scope_validity_lifecycle",
            lifecycle_ok,
            "scope, validity, and lifecycle remain bounded",
            "scope widened or validity/lifecycle changed unsafely",
        )

        coverage_ok = (
            not coverage_diff.added_source_ids
            and after_source_ids == active_source_ids
            and candidate_source_ids == active_source_ids
        )
        add(
            "source_coverage",
            coverage_ok,
            "every non-erased canonical source remains covered",
            "canonical source coverage was lost, added, or misreported",
        )

        protected_literals = tuple(
            sorted(
                {
                    literal
                    for record in source_records
                    if record.block_id not in erased_ids
                    for literal in _record_metadata_texts(record, "exact_literals")
                }
            )
        )
        rare_literals_ok = all(literal in candidate_strings for literal in protected_literals)
        add(
            "rare_exact_literals",
            rare_literals_ok,
            "rare and exact literals survive the candidate",
            "a rare or exact literal is missing or distorted",
        )

        correction_ids = {
            item
            for record in source_records
            for item in _record_metadata_texts(record, "correction_of")
        }
        contradiction_ids = {
            item
            for record in source_records
            for item in _record_metadata_texts(record, "contradiction_of")
        }
        correction_ok = correction_ids <= _candidate_text_set(candidate.get("correction_ids"))
        contradiction_ok = contradiction_ids <= _candidate_text_set(
            candidate.get("contradiction_ids")
        )
        add(
            "contradiction_correction_closure",
            correction_ok and contradiction_ok,
            "correction and contradiction relations remain explicit",
            "correction or contradiction lineage was silently dropped",
        )

        erased_literals = {
            literal
            for record in source_records
            if record.block_id in erased_ids
            for literal in (
                record.content,
                *_record_metadata_texts(record, "exact_literals"),
            )
            if literal
        }
        privacy_ok = (
            not (erased_ids & candidate_source_ids)
            and not (erased_ids & after_source_ids)
            and all(literal not in candidate_strings for literal in erased_literals)
        )
        add(
            "privacy_erase_closure",
            privacy_ok,
            "erased sources and literals are absent",
            "erased source identity or content leaked into the candidate",
        )

        views = candidate.get("materialized_views", [])
        views_by_id = (
            {
                str(item.get("view_id")): item
                for item in views
                if isinstance(item, dict) and item.get("view_id")
            }
            if isinstance(views, list)
            else {}
        )
        views_ok = set(transition.touched_view_ids) <= set(views_by_id) and all(
            _candidate_text_set(view.get("source_record_ids")) <= active_source_ids
            and bool(_candidate_text_set(view.get("source_record_ids")))
            for view in views_by_id.values()
        )
        add(
            "affected_materialized_views",
            views_ok,
            "every affected view has bounded transitive sources",
            "an affected view is missing or has invalid source coverage",
        )

        continuation_ok = (
            candidate.get("continuation_applicability") == decision.expected_applicability
        )
        add(
            "continuation_applicability",
            continuation_ok,
            "continuation applicability matches the decision",
            "continuation applicability widened or drifted",
        )

        procedure_ok = True
        if any("procedure" in operation for operation in transition.proposed_operations):
            procedure = candidate.get("procedure")
            required_fields = {
                "prerequisites",
                "steps",
                "applicability",
                "contraindications",
                "evidence_refs",
                "outcomes",
            }
            procedure_ok = (
                isinstance(procedure, dict)
                and required_fields <= set(procedure)
                and procedure.get("applicability") == decision.expected_applicability
                and all(procedure.get(field_name) for field_name in required_fields)
            )
        add(
            "procedure_applicability",
            procedure_ok,
            "procedure contract is closed and applicability is bounded",
            "procedure contract or applicability is incomplete",
        )

        semantic_ok = (
            semantic_verification.independent_from_proposer
            and semantic_verification.verifier_model_version != transition.model_version
            and semantic_verification.coverage
            and semantic_verification.preservation
            and semantic_verification.faithfulness
        )
        add(
            "independent_semantic_verification",
            semantic_ok,
            "independent verifier confirms coverage, preservation, and faithfulness",
            "semantic verifier is not independent or an assessment axis failed",
        )

        verdict: Literal["accept", "reject", "defer"] = (
            "accept" if all(check.passed for check in checks) else "reject"
        )
        reasons = tuple(check.reason for check in checks)
        evidence_refs = tuple(
            dict.fromkeys((*transition.evidence_refs, *semantic_verification.evidence_refs))
        )
        assessment = MemoryTransitionAssessment(
            assessment_id=assessment_id,
            transition_id=transition.transition_id,
            decision_id=transition.decision_id,
            pre_state_snapshot_sha256=transition.pre_state_snapshot_sha256,
            post_state_candidate_sha256=transition.post_state_candidate_sha256,
            source_coverage_diff=coverage_diff,
            policy_version=transition.policy_version,
            model_version=transition.model_version,
            prompt_version=transition.prompt_version,
            verdict=verdict,
            reasons=reasons,
            evidence_refs=evidence_refs,
        )
        return ConsolidationTribunalReport(
            report_id=f"tribunal-{transition.transition_id}",
            transition_id=transition.transition_id,
            verdict=verdict,
            deterministic_checks=tuple(checks),
            semantic_verification=semantic_verification,
            protected_literals=protected_literals,
            assessment=assessment,
        )


@dataclass(frozen=True)
class ConsolidationShadowSnapshot:
    snapshot_id: str
    workspace_id: str
    transition_id: str
    candidate_sha256: str
    view_payloads: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field_name in ("snapshot_id", "workspace_id", "transition_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if not self.candidate_sha256.startswith("sha256:"):
            raise ValueError("shadow candidate requires a sha256 digest")
        if not isinstance(self.view_payloads, tuple):
            raise ValueError("shadow view payloads must be a tuple")


@dataclass(frozen=True)
class ConsolidationRejectionReceipt:
    receipt_id: str
    workspace_id: str
    transition_id: str
    candidate_sha256: str
    disposition: Literal["rejected", "quarantined"]
    verifier_reasons: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        for field_name in ("receipt_id", "workspace_id", "transition_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if not self.candidate_sha256.startswith("sha256:"):
            raise ValueError("rejection receipt requires a candidate sha256 digest")
        if self.disposition not in {"rejected", "quarantined"}:
            raise ValueError("rejection receipt disposition is invalid")
        object.__setattr__(
            self,
            "verifier_reasons",
            _text_ids(self.verifier_reasons, field_name="verifier_reasons"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence_refs(self.evidence_refs, field_name="rejection evidence_refs"),
        )


@dataclass(frozen=True)
class AtomicConsolidationCommitReceipt:
    receipt_id: str
    workspace_id: str
    transition_id: str
    previous_head_sha256: str
    active_head_sha256: str
    committed_view_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        for field_name in ("receipt_id", "workspace_id", "transition_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if self.previous_head_sha256 and not self.previous_head_sha256.startswith("sha256:"):
            raise ValueError("previous head must be empty or a sha256 digest")
        if not self.active_head_sha256.startswith("sha256:"):
            raise ValueError("active head requires a sha256 digest")
        object.__setattr__(
            self,
            "committed_view_ids",
            _text_ids(
                self.committed_view_ids,
                field_name="committed_view_ids",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence_refs(self.evidence_refs, field_name="commit evidence_refs"),
        )


@dataclass(frozen=True)
class ConsolidationShadowResolution:
    snapshot_id: str
    disposition: Literal["committed", "rejected", "quarantined"]
    commit_receipt: AtomicConsolidationCommitReceipt | None = None
    rejection_receipt: ConsolidationRejectionReceipt | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_id",
            _required_text(self.snapshot_id, field_name="snapshot_id"),
        )
        if self.disposition == "committed":
            if self.commit_receipt is None or self.rejection_receipt is not None:
                raise ValueError("committed resolution requires only a commit receipt")
        elif self.disposition in {"rejected", "quarantined"}:
            if self.rejection_receipt is None or self.commit_receipt is not None:
                raise ValueError("rejected resolution requires only a rejection receipt")
        else:
            raise ValueError("shadow resolution disposition is invalid")


class ConsolidationShadowStore:
    """Isolate one bounded candidate and atomically swap accepted view state."""

    def __init__(self) -> None:
        self._pending_by_workspace: dict[str, ConsolidationShadowSnapshot] = {}
        self._pending_transition: dict[str, ConsolidationTransition] = {}
        self._active: dict[str, tuple[str, dict[str, str]]] = {}
        self._rejected_digests: dict[str, set[str]] = {}

    def stage(
        self,
        *,
        workspace_id: str,
        transition: ConsolidationTransition,
    ) -> ConsolidationShadowSnapshot:
        workspace = _required_text(workspace_id, field_name="workspace_id")
        if not isinstance(transition, ConsolidationTransition):
            raise ValueError("transition must be a ConsolidationTransition")
        if workspace in self._pending_by_workspace:
            raise ValueError("only one bounded candidate may be staged per workspace")
        if transition.post_state_candidate_sha256 in self._rejected_digests.get(workspace, set()):
            raise ValueError("previously rejected candidate revision cannot be staged again")
        candidate = json.loads(transition.post_state_candidate)
        if not isinstance(candidate, dict) or candidate.get("workspace_id") != workspace:
            raise ValueError("shadow candidate workspace mismatch")
        views = candidate.get("materialized_views", [])
        if not isinstance(views, list):
            raise ValueError("materialized_views must be a list")
        view_payloads: list[tuple[str, str]] = []
        for value in views:
            if not isinstance(value, dict):
                raise ValueError("materialized view payload must be an object")
            view_id = _required_text(value.get("view_id"), field_name="view_id")
            view_payloads.append(
                (
                    view_id,
                    _canonical_json(value, field_name="materialized view payload"),
                )
            )
        if len({view_id for view_id, _ in view_payloads}) != len(view_payloads):
            raise ValueError("materialized view IDs must be unique")
        snapshot = ConsolidationShadowSnapshot(
            snapshot_id=f"shadow-{transition.transition_id}",
            workspace_id=workspace,
            transition_id=transition.transition_id,
            candidate_sha256=transition.post_state_candidate_sha256,
            view_payloads=tuple(sorted(view_payloads)),
        )
        self._pending_by_workspace[workspace] = snapshot
        self._pending_transition[snapshot.snapshot_id] = transition
        return snapshot

    def resolve(
        self,
        snapshot_id: str,
        *,
        report: ConsolidationTribunalReport,
        quarantine_on_reject: bool = True,
    ) -> ConsolidationShadowResolution:
        snapshot_key = _required_text(snapshot_id, field_name="snapshot_id")
        transition = self._pending_transition.get(snapshot_key)
        if transition is None:
            raise ValueError("shadow snapshot is not pending")
        snapshot = self._pending_by_workspace[
            json.loads(transition.post_state_candidate)["workspace_id"]
        ]
        if snapshot.snapshot_id != snapshot_key:
            raise ValueError("shadow snapshot identity mismatch")
        if not isinstance(report, ConsolidationTribunalReport):
            raise ValueError("report must be a ConsolidationTribunalReport")
        if (
            report.transition_id != transition.transition_id
            or report.assessment.post_state_candidate_sha256 != snapshot.candidate_sha256
        ):
            raise ValueError("tribunal report does not match the staged candidate")

        del self._pending_transition[snapshot_key]
        del self._pending_by_workspace[snapshot.workspace_id]
        if report.verdict == "accept" and not report.failed_check_ids:
            previous_head, _ = self._active.get(snapshot.workspace_id, ("", {}))
            views = dict(snapshot.view_payloads)
            self._active[snapshot.workspace_id] = (snapshot.candidate_sha256, views)
            receipt = AtomicConsolidationCommitReceipt(
                receipt_id=f"commit-{transition.transition_id}",
                workspace_id=snapshot.workspace_id,
                transition_id=transition.transition_id,
                previous_head_sha256=previous_head,
                active_head_sha256=snapshot.candidate_sha256,
                committed_view_ids=tuple(views),
                evidence_refs=report.assessment.evidence_refs,
            )
            return ConsolidationShadowResolution(
                snapshot_id=snapshot_key,
                disposition="committed",
                commit_receipt=receipt,
            )

        self._rejected_digests.setdefault(snapshot.workspace_id, set()).add(
            snapshot.candidate_sha256
        )
        disposition: Literal["rejected", "quarantined"] = (
            "quarantined" if quarantine_on_reject else "rejected"
        )
        rejection_receipt = ConsolidationRejectionReceipt(
            receipt_id=f"reject-{transition.transition_id}",
            workspace_id=snapshot.workspace_id,
            transition_id=transition.transition_id,
            candidate_sha256=snapshot.candidate_sha256,
            disposition=disposition,
            verifier_reasons=report.assessment.reasons,
            evidence_refs=report.assessment.evidence_refs,
        )
        return ConsolidationShadowResolution(
            snapshot_id=snapshot_key,
            disposition=disposition,
            rejection_receipt=rejection_receipt,
        )

    def active_head(self, workspace_id: str) -> str:
        return self._active.get(workspace_id, ("", {}))[0]

    def active_views(self, workspace_id: str) -> dict[str, str]:
        return dict(self._active.get(workspace_id, ("", {}))[1])


@dataclass(frozen=True)
class PreservationManifest:
    """Shared preservation vocabulary with mandatory transitive raw sources."""

    raw_source_ids: tuple[str, ...]
    protected_literals: tuple[str, ...]
    correction_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        try:
            raw_source_ids = _text_ids(self.raw_source_ids, field_name="raw_source_ids")
        except ValueError as exc:
            raise ValueError("transitive raw source IDs are required") from exc
        object.__setattr__(self, "raw_source_ids", raw_source_ids)
        object.__setattr__(
            self,
            "protected_literals",
            _text_ids(
                self.protected_literals,
                field_name="protected_literals",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "correction_ids",
            _text_ids(
                self.correction_ids,
                field_name="correction_ids",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence_refs(self.evidence_refs, field_name="preservation evidence_refs"),
        )


@dataclass(frozen=True)
class ConsolidationSourceEnvelope:
    envelope_id: str
    record_kind: Literal["working_fold", "durable_consolidation"]
    lifetime: Literal["task", "session", "durable"]
    direct_source_ids: tuple[str, ...]
    transitive_raw_source_ids: tuple[str, ...]
    preservation: PreservationManifest

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "envelope_id",
            _required_text(self.envelope_id, field_name="envelope_id"),
        )
        if self.record_kind not in {"working_fold", "durable_consolidation"}:
            raise ValueError("consolidation source record kind is invalid")
        if self.lifetime not in {"task", "session", "durable"}:
            raise ValueError("consolidation source lifetime is invalid")
        if self.record_kind == "working_fold" and self.lifetime == "durable":
            raise ValueError("working fold cannot claim durable lifetime")
        if self.record_kind == "durable_consolidation" and self.lifetime != "durable":
            raise ValueError("durable consolidation requires durable lifetime")
        object.__setattr__(
            self,
            "direct_source_ids",
            _text_ids(self.direct_source_ids, field_name="direct_source_ids"),
        )
        object.__setattr__(
            self,
            "transitive_raw_source_ids",
            _text_ids(
                self.transitive_raw_source_ids,
                field_name="transitive_raw_source_ids",
            ),
        )
        if not isinstance(self.preservation, PreservationManifest):
            raise ValueError("source envelope requires PreservationManifest")
        if self.transitive_raw_source_ids != self.preservation.raw_source_ids:
            raise ValueError("source envelope raw coverage must match preservation")


def build_consolidation_source_envelope(
    *,
    envelope_id: str,
    record_kind: Literal["working_fold", "durable_consolidation"],
    lifetime: Literal["task", "session", "durable"],
    direct_source_ids: tuple[str, ...],
    preservation: PreservationManifest,
) -> ConsolidationSourceEnvelope:
    if not isinstance(preservation, PreservationManifest):
        raise ValueError("preservation must be a PreservationManifest")
    return ConsolidationSourceEnvelope(
        envelope_id=envelope_id,
        record_kind=record_kind,
        lifetime=lifetime,
        direct_source_ids=direct_source_ids,
        transitive_raw_source_ids=preservation.raw_source_ids,
        preservation=preservation,
    )


def _source_sha256(record: MemoryBlockRecord) -> str:
    return _sha256(
        _canonical_json(
            {
                "block_id": record.block_id,
                "content": record.content,
                "workspace_id": record.workspace_id,
                "scope": record.scope,
                "authority": record.authority,
                "valid_from": record.valid_from,
                "valid_to": record.valid_to,
                "status": record.status,
            },
            field_name="materialized source revision",
        )
    )


def _as_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class MaterializedSourceRevision:
    source_record_id: str
    revision_id: str
    source_sha256: str
    valid_from: str
    valid_to: str

    @classmethod
    def from_record(cls, record: MemoryBlockRecord) -> "MaterializedSourceRevision":
        if not isinstance(record, MemoryBlockRecord):
            raise ValueError("materialized source must be a MemoryBlockRecord")
        return cls(
            source_record_id=record.block_id,
            revision_id=record.revision_id,
            source_sha256=_source_sha256(record),
            valid_from=record.valid_from,
            valid_to=record.valid_to,
        )

    def __post_init__(self) -> None:
        for field_name in ("source_record_id", "revision_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if not self.source_sha256.startswith("sha256:"):
            raise ValueError("materialized source requires a sha256 digest")


@dataclass(frozen=True)
class MaterializedViewRevision:
    view_id: str
    workspace_id: str
    content: str
    sources: tuple[MaterializedSourceRevision, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    status: Literal["active", "invalidated"] = "active"
    invalidation_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("view_id", "workspace_id", "content"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if not isinstance(self.sources, tuple) or any(
            not isinstance(source, MaterializedSourceRevision) for source in self.sources
        ):
            raise ValueError("materialized view requires typed sources")
        if not self.sources:
            raise ValueError("materialized view requires sources")
        if len({source.source_record_id for source in self.sources}) != len(self.sources):
            raise ValueError("materialized view source IDs must be unique")
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence_refs(self.evidence_refs, field_name="materialized view evidence_refs"),
        )
        if self.status not in {"active", "invalidated"}:
            raise ValueError("materialized view status is invalid")
        object.__setattr__(
            self,
            "invalidation_reasons",
            _text_ids(
                self.invalidation_reasons,
                field_name="invalidation_reasons",
                required=False,
            ),
        )

    def revalidate(
        self,
        source_records: tuple[MemoryBlockRecord, ...],
        *,
        as_of: str,
    ) -> "MaterializedViewRevision":
        if not isinstance(source_records, tuple) or any(
            not isinstance(record, MemoryBlockRecord) for record in source_records
        ):
            raise ValueError("source_records must contain MemoryBlockRecord values")
        current = {record.block_id: record for record in source_records}
        instant = _as_utc(_required_text(as_of, field_name="as_of"))
        reasons: set[str] = set()
        for source in self.sources:
            record = current.get(source.source_record_id)
            if record is None:
                reasons.add(f"source_missing:{source.source_record_id}")
                continue
            if record.workspace_id != self.workspace_id:
                reasons.add(f"source_workspace_changed:{source.source_record_id}")
            if record.revision_id != source.revision_id:
                reasons.add(f"source_revision_changed:{source.source_record_id}")
            if _source_sha256(record) != source.source_sha256:
                reasons.add(f"source_hash_changed:{source.source_record_id}")
            if record.status in {"quarantined", "superseded", "deprecated", "retracted"}:
                reasons.add(f"source_lifecycle_invalid:{source.source_record_id}")
            if record.valid_from and instant < _as_utc(record.valid_from):
                reasons.add(f"source_outside_validity:{source.source_record_id}")
            if record.valid_to and instant >= _as_utc(record.valid_to):
                reasons.add(f"source_outside_validity:{source.source_record_id}")
        if reasons:
            return replace(
                self,
                status="invalidated",
                invalidation_reasons=tuple(sorted(reasons)),
            )
        return replace(self, status="active", invalidation_reasons=())


@dataclass(frozen=True)
class ConsolidationInputGateResult:
    input_mode: Literal["episodic_only", "heterogeneous", "streamed"]
    preservation_qualified: bool
    consolidation_allowed: bool
    shadow_only: bool
    fallback_action: Literal["retain", "defer"]
    reason: str

    def __post_init__(self) -> None:
        if self.input_mode not in {"episodic_only", "heterogeneous", "streamed"}:
            raise ValueError("consolidation input mode is invalid")
        for field_name in (
            "preservation_qualified",
            "consolidation_allowed",
            "shadow_only",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a bool")
        if self.fallback_action not in {"retain", "defer"}:
            raise ValueError("consolidation fallback action is invalid")
        object.__setattr__(self, "reason", _required_text(self.reason, field_name="reason"))


class ConsolidationInputGate:
    """Keep heterogeneous/streamed transitions default-off until qualified."""

    def evaluate(
        self,
        *,
        input_mode: Literal["episodic_only", "heterogeneous", "streamed"],
        preservation_qualified: bool,
    ) -> ConsolidationInputGateResult:
        if input_mode not in {"episodic_only", "heterogeneous", "streamed"}:
            raise ValueError("consolidation input mode is invalid")
        if not isinstance(preservation_qualified, bool):
            raise ValueError("preservation_qualified must be a bool")
        allowed = input_mode == "episodic_only" or preservation_qualified
        return ConsolidationInputGateResult(
            input_mode=input_mode,
            preservation_qualified=preservation_qualified,
            consolidation_allowed=allowed,
            shadow_only=not allowed,
            fallback_action="defer" if allowed else "retain",
            reason=(
                "episodic_or_preservation_qualified"
                if allowed
                else "heterogeneous_or_streamed_preservation_not_qualified"
            ),
        )


@dataclass(frozen=True)
class ConsolidationExitAssessment:
    passed: bool
    raw_record_count: int
    active_view_count: int
    reasons: tuple[str, ...]
    canonical_mutation_applied: bool = False
    host_action_authorized: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("exit assessment passed must be a bool")
        for field_name in ("raw_record_count", "active_view_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        object.__setattr__(
            self,
            "reasons",
            _text_ids(self.reasons, field_name="exit reasons"),
        )
        if self.canonical_mutation_applied or self.host_action_authorized:
            raise ValueError("exit assessment must remain non-authorizing")


@dataclass(frozen=True)
class ConsolidationExitMetrics:
    raw_record_ids: tuple[str, ...]
    active_view_ids: tuple[str, ...]
    covered_raw_record_ids: tuple[str, ...]
    lost_valid_fact_ids: tuple[str, ...]
    unsupported_addition_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "raw_record_ids",
            "active_view_ids",
            "covered_raw_record_ids",
            "lost_valid_fact_ids",
            "unsupported_addition_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _text_ids(
                    getattr(self, field_name),
                    field_name=field_name,
                    required=field_name in {"raw_record_ids", "active_view_ids"},
                ),
            )

    def evaluate(self) -> ConsolidationExitAssessment:
        reasons: list[str] = []
        raw_ids = set(self.raw_record_ids)
        covered_ids = set(self.covered_raw_record_ids)
        if len(self.active_view_ids) >= len(self.raw_record_ids):
            reasons.append("active_bank_not_smaller")
        if covered_ids != raw_ids:
            reasons.append("raw_coverage_incomplete")
        if self.lost_valid_fact_ids:
            reasons.append("lost_valid_fact")
        if self.unsupported_addition_ids:
            reasons.append("unsupported_addition")
        if not reasons:
            reasons.append("reduction_and_preservation_passed")
        return ConsolidationExitAssessment(
            passed=reasons == ["reduction_and_preservation_passed"],
            raw_record_count=len(self.raw_record_ids),
            active_view_count=len(self.active_view_ids),
            reasons=tuple(reasons),
        )


class ConsolidationScheduler:
    """Turn independent signals into deterministic, non-authorizing decisions."""

    _RISK_PRIORITY = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    def schedule(
        self,
        signals: tuple[ConsolidationSignal, ...],
    ) -> tuple[ConsolidationDecision, ...]:
        schedule = self.build_schedule(signals, window="immediate")
        return tuple(
            entry.decision for entry in schedule.entries if entry.disposition == "scheduled"
        )

    def build_schedule(
        self,
        signals: tuple[ConsolidationSignal, ...],
        *,
        window: str,
    ) -> ConsolidationSchedule:
        if not isinstance(signals, tuple) or any(
            not isinstance(signal, ConsolidationSignal) for signal in signals
        ):
            raise ValueError("signals must contain ConsolidationSignal values")
        bounded_window = _required_text(window, field_name="window")
        unique_signals: dict[str, ConsolidationSignal] = {}
        for signal in signals:
            canonical_signal = _canonical_json(
                asdict(signal),
                field_name="consolidation signal",
            )
            unique_signals[canonical_signal] = signal
        canonical_inputs = tuple(sorted(unique_signals))
        input_fingerprint = _sha256(
            _canonical_json(
                [json.loads(value) for value in canonical_inputs],
                field_name="consolidation schedule input set",
            )
        )
        scheduled: list[
            tuple[
                tuple[object, ...],
                ConsolidationSignal,
                ConsolidationDecision,
                float,
                Literal["scheduled", "deferred"],
                str,
            ]
        ] = []
        for canonical_signal in canonical_inputs:
            signal = unique_signals[canonical_signal]
            signal_fingerprint = _sha256(canonical_signal)
            decision_id = f"consolidation-{signal_fingerprint.removeprefix('sha256:')[:24]}"
            decision = ConsolidationDecision(
                decision_id=decision_id,
                workspace_id=signal.workspace_id,
                source_episode_ids=signal.source_episode_ids,
                grouping_key=signal.grouping_key,
                task_family=signal.task_family,
                trigger=signal.trigger,
                expected_applicability=signal.expected_applicability,
                risk=signal.risk,
                cost_budget_tokens=signal.cost_budget_tokens,
                rationale=signal.rationale,
                evidence_refs=signal.evidence_refs,
                action=signal.proposed_action,
            )
            if not signal.dependencies_closed:
                disposition: Literal["scheduled", "deferred"] = "deferred"
                reason = "dependency_closure_incomplete"
            elif (
                signal.trigger == "recurrence"
                and signal.recurrence_count < 2
                and not signal.rare_critical
            ):
                disposition = "deferred"
                reason = "recurrence_threshold_not_met"
            elif (
                signal.trigger == "recurrence"
                and signal.recurrence_count < 2
                and signal.rare_critical
            ):
                disposition = "scheduled"
                reason = "rare_critical_bypass"
            else:
                disposition = "scheduled"
                reason = "priority_eligible"
            risk_priority = self._RISK_PRIORITY[signal.risk]
            age_cost = signal.age_seconds / signal.estimated_cost_tokens
            priority_score = round(
                (10000.0 if signal.rare_critical else 0.0)
                + risk_priority * 1000.0
                + signal.expected_value * 100.0
                + age_cost,
                9,
            )
            sort_key: tuple[object, ...] = (
                0 if disposition == "scheduled" else 1,
                -int(signal.rare_critical),
                -risk_priority,
                -signal.expected_value,
                -age_cost,
                signal.estimated_cost_tokens,
                signal.trigger,
                signal.source_episode_ids,
                signal_fingerprint,
            )
            scheduled.append(
                (
                    sort_key,
                    signal,
                    decision,
                    priority_score,
                    disposition,
                    reason,
                )
            )
        entries = tuple(
            ConsolidationScheduleEntry(
                signal=item[1],
                decision=item[2],
                order=order,
                priority_score=item[3],
                disposition=item[4],
                reason=item[5],
            )
            for order, item in enumerate(sorted(scheduled, key=lambda item: item[0]), start=1)
        )
        schedule_basis = _canonical_json(
            {
                "window": bounded_window,
                "input_set_fingerprint_sha256": input_fingerprint,
            },
            field_name="consolidation schedule",
        )
        return ConsolidationSchedule(
            schedule_id=(f"schedule-{_sha256(schedule_basis).removeprefix('sha256:')[:24]}"),
            window=bounded_window,
            entries=entries,
            grouping_keys=tuple(
                sorted({signal.grouping_key for signal in unique_signals.values()})
            ),
            cluster_keys=tuple(sorted({signal.cluster_key for signal in unique_signals.values()})),
            input_set_fingerprint_sha256=input_fingerprint,
        )


class CanonicalConsolidationPipeline:
    """Build immutable shadow transitions from canonical records only."""

    def propose(
        self,
        decision: ConsolidationDecision,
        *,
        source_records: tuple[MemoryBlockRecord, ...],
        new_evidence_refs: tuple[EvidenceRef, ...],
        proposed_operations: tuple[str, ...],
        touched_view_ids: tuple[str, ...],
        post_state_candidate: Mapping[str, object],
        source_coverage_after: tuple[str, ...],
        policy_version: str,
        model_version: str,
        prompt_version: str,
    ) -> ConsolidationTransition:
        if not isinstance(decision, ConsolidationDecision):
            raise ValueError("decision must be a ConsolidationDecision")
        if not isinstance(source_records, tuple) or any(
            not isinstance(record, MemoryBlockRecord) for record in source_records
        ):
            raise ValueError("source_records must contain canonical MemoryBlockRecord values")
        if not source_records:
            raise ValueError("source_records must contain canonical MemoryBlockRecord values")
        by_id = {record.block_id: record for record in source_records}
        if len(by_id) != len(source_records):
            raise ValueError("canonical source record IDs must be unique")
        source_ids = tuple(sorted(by_id))
        if source_ids != decision.source_episode_ids:
            raise ValueError("canonical source episode set must match the decision")
        for record in source_records:
            if not record.canonical_key or not record.revision_id:
                raise ValueError("source_records must be canonical MemoryBlockRecord values")
            if record.workspace_id != decision.workspace_id:
                raise ValueError("canonical source workspace must match the decision")

        new_refs = _evidence_refs(new_evidence_refs, field_name="new_evidence_refs")
        operations = _text_ids(proposed_operations, field_name="proposed_operations")
        views = _text_ids(touched_view_ids, field_name="touched_view_ids", required=False)
        coverage_after = _text_ids(
            source_coverage_after,
            field_name="source_coverage_after",
            required=False,
        )
        policy = _required_text(policy_version, field_name="policy_version")
        model = _required_text(model_version, field_name="model_version")
        prompt = _required_text(prompt_version, field_name="prompt_version")
        pre_state = _canonical_json(
            [asdict(by_id[block_id]) for block_id in source_ids],
            field_name="canonical pre-state",
        )
        post_state = _canonical_json(dict(post_state_candidate), field_name="post_state_candidate")
        pre_state_sha256 = _sha256(pre_state)
        post_state_sha256 = _sha256(post_state)
        coverage_diff = SourceCoverageDiff(
            before_source_ids=source_ids,
            after_source_ids=coverage_after,
        )
        transition_basis = _canonical_json(
            {
                "decision_id": decision.decision_id,
                "pre_state_snapshot_sha256": pre_state_sha256,
                "post_state_candidate_sha256": post_state_sha256,
                "proposed_operations": operations,
                "touched_record_ids": source_ids,
                "touched_view_ids": views,
                "source_coverage_after": coverage_after,
                "policy_version": policy,
                "model_version": model,
                "prompt_version": prompt,
            },
            field_name="consolidation transition",
        )
        transition_id = f"transition-{_sha256(transition_basis).removeprefix('sha256:')[:24]}"
        evidence_refs = tuple(dict.fromkeys((*decision.evidence_refs, *new_refs)))
        return ConsolidationTransition(
            transition_id=transition_id,
            decision_id=decision.decision_id,
            from_status="proposed",
            to_status="assessed",
            pre_state_snapshot_sha256=pre_state_sha256,
            new_evidence_refs=new_refs,
            proposed_operations=operations,
            touched_record_ids=source_ids,
            touched_view_ids=views,
            post_state_candidate=post_state,
            post_state_candidate_sha256=post_state_sha256,
            source_coverage_diff=coverage_diff,
            policy_version=policy,
            model_version=model,
            prompt_version=prompt,
            evidence_refs=evidence_refs,
        )

    def build_provenance_dag(
        self,
        *,
        source_records: tuple[MemoryBlockRecord, ...],
        atomic_claims: tuple[AtomicMemoryClaim, ...],
        materialized_views: tuple[MaterializedMemoryView, ...],
    ) -> ConsolidationProvenanceDag:
        if not isinstance(source_records, tuple) or any(
            not isinstance(record, MemoryBlockRecord) for record in source_records
        ):
            raise ValueError("source_records must contain canonical MemoryBlockRecord values")
        if not source_records:
            raise ValueError("source_records must contain canonical MemoryBlockRecord values")
        by_id = {record.block_id: record for record in source_records}
        if len(by_id) != len(source_records):
            raise ValueError("canonical source record IDs must be unique")
        raw_ids = tuple(sorted(by_id))
        workspaces = {record.workspace_id for record in source_records}
        if len(workspaces) != 1 or not next(iter(workspaces)):
            raise ValueError("canonical raw sources must share one workspace")
        workspace_id = next(iter(workspaces))
        for record in source_records:
            if not record.canonical_key or not record.revision_id:
                raise ValueError("source_records must be canonical MemoryBlockRecord values")

        if not isinstance(atomic_claims, tuple) or any(
            not isinstance(claim, AtomicMemoryClaim) for claim in atomic_claims
        ):
            raise ValueError("atomic_claims must contain AtomicMemoryClaim values")
        if not atomic_claims:
            raise ValueError("atomic_claims are required")
        claims_by_id = {claim.claim_id: claim for claim in atomic_claims}
        if len(claims_by_id) != len(atomic_claims):
            raise ValueError("atomic claim IDs must be unique")
        covered_raw_ids: set[str] = set()
        for claim in atomic_claims:
            if claim.workspace_id != workspace_id:
                raise ValueError("atomic claim workspace must match raw sources")
            unknown_raw_ids = set(claim.source_record_ids) - set(raw_ids)
            if unknown_raw_ids:
                raise ValueError("atomic claim references an unknown raw source")
            covered_raw_ids.update(claim.source_record_ids)
        if covered_raw_ids != set(raw_ids):
            raise ValueError("atomic claims must cover every raw source")

        if not isinstance(materialized_views, tuple) or any(
            not isinstance(view, MaterializedMemoryView) for view in materialized_views
        ):
            raise ValueError("materialized_views must contain MaterializedMemoryView values")
        if not materialized_views:
            raise ValueError("materialized_views are required")
        views_by_id = {view.view_id: view for view in materialized_views}
        if len(views_by_id) != len(materialized_views):
            raise ValueError("materialized view IDs must be unique")
        used_claim_ids: set[str] = set()
        edges: set[tuple[str, str, str]] = set()
        for claim in atomic_claims:
            edges.update(
                (source_id, claim.claim_id, "supports_claim")
                for source_id in claim.source_record_ids
            )
        for view in materialized_views:
            if view.workspace_id != workspace_id:
                raise ValueError("materialized view workspace must match raw sources")
            unknown_claim_ids = set(view.claim_ids) - set(claims_by_id)
            if unknown_claim_ids:
                raise ValueError("materialized view references an unknown atomic claim")
            expected_raw_ids = {
                source_id
                for claim_id in view.claim_ids
                for source_id in claims_by_id[claim_id].source_record_ids
            }
            if set(view.transitive_raw_record_ids) != expected_raw_ids:
                raise ValueError("materialized view transitive raw coverage is invalid")
            used_claim_ids.update(view.claim_ids)
            edges.update(
                (claim_id, view.view_id, "materializes_view") for claim_id in view.claim_ids
            )
        if used_claim_ids != set(claims_by_id):
            raise ValueError("every atomic claim must reach a materialized view")

        ordered_claims = tuple(claims_by_id[claim_id] for claim_id in sorted(claims_by_id))
        ordered_views = tuple(views_by_id[view_id] for view_id in sorted(views_by_id))
        dag_input = _canonical_json(
            {
                "raw_records": [
                    {
                        "block_id": block_id,
                        "canonical_key": by_id[block_id].canonical_key,
                        "revision_id": by_id[block_id].revision_id,
                    }
                    for block_id in raw_ids
                ],
                "atomic_claims": [asdict(claim) for claim in ordered_claims],
                "materialized_views": [asdict(view) for view in ordered_views],
                "edges": sorted(edges),
            },
            field_name="consolidation provenance DAG",
        )
        fingerprint = _sha256(dag_input)
        return ConsolidationProvenanceDag(
            dag_id=f"provenance-{fingerprint.removeprefix('sha256:')[:24]}",
            workspace_id=workspace_id,
            raw_record_ids=raw_ids,
            claim_ids=tuple(sorted(claims_by_id)),
            view_ids=tuple(sorted(views_by_id)),
            atomic_claims=ordered_claims,
            materialized_views=ordered_views,
            edges=tuple(sorted(edges)),
            input_set_fingerprint_sha256=fingerprint,
        )

    def assess(
        self,
        transition: ConsolidationTransition,
        *,
        assessment_id: str,
        verdict: Literal["accept", "reject", "defer"],
        reasons: tuple[str, ...],
        evidence_refs: tuple[EvidenceRef, ...],
    ) -> MemoryTransitionAssessment:
        if not isinstance(transition, ConsolidationTransition):
            raise ValueError("transition must be a ConsolidationTransition")
        if not isinstance(transition.source_coverage_diff, SourceCoverageDiff):
            raise ValueError("transition source coverage is required")
        return MemoryTransitionAssessment(
            assessment_id=assessment_id,
            transition_id=transition.transition_id,
            decision_id=transition.decision_id,
            pre_state_snapshot_sha256=transition.pre_state_snapshot_sha256,
            post_state_candidate_sha256=transition.post_state_candidate_sha256,
            source_coverage_diff=transition.source_coverage_diff,
            policy_version=transition.policy_version,
            model_version=transition.model_version,
            prompt_version=transition.prompt_version,
            verdict=cast(Literal["accept", "reject", "defer"], verdict),
            reasons=reasons,
            evidence_refs=evidence_refs,
        )
