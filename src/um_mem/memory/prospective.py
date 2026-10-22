"""First-class advisory prospective obligations and trigger evaluation."""

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from um_mem.contracts import (
    AuthenticatedHostEvent,
    OriginRef,
    ProspectiveObligation,
    ProspectiveTriggerPredicate,
)
from um_mem.contracts.memory import (
    MemoryAuthority,
    ProspectivePredicateType,
    ProspectiveScope,
    ProspectiveStatus,
)
from um_mem.memory.backends.contracts import MemoryIndexNode


@dataclass(frozen=True)
class ProspectiveTriggerContext:
    time: dict[str, Any] = field(default_factory=dict)
    deadline: dict[str, Any] = field(default_factory=dict)
    host_event: AuthenticatedHostEvent | None = None
    tool_state: dict[str, Any] = field(default_factory=dict)
    test_state: dict[str, Any] = field(default_factory=dict)
    file_state: dict[str, Any] = field(default_factory=dict)
    registered_condition: dict[str, Any] = field(default_factory=dict)
    scope: ProspectiveScope = "workspace"
    scope_id: str = ""
    origin: OriginRef | None = None
    authority_ceiling: MemoryAuthority = "advisory"
    semantic_text: str = ""
    observed_at: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "time",
            "deadline",
            "tool_state",
            "test_state",
            "file_state",
            "registered_condition",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, dict) or any(
                not isinstance(key, str) or not key.strip() for key in value
            ):
                raise ValueError(f"{field_name} must be a mapping with non-empty string keys")
            object.__setattr__(self, field_name, dict(value))
        if self.host_event is not None and not isinstance(self.host_event, AuthenticatedHostEvent):
            raise ValueError("host_event must use AuthenticatedHostEvent")
        if self.origin is not None and not isinstance(self.origin, OriginRef):
            raise ValueError("prospective trigger origin must use OriginRef")
        if self.authority_ceiling not in {"advisory", "directive"}:
            raise ValueError("prospective trigger authority_ceiling is invalid")
        if self.scope not in {"workspace", "run", "task", "subtask"}:
            raise ValueError("prospective trigger scope is invalid")
        if not isinstance(self.scope_id, str):
            raise ValueError("scope_id must be a string")
        scope_id = self.scope_id.strip()
        if self.scope != "workspace" and not scope_id:
            raise ValueError("non-workspace prospective trigger scope requires scope_id")
        if self.scope == "workspace" and scope_id:
            raise ValueError("workspace prospective trigger scope cannot carry scope_id")
        object.__setattr__(self, "scope_id", scope_id)
        if not isinstance(self.semantic_text, str):
            raise ValueError("semantic_text must be a string")
        object.__setattr__(self, "semantic_text", self.semantic_text.strip())
        if self.observed_at:
            _parse_time(self.observed_at, field_name="observed_at")


@dataclass(frozen=True)
class ProspectiveTriggerDecision:
    obligation_id: str
    matched: bool
    match_kind: Literal["none", "typed", "semantic", "lifecycle"]
    reason: str
    trigger_type: str = ""
    candidate_class: Literal["none", "release", "experimental"] = "none"
    advertised: bool = False
    delivery_eligible: bool = False
    recommended_transition: Literal["", "cancelled", "expired"] = ""
    matched_cues: tuple[str, ...] = ()
    host_action_authorized: bool = field(default=False, init=False)


def _parse_time(value: str, *, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _context_value(
    predicate: ProspectiveTriggerPredicate,
    context: ProspectiveTriggerContext,
) -> tuple[bool, Any]:
    if predicate.predicate_type == "host_event":
        if context.host_event is None:
            return False, None
        values = context.host_event.trigger_values()
    else:
        values = getattr(context, predicate.predicate_type)
    if predicate.field not in values:
        return False, None
    return True, values[predicate.field]


def _ordered_value(value: Any) -> int | float | datetime | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return _parse_time(value, field_name="prospective trigger value")
        except ValueError:
            return None
    return None


def predicate_matches(
    predicate: ProspectiveTriggerPredicate,
    context: ProspectiveTriggerContext,
) -> bool:
    present, actual = _context_value(predicate, context)
    expected = predicate.value
    if predicate.operator == "exists":
        return present is bool(expected)
    if not present:
        return False
    if predicate.operator == "eq":
        return type(actual) is type(expected) and actual == expected
    if predicate.operator == "neq":
        return type(actual) is type(expected) and actual != expected
    if predicate.operator == "contains":
        if isinstance(actual, str) and isinstance(expected, str):
            return expected in actual
        if isinstance(actual, (list, tuple, set)):
            return expected in actual
        return False
    if predicate.operator == "prefix":
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.startswith(expected)
        if isinstance(actual, (list, tuple, set)) and isinstance(expected, str):
            return any(isinstance(value, str) and value.startswith(expected) for value in actual)
        return False
    actual_ordered = _ordered_value(actual)
    expected_ordered = _ordered_value(expected)
    if actual_ordered is None or expected_ordered is None:
        return False
    if type(actual_ordered) is not type(expected_ordered):
        return False
    if isinstance(actual_ordered, datetime) and isinstance(expected_ordered, datetime):
        if predicate.operator == "gte":
            return actual_ordered >= expected_ordered
        if predicate.operator == "lte":
            return actual_ordered <= expected_ordered
        return False
    if isinstance(actual_ordered, datetime) or isinstance(expected_ordered, datetime):
        return False
    if predicate.operator == "gte":
        return actual_ordered >= expected_ordered
    if predicate.operator == "lte":
        return actual_ordered <= expected_ordered
    return False


def _all_match(
    predicates: tuple[ProspectiveTriggerPredicate, ...],
    context: ProspectiveTriggerContext,
) -> bool:
    return bool(predicates) and all(predicate_matches(item, context) for item in predicates)


def _matched_trigger_type(
    predicates: tuple[ProspectiveTriggerPredicate, ...],
) -> ProspectivePredicateType | str:
    kinds = tuple(dict.fromkeys(predicate.predicate_type for predicate in predicates))
    return kinds[0] if len(kinds) == 1 else "multiple"


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[\w.-]+", value.casefold()))


def evaluate_prospective_trigger(
    obligation: ProspectiveObligation,
    context: ProspectiveTriggerContext,
) -> ProspectiveTriggerDecision:
    if obligation.status != "armed":
        return ProspectiveTriggerDecision(
            obligation_id=obligation.obligation_id,
            matched=False,
            match_kind="none",
            reason=f"status_{obligation.status}_not_armed",
        )
    if obligation.scope != "workspace" and (
        context.scope != obligation.scope or context.scope_id != obligation.scope_id
    ):
        return ProspectiveTriggerDecision(
            obligation_id=obligation.obligation_id,
            matched=False,
            match_kind="none",
            reason="scope_mismatch",
        )
    observed_at = (
        _parse_time(context.observed_at, field_name="observed_at")
        if context.observed_at
        else datetime.now(UTC)
    )
    if obligation.expires_at and observed_at >= _parse_time(
        obligation.expires_at, field_name="expires_at"
    ):
        return ProspectiveTriggerDecision(
            obligation_id=obligation.obligation_id,
            matched=True,
            match_kind="lifecycle",
            reason="expiry_reached",
            recommended_transition="expired",
        )
    if _all_match(obligation.cancel_predicates, context):
        return ProspectiveTriggerDecision(
            obligation_id=obligation.obligation_id,
            matched=True,
            match_kind="lifecycle",
            reason="cancel_condition_matched",
            recommended_transition="cancelled",
        )
    if _all_match(obligation.suppress_predicates, context):
        return ProspectiveTriggerDecision(
            obligation_id=obligation.obligation_id,
            matched=False,
            match_kind="none",
            reason="suppressed_by_current_state",
        )
    if obligation.earliest_at and observed_at < _parse_time(
        obligation.earliest_at, field_name="earliest_at"
    ):
        return ProspectiveTriggerDecision(
            obligation_id=obligation.obligation_id,
            matched=False,
            match_kind="none",
            reason="before_earliest_at",
        )
    if _all_match(obligation.trigger_predicates, context):
        return ProspectiveTriggerDecision(
            obligation_id=obligation.obligation_id,
            matched=True,
            match_kind="typed",
            reason="typed_predicates_matched",
            trigger_type=_matched_trigger_type(obligation.trigger_predicates),
            candidate_class="release",
            advertised=True,
            delivery_eligible=True,
        )
    if obligation.deadline_at and observed_at >= _parse_time(
        obligation.deadline_at, field_name="deadline_at"
    ):
        return ProspectiveTriggerDecision(
            obligation_id=obligation.obligation_id,
            matched=True,
            match_kind="typed",
            reason="deadline_reached",
            trigger_type="deadline",
            candidate_class="release",
            advertised=True,
            delivery_eligible=True,
        )
    semantic_text = _normalized_words(context.semantic_text)
    matched_cues = tuple(
        cue
        for cue in obligation.semantic_trigger_cues
        if _normalized_words(cue) and _normalized_words(cue) in semantic_text
    )
    if matched_cues:
        return ProspectiveTriggerDecision(
            obligation_id=obligation.obligation_id,
            matched=True,
            match_kind="semantic",
            reason="semantic_candidate_requires_typed_gate",
            candidate_class="experimental",
            advertised=False,
            delivery_eligible=False,
            matched_cues=matched_cues,
        )
    return ProspectiveTriggerDecision(
        obligation_id=obligation.obligation_id,
        matched=False,
        match_kind="none",
        reason="no_trigger_match",
    )


@dataclass(frozen=True)
class ProspectiveFireRevalidation:
    obligation_id: str
    allowed: bool
    reason: str
    decision: ProspectiveTriggerDecision


@dataclass(frozen=True)
class ProspectiveIntervention:
    obligation_id: str
    text: str
    surface: Literal["current_working_state"] = "current_working_state"
    label: Literal["prospective_advisory_intervention"] = "prospective_advisory_intervention"
    host_action_authorized: bool = field(default=False, init=False)
    executes_action: bool = field(default=False, init=False)
    consequential_action_requires_host_authorization: bool = field(default=True, init=False)


@dataclass(frozen=True)
class ProspectiveTelemetryEntry:
    obligation_id: str
    status: ProspectiveStatus
    false_activation_count: int
    missed_trigger_count: int
    overloaded_trigger_count: int
    attention_interruption_count: int
    duplicate_fire_count: int
    late_trigger_count: int


@dataclass(frozen=True)
class ProspectiveTelemetryReport:
    entries: tuple[ProspectiveTelemetryEntry, ...]
    channel: Literal["prospective_obligation_telemetry"] = field(
        default="prospective_obligation_telemetry", init=False
    )
    ordinary_recall: bool = field(default=False, init=False)
    host_action_authorized: bool = field(default=False, init=False)


def prospective_telemetry_report(
    obligations: tuple[ProspectiveObligation, ...],
) -> ProspectiveTelemetryReport:
    by_id: dict[str, ProspectiveObligation] = {}
    for obligation in obligations:
        if not isinstance(obligation, ProspectiveObligation):
            raise ValueError("prospective telemetry requires ProspectiveObligation values")
        if obligation.obligation_id in by_id:
            raise ValueError("prospective telemetry obligation ids must be unique")
        by_id[obligation.obligation_id] = obligation
    entries = tuple(
        ProspectiveTelemetryEntry(
            obligation_id=obligation.obligation_id,
            status=obligation.status,
            false_activation_count=obligation.telemetry.false_positive_count,
            missed_trigger_count=obligation.telemetry.missed_trigger_count,
            overloaded_trigger_count=obligation.telemetry.overloaded_trigger_count,
            attention_interruption_count=obligation.telemetry.attention_interruption_count,
            duplicate_fire_count=obligation.telemetry.duplicate_fire_count,
            late_trigger_count=obligation.telemetry.late_trigger_count,
        )
        for obligation in sorted(by_id.values(), key=lambda value: value.obligation_id)
    )
    return ProspectiveTelemetryReport(entries=entries)


def prospective_intervention(obligation: ProspectiveObligation) -> ProspectiveIntervention:
    if obligation.status != "fired":
        raise ValueError("status_fired_required")
    if not obligation.required_intervention:
        raise ValueError("fired obligation has no intervention payload")
    return ProspectiveIntervention(
        obligation_id=obligation.obligation_id,
        text=obligation.required_intervention,
    )


def revalidate_prospective_fire(
    obligation: ProspectiveObligation,
    context: ProspectiveTriggerContext,
) -> ProspectiveFireRevalidation:
    """Recheck one new fire without mutating obligation or host state."""

    decision = evaluate_prospective_trigger(obligation, context)
    if decision.match_kind != "typed" or not decision.delivery_eligible:
        reason = (
            decision.reason
            if decision.match_kind != "semantic"
            else "prospective fire requires a delivery-eligible typed trigger"
        )
        return ProspectiveFireRevalidation(
            obligation_id=obligation.obligation_id,
            allowed=False,
            reason=reason,
            decision=decision,
        )
    origin = context.origin
    if origin is None or origin.integrity_class not in {"observed", "verified", "endorsed"}:
        return ProspectiveFireRevalidation(
            obligation_id=obligation.obligation_id,
            allowed=False,
            reason="prospective fire requires observed origin integrity",
            decision=decision,
        )
    if context.host_event is not None and (
        origin.channel != "host_event" or origin.principal_id != context.host_event.principal_id
    ):
        return ProspectiveFireRevalidation(
            obligation_id=obligation.obligation_id,
            allowed=False,
            reason="prospective host-event origin does not match authenticated envelope",
            decision=decision,
        )
    if context.authority_ceiling != "advisory":
        return ProspectiveFireRevalidation(
            obligation_id=obligation.obligation_id,
            allowed=False,
            reason="prospective fire requires advisory authority ceiling",
            decision=decision,
        )
    return ProspectiveFireRevalidation(
        obligation_id=obligation.obligation_id,
        allowed=True,
        reason="typed_trigger_revalidated",
        decision=decision,
    )


def prospective_trigger_index_node_id(workspace_id: str, obligation_id: str) -> str:
    material = f"{workspace_id}\x00{obligation_id}"
    return f"prospective_trigger:{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


def prospective_trigger_index_node(
    obligation: ProspectiveObligation,
    *,
    workspace_id: str,
    version: int,
    created_at: str = "",
) -> MemoryIndexNode:
    predicate_fields = tuple(
        sorted(
            {
                f"{predicate.predicate_type}:{predicate.field}"
                for predicate in (
                    *obligation.trigger_predicates,
                    *obligation.suppress_predicates,
                    *obligation.cancel_predicates,
                )
            }
        )
    )
    tags = tuple(
        dict.fromkeys(
            (
                "prospective_obligation",
                obligation.status,
                *(("semantic_cue",) if obligation.semantic_trigger_cues else ()),
                *predicate_fields,
            )
        )
    )
    return MemoryIndexNode(
        node_id=prospective_trigger_index_node_id(workspace_id, obligation.obligation_id),
        title=f"Prospective trigger: {obligation.obligation_id}",
        node_type="prospective_trigger",
        workspace_id=workspace_id,
        tags=tags,
        metadata={
            "obligation_id": obligation.obligation_id,
            "scope": obligation.scope,
            "scope_id": obligation.scope_id,
            "status": obligation.status,
            "version": version,
            "trigger_predicates": [asdict(value) for value in obligation.trigger_predicates],
            "semantic_trigger_cues": list(obligation.semantic_trigger_cues),
            "earliest_at": obligation.earliest_at,
            "deadline_at": obligation.deadline_at,
            "expires_at": obligation.expires_at,
            "recurrence_policy": obligation.recurrence_policy,
            "recurrence_interval_seconds": obligation.recurrence_interval_seconds,
            "suppress_predicates": [asdict(value) for value in obligation.suppress_predicates],
            "cancel_predicates": [asdict(value) for value in obligation.cancel_predicates],
            "source_evidence_ref_ids": [ref.ref_id for ref in obligation.evidence_refs],
            "host_action_authorized": False,
        },
        created_at=created_at,
    )
