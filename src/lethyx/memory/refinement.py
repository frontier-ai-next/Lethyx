"""Verified utility refinement and operator-facing outcome reporting."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Iterable

from lethyx.contracts import (
    AdvisoryAssociation,
    EvidenceRef,
    EvidenceResolver,
    MemoryDecisionTrace,
    MemoryOutcomeReport,
    MemoryOutcomeReportEntry,
    OutcomeCreditEvent,
    ProcedurePromotionGate,
    ProcedureReuseDecision,
    TransferEnvelope,
    UtilitySelectionCandidate,
)
from lethyx.contracts.hashing import hash_value
from lethyx.memory.blocks.catalog import (
    FailureAttributionStage,
    RefinementCase,
    RefinementCaseType,
)
from lethyx.contracts.memory import AssociationStatus, MemoryOutcomeDisposition


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _parse_time(value: str, *, field_name: str) -> datetime:
    text = _required_text(value, field_name=field_name)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _unique_evidence(values: Iterable[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    refs: list[EvidenceRef] = []
    for ref in values:
        if not isinstance(ref, EvidenceRef):
            raise ValueError("refinement evidence must use EvidenceRef")
        if ref not in refs:
            refs.append(ref)
    return tuple(refs)


def procedure_promotion_gate(
    *,
    verified_outcome_ids: tuple[str, ...],
    independence_domains: tuple[str, ...],
    minimum_verified_outcomes: int = 2,
) -> ProcedurePromotionGate:
    """Require repeat independent verified outcomes before procedure promotion."""

    return ProcedurePromotionGate(
        verified_outcome_ids=verified_outcome_ids,
        independence_domains=independence_domains,
        minimum_verified_outcomes=minimum_verified_outcomes,
    )


def verified_procedure_promotion_gate(
    *,
    memory_id: str,
    profile_id: str,
    task_signature: str,
    outcome_credits: tuple[OutcomeCreditEvent, ...],
    resolver: EvidenceResolver,
) -> ProcedurePromotionGate:
    """Count distinct applied outcomes only when the host ledger verifies them."""

    memory = _required_text(memory_id, field_name="memory_id")
    profile = _required_text(profile_id, field_name="profile_id")
    signature = _required_text(task_signature, field_name="task_signature")
    if any(not isinstance(credit, OutcomeCreditEvent) for credit in outcome_credits):
        raise TypeError("procedure outcome credits must be typed")
    relevant = tuple(credit for credit in outcome_credits if memory in credit.memory_ids)
    outcome_ids: list[str] = []
    domains: list[str] = []
    decisions: set[str] = set()
    traces: set[str] = set()
    reports: set[str] = set()
    for credit in relevant:
        if credit.profile_id != profile or credit.outcome == "neutral":
            continue
        verified_reports = []
        for ref in credit.evidence_refs:
            if ref.ref_type != "verification_report" or ref.produced_by != "verifier":
                continue
            if resolver.validate_ref(ref):
                continue
            row = resolver.ledger_row(ref.ref_id)
            result = row.get("result") if row is not None else None
            if (
                row is not None
                and row.get("workspace_id") == credit.workspace_id
                and row.get("actor") == "verifier"
                and row.get("tool") == "memory_application_verification"
                and isinstance(result, dict)
                and result.get("passed") is True
                and result.get("verifier_id") == ref.producer_id
                and result.get("decision_id") == credit.decision_id
                and result.get("action_trace_id") == credit.action_trace_id
                and result.get("memory_id") == memory
                and result.get("usage_event_id") in credit.applied_usage_event_ids
                and result.get("task_signature") == credit.task_signature
                and result.get("outcome") == credit.outcome
                and row.get("result_hash") == hash_value(result)
            ):
                verified_reports.append(ref.ref_id)
        if credit.outcome == "hurt" and verified_reports:
            raise ValueError("procedure has a verified harmful application")
        if (
            credit.outcome != "helped"
            or credit.task_signature != signature
            or credit.memory_ids != (memory,)
            or len(credit.applied_usage_event_ids) != 1
            or credit.attribution_basis != "deterministic_verifier"
            or credit.verifier_result != "passed"
        ):
            continue
        if (
            not verified_reports
            or credit.decision_id in decisions
            or credit.action_trace_id in traces
            or credit.independence_domain in domains
            or any(report in reports for report in verified_reports)
        ):
            continue
        outcome_ids.append(credit.credit_id)
        domains.append(credit.independence_domain)
        decisions.add(credit.decision_id)
        traces.add(credit.action_trace_id)
        reports.update(verified_reports)
    return procedure_promotion_gate(
        verified_outcome_ids=tuple(outcome_ids),
        independence_domains=tuple(domains),
    )


def qualify_procedure_reuse(
    *,
    memory_id: str,
    source_task_signature: str,
    requested_task_signature: str,
    transfer_envelope: TransferEnvelope | None = None,
) -> ProcedureReuseDecision:
    """Admit reuse only in the original task signature or an accepted transfer."""

    memory = _required_text(memory_id, field_name="memory_id")
    source = _required_text(source_task_signature, field_name="source_task_signature")
    requested = _required_text(
        requested_task_signature,
        field_name="requested_task_signature",
    )
    if source == requested:
        return ProcedureReuseDecision(
            memory_id=memory,
            source_task_signature=source,
            requested_task_signature=requested,
            allowed=True,
            reason="same_task_signature",
        )
    if (
        isinstance(transfer_envelope, TransferEnvelope)
        and transfer_envelope.status == "accepted"
        and transfer_envelope.source_task_signature == source
        and transfer_envelope.target_task_signature == requested
        and memory in transfer_envelope.qualified_memory_ids
    ):
        return ProcedureReuseDecision(
            memory_id=memory,
            source_task_signature=source,
            requested_task_signature=requested,
            allowed=True,
            reason="qualified_transfer_envelope",
            transfer_envelope_id=transfer_envelope.envelope_id,
        )
    return ProcedureReuseDecision(
        memory_id=memory,
        source_task_signature=source,
        requested_task_signature=requested,
        allowed=False,
        reason="cross_task_signature_requires_qualified_transfer",
        transfer_envelope_id=(
            transfer_envelope.envelope_id if isinstance(transfer_envelope, TransferEnvelope) else ""
        ),
    )


def evaluate_advisory_association(
    *,
    association_id: str,
    memory_ids: tuple[str, str],
    coactivation_evidence_refs: tuple[EvidenceRef, ...],
    outcome_credits: tuple[OutcomeCreditEvent, ...],
    minimum_verified_outcomes: int = 2,
) -> AdvisoryAssociation:
    """Derive an advisory edge without turning co-activation into authority."""

    if minimum_verified_outcomes < 2:
        raise ValueError("minimum_verified_outcomes must be at least 2")
    pair = tuple(dict.fromkeys(str(memory_id or "").strip() for memory_id in memory_ids))
    if len(pair) != 2 or any(not memory_id for memory_id in pair):
        raise ValueError("advisory association requires exactly two memory_ids")
    if not isinstance(outcome_credits, tuple) or any(
        not isinstance(credit, OutcomeCreditEvent) for credit in outcome_credits
    ):
        raise ValueError("outcome_credits must contain OutcomeCreditEvent values")
    relevant = tuple(
        credit
        for credit in outcome_credits
        if set(pair).issubset(credit.memory_ids) and credit.outcome in {"helped", "hurt"}
    )
    helped_count = sum(credit.outcome == "helped" for credit in relevant)
    status: AssociationStatus
    hurt_count = sum(credit.outcome == "hurt" for credit in relevant)
    outcome_diversity = len({credit.independence_domain for credit in relevant})
    if hurt_count >= minimum_verified_outcomes and outcome_diversity >= minimum_verified_outcomes:
        status = "quarantined"
        weight = 0.0
    elif hurt_count:
        status = "weakened"
        weight = (helped_count - hurt_count) / (helped_count + hurt_count)
    elif (
        helped_count >= minimum_verified_outcomes and outcome_diversity >= minimum_verified_outcomes
    ):
        status = "strengthened"
        weight = helped_count / (helped_count + hurt_count)
    else:
        status = "candidate"
        weight = 0.0
    evidence = _unique_evidence(
        (
            *coactivation_evidence_refs,
            *(ref for credit in relevant for ref in credit.evidence_refs),
        )
    )
    return AdvisoryAssociation(
        association_id=association_id,
        memory_ids=(min(pair), max(pair)),
        status=status,
        helped_count=helped_count,
        hurt_count=hurt_count,
        outcome_diversity=outcome_diversity,
        advisory_weight=weight,
        evidence_refs=evidence,
    )


def select_diverse_utility_candidates(
    candidates: tuple[UtilitySelectionCandidate, ...],
    *,
    limit: int,
    max_per_task_family: int = 2,
    as_of: str,
    recent_window_days: int = 30,
) -> tuple[UtilitySelectionCandidate, ...]:
    """Apply deterministic family diversity and homogeneous-recency caps."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if (
        isinstance(max_per_task_family, bool)
        or not isinstance(max_per_task_family, int)
        or max_per_task_family < 1
    ):
        raise ValueError("max_per_task_family must be a positive integer")
    if (
        isinstance(recent_window_days, bool)
        or not isinstance(recent_window_days, int)
        or recent_window_days < 1
    ):
        raise ValueError("recent_window_days must be a positive integer")
    if not isinstance(candidates, tuple) or any(
        not isinstance(candidate, UtilitySelectionCandidate) for candidate in candidates
    ):
        raise ValueError("candidates must contain UtilitySelectionCandidate values")
    if len({candidate.memory_id for candidate in candidates}) != len(candidates):
        raise ValueError("utility candidate memory_ids must be unique")
    cutoff = _parse_time(as_of, field_name="as_of") - timedelta(days=recent_window_days)
    by_family: dict[str, list[UtilitySelectionCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_family[candidate.task_family].append(candidate)
    for family_candidates in by_family.values():
        family_candidates.sort(
            key=lambda item: (
                -item.shadow_utility,
                -item.support_count,
                -_parse_time(
                    item.newest_credit_at,
                    field_name="newest_credit_at",
                ).timestamp(),
                item.memory_id,
            )
        )
    families = sorted(
        by_family,
        key=lambda family: (
            min(
                _parse_time(item.newest_credit_at, field_name="newest_credit_at")
                for item in by_family[family]
            ),
            family,
        ),
    )
    selected: list[UtilitySelectionCandidate] = []
    family_counts: dict[str, int] = defaultdict(int)
    recent_family_counts: dict[str, int] = defaultdict(int)
    for family in families:
        if len(selected) >= limit:
            break
        candidate = by_family[family][0]
        selected.append(candidate)
        family_counts[family] += 1
        if _parse_time(candidate.newest_credit_at, field_name="newest_credit_at") >= cutoff:
            recent_family_counts[family] += 1
    remaining = sorted(
        (candidate for candidate in candidates if candidate not in selected),
        key=lambda item: (
            _parse_time(item.newest_credit_at, field_name="newest_credit_at") >= cutoff,
            -item.shadow_utility,
            -item.support_count,
            item.task_family,
            item.memory_id,
        ),
    )
    for candidate in remaining:
        if len(selected) >= limit:
            break
        family = candidate.task_family
        recent = _parse_time(candidate.newest_credit_at, field_name="newest_credit_at") >= cutoff
        if family_counts[family] >= max_per_task_family:
            continue
        if recent and recent_family_counts[family] >= 1:
            continue
        selected.append(candidate)
        family_counts[family] += 1
        if recent:
            recent_family_counts[family] += 1
    return tuple(selected)


def generate_refinement_case(
    *,
    case_id: str,
    case_type: RefinementCaseType,
    failure_stage: FailureAttributionStage,
    query: str,
    decision_id: str,
    task_signature: str,
    action_trace_id: str,
    evidence_refs: tuple[EvidenceRef, ...],
    reason: str,
    selected_descriptor_ids: tuple[str, ...] = (),
    expected_descriptor_ids: tuple[str, ...] = (),
) -> RefinementCase:
    """Create a candidate refinement case with an explicit failure owner."""

    return RefinementCase(
        case_id=case_id,
        case_type=case_type,
        failure_stage=failure_stage,
        query=query,
        decision_id=decision_id,
        task_signature=task_signature,
        action_trace_id=action_trace_id,
        evidence_refs=evidence_refs,
        reason=reason,
        selected_descriptor_ids=selected_descriptor_ids,
        expected_descriptor_ids=expected_descriptor_ids,
    )


def build_memory_outcome_report(
    *,
    report_id: str,
    trace: MemoryDecisionTrace,
    outcome_credits: tuple[OutcomeCreditEvent, ...],
) -> MemoryOutcomeReport:
    """Report outcome linkage without treating missing callbacks as evidence of nonuse."""

    if not isinstance(trace, MemoryDecisionTrace):
        raise ValueError("operator report requires MemoryDecisionTrace")
    if not isinstance(outcome_credits, tuple) or any(
        not isinstance(credit, OutcomeCreditEvent) for credit in outcome_credits
    ):
        raise ValueError("operator report requires OutcomeCreditEvent values")
    decision = trace.decision
    for credit in outcome_credits:
        if (
            credit.decision_id != decision.decision_id
            or credit.workspace_id != decision.workspace_id
            or credit.task_signature != decision.task_signature
            or credit.action_trace_id != decision.action_trace_id
        ):
            raise ValueError("operator report credit linkage is invalid")
    entries: list[MemoryOutcomeReportEntry] = []
    for memory_id in decision.candidate_ids:
        usage_events = tuple(event for event in trace.usage_events if event.memory_id == memory_id)
        credits = tuple(credit for credit in outcome_credits if memory_id in credit.memory_ids)
        hurt = next((credit for credit in credits if credit.outcome == "hurt"), None)
        helped = next((credit for credit in credits if credit.outcome == "helped"), None)
        applied = any(event.usage_type == "applied" for event in usage_events)
        disposition: MemoryOutcomeDisposition
        if hurt is not None:
            disposition = "hurt"
            reason = hurt.reason
        elif helped is not None:
            disposition = "helped"
            reason = helped.reason
        elif not applied and any(event.usage_type == "not_applied" for event in usage_events):
            disposition = "unnecessary"
            reason = "completed action-use observation explicitly records nonuse"
        elif not applied:
            disposition = "unknown"
            reason = (
                "action-use observation is incomplete; absent applied events do not prove nonuse"
            )
        else:
            disposition = "neutral"
            reason = "applied record has no explicit verified causal outcome"
        evidence = _unique_evidence(
            (
                *(ref for event in usage_events for ref in event.evidence_refs),
                *(ref for credit in credits for ref in credit.evidence_refs),
                *(() if usage_events or credits else decision.evidence_refs),
            )
        )
        entries.append(
            MemoryOutcomeReportEntry(
                memory_id=memory_id,
                disposition=disposition,
                usage_event_ids=tuple(event.event_id for event in usage_events),
                credit_ids=tuple(credit.credit_id for credit in credits),
                reason=reason,
                evidence_refs=evidence,
            )
        )
    return MemoryOutcomeReport(
        report_id=report_id,
        workspace_id=decision.workspace_id,
        decision_id=decision.decision_id,
        task_signature=decision.task_signature,
        action_trace_id=decision.action_trace_id,
        entries=tuple(entries),
    )
