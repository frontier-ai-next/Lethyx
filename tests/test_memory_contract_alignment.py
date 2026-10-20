from dataclasses import replace

import pytest

from lethyx.contracts import (
    ConsolidationAssessment,
    ConsolidationCommitReceipt,
    ConsolidationDecision,
    ConsolidationTransition,
    ContinuationSufficiencyAssessment,
    ContextualUtilitySummary,
    DerivationReceipt,
    EvidenceRef,
    FoldedBranch,
    MemoryBlockRecord,
    MemoryDecisionEvent,
    MemoryEvent,
    OriginRef,
    OriginSegment,
    OutcomeCreditEvent,
    ProspectiveObligation,
    RecallFrame,
    TransferEnvelope,
    VerificationReportRef,
    WorkingStateCheckpoint,
)
from lethyx.memory.kernel.models import block_record_to_event
from lethyx.memory.backends import contracts as backend_contracts
from lethyx.memory.backends.factory import (
    allowed_memory_backend_modes,
    selected_memory_backend_mode,
)
from lethyx.contracts.memory import CURRENT_SCHEMA_VERSION
from lethyx.memory.kernel import models as kernel_models


def _evidence() -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="artifact",
            ref_id="artifact:state",
            producer_id="harness:contract-alignment",
        ),
    )


def _checkpoint(evidence: tuple[EvidenceRef, ...]) -> WorkingStateCheckpoint:
    return WorkingStateCheckpoint(
        profile_id="default",
        owner_generation="owner-1",
        requested_block_id="chosen-leaf",
        working_view="Continue.",
        source_ref_sha256=tuple(
            (ref, "sha256:" + "1" * 64)
            for ref in (
                "event-1",
                "task-anchor-1",
                "system-anchor-1",
                "tail-event-1",
                "evidence-card-1",
            )
        ),
        workspace_id="alpha",
        host_event_id="event-1",
        session_id="session-1",
        parent_session_id="",
        task_id="task-1",
        source_event_refs=("event-1",),
        source_event_sha256="sha256:" + "1" * 64,
        task_anchor_refs=("task-anchor-1",),
        system_anchor_refs=("system-anchor-1",),
        exact_recent_tail_refs=("tail-event-1",),
        evidence_page_refs=(("evidence-card-1",),),
        rollback_or_page_back_ref="evidence-card-1",
        artifact_id="diff:contract",
        artifact_revision="abc",
        valid_from="2026-07-14T09:00:00Z",
        valid_to="2026-07-14T10:00:00Z",
        current_goal="close contracts",
        plan=("redirect contracts",),
        current_subgoal="verify state data",
        unresolved_constraints=("single canonical owner",),
        project_id="project:alpha",
        repository_id="repository:alpha",
        checkout="C:/repo/Lethyx",
        worktree="C:/repo/Lethyx",
        branch="main",
        revision="abc123",
        dirty_state="dirty:abc",
        schema_version=1,
        hard_policy_version=3,
        semantic_catalog_version="catalog/experimental",
        host_version="codex/2026-07-14",
        model_version="sol",
        prompt_version="master-plan/phase-3.5",
        touched_files=("src/lethyx/contracts/memory.py",),
        touched_symbols=("MemoryEvent",),
        current_diff="sha256:contract-diff",
        tests=("tests/test_memory_contract_alignment.py",),
        errors=(),
        open_hypotheses=("canonical projection remains advisory",),
        verified_hypotheses=("canonical consumer identity",),
        rejected_approaches=(),
        safety_state=("single canonical writer",),
        open_branch_ids=(),
        completed_branch_ids=("branch-1",),
        isolated_branch_ids=("branch-1",),
        pending_prospective_obligations=(),
        remaining_work=(),
        continuation_sufficiency="sufficient",
        evidence_refs=evidence,
    )


def _canonical_event(**overrides: object) -> MemoryEvent:
    values: dict[str, object] = {
        "content": "canonical identity binding",
        "canonical_key": "workspace:alpha:observation:canonical-binding",
        "revision_id": "rev_canonical_binding",
        "recorded_at": "2026-07-14T09:00:01Z",
        "workspace_id": "workspace:alpha",
        "session_id": "session-1",
        "task_id": "task-1",
        "parent_session_id": "session-parent",
        "artifact_id": "artifact:contract",
        "artifact_revision": "sha256:abc",
        "valid_from": "2026-07-14T09:00:00Z",
        "valid_to": "2026-07-14T10:00:00Z",
        "project_id": "project:alpha",
        "repository_id": "repository:alpha",
        "checkout": "C:/repo/Lethyx",
        "worktree": "C:/repo/Lethyx",
        "branch": "main",
        "revision": "abc123",
        "dirty_state": "sha256:dirty",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "hard_policy_version": 3,
        "semantic_catalog_version": "catalog/experimental",
    }
    values.update(overrides)
    return MemoryEvent(**values)


def test_canonical_identity_survives_normalize_record_event_round_trip() -> None:
    event = _canonical_event(stage_id="verify")
    restored = block_record_to_event(
        backend_contracts.memory_event_to_block_record(
            kernel_models.normalize_memory_event(event.__dict__)
        )
    )
    fields = (
        "canonical_key",
        "revision_id",
        "revision_of",
        "supersedes_ids",
        "duplicate_cluster_id",
        "recorded_at",
        "stage_id",
        "session_id",
        "task_id",
        "parent_session_id",
        "artifact_id",
        "artifact_revision",
        "valid_from",
        "valid_to",
        "project_id",
        "repository_id",
        "checkout",
        "worktree",
        "branch",
        "revision",
        "dirty_state",
        "schema_version",
        "hard_policy_version",
        "semantic_catalog_version",
    )
    assert tuple(getattr(restored, field) for field in fields) == tuple(
        getattr(event, field) for field in fields
    )


def test_serialized_record_requires_exact_current_schema_and_fields() -> None:
    record = MemoryBlockRecord(
        block_id="schema-record",
        content="current schema",
        workspace_id="workspace-a",
    )
    payload = backend_contracts.block_to_json(record)

    assert backend_contracts.block_from_json(payload) == record
    assert {"revision_of", "supersedes_ids"}.issubset(payload)
    assert not {
        "mutation_of",
        "supersedes",
        "superseded_by",
        "token_budget",
    }.intersection(payload)

    missing = dict(payload)
    missing.pop("schema_version")
    with pytest.raises(ValueError, match="missing schema_version"):
        backend_contracts.block_from_json(missing)

    unknown = {**payload, "legacy_phase": "write"}
    with pytest.raises(ValueError, match="unknown legacy_phase"):
        backend_contracts.block_from_json(unknown)

    old = {**payload, "schema_version": CURRENT_SCHEMA_VERSION - 1}
    with pytest.raises(ValueError, match="schema_version"):
        backend_contracts.block_from_json(old)

    for legacy_field in ("mutation_of", "supersedes", "superseded_by", "token_budget"):
        with pytest.raises(ValueError, match="retired memory event fields"):
            kernel_models.normalize_memory_event(
                {"content": "legacy", "workspace_id": "workspace-a", legacy_field: "old"}
            )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"artifact_id": ""}, "artifact_id and artifact_revision"),
        ({"artifact_revision": ""}, "artifact_id and artifact_revision"),
        (
            {
                "valid_from": "2026-07-14T11:00:00Z",
                "valid_to": "2026-07-14T10:00:00Z",
            },
            "valid_to",
        ),
    ),
)
def test_canonical_record_identity_rejects_incomplete_artifact_or_invalid_valid_time(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _canonical_event(**overrides)


@pytest.mark.parametrize(
    "field_name",
    (
        "workspace_id",
        "profile_id",
        "owner_generation",
        "requested_block_id",
        "working_view",
        "host_event_id",
        "session_id",
        "task_id",
        "valid_from",
        "valid_to",
        "semantic_catalog_version",
    ),
)
def test_checkpoint_identity_rejects_blank_critical_fields(field_name: str) -> None:
    values = _checkpoint(_evidence()).__dict__.copy()
    values[field_name] = ""

    with pytest.raises(ValueError, match=field_name):
        WorkingStateCheckpoint(**values)


def _origin_segment(
    claim_id: str,
    *,
    integrity: str,
    domain: str,
) -> OriginSegment:
    return OriginSegment(
        claim_id=claim_id,
        origin=OriginRef(
            channel="host_event",
            principal_id=f"host:{domain}",
            integrity_class=integrity,
        ),
        integrity_ceiling=integrity,
        independence_domain=domain,
        evidence_refs=(
            EvidenceRef(
                ref_type="artifact",
                ref_id=f"artifact:{claim_id}",
                producer_id="harness:contract-alignment",
            ),
        ),
    )


def test_selected_memory_backend_modes_are_closed() -> None:
    assert allowed_memory_backend_modes() == ("mempalace", "hindsight", "mnemosyne", "zvec")

    with pytest.raises(ValueError, match="unknown memory backend: unknown-backend"):
        selected_memory_backend_mode("unknown-backend")


def test_kernel_and_backend_consume_the_canonical_memory_contracts() -> None:
    assert kernel_models.MemoryEvent is MemoryEvent
    assert backend_contracts.MemoryBlockRecord is MemoryBlockRecord


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "unreviewed"),
        ("retention", "forever"),
        ("delivery_policy", "quiet"),
        ("temperature", "always_everywhere"),
        ("authority", "quiet_directive"),
        ("trust_level", "self_verified"),
        ("scope", "foreign_workspace"),
        ("verification_status", "self_verified"),
        ("promotion_decision", "maybe"),
    ),
)
def test_hard_memory_axes_reject_unknown_values(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        MemoryEvent(content="strict axis", **{field: value})


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({}, None),
        ({"status": "candidate"}, "status"),
        ({"verification_status": "unverified"}, "assurance"),
        ({"trust_level": "agent_claim"}, "trust"),
        ({"evidence_refs": ()}, "evidence"),
        ({"promotion_decision": "not_requested"}, "promotion"),
        ({"metadata": {"promotion_gate": "tribunal_accept"}}, "reserved governance"),
    ),
)
def test_directive_record_requires_independent_typed_governance_fields(
    overrides: dict[str, object], message: str | None
) -> None:
    values: dict[str, object] = {
        "block_id": "directive-governance",
        "content": "Use only reviewed evidence.",
        "workspace_id": "workspace-a",
        "status": "accepted",
        "retention": "durable",
        "delivery_policy": "on_demand",
        "temperature": "warm",
        "authority": "directive",
        "trust_level": "workspace_verified",
        "verification_status": "verified",
        "verification_methods": ("test_run",),
        "promotion_decision": "accepted",
        "evidence_refs": (
            EvidenceRef(
                ref_type="test_run",
                ref_id="directive-governance-test",
                produced_by="verifier",
                producer_id="verifier:contract-alignment",
            ),
        ),
    }
    values.update(overrides)

    if message is None:
        record = MemoryBlockRecord(**values)
        assert record.authority == "directive"
    else:
        with pytest.raises(ValueError, match=message):
            MemoryBlockRecord(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"status": "retracted"}, "archive retention"),
        (
            {"status": "retracted", "retention": "archive"},
            "hidden delivery_policy",
        ),
        ({"verification_status": "evidence_backed"}, "verification_methods"),
        (
            {"verification_status": "verified", "verification_methods": ("test_run",)},
            "typed evidence_refs",
        ),
    ),
)
def test_canonical_constructor_rejects_invalid_state_combinations(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MemoryBlockRecord(block_id="invalid-state", content="unsafe state", **overrides)


@pytest.mark.parametrize(
    "metadata",
    (
        {"authority": "directive"},
        {"sensitivity": "secret"},
        {"selected_backend": "hindsight"},
    ),
)
def test_metadata_cannot_set_authority_safety_or_backend(metadata: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="reserved governance fields"):
        MemoryBlockRecord(block_id="metadata-collision", content="safe", metadata=metadata)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"ref_type": "untyped", "ref_id": "artifact:one"},
        {"ref_type": "artifact", "ref_id": "artifact:one", "produced_by": "pytest"},
    ),
)
def test_evidence_ref_requires_closed_type_and_actor(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="evidence"):
        EvidenceRef(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "payload", "message"),
    (
        (
            "evidence",
            {
                "ref_type": "artifact",
                "ref_id": "artifact:one",
                "producer_id": "harness:contract-alignment",
                "phase": "execute",
            },
            "unknown evidence fields",
        ),
        (
            "verification",
            {
                "report_id": "report:one",
                "report_hash": "hash:one",
                "workspace_hash": "workspace:one",
                "diff_hash": "diff:one",
                "produced_after_event_id": "event:one",
                "verifier_id": "verifier:one",
                "passed": "true",
            },
            "passed must be a bool",
        ),
    ),
)
def test_contract_mapping_rejects_unknown_or_coerced_fields(
    kind: str, payload: dict[str, object], message: str
) -> None:
    from_mapping = (
        EvidenceRef.from_mapping if kind == "evidence" else VerificationReportRef.from_mapping
    )

    with pytest.raises(ValueError, match=message):
        from_mapping(payload)


def test_semantic_memory_kind_remains_a_data_driven_catalog_id() -> None:
    event = MemoryEvent(
        content="catalog entry",
        memory_kind="workspace/custom-guidance",
        workspace_id="workspace-a",
    )
    record = MemoryBlockRecord(
        block_id="catalog-entry",
        content="catalog entry",
        kind="workspace/custom-guidance",
        workspace_id="workspace-a",
    )

    assert event.memory_kind == "workspace/custom-guidance"
    assert record.kind == "workspace/custom-guidance"


def test_claim_derivation_preserves_mixed_origin_segments_and_integrity_ceiling() -> None:
    observed = _origin_segment("claim-observed", integrity="observed", domain="host-a")
    verified = _origin_segment("claim-verified", integrity="verified", domain="host-b")
    receipt = DerivationReceipt(
        derived_id="summary-1",
        source_segments=(observed, verified),
        transformation="summarize",
        integrity_ceiling="observed",
    )

    assert tuple(segment.claim_id for segment in receipt.source_segments) == (
        "claim-observed",
        "claim-verified",
    )
    assert receipt.integrity_ceiling == "observed"

    with pytest.raises(ValueError, match="minimum source ceiling"):
        DerivationReceipt(
            derived_id="laundered-summary",
            source_segments=(observed, verified),
            transformation="tool_echo",
            integrity_ceiling="verified",
        )


def test_state_transition_shapes_expose_fields_and_safe_defaults() -> None:
    evidence = _evidence()
    checkpoint = _checkpoint(evidence)
    assessment = ContinuationSufficiencyAssessment(
        workspace_id=checkpoint.workspace_id,
        session_id=checkpoint.session_id,
        task_id=checkpoint.task_id,
        branch_id="branch-1",
        branch_completed=True,
        branch_isolated=True,
        raw_page_back_verified=True,
        projection_sufficiency=checkpoint.continuation_sufficiency,
        source_event_sha256=checkpoint.source_event_sha256,
        exact_literal_manifest=checkpoint.exact_literal_manifest,
        evidence_refs=evidence,
    )
    obligation = ProspectiveObligation(
        obligation_id="follow-up",
        scope="workspace",
        evidence_refs=evidence,
    ).transition_to("armed", evidence_refs=evidence)
    decision = _consolidation_decision(evidence)
    commit = ConsolidationCommitReceipt(
        receipt_id="receipt-1",
        evidence_refs=evidence,
        committed_ids=("memory-1",),
    )
    transfer = TransferEnvelope(envelope_id="transfer-1")

    assert checkpoint.task_id == "task-1"
    assert obligation.status == "armed"
    assert (
        FoldedBranch(
            workspace_id="alpha",
            session_id="session-1",
            task_id="task-1",
            branch_id="branch-1",
            source_event_refs=("event-1",),
            source_event_sha256=checkpoint.source_event_sha256,
            schema_version=CURRENT_SCHEMA_VERSION,
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
            result="branch complete",
            raw_span_refs=("event-1",),
            evidence_refs=evidence,
        ).branch_id
        == "branch-1"
    )
    assert RecallFrame(query="canonical contract", task_id="task-1").query == "canonical contract"
    assert (
        MemoryDecisionEvent(
            decision_id="decision-1",
            workspace_id="alpha",
            run_id="run-1",
            task_signature="contract-alignment",
            action_trace_id="action-1",
            action="recall",
            canonical_snapshot_sha256=f"sha256:{'a' * 64}",
            context_snapshot_sha256=f"sha256:{'b' * 64}",
            live_state_snapshot_sha256=f"sha256:{'c' * 64}",
            live_trigger="contract_test",
            candidate_ids=("memory-1",),
            selected_candidate_ids=("memory-1",),
            policy_revision="policy-1",
            catalog_revision="catalog-1",
            cost_tokens=1,
            cost_latency_ms=1,
            downstream_outcome="pending",
            evidence_refs=evidence,
        ).action
        == "recall"
    )
    assert _credit_fixture(evidence, outcome="neutral").outcome == "neutral"
    assert _credit_fixture(evidence).outcome == "helped"
    assert (
        ContextualUtilitySummary(
            summary_id="utility-1",
            workspace_id="alpha",
            memory_id="memory-1",
            task_family="alignment",
            stage="test",
            project_id="lethyx",
            host_id="codex",
            profile_id="test",
            minimum_support=1,
            minimum_independent_outcomes=1,
            helped_count=0,
            hurt_count=0,
            neutral_count=0,
            outcome_diversity=0,
        ).support_count
        == 0
    )
    assert decision.action == "consolidate"
    assert ConsolidationAssessment(assessment_id="assessment-1").verdict == "defer"
    assert commit.committed_ids == ("memory-1",)
    assert transfer.status == "pending"


def _credit_fixture(
    evidence: tuple[EvidenceRef, ...],
    *,
    outcome: str = "helped",
) -> OutcomeCreditEvent:
    neutral = outcome == "neutral"
    credit_evidence = evidence
    if not neutral:
        credit_evidence = (
            EvidenceRef(
                ref_type="test_run",
                ref_id="alignment-credit",
                produced_by="verifier",
                producer_id="verifier:alignment",
            ),
        )
    return OutcomeCreditEvent(
        credit_id=f"credit-{outcome}",
        decision_id="decision-1",
        workspace_id="alpha",
        task_signature="alignment",
        action_trace_id="action-1",
        memory_ids=("memory-1",),
        applied_usage_event_ids=() if neutral else ("usage-applied",),
        outcome=outcome,
        attribution_basis="insufficient_linkage" if neutral else "deterministic_verifier",
        verifier_result="" if neutral else "passed",
        approval_principal_id="",
        task_family="alignment",
        stage="test",
        project_id="lethyx",
        host_id="codex",
        profile_id="test",
        independence_domain="run-1",
        evidence_refs=credit_evidence,
        reason="contract alignment",
    )


def _consolidation_decision(
    evidence: tuple[EvidenceRef, ...],
    *,
    action: str = "consolidate",
) -> ConsolidationDecision:
    return ConsolidationDecision(
        decision_id="consolidation-1",
        workspace_id="alpha",
        source_episode_ids=("episode-1",),
        grouping_key="alignment",
        task_family="contract",
        trigger="recurrence",
        expected_applicability="same contract family",
        risk="low",
        cost_budget_tokens=64,
        rationale="Exercise the complete consolidation decision shape.",
        evidence_refs=evidence,
        action=action,  # type: ignore[arg-type]
    )


def _outcome_boundary(evidence: tuple[EvidenceRef, ...]) -> OutcomeCreditEvent:
    with pytest.raises(ValueError, match="outcome credit outcome"):
        replace(_credit_fixture(evidence), outcome="unknown")  # type: ignore[arg-type]
    return replace(_credit_fixture(evidence), evidence_refs=())


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (
            lambda evidence: ProspectiveObligation(
                obligation_id="follow-up", scope="workspace", evidence_refs=evidence
            ).transition_to("satisfied", evidence_refs=evidence),
            "invalid prospective transition",
        ),
        (
            lambda evidence: ConsolidationTransition(
                transition_id="transition-1",
                from_status="proposed",
                to_status="accepted",
                evidence_refs=evidence,
            ),
            "invalid consolidation transition",
        ),
        (
            lambda evidence: ConsolidationCommitReceipt(
                receipt_id="receipt-1", evidence_refs=evidence
            ),
            "consolidation commit requires committed_ids",
        ),
        (
            lambda evidence: TransferEnvelope(
                envelope_id="transfer-1", status="accepted", evidence_refs=evidence
            ),
            "accepted transfer source_scope",
        ),
        (
            lambda evidence: _consolidation_decision(evidence, action="erase"),
            "consolidation action",
        ),
        (_outcome_boundary, "outcome credit requires evidence_refs"),
    ),
)
def test_control_bearing_state_contracts_reject_invalid_transition_or_missing_evidence(
    factory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory(_evidence())
