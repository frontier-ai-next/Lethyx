from dataclasses import FrozenInstanceError

import pytest

from um_mem.contracts import ConsolidationDecision, EvidenceRef, MemoryBlockRecord
from um_mem.memory.consolidation import (
    CanonicalConsolidationPipeline,
    ConsolidationShadowStore,
    ConsolidationTransitionTribunal,
    MaterializedSourceRevision,
    MaterializedViewRevision,
    PreservationManifest,
    TransitionSemanticVerification,
    build_consolidation_source_envelope,
)


def _evidence(name: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="verification_report",
            ref_id=name,
            produced_by="verifier",
            producer_id="verifier:phase-7-shadow",
        ),
    )


def _record(
    block_id: str = "episode-1",
    *,
    revision_id: str = "episode-1@1",
    content: str = "Preserve exact literal RELEASE_PORT_4318.",
    valid_from: str = "2026-07-15T00:00:00Z",
    valid_to: str = "2026-07-16T00:00:00Z",
) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        canonical_key=f"episode:{block_id}",
        revision_id=revision_id,
        content=content,
        workspace_id="ws",
        kind="episode",
        status="candidate",
        authority="advisory",
        scope="workspace",
        valid_from=valid_from,
        valid_to=valid_to,
        evidence_refs=_evidence(block_id),
        metadata={"rare_critical": True, "exact_literals": ["RELEASE_PORT_4318"]},
    )


def _decision(record: MemoryBlockRecord, suffix: str) -> ConsolidationDecision:
    return ConsolidationDecision(
        decision_id=f"decision-{suffix}",
        workspace_id="ws",
        source_episode_ids=(record.block_id,),
        grouping_key="release",
        task_family="release",
        trigger="rare_irreversible_fact",
        expected_applicability="same release procedure",
        risk="critical",
        cost_budget_tokens=256,
        rationale="preserve release constraint",
        evidence_refs=_evidence(f"decision-{suffix}"),
        action="consolidate",
    )


def _candidate(record: MemoryBlockRecord, suffix: str) -> dict[str, object]:
    return {
        "workspace_id": "ws",
        "authority": "advisory",
        "scope": "workspace",
        "status": "candidate",
        "source_record_ids": [record.block_id],
        "claims": [{"content": record.content, "source_record_ids": [record.block_id]}],
        "materialized_views": [
            {
                "view_id": f"view-{suffix}",
                "content": record.content,
                "source_record_ids": [record.block_id],
            }
        ],
        "continuation_applicability": "same release procedure",
    }


def _transition(record: MemoryBlockRecord, suffix: str):
    decision = _decision(record, suffix)
    transition = CanonicalConsolidationPipeline().propose(
        decision,
        source_records=(record,),
        new_evidence_refs=_evidence(f"transition-{suffix}"),
        proposed_operations=("materialize_view",),
        touched_view_ids=(f"view-{suffix}",),
        post_state_candidate=_candidate(record, suffix),
        source_coverage_after=(record.block_id,),
        policy_version="policy-v1",
        model_version="proposer-v1",
        prompt_version="prompt-v1",
    )
    return decision, transition


def _report(record: MemoryBlockRecord, suffix: str, *, faithfulness: bool = True):
    decision, transition = _transition(record, suffix)
    semantic = TransitionSemanticVerification(
        verifier_id="independent-reader",
        verifier_model_version="reader-v1",
        coverage=True,
        preservation=True,
        faithfulness=faithfulness,
        independent_from_proposer=True,
        reasons=("canonical source comparison complete",),
        evidence_refs=_evidence(f"semantic-{suffix}"),
    )
    report = ConsolidationTransitionTribunal().assess(
        transition,
        decision=decision,
        source_records=(record,),
        semantic_verification=semantic,
        assessment_id=f"assessment-{suffix}",
    )
    return transition, report


def test_shadow_store_allows_one_candidate_and_commits_only_accepted_tribunal() -> None:
    record = _record()
    transition, report = _report(record, "accepted")
    store = ConsolidationShadowStore()

    snapshot = store.stage(workspace_id="ws", transition=transition)
    with pytest.raises(ValueError, match="one bounded candidate"):
        store.stage(workspace_id="ws", transition=transition)

    resolution = store.resolve(snapshot.snapshot_id, report=report)

    assert resolution.disposition == "committed"
    assert store.active_head("ws") == transition.post_state_candidate_sha256
    assert tuple(store.active_views("ws")) == ("view-accepted",)
    assert resolution.commit_receipt is not None
    assert resolution.commit_receipt.previous_head_sha256 == ""
    assert resolution.commit_receipt.active_head_sha256 == transition.post_state_candidate_sha256


def test_reject_or_quarantine_keeps_active_head_and_views_and_blocks_same_revision() -> None:
    record = _record()
    accepted_transition, accepted_report = _report(record, "base")
    store = ConsolidationShadowStore()
    accepted_snapshot = store.stage(workspace_id="ws", transition=accepted_transition)
    store.resolve(accepted_snapshot.snapshot_id, report=accepted_report)
    old_head = store.active_head("ws")
    old_views = store.active_views("ws")

    rejected_transition, rejected_report = _report(
        record,
        "unsafe",
        faithfulness=False,
    )
    rejected_snapshot = store.stage(workspace_id="ws", transition=rejected_transition)
    resolution = store.resolve(rejected_snapshot.snapshot_id, report=rejected_report)

    assert resolution.disposition == "quarantined"
    assert store.active_head("ws") == old_head
    assert store.active_views("ws") == old_views
    assert resolution.rejection_receipt is not None
    assert resolution.rejection_receipt.verifier_reasons
    assert not hasattr(resolution.rejection_receipt, "judge_score")
    with pytest.raises(FrozenInstanceError):
        resolution.rejection_receipt.disposition = "accepted"  # type: ignore[misc]
    with pytest.raises(ValueError, match="previously rejected"):
        store.stage(workspace_id="ws", transition=rejected_transition)


def test_working_fold_and_durable_consolidation_share_preservation_not_identity() -> None:
    manifest = PreservationManifest(
        raw_source_ids=("episode-1", "episode-2"),
        protected_literals=("RELEASE_PORT_4318",),
        correction_ids=("episode-old",),
        evidence_refs=_evidence("preservation"),
    )

    fold = build_consolidation_source_envelope(
        envelope_id="fold-1",
        record_kind="working_fold",
        lifetime="task",
        direct_source_ids=("folded-branch-1",),
        preservation=manifest,
    )
    durable = build_consolidation_source_envelope(
        envelope_id="durable-1",
        record_kind="durable_consolidation",
        lifetime="durable",
        direct_source_ids=("episode-1", "episode-2"),
        preservation=manifest,
    )

    assert fold.record_kind != durable.record_kind
    assert fold.lifetime != durable.lifetime
    assert fold.preservation == durable.preservation
    assert fold.transitive_raw_source_ids == ("episode-1", "episode-2")
    with pytest.raises(ValueError, match="transitive raw"):
        PreservationManifest(
            raw_source_ids=(),
            protected_literals=("RELEASE_PORT_4318",),
            correction_ids=(),
            evidence_refs=_evidence("missing-raw"),
        )


def test_materialized_view_invalidates_on_source_hash_revision_validity_or_lifecycle() -> None:
    source = _record()
    source_revision = MaterializedSourceRevision.from_record(source)
    view = MaterializedViewRevision(
        view_id="view-release",
        workspace_id="ws",
        content=source.content,
        sources=(source_revision,),
        evidence_refs=_evidence("view"),
    )

    assert view.revalidate((source,), as_of="2026-07-15T12:00:00Z").status == "active"

    changed_revision = _record(revision_id="episode-1@2")
    revision_result = view.revalidate((changed_revision,), as_of="2026-07-15T12:00:00Z")
    assert revision_result.status == "invalidated"
    assert "source_revision_changed:episode-1" in revision_result.invalidation_reasons

    changed_content = _record(content="Preserve different literal RELEASE_PORT_4318.")
    hash_result = view.revalidate((changed_content,), as_of="2026-07-15T12:00:00Z")
    assert hash_result.status == "invalidated"
    assert "source_hash_changed:episode-1" in hash_result.invalidation_reasons

    expired_result = view.revalidate((source,), as_of="2026-07-17T00:00:00Z")
    assert expired_result.status == "invalidated"
    assert "source_outside_validity:episode-1" in expired_result.invalidation_reasons

    boundary_result = view.revalidate((source,), as_of=source.valid_to)
    assert boundary_result.status == "invalidated"
    assert "source_outside_validity:episode-1" in boundary_result.invalidation_reasons
