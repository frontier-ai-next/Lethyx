import copy

import pytest

from lethyx.contracts import ConsolidationDecision, EvidenceRef, MemoryBlockRecord
from lethyx.memory.consolidation import (
    CanonicalConsolidationPipeline,
    ConsolidationTransitionTribunal,
    TransitionSemanticVerification,
)


def _evidence(name: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="verification_report",
            ref_id=name,
            produced_by="verifier",
            producer_id="verifier:phase-7-tribunal",
        ),
    )


def _record(
    block_id: str = "episode-rare",
    *,
    content: str = "Keep literal DO_NOT_ROTATE_KEY_7 for the release procedure.",
    metadata: dict[str, object] | None = None,
) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        canonical_key=f"episode:{block_id}",
        revision_id=f"{block_id}@1",
        content=content,
        workspace_id="ws",
        kind="episode",
        status="candidate",
        authority="advisory",
        scope="workspace",
        evidence_refs=_evidence(block_id),
        metadata=metadata
        or {
            "rare_critical": True,
            "exact_literals": ["DO_NOT_ROTATE_KEY_7"],
        },
    )


def _decision(record: MemoryBlockRecord) -> ConsolidationDecision:
    return ConsolidationDecision(
        decision_id="decision-rare",
        workspace_id="ws",
        source_episode_ids=(record.block_id,),
        grouping_key="release-constraints",
        task_family="release",
        trigger="rare_irreversible_fact",
        expected_applicability="same release procedure",
        risk="critical",
        cost_budget_tokens=512,
        rationale="preserve a one-off irreversible constraint",
        evidence_refs=_evidence("decision"),
        action="consolidate",
    )


def _candidate(record: MemoryBlockRecord) -> dict[str, object]:
    return {
        "workspace_id": "ws",
        "authority": "advisory",
        "scope": "workspace",
        "status": "candidate",
        "source_record_ids": [record.block_id],
        "claims": [
            {
                "content": record.content,
                "source_record_ids": [record.block_id],
            }
        ],
        "materialized_views": [
            {
                "view_id": "view-release",
                "content": record.content,
                "source_record_ids": [record.block_id],
            }
        ],
        "continuation_applicability": "same release procedure",
        "procedure": {
            "prerequisites": ["release owner present"],
            "steps": ["retain the exact constraint"],
            "applicability": "same release procedure",
            "contraindications": ["source correction pending"],
            "evidence_refs": ["episode-rare"],
            "outcomes": ["constraint preserved"],
        },
    }


def _transition(
    record: MemoryBlockRecord,
    candidate: dict[str, object],
    *,
    source_coverage_after: tuple[str, ...] | None = None,
):
    return CanonicalConsolidationPipeline().propose(
        _decision(record),
        source_records=(record,),
        new_evidence_refs=_evidence("transition"),
        proposed_operations=("materialize_view", "materialize_procedure"),
        touched_view_ids=("view-release",),
        post_state_candidate=candidate,
        source_coverage_after=(
            source_coverage_after if source_coverage_after is not None else (record.block_id,)
        ),
        policy_version="consolidation-policy-v1",
        model_version="proposer-model-v1",
        prompt_version="consolidation-prompt-v1",
    )


def _semantic(
    *,
    coverage: bool = True,
    preservation: bool = True,
    faithfulness: bool = True,
    independent: bool = True,
    verifier_model_version: str = "independent-verifier-v1",
) -> TransitionSemanticVerification:
    return TransitionSemanticVerification(
        verifier_id="semantic-verifier",
        verifier_model_version=verifier_model_version,
        coverage=coverage,
        preservation=preservation,
        faithfulness=faithfulness,
        independent_from_proposer=independent,
        reasons=("semantic axes evaluated against canonical sources",),
        evidence_refs=_evidence("semantic"),
    )


def test_tribunal_accepts_only_when_rare_exact_fact_and_semantic_axes_survive() -> None:
    record = _record()
    transition = _transition(record, _candidate(record))

    report = ConsolidationTransitionTribunal().assess(
        transition,
        decision=_decision(record),
        source_records=(record,),
        semantic_verification=_semantic(),
        assessment_id="assessment-pass",
    )

    assert report.verdict == "accept"
    assert report.assessment.verdict == "accept"
    assert report.protected_literals == ("DO_NOT_ROTATE_KEY_7",)
    assert all(check.passed for check in report.deterministic_checks)
    assert report.canonical_mutation_applied is False
    assert report.host_action_authorized is False


def test_tribunal_rejects_lost_source_and_rare_exact_literal() -> None:
    record = _record()
    candidate = _candidate(record)
    candidate["claims"] = [{"content": "A lossy summary.", "source_record_ids": []}]
    candidate["materialized_views"] = [
        {"view_id": "view-release", "content": "A lossy summary.", "source_record_ids": []}
    ]
    transition = _transition(record, candidate, source_coverage_after=())

    report = ConsolidationTransitionTribunal().assess(
        transition,
        decision=_decision(record),
        source_records=(record,),
        semantic_verification=_semantic(),
        assessment_id="assessment-loss",
    )

    assert report.verdict == "reject"
    assert "source_coverage" in report.failed_check_ids
    assert "rare_exact_literals" in report.failed_check_ids


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ({"authority": "directive"}, "no_authority_uplift"),
        ({"workspace_id": "other"}, "origin_integrity"),
        ({"scope": "manager"}, "scope_validity_lifecycle"),
        ({"status": "accepted"}, "scope_validity_lifecycle"),
        ({"materialized_views": []}, "affected_materialized_views"),
        ({"continuation_applicability": "all tasks"}, "continuation_applicability"),
        ({"procedure": {"steps": ["unsafe"]}}, "procedure_applicability"),
    ],
)
def test_tribunal_rejects_governance_and_applicability_violations(
    mutation: dict[str, object],
    failed_check: str,
) -> None:
    record = _record()
    candidate = _candidate(record)
    candidate.update(copy.deepcopy(mutation))

    report = ConsolidationTransitionTribunal().assess(
        _transition(record, candidate),
        decision=_decision(record),
        source_records=(record,),
        semantic_verification=_semantic(),
        assessment_id=f"assessment-{failed_check}",
    )

    assert report.verdict == "reject"
    assert failed_check in report.failed_check_ids


def test_tribunal_requires_independent_semantic_coverage_preservation_and_faithfulness() -> None:
    record = _record()
    transition = _transition(record, _candidate(record))

    for semantic in (
        _semantic(independent=False),
        _semantic(verifier_model_version="proposer-model-v1"),
        _semantic(coverage=False),
        _semantic(preservation=False),
        _semantic(faithfulness=False),
    ):
        report = ConsolidationTransitionTribunal().assess(
            transition,
            decision=_decision(record),
            source_records=(record,),
            semantic_verification=semantic,
            assessment_id=f"assessment-semantic-{semantic.verifier_model_version}",
        )
        assert report.verdict == "reject"
        assert "independent_semantic_verification" in report.failed_check_ids


def test_tribunal_enforces_privacy_erase_and_latest_correction_closure() -> None:
    erased = _record(
        "episode-erased",
        content="secret literal ERASE_ME_42",
        metadata={"privacy_erased": True, "exact_literals": ["ERASE_ME_42"]},
    )
    erased_candidate = _candidate(erased)
    erased_report = ConsolidationTransitionTribunal().assess(
        _transition(erased, erased_candidate),
        decision=_decision(erased),
        source_records=(erased,),
        semantic_verification=_semantic(),
        assessment_id="assessment-erased",
    )
    assert erased_report.verdict == "reject"
    assert "privacy_erase_closure" in erased_report.failed_check_ids

    correction = _record(
        "episode-correction",
        content="The supported port is exactly 4318.",
        metadata={
            "exact_literals": ["4318"],
            "correction_of": ["episode-old-port"],
        },
    )
    candidate = _candidate(correction)
    candidate["correction_ids"] = []
    correction_report = ConsolidationTransitionTribunal().assess(
        _transition(correction, candidate),
        decision=_decision(correction),
        source_records=(correction,),
        semantic_verification=_semantic(),
        assessment_id="assessment-correction",
    )
    assert correction_report.verdict == "reject"
    assert "contradiction_correction_closure" in correction_report.failed_check_ids
