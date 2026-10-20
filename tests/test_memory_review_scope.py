"""Context cards do not infer independent review from historical lifecycle labels."""

from dataclasses import asdict

from lethyx.contracts import EvidenceRef, MemoryBlockRecord, MemorySemanticReview
from lethyx.memory.context_payload import compact_context_metadata
from lethyx.memory.mcp_facade import _packed_payload_from_records


def test_historical_verified_label_does_not_claim_independent_review() -> None:
    recorded = {
        "status": "accepted",
        "verification_status": "verified",
        "trust_level": "workspace_verified",
    }
    projected = compact_context_metadata(recorded)
    assert projected["status"] == "accepted"
    assert projected["verification_status"] == "unverified"
    assert projected["trust_level"] == "agent_reported"
    assert recorded["verification_status"] == "verified"


def test_declared_semantic_scope_alone_cannot_grant_review_assurance() -> None:
    projected = compact_context_metadata(
        {
            "verification_scope": "semantic",
            "verification_status": "verified",
            "trust_level": "workspace_verified",
        }
    )
    assert projected["verification_status"] == "unverified"


def test_legacy_directive_projection_is_advisory_without_exact_review() -> None:
    record = MemoryBlockRecord(
        block_id="legacy-directive",
        workspace_id="alpha",
        content="An earlier accepted record without exact semantic review.",
        status="accepted",
        authority="directive",
        trust_level="workspace_verified",
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id="legacy-test",
                produced_by="verifier",
                producer_id="legacy-verifier",
            ),
        ),
    )
    (item,) = _packed_payload_from_records([record], max_tokens=1024).items

    assert item["authority"] == "advisory"
    assert item["metadata"]["authority"] == "advisory"
    assert item["metadata"]["source_authority"] == "directive"
    assert item["metadata"]["verification_status"] == "unverified"
    assert record.authority == "directive"
    assert record.verification_status == "verified"


def test_current_exact_semantic_proof_retains_its_recorded_review_assurance() -> None:
    review = MemorySemanticReview(
        record_id="candidate",
        revision_id="revision",
        content_hash="a" * 64,
        source_authority_id="work",
        source_authority_hash="b" * 64,
        review_authority_id="review",
        review_authority_hash="c" * 64,
        manifest_hash="d" * 64,
    )
    projected = compact_context_metadata(
        {
            "verification_scope": "semantic",
            "verification_status": "verified",
            "trust_level": "workspace_verified",
            "authority": "directive",
            "semantic_review": asdict(review),
            "promotion_source_id": review.record_id,
            "evidence_refs": [
                {
                    "ref_type": "verification_report",
                    "producer_id": review.review_authority_id,
                    "hash": "exact-host-ledger-hash",
                }
            ],
        }
    )
    assert projected["verification_status"] == "verified"
    assert projected["authority"] == "directive"
