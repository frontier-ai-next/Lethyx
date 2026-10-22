from dataclasses import asdict

from um_mem.contracts import (
    EvidenceRef,
    MemoryBlockRecord,
    OriginRef,
    OriginSegment,
    TurnChildClaim,
)
from um_mem.memory.backends.contracts import block_record_to_memory_hit
from um_mem.memory.context_payload import (
    compact_context_item,
    compact_context_metadata,
    excerpt_to_tokens,
    pack_evidence_payload,
    truncate_to_tokens,
)
from um_mem.memory.query_analysis import EvidenceRequirement


def _context_item(block_id: str, text: str) -> dict[str, object]:
    return {
        "text": text,
        "title": f"Source card {block_id}",
        "source": block_id,
        "source_path": f"fixtures/{block_id}.txt",
        "status": "accepted",
        "authority": "advisory",
        "metadata": {
            "block_id": block_id,
            "workspace_id": "packing",
            "source_backend": "hindsight",
            "sensitivity": "internal",
            "content_mode": "full",
            "operational_risk": "low",
        },
    }


def test_compact_context_metadata_keeps_only_canonical_provenance() -> None:
    metadata = {
        "source_backend": "hindsight",
        "block_membership": {
            "primary_block_id": "stable_project_facts",
            "secondary_block_ids": [],
            "catalog_revision_id": "catalog:test",
        },
        "unrelated_debug_field": "omit",
    }

    assert compact_context_metadata(metadata) == {
        "block_membership": {
            "primary_block_id": "stable_project_facts",
            "secondary_block_ids": [],
            "catalog_revision_id": "catalog:test",
        },
        "source_backend": "hindsight",
    }
    assert metadata["unrelated_debug_field"] == "omit"


def test_historical_evidence_keeps_its_validity_window_in_packed_context() -> None:
    record = MemoryBlockRecord(
        block_id="calibration-history",
        content="The calibration coefficient was 1.25.",
        workspace_id="packing",
        status="accepted",
        verification_status="verified",
        verification_methods=("test_run",),
        trust_level="workspace_verified",
        promotion_decision="accepted",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id="calibration-fixture",
                produced_by="verifier",
                producer_id="qa:calibration-fixture",
            ),
        ),
        valid_from="2020-01-01T00:00:00Z",
        valid_to="2020-02-01T00:00:00Z",
        expires_at="2020-03-01T00:00:00Z",
    )
    hit = block_record_to_memory_hit(record)
    assert hit.metadata["expires_at"] == record.expires_at
    item = asdict(hit)
    item["metadata"].update(historical=True, as_of="2020-01-15T00:00:00Z")
    packed = pack_evidence_payload([item], max_tokens=800)
    assert len(packed.items) == 1
    assert packed.estimated_tokens <= 800
    assert packed.items[0]["metadata"]["historical"] is True
    assert "temporal=historical" in packed.rendered_context
    assert "status=current/accepted" not in packed.rendered_context
    for field in ("as_of", "valid_from", "valid_to", "expires_at"):
        assert f"{field}={item['metadata'][field]}" in packed.rendered_context


def test_compact_context_item_omits_unknown_top_level_fields() -> None:
    compact = compact_context_item(
        {
            "text": "Selected source provenance remains advisory.",
            "source": "record-1",
            "unrelated_debug_field": "omit",
        }
    )

    assert compact == {
        "metadata": {},
        "source": "record-1",
        "text": "Selected source provenance remains advisory.",
    }


def test_compact_context_preserves_mixed_origin_child_claim_ceilings() -> None:
    def origin(
        claim_id: str,
        channel: str,
        principal_id: str,
        integrity: str,
    ) -> OriginSegment:
        return OriginSegment(
            claim_id=claim_id,
            origin=OriginRef(
                channel=channel,
                principal_id=principal_id,
                integrity_class=integrity,
            ),
            integrity_ceiling=integrity,
            independence_domain=principal_id,
        )

    record = MemoryBlockRecord(
        block_id="mixed-origin",
        content="A record with mixed child-claim provenance.",
        status="accepted",
        authority="directive",
        trust_level="workspace_verified",
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
        workspace_id="alpha",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id="mixed-origin-test",
                producer_id="verifier:mixed-origin",
            ),
        ),
        child_claims=(
            TurnChildClaim(
                claim_type="observed_fact",
                source_role="conversation",
                content="Observed child claim",
                parent_session_id="session-mixed",
                origin_segments=(
                    origin("observed-child", "user_message", "user:alpha", "observed"),
                ),
            ),
            TurnChildClaim(
                claim_type="verified_fact",
                source_role="conversation",
                content="Verified child claim",
                parent_session_id="session-mixed",
                origin_segments=(origin("verified-child", "tool_result", "tool:beta", "verified"),),
            ),
        ),
    )
    hit = block_record_to_memory_hit(record)

    (item,) = pack_evidence_payload(
        [{"text": hit.text, "source": hit.source, "metadata": hit.metadata}],
        max_tokens=512,
    ).items

    assert item["metadata"]["authority"] == "advisory"
    assert record.authority == "directive"
    claims = item["metadata"]["child_claims"]
    assert [(claim["content"], claim["integrity_ceiling"]) for claim in claims] == [
        ("Observed child claim", "observed"),
        ("Verified child claim", "verified"),
    ]
    assert claims[0]["origin_segments"][0]["source"]["channel"] == "user_message"
    assert claims[0]["origin_segments"][0]["independence_domain"] == "user:alpha"
    assert claims[1]["origin_segments"][0]["source"]["channel"] == "tool_result"
    assert claims[1]["origin_segments"][0]["independence_domain"] == "tool:beta"
    assert {(claim["authority"], claim["trust_level"], claim["status"]) for claim in claims} == {
        ("advisory", "agent_reported", "candidate")
    }


def test_excerpt_ignores_semantic_query_words_and_returns_same_prefix() -> None:
    original = "alpha start marker\n" + ("plain material. " * 180) + "late target"

    first = excerpt_to_tokens(original, 48, query="find the late target")
    second = excerpt_to_tokens(original, 48, query="orchid waveform constellation")

    assert first == second == truncate_to_tokens(original, 48)
    assert "late target" not in first


def test_structural_exact_path_can_use_a_bounded_capsule() -> None:
    exact_path = "src/um_mem/memory/context_payload.py"
    text = "Architecture source.\n" + ("Archived material.\n" * 180) + exact_path

    payload = pack_evidence_payload(
        [_context_item("context-source", text)],
        max_tokens=100,
        query=f"Inspect {exact_path}",
        requirements=(EvidenceRequirement(exact_path, "source", (exact_path,)),),
    )

    assert payload.exact_literal_capsule == (exact_path,)
    assert payload.rendered_context.count(exact_path) == 1


def test_packer_never_promotes_a_later_semantic_match() -> None:
    payload = pack_evidence_payload(
        [
            _context_item("backend-first", "Backend first evidence. " * 80),
            _context_item("later-match", "orchid waveform answer. " * 80),
        ],
        max_tokens=64,
        query="orchid waveform",
    )

    assert [item["source"] for item in payload.items] == ["backend-first"]
    assert "orchid waveform" not in payload.rendered_context


def test_compact_current_context_retains_fact_before_optional_start_date() -> None:
    item = _context_item("calibration", "Aurora calibration is 1.25.")
    item["metadata"]["valid_from"] = "2026-01-01T00:00:00+00:00"
    payload = pack_evidence_payload([item], max_tokens=64)
    assert "Aurora" in payload.rendered_context
    assert payload.items[0]["metadata"]["valid_from"] == "2026-01-01T00:00:00+00:00"
    assert payload.estimated_tokens <= 64


def test_evidence_header_always_exposes_identity_block_revision_source_and_trust() -> None:
    payload = pack_evidence_payload(
        [
            {
                "text": "The API contract is current.",
                "title": "API contract",
                "source": "mempalace:record-7",
                "tags": ["decision", "api"],
                "metadata": {
                    "block_id": "record-7",
                    "revision_id": "revision-3",
                    "recorded_at": "2026-07-22T10:00:00+00:00",
                    "status": "accepted",
                    "source_backend": "mempalace",
                    "trust_level": "workspace_verified",
                    "verification_status": "verified",
                    "sensitivity": "internal",
                    "content_mode": "full",
                    "operational_risk": "low",
                    "temperature": "hot",
                    "importance": 0.99,
                    "native_score": 0.99,
                    "native_rank": 1,
                    "candidate_pool_ids": ["record-7", "record-8"],
                    "block_membership": {
                        "primary_block_id": "decisions_and_contracts",
                        "secondary_block_ids": ["stable_project_facts"],
                        "catalog_revision_id": "catalog:test",
                    },
                },
            }
        ],
        max_tokens=256,
    )

    header = payload.rendered_context.splitlines()[1]
    assert "block=decisions_and_contracts" in header
    assert "record=record-7" in header
    assert "title=API contract" in header
    assert "status=current/accepted" in header
    assert "revision=revision-3" in header
    assert "source=mempalace@2026-07-22T10:00:00+00:00" in header
    assert "trust=agent_reported/unverified" in header
    assert "tags=decision,api" in header
    assert "importance=" not in header
    assert "native_score" not in payload.rendered_context
    assert "native_rank" not in payload.rendered_context
    assert "candidate_pool_ids" not in payload.rendered_context
