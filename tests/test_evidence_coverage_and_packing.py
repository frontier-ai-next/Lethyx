import json

from lethyx.integrations.hermes_provider import UmMemMemoryProvider
from lethyx.llm_budget import estimate_tokens
from lethyx.memory.context_payload import PackedEvidencePayload, pack_evidence_payload
from lethyx.memory.query_analysis import EvidenceRequirement


def _item(
    block_id: str,
    text: str,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "title": block_id,
        "source": block_id,
        "status": "supplemental non-directive",
        "authority": "advisory",
        "metadata": {
            "block_id": block_id,
            "sensitivity": "internal",
            "content_mode": "full",
            "operational_risk": "low",
            **(metadata or {}),
        },
    }


def test_negative_evidence_keeps_subject_attribution_in_card_and_metadata() -> None:
    payload = pack_evidence_payload(
        [
            _item(
                "dependency-audit",
                "No verified dependency declaration names ReportKit.",
                metadata={
                    "evidence_roles": ["negative_evidence"],
                    "subject_id": "repository:alpha",
                    "speaker_id": "tool:manifest-audit",
                    "belief_holder_id": "repository:alpha",
                },
            )
        ],
        max_tokens=160,
        query="What dependency evidence is missing for repository alpha?",
    )

    assert payload.items[0]["metadata"]["subject_id"] == "repository:alpha"
    assert payload.items[0]["metadata"]["speaker_id"] == "tool:manifest-audit"
    assert "subject=repository:alpha" in payload.rendered_context
    assert "attribution=subject_id" in payload.rendered_context


def test_unattributed_negative_evidence_is_marked_instead_of_inheriting_query_subject() -> None:
    payload = pack_evidence_payload(
        [
            _item(
                "unattributed-audit",
                "No matching declaration was found.",
                metadata={"evidence_roles": ["negative_evidence"]},
            )
        ],
        max_tokens=120,
        query="What dependency evidence is missing for repository alpha?",
    )

    assert "subject=unattributed" in payload.rendered_context
    assert "subject=repository:alpha" not in payload.rendered_context


def test_large_record_query_span_preserves_supported_exact_literal_capsule_once() -> None:
    exact_path = "src/lethyx/memory/context_payload.py"
    text = (
        "Architecture source card.\n"
        + ("Unrelated historical material.\n" * 180)
        + f"Canonical implementation path: {exact_path}\n"
        + ("Trailing archival material.\n" * 40)
    )

    payload = pack_evidence_payload(
        [_item("context-source", text)],
        max_tokens=120,
        query=f"Inspect {exact_path} and name the final packer.",
        requirements=(EvidenceRequirement(exact_path, "source", (exact_path,)),),
    )

    assert payload.exact_literal_capsule == (exact_path,)
    assert payload.items[0]["metadata"]["exact_literal_capsule"] == [exact_path]
    assert payload.rendered_context.count(exact_path) == 1
    assert payload.items[0]["text"].count(exact_path) == 1


def test_packer_does_not_claim_short_literal_inside_a_larger_word() -> None:
    payload = pack_evidence_payload(
        [_item("units", "The available units are documented here.")],
        max_tokens=120,
        query="What does its configuration say?",
        requirements=(EvidenceRequirement("its", "source", ("its",)),),
    )

    assert payload.exact_literal_capsule == ()
    assert "exact_literal_capsule" not in payload.items[0]["metadata"]


def test_canonical_packer_returns_final_cards_and_accounting() -> None:
    payload = pack_evidence_payload(
        [
            _item("one", "First selected evidence " * 40),
            _item("two", "Second selected evidence " * 40),
        ],
        max_tokens=150,
        query="selected evidence",
    )

    assert isinstance(payload, PackedEvidencePayload)
    assert payload.token_budget == 150
    assert payload.estimated_tokens == estimate_tokens(payload.rendered_context)
    assert payload.estimated_tokens <= payload.token_budget
    assert payload.truncated_item_count == 1
    assert all(f"fact={item['text']}" in payload.rendered_context for item in payload.items)


def test_packer_preserves_first_complete_fact_instead_of_many_metadata_stubs() -> None:
    fact = (
        "A source-dated account of the service and its background. " * 4
        + "The service includes individual mentoring, workshops and training."
    )
    items = [
        _item(
            f"mem_{index:016x}",
            fact + " Further source detail." * 20,
            metadata={
                "revision_id": "rev_" + "a" * 64,
                "block_membership": {"primary_block_id": "b" * 36},
                "logical_block_path": "b" * 36,
                "recorded_at": "2026-10-15T00:00:00Z",
                "trust_level": "agent_reported",
                "verification_status": "unverified",
            },
        )
        for index in range(20)
    ]
    payload = pack_evidence_payload(items, max_tokens=2048)

    assert payload.items[0]["text"] == items[0]["text"]
    assert payload.items[0]["truncated"] is False
    assert payload.estimated_tokens <= 2048
    assert payload.metadata_tokens < payload.estimated_tokens // 3
    assert [item["source"] for item in payload.items] == [
        item["source"] for item in items[: len(payload.items)]
    ]
    assert payload.items[0]["metadata"]["revision_id"] == "rev_" + "a" * 64
    assert any(v.state == "not_visible" for v in payload.visibility)


def test_context_budget_keeps_backend_prefix_without_query_reranking() -> None:
    payload = pack_evidence_payload(
        [
            _item("backend-first", "The backend ranked this record first. " * 20),
            _item("later-exact", "OPAQUE_VALUE_7f3a appears only in the later record. " * 20),
        ],
        max_tokens=64,
        query="OPAQUE_VALUE_7f3a",
    )

    assert [item["source"] for item in payload.items] == ["backend-first"]


def test_packer_reserves_each_requirement_literal_and_keeps_core_order() -> None:
    first_path = "src/packages/alpha/manifest.py"
    second_path = "src/packages/beta/manifest.py"
    requirements = (
        EvidenceRequirement("alpha", "source", (first_path,)),
        EvidenceRequirement("beta", "source", (second_path,)),
    )
    payload = pack_evidence_payload(
        [
            _item(
                "alpha",
                "Alpha evidence. " + ("long background material " * 100) + first_path,
            ),
            _item(
                "beta",
                "Beta evidence. " + ("long background material " * 100) + second_path,
            ),
        ],
        max_tokens=180,
        query=f"Compare {first_path} and {second_path}",
        requirements=requirements,
    )

    assert [item["source"] for item in payload.items] == ["alpha", "beta"]
    assert first_path in payload.rendered_context
    assert second_path in payload.rendered_context
    assert payload.exact_literal_losses == ()


def test_canonical_packer_keeps_six_long_requirement_cards_within_12k() -> None:
    literals = tuple(f"src/providers/provider_{index}/client.py" for index in range(6))
    requirements = tuple(
        EvidenceRequirement(
            subject=f"provider-{index}",
            role="source",
            accepted_literals=(literal,),
        )
        for index, literal in enumerate(literals)
    )
    items = [
        _item(
            f"provider-{index}",
            f"Provider {index} source evidence. "
            + (f"typed dependency detail {index} " * 150)
            + literal,
        )
        for index, literal in enumerate(literals)
    ]

    payload = pack_evidence_payload(
        items,
        max_tokens=12_000,
        query="Final synthesis across all six provider source cards.",
        requirements=requirements,
    )

    assert len(payload.items) == 6
    assert payload.estimated_tokens <= 12_000
    assert payload.exact_literal_losses == ()
    assert all(literal in payload.rendered_context for literal in literals)


def test_hermes_accepts_only_final_payload_and_never_repacks_items() -> None:
    provider = UmMemMemoryProvider()
    provider._config = {"recall_max_tokens": 160, "recall_limit": 2}
    final = pack_evidence_payload(
        [_item("exact", "EXACT_LITERAL_9b1f13 is final evidence.")],
        max_tokens=160,
        query="EXACT_LITERAL_9b1f13",
    )

    rendered = provider._format_context(
        {
            "ok": True,
            "items": list(final.items),
            "rendered_context": final.rendered_context,
            "diagnostics": {"rendered_item_count": 1},
        },
        max_tokens=160,
        query="EXACT_LITERAL_9b1f13",
    )

    assert rendered == final.rendered_context
    assert rendered.count("EXACT_LITERAL_9b1f13") == 1
    assert provider._format_context({"ok": True, "items": list(final.items)}) == ""


def test_hermes_tool_wrapper_keeps_final_context_byte_for_byte() -> None:
    provider = UmMemMemoryProvider()
    final = pack_evidence_payload(
        [_item("route", "The route is src/routes/tasks.py.")],
        max_tokens=120,
        query="src/routes/tasks.py",
    )

    rendered = provider._compact_recall_tool_result(
        {
            "ok": True,
            "backend": "mempalace",
            "items": list(final.items),
            "rendered_context": final.rendered_context,
            "diagnostics": {
                "rendered_item_count": 1,
                "truncated_item_count": final.truncated_item_count,
            },
        },
        query_hash="abc123",
        max_tokens=180,
        context_tokens=120,
        query="src/routes/tasks.py",
    )

    assert json.loads(rendered)["context"] == final.rendered_context


def test_hermes_tool_wrapper_exposes_unrendered_exact_record_refs() -> None:
    provider = UmMemMemoryProvider()
    result = json.loads(
        provider._compact_recall_tool_result(
            {
                "ok": True,
                "backend": "mempalace",
                "rendered_context": "E1: record=visible block=leaf\nfact=visible evidence",
                "items": [{"text": "visible evidence"}],
                "diagnostics": {
                    "rendered_item_count": 1,
                    "visibility": [
                        {
                            "record_id": "visible",
                            "logical_block_id": "leaf",
                            "state": "full_visible",
                        },
                        {"record_id": "hidden", "logical_block_id": "leaf", "state": "not_visible"},
                    ],
                },
            },
            query_hash="query",
            max_tokens=256,
            context_tokens=128,
        )
    )
    assert result["context"].endswith("visible evidence")
    assert result["unrendered_record_refs"] == [{"record_id": "hidden", "logical_block_id": "leaf"}]
    assert result["unrendered_record_count"] == 1


def test_hermes_tool_wrapper_never_reports_empty_evidence_as_success() -> None:
    provider = UmMemMemoryProvider()
    result = json.loads(
        provider._compact_recall_tool_result(
            {
                "ok": True,
                "backend": "mempalace",
                "rendered_context": "\\" * 950,
                "items": [{"text": "evidence"}],
                "diagnostics": {"rendered_item_count": 1},
            },
            query_hash="query",
            max_tokens=256,
            context_tokens=256,
        )
    )

    assert result["ok"] is True
    assert result["context"]
    assert result["truncated"] is True
    assert (
        len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= 1024
    )
