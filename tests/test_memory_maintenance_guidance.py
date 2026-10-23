from um_mem.memory.mcp_compact import compact_response


def test_compact_exact_record_keeps_evidence_references(monkeypatch) -> None:
    monkeypatch.setenv("UM_MEM_MCP_RESPONSE_MODE", "compact")
    evidence_refs = [
        {
            "ref_type": "test_run",
            "ref_id": "pytest:tests/test_example.py",
            "produced_by": "verifier",
            "producer_id": "pytest",
        }
    ]

    result = compact_response(
        "read_exact_record",
        {
            "ok": True,
            "record": {
                "block_id": "record-2",
                "revision_id": "revision-2",
                "revision_of": "record-1",
                "title": "Current parser behavior",
                "content": "The parser uses the checked writer.",
                "evidence_refs": evidence_refs,
            },
        },
    )

    assert result["record"]["evidence_refs"] == evidence_refs
    assert result["record"]["revision_of"] == "record-1"

    exact_batch = compact_response(
        "read_exact_records",
        {
            "ok": True,
            "records": [
                {
                    "block_id": "record-2",
                    "revision_id": "revision-2",
                    "evidence_refs": evidence_refs,
                }
            ],
        },
    )
    assert exact_batch["records"][0]["evidence_refs"] == evidence_refs

    listed = compact_response(
        "list_records",
        {
            "ok": True,
            "records": [
                {
                    "block_id": "record-2",
                    "revision_id": "revision-2",
                    "evidence_refs": evidence_refs,
                }
            ],
        },
    )
    assert "evidence_refs" not in listed["records"][0]
