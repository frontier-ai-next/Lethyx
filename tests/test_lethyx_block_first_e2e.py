from lethyx.api import MemoryClient
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex


def test_e2e_block_first_routed_write_and_context_filters_logical_blocks(tmp_path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite", backend_id="sqlite", workspace_id="alpha"
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        client.remember(
            "Parser invariant validated by focused pytest run",
            title="Parser invariant",
            kind="lesson",
            evidence_kind="test_run",
            evidence_refs=[
                {
                    "ref_type": "test_run",
                    "ref_id": "test-parser",
                    "produced_by": "verifier",
                    "producer_id": "verifier:test-parser",
                }
            ],
            tags=["parser"],
            requested_block_id="artifact_claims",
        )
        client.remember(
            "Parser brainstorm candidate not yet verified",
            title="Parser brainstorm",
            kind="idea",
            tags=["parser"],
            requested_block_id="project_ideas",
        )

        items = client.build_context(
            "parser",
            source_horizon="durable",
            include_candidates=True,
            logical_block_ids=["artifact_claims"],
        )

        assert [item["metadata"]["block_id"] for item in items]
        assert all(
            item["metadata"]["block_membership"]["primary_block_id"] == "artifact_claims"
            for item in items
        )
        assert "validated" in items[0]["text"]
    finally:
        index.close()
