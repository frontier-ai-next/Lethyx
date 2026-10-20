"""Exact retraction remains available after a leaf is archived; history stays immutable."""

from pathlib import Path

from lethyx.api import MemoryClient
from lethyx.memory.blocks.registry import load_workspace_topology


def test_archived_leaf_retraction_is_exact_recoverable_and_idempotent(tmp_path: Path) -> None:
    client = MemoryClient(tmp_path, workspace_id="archive-test", backend="mempalace")
    try:
        root = client.block_map()[0]
        leaf = client.create_leaf_block(
            name="Measured result",
            synopsis="One independently checked result",
            parent_id=root["block_id"],
            reason="explicit test scope",
        )
        saved = client.remember(
            "A retained result",
            title="Result",
            requested_block_id=leaf["block_id"],
        )
        assert saved.saved
        original = client.backend.get_block(saved.memory_id)
        assert original is not None
        archive = client.archive_block(
            leaf["block_id"],
            expected_revision=leaf["revision"],
            reason="close completed scope",
        )
        assert archive["archived"]
        wrong = client.forget(saved.memory_id, expected_revision="different", reason="stale")
        assert wrong["ok"] is False
        result = client.forget(
            original.block_id,
            expected_revision=original.revision_id,
            reason="withdraw exact result",
        )
        assert result["ok"] is True and result["saved"] is True
        retracted = client.backend.get_block(result["block_id"])
        assert retracted is not None and retracted.status == "retracted"
        assert retracted.revision_of == original.block_id
        historical = client.backend.get_block(original.block_id)
        assert historical is not None
        # Current validity bounds are a projection; retained authored bytes/revision are immutable.
        assert (
            historical.content,
            historical.title,
            historical.evidence_refs,
            historical.revision_id,
            historical.canonical_key,
        ) == (
            original.content,
            original.title,
            original.evidence_refs,
            original.revision_id,
            original.canonical_key,
        )
        assert result["logical_index_sync"]["status"] == "not_applicable"
        assert retracted.block_membership.primary_block_id == leaf["block_id"]
        repeated = client.forget(
            original.block_id,
            expected_revision=original.revision_id,
            reason="withdraw exact result",
        )
        assert repeated["ok"] is True and repeated["already_retracted"] is True
        assert repeated["block_id"] == retracted.block_id
        assert client.backend.get_block(retracted.block_id) == retracted
        # Retraction does not revive the leaf or authorize new knowledge there.
        archived = next(
            state
            for state in load_workspace_topology(
                tmp_path,
                "archive-test",
                include_archived=True,
            )
            if state.block_id == leaf["block_id"]
        )
        assert archived.status == "archived"
        denied = client.remember("New result", title="Other", requested_block_id=leaf["block_id"])
        assert denied.saved is False
    finally:
        client.backend.close()
