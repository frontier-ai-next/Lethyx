from pathlib import Path

import pytest

from um_mem import MemoryClient
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex


def _shared_backend(tmp_path: Path) -> BackendSQLiteIndex:
    return BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="profile-isolation",
        workspace_id="alpha",
    )


def test_non_default_profile_can_write_to_its_dynamic_leaf(tmp_path: Path) -> None:
    backend = _shared_backend(tmp_path)
    try:
        client = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            profile_id="researcher",
            backend_instance=backend,
        )
        root_id = client.block_map()[0]["block_id"]
        leaf = client.create_leaf_block(
            name="Research notes",
            synopsis="Agent-authored notes for this isolated profile.",
            parent_id=root_id,
            reason="profile-local semantic region",
        )

        result = client.remember(
            "The isolated profile retained this note.",
            title="Isolated research note",
            kind="fact",
            requested_block_id=leaf["block_id"],
        )

        assert result.saved is True
        stored = backend.get_block(result.memory_id)
        assert stored is not None
        assert stored.profile_id == "researcher"
        assert stored.block_membership is not None
        assert stored.block_membership.primary_block_id == leaf["block_id"]
    finally:
        backend.close()


def test_profiles_do_not_share_canonical_records_or_recall(tmp_path: Path) -> None:
    backend = _shared_backend(tmp_path)
    try:
        first = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            profile_id="one",
            backend_instance=backend,
        )
        second = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            profile_id="two",
            backend_instance=backend,
        )

        first_result = first.remember(
            "Profile one secret marker.",
            title="Shared title",
            kind="fact",
            requested_block_id="workspace_episodes",
        )
        second_before = second.recall(
            "secret marker",
            logical_block_ids=["workspace_episodes"],
            include_candidates=True,
        )
        second_result = second.remember(
            "Profile two independent marker.",
            title="Shared title",
            kind="fact",
            requested_block_id="workspace_episodes",
        )

        first_hits = first.recall(
            "marker",
            logical_block_ids=["workspace_episodes"],
            include_candidates=True,
        )
        second_hits = second.recall(
            "marker",
            logical_block_ids=["workspace_episodes"],
            include_candidates=True,
        )

        assert first_result.saved is True
        assert second_result.saved is True
        assert first_result.memory_id != second_result.memory_id
        assert second_before == []
        assert [hit.text for hit in first_hits] == ["Profile one secret marker."]
        assert [hit.text for hit in second_hits] == ["Profile two independent marker."]
        assert {hit.metadata["profile_id"] for hit in first_hits} == {"one"}
        assert {hit.metadata["profile_id"] for hit in second_hits} == {"two"}
        assert backend.record_ids_for_logical_blocks(
            ("workspace_episodes",),
            workspace_id="alpha",
            profile_id="one",
        ) == (first_result.memory_id,)
        assert backend.record_ids_for_logical_blocks(
            ("workspace_episodes",),
            workspace_id="alpha",
            profile_id="two",
        ) == (second_result.memory_id,)
        with pytest.raises(ValueError, match="profile"):
            second.forget(first_result.memory_id)
    finally:
        backend.close()
