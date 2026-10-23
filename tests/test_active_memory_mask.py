from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from um_mem import MemoryClient
from um_mem.memory.active_mask import ActiveMemorySelector
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.catalog_store import WorkspaceCatalogStore


def _client(tmp_path: Path, index: BackendSQLiteIndex, *, task_id: str = "task-a") -> MemoryClient:
    return MemoryClient(
        tmp_path,
        workspace_id="alpha",
        profile_id="default",
        session_id="session-a",
        task_id=task_id,
        backend_instance=index,
    )


def test_active_mask_is_durable_and_resolves_new_canonical_head_each_round(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        client = _client(tmp_path, index)
        root = client.block_map()[0]
        leaf = client.create_leaf_block(
            name="Active facts",
            synopsis="Facts that should persist between rounds.",
            parent_id=root["block_id"],
            reason="test active context",
        )
        saved = client.remember(
            "The current opaque value is N7Q.",
            title="Current opaque value",
            kind="fact",
            requested_block_id=leaf["block_id"],
        )
        mask = client.set_active_mask(
            selected_block_ids=(leaf["block_id"],),
            selectors=(
                ActiveMemorySelector(
                    block_id=leaf["block_id"],
                    selector_type="canonical_heads",
                ),
            ),
            token_budget=400,
            activation_reason="needed across rounds",
        )

        first = client.resolve_active_mask()
        assert first.trace.event_type == "active_resolve"
        assert first.trace.mask_id == mask.mask_id
        assert "N7Q" in first.render()

        current = index.get_block(saved.memory_id)
        assert current is not None
        amended = client.amend(
            saved.memory_id,
            expected_revision=current.revision_id,
            content="The current opaque value is R4M.",
            reason="verified update",
            requested_block_id=leaf["block_id"],
        )
        second = client.resolve_active_mask()

        assert "R4M" in second.render()
        assert "N7Q" not in second.render()
        assert amended["new_id"] in second.trace.record_ids

        reloaded = _client(tmp_path, index)
        assert reloaded.get_active_mask() == mask
        assert "R4M" in reloaded.resolve_active_mask().render()

        bounded = reloaded.resolve_active_mask(
            mask_id=mask.mask_id,
            expected_revision=mask.revision,
            max_tokens=40,
        )
        assert bounded.token_budget == 40
        assert bounded.estimated_tokens <= 40
        assert reloaded.get_active_mask() == mask
        stale = reloaded.resolve_active_mask(
            mask_id=mask.mask_id,
            expected_revision=mask.revision + 1,
            max_tokens=40,
        )
        assert stale.trace.resolution_status == "revision_mismatch"
        assert stale.render() == ""
        assert stale.items == ()
    finally:
        index.close()


def test_project_scope_filters_latest_selector_before_its_limit(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            profile_id="default",
            project_id="project-a",
            session_id="session-a",
            task_id="task-a",
            backend_instance=index,
        )
        root_id = client.block_map()[0]["block_id"]
        project_leaf = client.create_leaf_block(
            name="Project context",
            synopsis="Facts for project A.",
            parent_id=root_id,
            scope="project",
            reason="test scoped selector",
        )
        project = client.remember(
            "Project A value is P7.", requested_block_id=project_leaf["block_id"]
        )
        other_project = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            profile_id="default",
            project_id="project-b",
            session_id="session-b",
            task_id="task-b",
            backend_instance=index,
        )
        foreign = other_project.remember(
            "Project B value is W9.", requested_block_id=project_leaf["block_id"]
        )
        assert project.saved and foreign.saved
        mask = client.set_active_mask(
            selected_block_ids=(project_leaf["block_id"],),
            selectors=(
                ActiveMemorySelector(
                    block_id=project_leaf["block_id"],
                    selector_type="latest_records",
                    limit=1,
                ),
            ),
            token_budget=400,
            activation_reason="project context",
        )
        lease = client.resolve_active_mask(
            mask_id=mask.mask_id,
            scope_chain=("project:project-a",),
        )
        assert lease.trace.record_ids == (project.memory_id,)
        assert "W9" not in lease.render()
        query_client = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            profile_id="default",
            project_id="project-a",
            session_id="session-a",
            task_id="task-query",
            backend_instance=index,
        )
        query_mask = query_client.set_active_mask(
            selected_block_ids=(project_leaf["block_id"],),
            selectors=(
                ActiveMemorySelector(
                    block_id=project_leaf["block_id"],
                    selector_type="backend_query",
                    query="Project value",
                    limit=1,
                ),
            ),
            token_budget=400,
            activation_reason="project search",
        )
        query_lease = query_client.resolve_active_mask(
            mask_id=query_mask.mask_id,
            scope_chain=("project:project-a",),
        )
        assert query_lease.trace.record_ids == (project.memory_id,)
        assert "W9" not in query_lease.render()
    finally:
        index.close()


def test_scope_mismatch_returns_empty_typed_resolution_without_changing_mask(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        owner = _client(tmp_path, index, task_id="task-a")
        root = owner.block_map()[0]
        leaf = owner.create_leaf_block(
            name="Task context",
            synopsis="Task-scoped active context.",
            parent_id=root["block_id"],
            reason="test",
        )
        mask = owner.set_active_mask(
            selected_block_ids=(leaf["block_id"],),
            token_budget=200,
            activation_reason="task context",
        )

        other_task = _client(tmp_path, index, task_id="task-b")
        rejected = other_task.resolve_active_mask(mask_id=mask.mask_id)

        assert rejected.render() == ""
        assert rejected.trace.scope_validation_result == "scope_mismatch"
        assert owner.get_active_mask() == mask
    finally:
        index.close()


def test_suspend_resume_and_deactivate_control_automatic_delivery(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        client = _client(tmp_path, index)
        root = client.block_map()[0]
        leaf = client.create_leaf_block(
            name="Lifecycle",
            synopsis="Mask lifecycle test.",
            parent_id=root["block_id"],
            reason="test",
        )
        mask = client.set_active_mask(
            selected_block_ids=(leaf["block_id"],),
            token_budget=100,
            activation_reason="test",
        )

        suspended = client.suspend_active_mask(
            mask.mask_id,
            expected_revision=mask.revision,
            reason="pause",
        )
        assert client.resolve_active_mask().trace.resolution_status == "suspended"
        resumed = client.resume_active_mask(
            mask.mask_id,
            expected_revision=suspended.revision,
            reason="continue",
        )
        assert resumed.status == "active"
        expired = client.deactivate_active_mask(
            mask.mask_id,
            expected_revision=resumed.revision,
            reason="done",
        )
        assert expired.status == "expired"
        assert client.resolve_active_mask().render() == ""
    finally:
        index.close()


def test_active_mask_rejects_archived_blocks_on_create_and_amend(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        client = _client(tmp_path, index)
        root_id = client.block_map()[0]["block_id"]
        selected = client.create_leaf_block(
            name="Selected",
            synopsis="Current active context.",
            parent_id=root_id,
            reason="test",
        )
        archived = client.create_leaf_block(
            name="Archived",
            synopsis="No longer valid active context.",
            parent_id=root_id,
            reason="test",
        )
        client.archive_block(
            archived["block_id"],
            expected_revision=archived["revision"],
            reason="test lifecycle guard",
        )

        with pytest.raises(ValueError, match="active selected blocks"):
            client.set_active_mask(
                selected_block_ids=(archived["block_id"],),
                token_budget=100,
                activation_reason="must fail",
            )

        mask = client.set_active_mask(
            selected_block_ids=(selected["block_id"],),
            token_budget=100,
            activation_reason="valid selection",
        )
        with pytest.raises(ValueError, match="active selected blocks"):
            client.amend_active_mask(
                mask.mask_id,
                expected_revision=mask.revision,
                selected_block_ids=(archived["block_id"],),
                activation_reason="must fail",
            )
    finally:
        index.close()


def test_elapsed_mask_does_not_block_archive_or_a_new_mask(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        client = _client(tmp_path, index)
        root_id = client.block_map()[0]["block_id"]
        elapsed_leaf = client.create_leaf_block(
            name="Elapsed context",
            synopsis="Context whose mask deadline elapsed.",
            parent_id=root_id,
            reason="test",
        )
        next_leaf = client.create_leaf_block(
            name="Next context",
            synopsis="Context selected after the old deadline.",
            parent_id=root_id,
            reason="test",
        )
        elapsed = client.set_active_mask(
            selected_block_ids=(elapsed_leaf["block_id"],),
            token_budget=100,
            activation_reason="short-lived context",
            expires_at="2020-01-01T00:00:00Z",
        )

        assert client.resolve_active_mask().trace.resolution_status == "expired"
        archived = client.archive_block(
            elapsed_leaf["block_id"],
            expected_revision=elapsed_leaf["revision"],
            reason="elapsed lease no longer protects the block",
        )
        assert archived["status"] == "archived"

        replacement = client.set_active_mask(
            selected_block_ids=(next_leaf["block_id"],),
            token_budget=100,
            activation_reason="replace elapsed mask",
        )
        assert replacement.mask_id != elapsed.mask_id
        assert replacement.status == "active"
    finally:
        index.close()


def test_suspended_mask_cannot_resume_over_another_mask_in_exact_scope(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        client = _client(tmp_path, index)
        root_id = client.block_map()[0]["block_id"]
        leaf = client.create_leaf_block(
            name="Scoped context",
            synopsis="Context shared by two stored mask revisions.",
            parent_id=root_id,
            reason="test",
        )
        original = client.set_active_mask(
            selected_block_ids=(leaf["block_id"],),
            token_budget=100,
            activation_reason="original",
        )
        suspended = client.suspend_active_mask(
            original.mask_id,
            expected_revision=original.revision,
            reason="pause",
        )

        now = datetime.now(UTC).isoformat()
        store = WorkspaceCatalogStore(tmp_path, "alpha", "default")
        try:
            replacement = replace(
                suspended,
                mask_id="7b1b2c34-9c8c-4b63-8d8d-349917d259ab",
                revision=1,
                status="active",
                activation_reason="replacement",
                created_at=now,
                updated_at=now,
            )
            store.put_mask(replacement)
        finally:
            store.close()

        with pytest.raises(ValueError, match="another active memory mask"):
            client.resume_active_mask(
                suspended.mask_id,
                expected_revision=suspended.revision,
                reason="must not create two active masks",
            )
    finally:
        index.close()
