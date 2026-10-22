from pathlib import Path

import pytest

from um_mem import MemoryClient
from um_mem.contracts import EvidenceRef
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.blocks.registry import load_memory_blocks


def test_amend_appends_revision_and_rejects_stale_or_governance_patch(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        client = MemoryClient(
            repo_root=tmp_path,
            workspace_id="alpha",
            backend_instance=index,
        )
        first = client.remember(
            "The service uses Flask.",
            title="Service framework",
            kind="project_fact",
            tags=["framework", "old"],
            metadata={"importance": 0.4, "source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        original = index.get_block(first.memory_id)
        assert original is not None

        amended = client.amend(
            first.memory_id,
            expected_revision=original.revision_id,
            content="The service uses FastAPI.",
            tags=["framework", "current"],
            metadata_patch={"importance": 0.9, "source_path": "docs/architecture.md"},
            expires_at="2027-01-01T00:00:00Z",
            reason="architecture corrected",
            evidence_refs=(
                EvidenceRef(
                    ref_type="test_run",
                    ref_id="pytest:framework",
                    produced_by="verifier",
                    producer_id="pytest",
                ),
            ),
            requested_block_id="stable_project_facts",
        )

        assert amended["ok"] is True
        assert amended["old_id"] == first.memory_id
        assert amended["new_id"] != first.memory_id
        assert amended["old_revision"] == original.revision_id
        assert amended["new_revision"] != original.revision_id
        assert amended["lifecycle"] == {
            "old_status": "superseded",
            "new_status": "candidate",
        }
        assert amended["graph_edge"]["edge_type"] == "supersedes"
        assert amended["graph_edge"]["src_id"] == amended["new_id"]
        assert amended["graph_edge"]["dst_id"] == first.memory_id

        previous = index.get_block(first.memory_id)
        current = index.get_block(amended["new_id"])
        assert previous is not None and previous.content == "The service uses Flask."
        assert current is not None and current.content == "The service uses FastAPI."
        assert current.tags == ("framework", "current")
        assert current.metadata["importance"] == 0.9
        assert current.expires_at == "2027-01-01T00:00:00Z"
        assert current.revision_of == first.memory_id
        assert current.supersedes_ids == (first.memory_id,)

        hits = client.recall(
            "service framework FastAPI",
            include_candidates=True,
            logical_block_ids=["stable_project_facts"],
        )
        assert [hit.metadata["block_id"] for hit in hits] == [amended["new_id"]]
        assert all("Flask" not in hit.text for hit in hits)

        with pytest.raises(ValueError, match="expected_revision"):
            client.amend(
                first.memory_id,
                expected_revision=original.revision_id,
                content="A stale correction.",
                reason="stale writer",
                requested_block_id="stable_project_facts",
            )
        with pytest.raises(ValueError, match="governance"):
            client.amend(
                amended["new_id"],
                expected_revision=amended["new_revision"],
                metadata_patch={"authority": "directive"},
                reason="invalid authority escalation",
                requested_block_id="stable_project_facts",
            )
    finally:
        index.close()


def test_dynamic_block_tree_is_workspace_scoped_reversible_and_cycle_safe(
    tmp_path: Path,
) -> None:
    alpha = MemoryClient(repo_root=tmp_path, workspace_id="alpha", backend_instance=object())
    root_id = alpha.block_map()[0]["block_id"]

    created = alpha.create_leaf_block(
        name="Research findings",
        synopsis="Evidence collected for the current research program.",
        parent_id=root_id,
        reason="independent research area",
        tags=["research", "evidence"],
        metadata={"domain": "retrieval"},
    )
    assert created["ok"] is True
    assert created["status"] == "active"
    assert any(
        block.block_id == created["block_id"]
        for block in load_memory_blocks(tmp_path, "alpha", strict=True)
    )
    assert all(
        block.block_id != created["block_id"]
        for block in load_memory_blocks(tmp_path, "beta", strict=True)
    )

    amended = alpha.amend_block(
        created["block_id"],
        expected_revision=created["revision"],
        name="Retrieval research",
        synopsis="Verified retrieval experiments and their conclusions.",
        tags=["research", "retrieval"],
        metadata_patch={"domain": "retrieval", "reviewed": True},
        reason="clarify the area",
    )
    assert amended["ok"] is True
    expanded = alpha.expand_block(created["block_id"])["root"]
    assert expanded["name"] == "Retrieval research"
    assert expanded["description"] == "Verified retrieval experiments and their conclusions."
    assert expanded["parent_id"] == root_id
    assert expanded["metadata"]["reviewed"] is True

    with pytest.raises(ValueError, match="protected"):
        alpha.amend_block(
            root_id,
            expected_revision=1,
            name="Replacement root",
            reason="must fail",
        )

    branch = alpha.create_index_block(
        name="Research branch",
        synopsis="Agent-managed branch.",
        parent_id=root_id,
        reason="create branch",
    )
    child = alpha.create_index_block(
        name="Research child",
        synopsis="Agent-managed child branch.",
        parent_id=branch["block_id"],
        reason="create child",
    )
    with pytest.raises(ValueError, match="cycle"):
        alpha.move_block(
            branch["block_id"],
            expected_revision=branch["revision"],
            parent_id=child["block_id"],
            reason="must fail",
        )

    archived = alpha.archive_block(
        created["block_id"],
        expected_revision=amended["revision"],
        reason="temporarily inactive",
    )
    assert archived["ok"] is True
    assert archived["archived"] is True
    with pytest.raises(ValueError, match="visible catalog block"):
        alpha.expand_block(created["block_id"])

    restored = alpha.revive_block(
        created["block_id"],
        expected_revision=archived["revision"],
        reason="resume research",
    )
    assert restored["ok"] is True
    assert restored["archived"] is False
    assert alpha.expand_block(created["block_id"])["root"]["name"] == "Retrieval research"
    assert child["revision"]


def test_host_can_repair_only_the_expected_protected_block_parent(tmp_path: Path) -> None:
    client = MemoryClient(repo_root=tmp_path, workspace_id="alpha", backend_instance=object())
    root_id = client.block_map()[0]["block_id"]
    misplaced_parent = client.create_index_block(
        name="Legacy container",
        synopsis="The old parent retained for a topology repair fixture.",
        parent_id=root_id,
        reason="create the old parent",
    )
    canonical_parent = client.create_index_block(
        name="Canonical container",
        synopsis="The expected parent for a protected host scaffold.",
        parent_id=root_id,
        reason="create the canonical parent",
        protected=True,
    )
    scaffold = client.create_index_block(
        name="Sessions",
        synopsis="A protected host-owned scaffold with a stale parent.",
        parent_id=misplaced_parent["block_id"],
        reason="create the misplaced protected scaffold",
        metadata={
            "umbrella_scaffold_key": "sessions-root",
            "umbrella_scaffold_scope": "workspace",
            "umbrella_scaffold_identity": "sessions:alpha",
        },
        protected=True,
    )
    child = client.create_index_block(
        name="Preserved child",
        synopsis="Existing memory below the scaffold must remain attached.",
        parent_id=scaffold["block_id"],
        reason="prove that repair preserves descendants",
    )

    with pytest.raises(ValueError, match="metadata"):
        client.repair_protected_block_parent(
            scaffold["block_id"],
            expected_revision=scaffold["revision"],
            expected_parent_id=misplaced_parent["block_id"],
            parent_id=canonical_parent["block_id"],
            expected_metadata={"umbrella_scaffold_identity": "another-project"},
            reason="a foreign host must not move this scaffold",
        )
    with pytest.raises(ValueError, match="metadata"):
        client.repair_protected_block_parent(
            scaffold["block_id"],
            expected_revision=scaffold["revision"],
            expected_parent_id=misplaced_parent["block_id"],
            parent_id=canonical_parent["block_id"],
            expected_metadata={"missing_owner_marker": None},
            reason="a missing ownership marker must not match a null guard",
        )

    repaired = client.repair_protected_block_parent(
        scaffold["block_id"],
        expected_revision=scaffold["revision"],
        expected_parent_id=misplaced_parent["block_id"],
        parent_id=canonical_parent["block_id"],
        expected_metadata={
            "umbrella_scaffold_key": "sessions-root",
            "umbrella_scaffold_scope": "workspace",
            "umbrella_scaffold_identity": "sessions:alpha",
        },
        reason="restore the host-owned canonical topology",
    )

    assert repaired["parent_id"] == canonical_parent["block_id"]
    assert repaired["protected"] is True
    assert repaired["revision"] == scaffold["revision"] + 1
    assert client.expand_block(child["block_id"])["root"]["parent_id"] == scaffold["block_id"]
    with pytest.raises(ValueError, match="expected_revision"):
        client.repair_protected_block_parent(
            scaffold["block_id"],
            expected_revision=scaffold["revision"],
            expected_parent_id=canonical_parent["block_id"],
            parent_id=root_id,
            expected_metadata={"umbrella_scaffold_key": "sessions-root"},
            reason="a stale repair must not overwrite a newer topology",
        )
    with pytest.raises(ValueError, match="cycle"):
        client.repair_protected_block_parent(
            scaffold["block_id"],
            expected_revision=repaired["revision"],
            expected_parent_id=canonical_parent["block_id"],
            parent_id=child["block_id"],
            expected_metadata={"umbrella_scaffold_key": "sessions-root"},
            reason="a repair must preserve tree integrity",
        )


def test_recall_expands_selected_index_regions_to_content_leaves(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        client = MemoryClient(
            repo_root=tmp_path,
            workspace_id="alpha",
            backend_instance=index,
        )
        root_id = client.block_map()[0]["block_id"]
        branch = client.create_index_block(
            name="Plans",
            synopsis="Plan memory region.",
            parent_id=root_id,
            reason="test region",
        )
        leaf = client.create_leaf_block(
            name="Accepted plans",
            synopsis="Accepted reusable plans.",
            parent_id=branch["block_id"],
            reason="test leaf",
        )
        written = client.remember(
            "Use the cancellation-safe numerical identity.",
            title="Stable numerical plan",
            requested_block_id=leaf["block_id"],
        )

        hits = client.recall(
            "numerical identity",
            logical_block_ids=[branch["block_id"]],
            include_candidates=True,
        )

        assert [hit.metadata["block_id"] for hit in hits] == [written.memory_id]
        empty = client.create_index_block(
            name="Empty region",
            synopsis="No content leaves yet.",
            parent_id=root_id,
            reason="test empty region",
        )
        assert client.recall("anything", logical_block_ids=[empty["block_id"]]) == []
    finally:
        index.close()


def test_explicit_block_ttl_archives_empty_session_scaffolds_but_not_guesses(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        client = MemoryClient(
            repo_root=tmp_path,
            workspace_id="alpha",
            profile_id="umbrella",
            session_id="um-session-chat-ttl",
            task_id="um-task-chat-ttl",
            backend_instance=index,
        )
        root_id = client.block_map()[0]["block_id"]
        expired = client.create_leaf_block(
            name="Expired chat scaffold",
            synopsis="An explicitly time-bounded empty session block.",
            parent_id=root_id,
            scope="session",
            protected=True,
            lifecycle_expires_at="2020-01-01T00:00:00Z",
            metadata={"umbrella_memory_session_id": "chat-ttl"},
            reason="TTL fixture",
        )
        legacy = client.create_leaf_block(
            name="Legacy chat scaffold",
            synopsis="A host-owned session scaffold without an expiry contract.",
            parent_id=root_id,
            scope="session",
            protected=True,
            metadata={
                "umbrella_memory_session_id": "chat-legacy",
                "umbrella_scaffold_key": "session-context",
            },
            reason="Legacy TTL fixture",
        )
        legacy_requirement = client.create_leaf_block(
            name="Legacy phase memory",
            synopsis="A host-owned phase block without an expiry contract.",
            parent_id=root_id,
            scope="session",
            protected=True,
            metadata={
                "umbrella_memory_session_id": "chat-legacy-requirement",
                "umbrella_requirement_id": "session-progress",
                "umbrella_requirement_scope": "session",
            },
            reason="Legacy requirement TTL fixture",
        )
        durable = client.create_leaf_block(
            name="Durable empty block",
            synopsis="Empty by design and without an expiry.",
            parent_id=root_id,
            reason="TTL fixture",
        )
        occupied = client.create_leaf_block(
            name="Expired block with a retained finding",
            synopsis="Expiry alone does not erase records.",
            parent_id=root_id,
            lifecycle_expires_at="2020-01-01T00:00:00Z",
            reason="TTL fixture",
        )
        record = client.remember(
            "A retained finding remains addressable after topology expiry.",
            requested_block_id=occupied["block_id"],
        )
        assert record.saved

        result = client.expire_due_blocks(
            as_of="2099-01-01T00:00:00Z",
            reason="expire test fixture",
        )

        assert result["archived_block_ids"] == [expired["block_id"]]
        for preserved in (legacy, legacy_requirement, durable):
            current = client.expand_block(preserved["block_id"])["root"]
            assert current["revision"] == preserved["revision"]
            assert current["lifecycle"].get("expires_at") is None
        assert result["blockers"][occupied["block_id"]] == ["block_has_current_records"]
        assert index.get_block(record.memory_id) is not None
        assert durable["block_id"] not in result["archived_block_ids"]
        assert client.catalog_snapshot()["blocks"]
        with pytest.raises(ValueError, match="visible catalog block"):
            client.expand_block(expired["block_id"])
    finally:
        index.close()
