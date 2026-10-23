import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import pytest

from um_mem import MemoryClient
from um_mem.memory.backends.mempalace import MemPalaceBackend
from um_mem.memory.backends.mnemosyne import MnemosyneBackend
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.catalog_store import (
    BlockEvolutionTarget,
    BlockMergeSource,
    WorkspaceCatalogStore,
)


def test_free_topology_uses_service_generated_stable_ids_and_durable_catalog(
    tmp_path: Path,
) -> None:
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        profile_id="researcher",
        backend_instance=object(),
    )
    root = client.block_map()[0]

    assert root["protected"] is True
    assert root["parent_id"] == ""
    assert root["child_ids"] == []
    assert "project" not in root["name"].lower()

    branch = client.create_index_block(
        name="Наблюдения о поведении",
        synopsis="Устойчивая область наблюдений, выбранная агентом.",
        parent_id=root["block_id"],
        reason="область будет развиваться независимо",
    )
    leaf = client.create_leaf_block(
        name="Неожиданные эффекты",
        synopsis="Проверяемые эффекты без предметной системной классификации.",
        parent_id=branch["block_id"],
        reason="для области нужны собственные записи",
    )

    UUID(branch["block_id"])
    UUID(leaf["block_id"])
    assert branch["block_id"] != branch["name"]
    assert leaf["backend_partition_id"]

    renamed = client.amend_block(
        leaf["block_id"],
        expected_revision=leaf["revision"],
        name="Побочные эффекты",
        reason="уточнён смысл области",
    )
    assert renamed["block_id"] == leaf["block_id"]

    reloaded = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        profile_id="researcher",
        backend_instance=object(),
    )
    branch_view = reloaded.expand_block(branch["block_id"])["root"]
    leaf_view = reloaded.expand_block(leaf["block_id"])["root"]
    assert branch_view["index_only"] is True
    assert branch_view["agent_writable"] is False
    assert leaf_view["name"] == "Побочные эффекты"
    assert leaf_view["index_only"] is False
    assert leaf_view["agent_writable"] is True


def test_dynamic_topology_preserves_typed_project_and_user_scopes(tmp_path: Path) -> None:
    client = MemoryClient(
        tmp_path,
        workspace_id="shared-user-store",
        profile_id="researcher",
        user_id="user-1",
        project_id="project-a",
        backend_instance=object(),
    )
    root_id = client.block_map()[0]["block_id"]

    project = client.create_index_block(
        name="Project research",
        synopsis="Knowledge isolated to one selected project.",
        parent_id=root_id,
        reason="bind model-authored research to project lineage",
        scope="project",
    )
    preference = client.create_leaf_block(
        name="User preferences",
        synopsis="Preferences shared for the authenticated user.",
        parent_id=root_id,
        reason="bind explicit preferences to user-global lineage",
        scope="user_global",
    )

    assert project["scope"] == "project"
    assert project["isolation_policy"] == {
        "dimension": "project",
        "cross_workspace": False,
    }
    assert preference["scope"] == "user_global"
    assert preference["isolation_policy"] == {
        "dimension": "user_global",
        "cross_workspace": True,
    }
    with pytest.raises(ValueError, match="scope is unsupported"):
        client.create_leaf_block(
            name="Invalid",
            synopsis="Unsupported scope must fail closed.",
            parent_id=root_id,
            reason="test validation",
            scope="global-ish",
        )


def test_typed_block_edges_fail_closed_across_profiles(tmp_path: Path) -> None:
    alpha = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        profile_id="one",
        backend_instance=object(),
    )
    beta = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        profile_id="two",
        backend_instance=object(),
    )
    alpha_root = alpha.block_map()[0]["block_id"]
    beta_root = beta.block_map()[0]["block_id"]
    left = alpha.create_leaf_block(
        name="Left",
        synopsis="Left side.",
        parent_id=alpha_root,
        reason="test",
    )
    right = alpha.create_leaf_block(
        name="Right",
        synopsis="Right side.",
        parent_id=alpha_root,
        reason="test",
    )

    edge = alpha.add_block_edge(
        left["block_id"],
        right["block_id"],
        edge_type="supports",
        reason="typed relation",
    )
    assert edge["edge_type"] == "supports"
    assert alpha.related_blocks(left["block_id"])[0]["block_id"] == right["block_id"]
    related_page = alpha.related_blocks_page(left["block_id"])
    assert related_page["related_blocks"][0]["block_id"] == right["block_id"]
    assert related_page["related_count"] == 1
    assert related_page["related_next_cursor"] == ""

    foreign = beta.create_leaf_block(
        name="Foreign",
        synopsis="Other profile.",
        parent_id=beta_root,
        reason="test",
    )
    with pytest.raises(ValueError, match="profile"):
        alpha.add_block_edge(
            left["block_id"],
            foreign["block_id"],
            edge_type="related_to",
            reason="must fail",
        )


def test_observation_is_advisory_and_never_mutates_topology(tmp_path: Path) -> None:
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=object())
    root = client.block_map()[0]
    leaf = client.create_leaf_block(
        name="Capacity sample",
        synopsis="Observation target.",
        parent_id=root["block_id"],
        token_budget=1,
        reason="test",
    )
    revision_before = client.catalog_revision()

    observation = client.observe_maintenance(
        leaf["block_id"],
        observation_type="capacity_pressure",
        values={"estimated_tokens": 2, "token_budget": 1},
        threshold_revision="thresholds:1",
        evidence_ids=("health:sample",),
    )

    assert observation["observation_type"] == "capacity_pressure"
    assert client.catalog_revision() == revision_before
    assert client.expand_block(leaf["block_id"])["root"]["status"] == "active"

    envelope = client.navigation_envelope(leaf["block_id"])
    assert envelope["tree_path"] == [root["block_id"], leaf["block_id"]]
    assert envelope["block"]["block_id"] == leaf["block_id"]
    assert envelope["maintenance_observations"] == [observation]
    assert envelope["proposal_options"] == ["propose_split", "propose_budget_change"]
    assert envelope["mutation_performed"] is False


def test_navigation_envelope_preserves_timestamp_freshness(tmp_path: Path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        root_id = client.block_map()[0]["block_id"]
        leaf = client.create_leaf_block(
            name="Fresh observations",
            synopsis="Observations with a visible latest-record timestamp.",
            parent_id=root_id,
            reason="exercise navigation health hints",
        )
        client.remember(
            "The attractor embedding improved the bounded comparison.",
            requested_block_id=leaf["block_id"],
        )

        envelope = client.navigation_envelope(leaf["block_id"])

        assert envelope["health_hints"]["freshness"]
        assert envelope["health_hints"]["freshness"] == envelope["block"]["freshness"]
    finally:
        backend.close()


def test_navigation_envelope_pages_dense_graph_edges_with_scope_and_revision(
    tmp_path: Path,
) -> None:
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=object())
    root_id = client.block_map()[0]["block_id"]
    source = client.create_index_block(
        name="Graph source", synopsis="Dense graph page source.", parent_id=root_id, reason="test"
    )
    targets = [
        client.create_leaf_block(
            name=f"Graph target {index}",
            synopsis="Related target.",
            parent_id=source["block_id"],
            reason="test",
        )
        for index in range(5)
    ]
    for target in targets:
        client.add_block_edge(
            source["block_id"], target["block_id"], edge_type="related_to", reason="test"
        )

    first = client.navigation_envelope(source["block_id"], limit=2)
    assert len(first["related_blocks"]) == 2
    assert first["related_count"] == 5
    assert first["related_next_cursor"]
    assert first["next_cursor"]
    with pytest.raises(ValueError, match="scope"):
        client.navigation_envelope(source["block_id"], limit=2, related_cursor=first["next_cursor"])
    second = client.navigation_envelope(
        source["block_id"], limit=2, related_cursor=first["related_next_cursor"]
    )
    third = client.navigation_envelope(
        source["block_id"], limit=2, related_cursor=second["related_next_cursor"]
    )
    ids = [row["block_id"] for page in (first, second, third) for row in page["related_blocks"]]
    assert len(ids) == 5
    assert set(ids) == {target["block_id"] for target in targets}
    assert third["related_next_cursor"] == ""

    added = client.create_leaf_block(
        name="Later graph target",
        synopsis="Invalidates graph page cursors.",
        parent_id=root_id,
        reason="test",
    )
    client.add_block_edge(
        source["block_id"], added["block_id"], edge_type="related_to", reason="test"
    )
    with pytest.raises(ValueError, match="revision changed"):
        client.related_blocks_page(source["block_id"], cursor=first["related_next_cursor"])

    other_profile = MemoryClient(
        tmp_path, workspace_id="alpha", profile_id="other", backend_instance=object()
    )
    with pytest.raises(ValueError, match="scope"):
        other_profile.related_blocks_page(source["block_id"], cursor=first["related_next_cursor"])


@pytest.mark.parametrize("backend_type", [MemPalaceBackend, MnemosyneBackend])
def test_split_and_merge_require_exact_record_coverage_and_preserve_provenance(
    tmp_path: Path,
    backend_type: type[MemPalaceBackend] | type[MnemosyneBackend],
) -> None:
    backend = backend_type(repo_root=tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        root_id = client.block_map()[0]["block_id"]
        source = client.create_leaf_block(
            name="Mixed observations",
            synopsis="Observations awaiting an explicit semantic split.",
            parent_id=root_id,
            reason="test",
        )
        first = client.remember(
            "Opaque fact A.",
            title="Fact A",
            kind="fact",
            requested_block_id=source["block_id"],
        )
        second = client.remember(
            "Opaque fact B.",
            title="Fact B",
            kind="fact",
            requested_block_id=source["block_id"],
        )
        assert client.projection_barrier(first.memory_id)["ok"]
        assert client.projection_barrier(second.memory_id)["ok"]
        source_payloads = {
            record_id: backend._index._conn.execute(
                "SELECT payload_json FROM blocks WHERE block_id=?", (record_id,)
            ).fetchone()[0]
            for record_id in (first.memory_id, second.memory_id)
        }
        revision_before = client.catalog_revision()

        with pytest.raises(ValueError, match="exactly cover"):
            client.split_block(
                source["block_id"],
                expected_revision=source["revision"],
                targets=(
                    BlockEvolutionTarget(
                        name="Only A",
                        synopsis="Incomplete mapping.",
                        record_ids=(first.memory_id,),
                    ),
                ),
                reason="must fail",
            )
        assert client.catalog_revision() == revision_before

        split = client.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=(
                BlockEvolutionTarget(
                    name="Area A",
                    synopsis="Agent-selected area A.",
                    record_ids=(first.memory_id,),
                ),
                BlockEvolutionTarget(
                    name="Area B",
                    synopsis="Agent-selected area B.",
                    record_ids=(second.memory_id,),
                ),
            ),
            reason="explicit full split",
        )
        target_a, target_b = split["targets"]
        assert split["source"]["status"] == "archived"
        assert source["block_id"] not in {
            identity[0]
            for identity in backend._index.native_partition_memberships(
                workspace_id="alpha",
                profile_id="default",
            )
        }
        assert (
            backend._index.logical_block_descriptor_snapshot(
                source["block_id"], workspace_id="alpha", profile_id="default"
            )["record_count"]
            == 0
        )
        assert client.expand_block(target_a["block_id"])["root"]["record_count"] == 1
        assert client.expand_block(target_a["block_id"])["root"]["metrics"]["record_count"] == 1
        assert client.expand_block(target_b["block_id"])["root"]["record_count"] == 1
        assert (
            backend.get_block(first.memory_id).block_membership.primary_block_id
            == target_a["block_id"]
        )
        assert (
            backend.get_block(second.memory_id).block_membership.primary_block_id
            == target_b["block_id"]
        )
        assert {edge["edge_type"] for edge in split["provenance_edges"]} == {"derived_from"}
        assert any(
            "Opaque fact A." in hit.text
            for hit in client.recall(
                "Opaque fact A",
                logical_block_ids=(target_a["block_id"],),
                include_candidates=True,
            )
        )
        assert {
            record_id: backend._index._conn.execute(
                "SELECT payload_json FROM blocks WHERE block_id=?", (record_id,)
            ).fetchone()[0]
            for record_id in source_payloads
        } == source_payloads

        merged = client.merge_blocks(
            (
                BlockMergeSource(target_a["block_id"], target_a["revision"]),
                BlockMergeSource(target_b["block_id"], target_b["revision"]),
            ),
            target=BlockEvolutionTarget(
                name="Unified area",
                synopsis="Explicitly merged observations.",
                record_ids=(first.memory_id, second.memory_id),
            ),
            parent_id=root_id,
            reason="explicit full merge",
        )
        merged_id = merged["target"]["block_id"]
        assert client.expand_block(merged_id)["root"]["record_count"] == 2
        assert (
            backend._index.logical_block_descriptor_snapshot(
                target_a["block_id"], workspace_id="alpha", profile_id="default"
            )["record_count"]
            == 0
        )
        assert {
            record_id: backend._index._conn.execute(
                "SELECT payload_json FROM blocks WHERE block_id=?", (record_id,)
            ).fetchone()[0]
            for record_id in source_payloads
        } == source_payloads
        assert all(source_state["status"] == "archived" for source_state in merged["sources"])
        assert backend.get_block(first.memory_id).block_membership.primary_block_id == merged_id
        assert backend.get_block(second.memory_id).block_membership.primary_block_id == merged_id
        assert {edge["dst_id"] for edge in merged["provenance_edges"]} == {
            target_a["block_id"],
            target_b["block_id"],
        }
        assert any(
            "Opaque fact B." in hit.text
            for hit in client.recall(
                "Opaque fact B", logical_block_ids=(merged_id,), include_candidates=True
            )
        )
        backend.close()
        backend = backend_type(repo_root=tmp_path, workspace_id="alpha")
        reopened = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        assert reopened.read_exact_record(record_id=first.memory_id).content == "Opaque fact A."
        assert reopened.read_exact_record(record_id=second.memory_id).content == "Opaque fact B."
        assert backend.get_block(first.memory_id).block_membership.primary_block_id == merged_id
        assert any(
            "Opaque fact A." in hit.text
            for hit in reopened.recall(
                "Opaque fact A", logical_block_ids=(merged_id,), include_candidates=True
            )
        )
        second_split = reopened.split_block(
            merged_id,
            expected_revision=merged["target"]["revision"],
            targets=(
                BlockEvolutionTarget(
                    name="Refined A",
                    synopsis="Agent refined this branch again.",
                    record_ids=(first.memory_id,),
                ),
                BlockEvolutionTarget(
                    name="Refined B",
                    synopsis="Agent refined the other branch.",
                    record_ids=(second.memory_id,),
                ),
            ),
            reason="second explicit generation",
        )
        assert second_split["generation_status"] == "published"
        assert reopened.read_exact_record(record_id=first.memory_id).content == "Opaque fact A."
        assert any(
            "Opaque fact B." in hit.text
            for hit in reopened.recall(
                "Opaque fact B",
                logical_block_ids=(second_split["targets"][1]["block_id"],),
                include_candidates=True,
            )
        )
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", [MemPalaceBackend, MnemosyneBackend])
def test_topology_publication_failure_keeps_old_snapshot_and_retry_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_type: type[MemPalaceBackend] | type[MnemosyneBackend],
) -> None:
    backend = backend_type(repo_root=tmp_path, workspace_id="alpha")
    operation_id = "agent-split-after-projection"
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        source = client.create_leaf_block(
            name="Source",
            synopsis="Agent-authored source.",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        written = client.remember("Durable source bytes.", requested_block_id=source["block_id"])
        assert client.projection_barrier(written.memory_id)["ok"]
        target = BlockEvolutionTarget(
            name="Target",
            synopsis="Agent-authored destination.",
            record_ids=(written.memory_id,),
        )
        original_publish = WorkspaceCatalogStore.publish_topology_evolution

        def interrupted_publish(store: WorkspaceCatalogStore, selected_operation: str) -> dict:
            monkeypatch.setattr(
                WorkspaceCatalogStore, "publish_topology_evolution", original_publish
            )
            raise OSError("simulated crash before publication")

        monkeypatch.setattr(
            WorkspaceCatalogStore, "publish_topology_evolution", interrupted_publish
        )
        with pytest.raises(OSError, match="simulated crash"):
            client.split_block(
                source["block_id"],
                expected_revision=source["revision"],
                targets=(target,),
                reason="agent split",
                operation_id=operation_id,
            )
        store = WorkspaceCatalogStore(tmp_path, "alpha", "default")
        try:
            assert store.topology_evolution(operation_id)["state"] == "projections_ready"
            assert store.get_block(source["block_id"]).status == "active"
            assert len(store.list_blocks()) == 2
        finally:
            store.close()
        assert (
            backend.get_block(written.memory_id).block_membership.primary_block_id
            == source["block_id"]
        )
        backend.close()
        backend = backend_type(repo_root=tmp_path, workspace_id="alpha")
        reopened = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        result = reopened.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=(target,),
            reason="agent split",
            operation_id=operation_id,
        )
        target_id = result["targets"][0]["block_id"]
        assert result["generation_status"] == "published"
        assert backend.get_block(written.memory_id).block_membership.primary_block_id == target_id
        assert any(
            "Durable source bytes." in hit.text
            for hit in reopened.recall(
                "Durable source bytes", logical_block_ids=(target_id,), include_candidates=True
            )
        )
        assert (
            reopened.split_block(
                source["block_id"],
                expected_revision=source["revision"],
                targets=(target,),
                reason="agent split",
                operation_id=operation_id,
            )["targets"][0]["block_id"]
            == target_id
        )
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", [MemPalaceBackend, MnemosyneBackend])
def test_topology_projection_failure_can_resume_exact_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_type: type[MemPalaceBackend] | type[MnemosyneBackend],
) -> None:
    backend = backend_type(repo_root=tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        source = client.create_leaf_block(
            name="Source",
            synopsis="Two independently preserved facts.",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        records = tuple(
            client.remember(f"Exact source {index}.", requested_block_id=source["block_id"])
            for index in range(2)
        )
        for record in records:
            assert client.projection_barrier(record.memory_id)["ok"]
        targets = tuple(
            BlockEvolutionTarget(
                name=f"Target {index}",
                synopsis=f"Agent target {index}.",
                record_ids=(record.memory_id,),
            )
            for index, record in enumerate(records)
        )
        original_stage = backend.stage_topology_projection
        calls = 0

        def fail_second(record: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated native outage")
            original_stage(record)

        monkeypatch.setattr(backend, "stage_topology_projection", fail_second)
        with pytest.raises(OSError, match="native outage"):
            client.split_block(
                source["block_id"],
                expected_revision=source["revision"],
                targets=targets,
                reason="agent split",
                operation_id="split-retry",
            )
        assert client.pending_topology_evolutions() == (
            {"operation_id": "split-retry", "state": "prepared", "operation": "split"},
        )
        assert client.expand_block(source["block_id"])["root"]["status"] == "active"
        backend.close()
        backend = backend_type(repo_root=tmp_path, workspace_id="alpha")
        reopened = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        split = reopened.resume_topology_evolution("split-retry")
        assert split["generation_status"] == "published"
        assert reopened.pending_topology_evolutions() == ()
        assert len(split["targets"]) == 2
        for record, target in zip(records, split["targets"], strict=True):
            assert (
                backend.get_block(record.memory_id).block_membership.primary_block_id
                == target["block_id"]
            )
    finally:
        backend.close()


def test_split_replay_rejects_changed_request_and_conflict_releases_source(tmp_path: Path) -> None:
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        source = client.create_leaf_block(
            name="Source",
            synopsis="Facts awaiting agent placement.",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        first = client.remember("First exact fact.", requested_block_id=source["block_id"])
        assert client.projection_barrier(first.memory_id)["ok"]
        target = BlockEvolutionTarget(
            name="First target",
            synopsis="Selected by the agent.",
            record_ids=(first.memory_id,),
        )
        original_stage = backend.stage_topology_projection

        def outage(record: object) -> None:
            raise OSError("native stage unavailable")

        backend.stage_topology_projection = outage
        with pytest.raises(OSError, match="native stage unavailable"):
            client.split_block(
                source["block_id"],
                expected_revision=source["revision"],
                targets=(target,),
                reason="first decision",
                operation_id="reused-id",
            )
        backend.stage_topology_projection = original_stage
        with pytest.raises(ValueError, match="different topology request"):
            client.split_block(
                source["block_id"],
                expected_revision=source["revision"],
                targets=(target,),
                reason="changed decision",
                operation_id="reused-id",
            )
        second = client.remember("Second exact fact.", requested_block_id=source["block_id"])
        assert client.projection_barrier(second.memory_id)["ok"]
        with pytest.raises(ValueError, match="record set changed"):
            client.resume_topology_evolution("reused-id")
        assert client.pending_topology_evolutions() == ()
        assert client.expand_block(source["block_id"])["root"]["status"] == "active"
        fresh = client.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=(
                BlockEvolutionTarget(
                    name="Both facts",
                    synopsis="New agent decision.",
                    record_ids=(first.memory_id, second.memory_id),
                ),
            ),
            reason="new decision",
            operation_id="fresh-id",
        )
        assert fresh["generation_status"] == "published"
        with pytest.raises(ValueError, match="different topology request"):
            client.split_block(
                source["block_id"],
                expected_revision=source["revision"],
                targets=(target,),
                reason="changed decision",
                operation_id="fresh-id",
            )
    finally:
        backend.close()


def test_agent_can_abort_unrecoverable_native_stage_without_losing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        source = client.create_leaf_block(
            name="Source",
            synopsis="Durable source before a failed stage.",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        saved = client.remember("Exact source fact.", requested_block_id=source["block_id"])
        assert client.projection_barrier(saved.memory_id)["ok"]

        def fail_stage(_record: object) -> None:
            raise OSError("native unavailable")

        monkeypatch.setattr(backend, "stage_topology_projection", fail_stage)
        with pytest.raises(OSError, match="native unavailable"):
            client.split_block(
                source["block_id"],
                expected_revision=source["revision"],
                targets=(
                    BlockEvolutionTarget(
                        name="Target",
                        synopsis="Explicit destination.",
                        record_ids=(saved.memory_id,),
                    ),
                ),
                reason="agent decision",
                operation_id="abandon-me",
            )
        aborted = client.abort_topology_evolution(
            "abandon-me",
            reason="native service permanently unavailable",
        )
        assert aborted == {
            "ok": True,
            "operation_id": "abandon-me",
            "state": "aborted",
            "reason": "native service permanently unavailable",
        }
        assert client.pending_topology_evolutions() == ()
        assert client.expand_block(source["block_id"])["root"]["status"] == "active"
        assert client.read_exact_record(record_id=saved.memory_id).content == "Exact source fact."
        with pytest.raises(ValueError, match="aborted"):
            client.resume_topology_evolution("abandon-me")
    finally:
        backend.close()


def test_same_topology_operation_id_is_scoped_to_profile(tmp_path: Path) -> None:
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
    try:
        for profile in ("agent-one", "agent-two"):
            client = MemoryClient(
                tmp_path,
                workspace_id="alpha",
                profile_id=profile,
                backend_instance=backend,
            )
            source = client.create_leaf_block(
                name="Source",
                synopsis="Profile-owned facts.",
                parent_id=client.block_map()[0]["block_id"],
                reason="test",
            )
            record = client.remember(
                f"Fact for {profile}.",
                requested_block_id=source["block_id"],
            )
            assert client.projection_barrier(record.memory_id)["ok"]
            result = client.split_block(
                source["block_id"],
                expected_revision=source["revision"],
                targets=(
                    BlockEvolutionTarget(
                        name="Target",
                        synopsis="Profile-owned target.",
                        record_ids=(record.memory_id,),
                    ),
                ),
                reason="agent decision",
                operation_id="same-scoped-id",
            )
            assert result["generation_status"] == "published"
    finally:
        backend.close()


def test_split_projects_distinct_target_leaves_concurrently(tmp_path: Path) -> None:
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        source = client.create_leaf_block(
            name="Source",
            synopsis="Two facts for independent leaves.",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        records = tuple(
            client.remember(f"Fact {index}.", requested_block_id=source["block_id"])
            for index in range(2)
        )
        for record in records:
            assert client.projection_barrier(record.memory_id)["ok"]
        barrier = threading.Barrier(2, timeout=10)
        staged_threads: set[str] = set()
        original_stage = backend.stage_topology_projection

        def stage_concurrently(record: object) -> None:
            staged_threads.add(threading.current_thread().name)
            barrier.wait()
            original_stage(record)

        backend.stage_topology_projection = stage_concurrently
        result = client.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=tuple(
                BlockEvolutionTarget(
                    name=f"Target {index}",
                    synopsis="Agent-selected target.",
                    record_ids=(record.memory_id,),
                )
                for index, record in enumerate(records)
            ),
            reason="parallel target projection",
        )
        assert len(staged_threads) == 2
        assert result["generation_status"] == "published"
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", [MemPalaceBackend, MnemosyneBackend])
def test_amend_after_split_inherits_published_leaf_membership(
    tmp_path: Path,
    backend_type: type[MemPalaceBackend] | type[MnemosyneBackend],
) -> None:
    backend = backend_type(repo_root=tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        source = client.create_leaf_block(
            name="Source",
            synopsis="Initial agent placement.",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        created = client.remember("Original detail.", requested_block_id=source["block_id"])
        assert client.projection_barrier(created.memory_id)["ok"]
        split = client.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=(
                BlockEvolutionTarget(
                    name="New area",
                    synopsis="Explicit refined meaning.",
                    record_ids=(created.memory_id,),
                ),
            ),
            reason="agent refined placement",
        )
        target_id = split["targets"][0]["block_id"]
        original = backend.get_block(created.memory_id)
        assert original is not None
        amended = client.amend(
            created.memory_id,
            expected_revision=original.revision_id,
            content="Corrected exact detail.",
            reason="source correction",
        )
        assert amended["ok"] is True
        successor = backend.get_block(amended["new_id"])
        assert successor is not None
        assert successor.block_membership.primary_block_id == target_id
        assert client.projection_barrier(successor.block_id)["ok"]
        assert any(
            "Corrected exact detail." in hit.text
            for hit in client.recall(
                "Corrected exact detail",
                logical_block_ids=(target_id,),
                include_candidates=True,
            )
        )
    finally:
        backend.close()


def test_split_preserves_lossless_source_bytes_after_reopen(tmp_path: Path) -> None:
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
    source_text = "A🧠B\nExact quoted detail."
    try:
        client = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            session_id="session-a",
            backend_instance=backend,
        )
        leaf = client.create_leaf_block(
            name="Source",
            synopsis="An exact source record.",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        source = client.register_source(
            source_id="turn-1",
            text=source_text,
            producer_id="trusted-host",
            speaker="Ada",
        )
        written = client.remember(
            source_text,
            requested_block_id=leaf["block_id"],
            write_mode="lossless_atomic",
            metadata={
                "source_segments": [
                    {
                        "source_id": "turn-1",
                        "source_sha256": source["source_sha256"],
                        "byte_start": 0,
                        "byte_end": len(source_text.encode("utf-8")),
                        "text": source_text,
                    }
                ]
            },
        )
        assert written.saved and written.raw["exact_source_coverage"] == 1
        assert client.projection_barrier(written.memory_id)["ok"]
        result = client.split_block(
            leaf["block_id"],
            expected_revision=leaf["revision"],
            targets=(
                BlockEvolutionTarget(
                    name="Exact source area",
                    synopsis="Agent chose the destination.",
                    record_ids=(written.memory_id,),
                ),
            ),
            reason="agent placement",
        )
        target_id = result["targets"][0]["block_id"]
        backend.close()
        backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
        reopened = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            session_id="session-b",
            backend_instance=backend,
        )
        exact = reopened.read_exact_record(record_id=written.memory_id)
        assert exact.content.encode("utf-8") == source_text.encode("utf-8")
        assert exact.metadata["source_segments"][0]["text"] == source_text
        assert exact.metadata["summary_count"] == 0
        assert exact.block_membership.primary_block_id == target_id
    finally:
        backend.close()


def test_independent_clients_publish_disjoint_leaves_and_conflict_on_same_leaf(
    tmp_path: Path,
) -> None:
    first_backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
    second_backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
    try:
        first_client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=first_backend)
        second_client = MemoryClient(
            tmp_path, workspace_id="alpha", backend_instance=second_backend
        )
        root_id = first_client.block_map()[0]["block_id"]
        leaves = tuple(
            first_client.create_leaf_block(
                name=f"Source {index}",
                synopsis="Separate agent-owned leaf.",
                parent_id=root_id,
                reason="test",
            )
            for index in range(2)
        )
        records = tuple(
            first_client.remember(
                f"Fact {index}.",
                requested_block_id=leaf["block_id"],
            )
            for index, leaf in enumerate(leaves)
        )
        for record in records:
            assert first_client.projection_barrier(record.memory_id)["ok"]

        def split_one(index: int) -> dict:
            client = first_client if index == 0 else second_client
            leaf, record = leaves[index], records[index]
            return client.split_block(
                leaf["block_id"],
                expected_revision=leaf["revision"],
                targets=(
                    BlockEvolutionTarget(
                        name=f"Target {index}",
                        synopsis="Agent-selected independent target.",
                        record_ids=(record.memory_id,),
                    ),
                ),
                reason=f"independent decision {index}",
                operation_id=f"independent-{index}",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(split_one, range(2)))
        assert all(result["generation_status"] == "published" for result in results)
        for record, result in zip(records, results, strict=True):
            assert (
                first_backend.get_block(record.memory_id).block_membership.primary_block_id
                == (result["targets"][0]["block_id"])
            )

        current = results[0]["targets"][0]

        def compete(index: int) -> object:
            client = first_client if index == 0 else second_client
            try:
                return client.split_block(
                    current["block_id"],
                    expected_revision=current["revision"],
                    targets=(
                        BlockEvolutionTarget(
                            name=f"Competing {index}",
                            synopsis="Explicit competing placement.",
                            record_ids=(records[0].memory_id,),
                        ),
                    ),
                    reason="competing agent decision",
                    operation_id=f"competing-{index}",
                )
            except ValueError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as executor:
            competing = tuple(executor.map(compete, range(2)))
        assert sum(isinstance(result, dict) for result in competing) == 1
        assert sum(isinstance(result, ValueError) for result in competing) == 1
        assert first_client.pending_topology_evolutions() == ()
    finally:
        first_backend.close()
        second_backend.close()


def test_evolved_leaf_descriptor_uses_index_without_decoding_all_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        source = client.create_leaf_block(
            name="Source",
            synopsis="Indexed descriptor source.",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        records = tuple(
            client.remember(f"Fact {index}.", requested_block_id=source["block_id"])
            for index in range(3)
        )
        for record in records:
            assert client.projection_barrier(record.memory_id)["ok"]
        result = client.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=(
                BlockEvolutionTarget(
                    name="Target",
                    synopsis="Agent-selected target.",
                    record_ids=tuple(record.memory_id for record in records),
                ),
            ),
            reason="indexed descriptor",
        )

        def no_record_decode(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("descriptor must not decode canonical records")

        monkeypatch.setattr(backend._index, "get_block", no_record_decode)
        descriptor = backend._index.logical_block_descriptor_snapshot(
            result["targets"][0]["block_id"],
            workspace_id="alpha",
            profile_id="default",
        )
        assert descriptor["record_count"] == 3
        assert len(descriptor["related_block_ids"]) == 3
    finally:
        backend.close()


def test_split_requires_agent_to_move_live_mask_then_new_leaf_can_be_activated(
    tmp_path: Path,
) -> None:
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            session_id="session-a",
            backend_instance=backend,
        )
        source = client.create_leaf_block(
            name="Source",
            synopsis="An active source leaf.",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        saved = client.remember("Active fact.", requested_block_id=source["block_id"])
        assert client.projection_barrier(saved.memory_id)["ok"]
        mask = client.set_active_mask(
            selected_block_ids=(source["block_id"],),
            token_budget=200,
            activation_reason="agent selected source",
        )
        target = BlockEvolutionTarget(
            name="Target",
            synopsis="Agent-selected next leaf.",
            record_ids=(saved.memory_id,),
        )
        with pytest.raises(ValueError, match="active mask"):
            client.split_block(
                source["block_id"],
                expected_revision=source["revision"],
                targets=(target,),
                reason="requires agent mask review",
            )
        client.deactivate_active_mask(
            mask.mask_id,
            expected_revision=mask.revision,
            reason="agent will select evolved target",
        )
        split = client.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=(target,),
            reason="agent reviewed mask",
        )
        new_mask = client.set_active_mask(
            selected_block_ids=(split["targets"][0]["block_id"],),
            token_budget=200,
            activation_reason="agent selected evolved target",
        )
        assert new_mask.selected_block_ids == (split["targets"][0]["block_id"],)
    finally:
        backend.close()


def test_split_rejects_writer_that_validated_an_archived_source(tmp_path: Path) -> None:
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        source = client.create_leaf_block(
            name="Source",
            synopsis="Records before the agent's split.",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        first = client.remember("Initial fact.", requested_block_id=source["block_id"])
        assert client.projection_barrier(first.memory_id)["ok"]
        waiting = threading.Event()
        proceed = threading.Event()
        original_gate = backend.atomic_canonical_write_gate

        @contextmanager
        def delayed_gate():
            if threading.current_thread().name == "late-memory-writer":
                waiting.set()
                if not proceed.wait(timeout=10):
                    raise RuntimeError("late writer was not released")
            with original_gate():
                yield

        backend.atomic_canonical_write_gate = delayed_gate
        result: list[object] = []

        def write_late() -> None:
            result.append(client.remember("Late fact.", requested_block_id=source["block_id"]))

        writer = threading.Thread(target=write_late, name="late-memory-writer")
        writer.start()
        assert waiting.wait(timeout=10)
        split = client.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=(
                BlockEvolutionTarget(
                    name="Target",
                    synopsis="Only the committed first fact.",
                    record_ids=(first.memory_id,),
                ),
            ),
            reason="agent split before late durable commit",
        )
        proceed.set()
        writer.join(timeout=10)
        assert not writer.is_alive()
        assert len(result) == 1 and result[0].saved is False
        assert (
            split["targets"][0]["block_id"]
            == backend.get_block(first.memory_id).block_membership.primary_block_id
        )
        assert backend._index._conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 1
    finally:
        backend.close()


def test_hard_delete_rejects_archived_children_and_outgoing_edges(tmp_path: Path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        root_id = client.block_map()[0]["block_id"]
        parent = client.create_index_block(
            name="Parent",
            synopsis="Parent retained while any child references it.",
            parent_id=root_id,
            reason="test",
        )
        child = client.create_leaf_block(
            name="Child",
            synopsis="Archived child still retains its parent identity.",
            parent_id=parent["block_id"],
            reason="test",
        )
        client.archive_block(
            child["block_id"],
            expected_revision=child["revision"],
            reason="test",
        )
        archived_parent = client.archive_block(
            parent["block_id"],
            expected_revision=parent["revision"],
            reason="test",
        )

        with pytest.raises(ValueError, match="no children"):
            client.delete_block(
                parent["block_id"],
                expected_revision=archived_parent["revision"],
                reason="must preserve child reference",
            )

        source = client.create_leaf_block(
            name="Edge source",
            synopsis="Outgoing graph edge owner.",
            parent_id=root_id,
            reason="test",
        )
        destination = client.create_leaf_block(
            name="Edge destination",
            synopsis="Outgoing graph edge destination.",
            parent_id=root_id,
            reason="test",
        )
        edge = client.add_block_edge(
            source["block_id"],
            destination["block_id"],
            edge_type="related_to",
            reason="test",
        )
        archived_source = client.archive_block(
            source["block_id"],
            expected_revision=source["revision"],
            reason="test",
        )

        with pytest.raises(ValueError, match="no graph edges"):
            client.delete_block(
                source["block_id"],
                expected_revision=archived_source["revision"],
                reason="must preserve graph integrity",
            )
        store = WorkspaceCatalogStore(tmp_path, "alpha", "default")
        try:
            assert any(
                item.edge_id == edge["edge_id"] for item in store.edges_for(destination["block_id"])
            )
        finally:
            store.close()
    finally:
        backend.close()
