from dataclasses import replace
from pathlib import Path

import pytest

from lethyx import MemoryClient
from lethyx.contracts import MemoryBlockRecord, RecordBlockMembership
from lethyx.memory.backends import contracts as backend_contracts
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.backends.mempalace import _mempalace_partition_identity
from lethyx.memory.backends.hindsight import HindsightBackend
from lethyx.memory.backends.mnemosyne import MnemosyneBackend
from lethyx.memory.blocks.catalog import compile_semantic_catalog
from lethyx.memory.blocks.registry import load_memory_blocks
from lethyx.memory.kernel.models import block_record_to_event
from lethyx.memory.kernel.writer import _write_memory_event
from lethyx.memory.mutation import plan_memory_mutation
from lethyx.memory.selection import MemorySelectionPolicy, SelectionContext


def _catalog_revision(repo_root: Path, workspace_id: str) -> str:
    return compile_semantic_catalog(
        load_memory_blocks(repo_root, workspace_id=workspace_id, strict=True)
    ).revision_id


def test_record_block_membership_binds_record_identity_and_round_trips() -> None:
    membership = RecordBlockMembership(
        primary_block_id="stable_project_facts",
        secondary_block_ids=("artifact_claims",),
        catalog_revision_id="catalog:test",
    )
    record = MemoryBlockRecord(
        block_id="record-1",
        revision_id="revision-1",
        workspace_id="alpha",
        valid_from="2026-07-15T09:00:00Z",
        content="Typed membership survives exact serialization.",
        block_membership=membership,
    )

    bound = record.block_membership
    assert bound is not None
    assert bound.record_id == record.block_id
    assert bound.record_revision_id == record.revision_id
    assert bound.workspace_id == record.workspace_id
    assert bound.membership_revision_id.startswith("membership_")
    serialized = backend_contracts.block_to_json(record)
    assert backend_contracts.block_from_json(serialized) == record

    missing = dict(serialized)
    missing["block_membership"] = dict(serialized["block_membership"])
    missing["block_membership"].pop("catalog_revision_id")
    with pytest.raises(ValueError, match="block_membership fields must be exact"):
        backend_contracts.block_from_json(missing)

    unknown = dict(serialized)
    unknown["block_membership"] = {
        **serialized["block_membership"],
        "logical_block_id": "free-form-injection",
    }
    with pytest.raises(ValueError, match="block_membership fields must be exact"):
        backend_contracts.block_from_json(unknown)


def test_record_block_membership_rejects_an_inverted_validity_window() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        RecordBlockMembership(
            primary_block_id="stable_project_facts",
            catalog_revision_id="catalog:test",
            valid_from="2026-07-15T10:00:00Z",
            valid_to="2026-07-15T09:00:00Z",
        )


def test_static_leaf_partition_is_scoped_to_profile() -> None:
    membership = RecordBlockMembership(
        primary_block_id="stable_project_facts", catalog_revision_id="catalog:test"
    )
    default = MemoryBlockRecord(
        block_id="default-record",
        content="default",
        workspace_id="alpha",
        profile_id="default",
        block_membership=membership,
    )
    alternate = MemoryBlockRecord(
        block_id="alternate-record",
        content="alternate",
        workspace_id="alpha",
        profile_id="alternate",
        block_membership=membership,
    )
    assert default.block_membership is not None
    assert alternate.block_membership is not None
    assert (
        default.block_membership.backend_partition_id
        != alternate.block_membership.backend_partition_id
    )


def test_routed_writer_persists_primary_membership_without_free_metadata(tmp_path: Path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        saved = client.remember(
            "The parser repair requires a verified regression test.",
            title="Parser regression lesson",
            kind="lesson",
            requested_block_id="artifact_claims",
            evidence_refs=[
                {
                    "ref_type": "test_run",
                    "ref_id": "test-parser-regression",
                    "produced_by": "verifier",
                    "producer_id": "verifier:parser-regression",
                }
            ],
        )
        record = backend.get_block(saved.memory_id, workspace_id="alpha")

        assert saved.saved is True
        assert record is not None
        assert record.block_membership is not None
        assert record.block_membership.primary_block_id == "artifact_claims"
        assert record.block_membership.catalog_revision_id == _catalog_revision(tmp_path, "alpha")
        assert record.block_membership.valid_from == record.recorded_at
        assert backend._conn.execute(
            "SELECT backend_partition_id FROM record_block_memberships WHERE record_id=? AND role='primary'",
            (record.block_id,),
        ).fetchone() == (record.block_membership.backend_partition_id,)
        assert "logical_block_id" not in record.metadata
        assert backend.record_ids_for_logical_blocks(
            ("artifact_claims",), workspace_id="alpha"
        ) == (record.block_id,)

        [hit] = client.recall(
            "parser regression",
            logical_block_ids=("artifact_claims",),
            include_candidates=True,
        )
        assert hit.metadata["block_membership"]["primary_block_id"] == "artifact_claims"
    finally:
        backend.close()


def test_leaf_partition_identity_survives_unrelated_catalog_revision(tmp_path: Path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        root = client.block_map()[0]["block_id"]
        leaf = client.create_leaf_block(
            name="Stable leaf",
            synopsis="Partition identity fixture",
            parent_id=root,
            reason="test stable partition identity",
        )
        first = client.remember("first", requested_block_id=leaf["block_id"])
        client.create_leaf_block(
            name="Unrelated leaf",
            synopsis="Changes catalog revision",
            parent_id=root,
            reason="test unrelated catalog change",
        )
        second = client.remember("second", requested_block_id=leaf["block_id"])
        client.amend_block(
            leaf["block_id"],
            expected_revision=leaf["revision"],
            name="Renamed stable leaf",
            reason="test rename preserves partition",
        )
        third = client.remember("third", requested_block_id=leaf["block_id"])
        first_record = backend.get_block(first.memory_id)
        second_record = backend.get_block(second.memory_id)
        third_record = backend.get_block(third.memory_id)
        assert first_record is not None and second_record is not None and third_record is not None
        assert first_record.block_membership is not None
        assert second_record.block_membership is not None
        assert third_record.block_membership is not None
        assert (
            first_record.block_membership.catalog_revision_id
            != second_record.block_membership.catalog_revision_id
        )
        assert (
            first_record.block_membership.backend_partition_id
            == second_record.block_membership.backend_partition_id
        )
        assert first_record.block_membership.backend_partition_id == leaf["backend_partition_id"]
        assert third_record.block_membership.backend_partition_id == leaf["backend_partition_id"]
    finally:
        backend.close()


def test_all_native_partition_ids_ignore_unrelated_catalog_revision() -> None:
    membership = RecordBlockMembership(
        primary_block_id="leaf-1",
        catalog_revision_id="catalog:a",
        backend_partition_id="leaf-partition-1",
    )
    for identity in (
        _mempalace_partition_identity,
        HindsightBackend.native_partition_identity,
        MnemosyneBackend.native_partition_identity,
    ):
        first = identity(
            workspace_id="alpha",
            primary_block_id=membership.primary_block_id,
            catalog_revision_id="catalog:a",
            backend_partition_id=membership.backend_partition_id,
        )
        second = identity(
            workspace_id="alpha",
            primary_block_id=membership.primary_block_id,
            catalog_revision_id="catalog:b",
            backend_partition_id=membership.backend_partition_id,
        )
        assert first.partition_id == second.partition_id
        assert first == second


def test_writer_rebases_membership_after_unrelated_parallel_catalog_change(
    tmp_path: Path,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        original_revision = _catalog_revision(tmp_path, "alpha")
        planned = plan_memory_mutation(
            MemoryBlockRecord(
                block_id="",
                title="Parallel-safe membership",
                content="An unrelated topology update must not reject this write.",
                kind="project_fact",
                workspace_id="alpha",
                block_membership=RecordBlockMembership(
                    primary_block_id="stable_project_facts",
                    catalog_revision_id=original_revision,
                ),
            ),
            existing=None,
        )
        [root] = [item for item in client.block_map() if not item["parent_id"]]
        client.create_leaf_block(
            name="Concurrent unrelated notes",
            synopsis="A separate phase-created branch.",
            parent_id=root["block_id"],
            reason="simulate a parallel phase topology mutation",
        )
        current_revision = _catalog_revision(tmp_path, "alpha")
        assert current_revision != original_revision

        result = _write_memory_event(
            tmp_path,
            block_record_to_event(planned.record),
            workspace_id="alpha",
            backend_instance=backend,
            mutation=planned,
        )

        assert result.saved is True
        stored = backend.get_block(result.canonical_id, workspace_id="alpha")
        assert stored is not None and stored.block_membership is not None
        assert stored.block_membership.catalog_revision_id == current_revision
        assert stored.block_membership.primary_block_id == "stable_project_facts"
    finally:
        backend.close()


def test_writer_rebases_a_planned_promotion_after_parallel_topology_change(
    tmp_path: Path,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        candidate = client.remember(
            "The promoted result remains valid across unrelated topology edits.",
            title="Parallel promotion",
            kind="project_fact",
            requested_block_id="stable_project_facts",
        )
        source = backend.get_block(candidate.memory_id, workspace_id="alpha")
        assert source is not None and source.block_membership is not None
        planned = plan_memory_mutation(
            source,
            existing=source,
            action="promote",
        )
        planned_revision = planned.record.block_membership.catalog_revision_id
        [root] = [item for item in client.block_map() if not item["parent_id"]]
        client.create_leaf_block(
            name="Concurrent promotion notes",
            synopsis="A separate phase-created leaf.",
            parent_id=root["block_id"],
            reason="simulate topology mutation before promotion commit",
        )
        current_revision = _catalog_revision(tmp_path, "alpha")
        assert current_revision != planned_revision

        result = _write_memory_event(
            tmp_path,
            block_record_to_event(planned.record),
            workspace_id="alpha",
            backend_instance=backend,
            mutation=planned,
        )

        assert result.saved is True
        promoted = backend.get_block(result.canonical_id, workspace_id="alpha")
        assert promoted is not None and promoted.block_membership is not None
        assert promoted.status == source.status
        assert promoted.revision_of == source.block_id
        assert promoted.block_membership.catalog_revision_id == current_revision
    finally:
        backend.close()


def test_caller_cannot_inject_free_logical_block_metadata(tmp_path: Path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        saved = client.remember(
            "Caller metadata must not select a canonical block.",
            requested_block_id="project_ideas",
            metadata={"logical_block_id": "workspace_core_directives"},
        )

        assert saved.saved is False
        assert "reserved_metadata:logical_block_id" in saved.raw["policy_issues"]
    finally:
        backend.close()


def test_membership_lookup_and_rebuild_remain_structural_indexes(tmp_path: Path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    revision = _catalog_revision(tmp_path, "alpha")
    records = (
        MemoryBlockRecord(
            block_id="record-a",
            revision_id="revision-a",
            workspace_id="alpha",
            content="Primary membership record.",
            block_membership=RecordBlockMembership(
                primary_block_id="stable_project_facts",
                secondary_block_ids=("artifact_claims",),
                catalog_revision_id=revision,
            ),
        ),
        MemoryBlockRecord(
            block_id="record-b",
            revision_id="revision-b",
            workspace_id="alpha",
            content="Second primary membership record.",
            block_membership=RecordBlockMembership(
                primary_block_id="stable_project_facts",
                catalog_revision_id=revision,
            ),
        ),
    )
    try:
        for record in records:
            backend.retain_block(record)

        assert (
            backend.record_ids_for_logical_blocks(("artifact_claims",), workspace_id="alpha") == ()
        )
        catalog = compile_semantic_catalog(
            load_memory_blocks(tmp_path, workspace_id="alpha", strict=True)
        )
        backend.retain_index_node(
            backend_contracts.MemoryIndexNode(
                node_id="stale-exact-facet",
                title="dependency: stale-dependency",
                node_type="exact_facet",
                workspace_id="alpha",
                tags=("exact_facet", "dependency"),
                metadata={
                    "term_type": "dependency",
                    "term_value": "stale-dependency",
                    "related_record_ids": ["record-a"],
                },
            )
        )
        first_digest = backend.rebuild_memory_indexes(catalog=catalog)["index_sha256"]
        second_digest = backend.rebuild_memory_indexes(catalog=catalog)["index_sha256"]
        assert first_digest == second_digest
        assert backend.get_index_node("stale-exact-facet", workspace_id="alpha") is None
        assert backend.record_ids_for_logical_blocks(
            ("stable_project_facts",), workspace_id="alpha"
        ) == ("record-a", "record-b")
    finally:
        backend.close()


def test_canonical_successor_closes_previous_membership(tmp_path: Path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        saved = client.remember(
            "The release writer uses the original implementation.",
            title="Release writer",
            kind="project_fact",
            requested_block_id="stable_project_facts",
        )
        correction = client.correction(
            saved.memory_id,
            "The release writer uses the corrected implementation.",
            requested_block_id="stable_project_facts",
        )

        original = backend.get_block(saved.memory_id, workspace_id="alpha")
        assert original is not None and original.block_membership is not None
        primary_block_id = original.block_membership.primary_block_id
        current_ids = backend.record_ids_for_logical_blocks(
            (primary_block_id,), workspace_id="alpha"
        )
        historical_ids = backend.record_ids_for_logical_blocks(
            (primary_block_id,), workspace_id="alpha", historical=True
        )
        assert current_ids == (correction["block_id"],)
        assert set(historical_ids) == {saved.memory_id, correction["block_id"]}

        projected_before = backend._conn.execute(
            """
            SELECT record_id, is_current, valid_to
            FROM record_block_memberships
            WHERE role='primary'
            ORDER BY record_id
            """
        ).fetchall()
        catalog = compile_semantic_catalog(
            load_memory_blocks(tmp_path, workspace_id="alpha", strict=True)
        )
        backend.rebuild_memory_indexes(catalog=catalog)
        projected_after = backend._conn.execute(
            """
            SELECT record_id, is_current, valid_to
            FROM record_block_memberships
            WHERE role='primary'
            ORDER BY record_id
            """
        ).fetchall()

        assert projected_after == projected_before
        by_record_id = {str(row[0]): (int(row[1]), str(row[2])) for row in projected_after}
        assert by_record_id[saved.memory_id][0] == 0
        assert by_record_id[saved.memory_id][1]
        assert by_record_id[correction["block_id"]] == (1, "")
    finally:
        backend.close()


def test_retracted_canonical_head_is_not_a_current_block_member(tmp_path: Path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        saved = client.remember(
            "This transient session fact must be forgotten.",
            title="Transient fact",
            kind="task_state",
            requested_block_id="stable_project_facts",
        )
        original = backend.get_block(saved.memory_id, workspace_id="alpha")
        assert original is not None and original.block_membership is not None
        primary_block_id = original.block_membership.primary_block_id

        forgotten = client.forget(saved.memory_id, reason="dialog memory reset")

        assert forgotten["ok"] is True
        retracted_id = str(forgotten["block_id"])
        assert (
            backend.record_ids_for_logical_blocks((primary_block_id,), workspace_id="alpha") == ()
        )
        historical_ids = backend.record_ids_for_logical_blocks(
            (primary_block_id,), workspace_id="alpha", historical=True
        )
        assert set(historical_ids) == {saved.memory_id, retracted_id}
        [(is_current,)] = backend._conn.execute(
            "SELECT is_current FROM record_block_memberships WHERE record_id=?",
            (retracted_id,),
        ).fetchall()
        assert is_current == 0
    finally:
        backend.close()


def test_recall_can_inspect_reflection_but_overlay_and_hidden_stay_blocked() -> None:
    membership = RecordBlockMembership(
        primary_block_id="reflection_candidates",
        catalog_revision_id="catalog:test",
    )
    policy = MemorySelectionPolicy()
    context = SelectionContext(
        scope_chain=("workspace:alpha",),
        source_horizon="archive",
        include_candidates=True,
    )
    base = MemoryBlockRecord(
        block_id="reflection",
        revision_id="revision-reflection",
        workspace_id="alpha",
        content="Reflection-only evidence.",
        delivery_policy="reflection_only",
        block_membership=membership,
    )

    assert policy.eligible_for_recall(base, context) is True
    assert policy.eligible_for_overlay(base, context) is False
    assert policy.eligible_for_recall(replace(base, delivery_policy="hidden"), context) is False
