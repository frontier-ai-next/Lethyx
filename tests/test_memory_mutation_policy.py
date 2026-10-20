import hashlib
import json
import sqlite3
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from lethyx import MemoryClient
from lethyx.contracts import (
    ConsolidationCommitReceipt,
    EvidenceRef,
    MemoryBlockRecord,
    MemoryEvent,
    RecordBlockMembership,
)
from lethyx.memory.backends.contracts import (
    EdgeType,
    MemoryEdgeRecord,
    memory_event_to_block_record,
)
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.blocks.catalog import compile_semantic_catalog
from lethyx.memory.blocks.registry import load_memory_blocks
from lethyx.memory.kernel.models import block_record_to_event
from lethyx.memory.kernel.telemetry import read_memory_events
from lethyx.memory.kernel.writer import _write_memory_event
from lethyx.memory import mcp_facade
from lethyx.memory.mutation import (
    CanonicalMutationIdentity,
    plan_lifecycle_mutation,
    plan_memory_mutation,
)


def _write_planned_record(
    tmp_path,
    index: BackendSQLiteIndex,
    *,
    content: str,
    title: str,
    expires_at: str = "",
) -> MemoryBlockRecord:
    catalog = compile_semantic_catalog(
        load_memory_blocks(tmp_path, workspace_id="alpha", strict=True)
    )
    event = MemoryEvent(
        content=content,
        block_membership=RecordBlockMembership(
            primary_block_id="project_ideas",
            catalog_revision_id=catalog.revision_id,
        ),
        title=title,
        memory_kind="fact",
        workspace_id="alpha",
        scope="workspace",
    )
    source = replace(memory_event_to_block_record(event), expires_at=expires_at)
    mutation = plan_memory_mutation(source, existing=None)
    result = _write_memory_event(
        tmp_path,
        block_record_to_event(mutation.record),
        workspace_id="alpha",
        backend_instance=index,
        mutation=mutation,
    )
    assert result.saved is True
    return mutation.record


def test_only_explicit_correction_can_cross_primary_block(tmp_path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        current = _write_planned_record(
            tmp_path,
            index,
            content="Original bounded idea.",
            title="Bounded idea",
        )
        moved = replace(
            current,
            content="Attempted cross-block revision.",
            block_membership=RecordBlockMembership(
                primary_block_id="decisions_and_contracts",
                catalog_revision_id=current.block_membership.catalog_revision_id,
            ),
        )
        with pytest.raises(ValueError, match="explicit correction"):
            plan_memory_mutation(moved, existing=current)
        correction = plan_memory_mutation(moved, existing=current, action="correction")
        assert correction.record.block_membership is not None
        assert correction.record.block_membership.primary_block_id == "decisions_and_contracts"
        assert correction.edge is not None
        assert correction.edge.edge_type == "supersedes"
    finally:
        index.close()


def test_new_candidates_do_not_infer_identity_from_title_or_source(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        first = client.remember(
            "delta = +5",
            title="Measurement",
            metadata={"source_path": "measurements.txt"},
            requested_block_id="stable_project_facts",
        )
        second = client.remember(
            "delta = -5",
            title="Measurement",
            metadata={"source_path": "measurements.txt"},
            requested_block_id="stable_project_facts",
        )
        assert first.memory_id != second.memory_id
        assert index.get_block(second.memory_id).revision_of == ""
        assert len(index.export_blocks()) == 2
    finally:
        index.close()


@pytest.mark.parametrize("temporal_mode", ["current", "historical"])
def test_event_time_unknown_survives_historical_recall_and_reopen(tmp_path, temporal_mode) -> None:
    path = tmp_path / "memory.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
    saved = client.remember(
        "The expedition lasted 17 days.",
        requested_block_id="stable_project_facts",
        _recorded_at="2026-10-17T12:00:00Z",
    )
    raw = index._conn.execute(
        "SELECT payload_json FROM blocks WHERE block_id=?", (saved.memory_id,)
    ).fetchone()[0]
    index.close()
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        query = dict(
            temporal_mode=temporal_mode,
            as_of="2023-05-30T23:42:00Z",
            record_ids=[saved.memory_id],
            logical_block_ids=["stable_project_facts"],
            include_candidates=True,
        )
        hits = client.recall("expedition", **query)
        assert [hit.metadata["block_id"] for hit in hits] == [saved.memory_id]
        assert hits[0].metadata["event_time_status"] == "unknown"
        assert not hits[0].metadata.get("valid_from")
        assert not client.recall("expedition", **query, known_as_of="2023-05-30T23:42:00Z")
        assert (
            index._conn.execute(
                "SELECT payload_json FROM blocks WHERE block_id=?", (saved.memory_id,)
            ).fetchone()[0]
            == raw
        )
    finally:
        index.close()


@pytest.mark.parametrize(
    ("valid_from", "valid_to", "expires_at", "expected"),
    [
        ("", "", "", True),
        ("2023-01-01T00:00:00Z", "", "", True),
        ("2023-06-01T00:00:00Z", "", "", False),
        ("2023-01-01T00:00:00Z", "2023-05-30T00:00:00Z", "", False),
        ("", "", "2023-05-30T00:00:00Z", False),
    ],
)
def test_historical_filter_uses_event_interval_not_provenance(
    tmp_path, valid_from, valid_to, expires_at, expected
) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        record = MemoryBlockRecord(
            block_id="source",
            content="Exact source",
            workspace_id="alpha",
            recorded_at="2026-10-17T12:00:00Z",
            valid_from=valid_from,
            valid_to=valid_to,
            expires_at=expires_at,
        )
        assert (
            index.record_matches_temporal(
                record, {"temporal_mode": "historical", "as_of": "2023-05-30T00:00:00Z"}
            )
            is expected
        )
        with pytest.raises(ValueError, match="as_of"):
            index.record_matches_temporal(record, {"temporal_mode": "historical", "as_of": "bad"})
    finally:
        index.close()


def test_unknown_start_keeps_correction_end_without_changing_knowledge_snapshot(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
    first = client.remember("Original value.", requested_block_id="stable_project_facts")
    derived = client.remember("Derived value.", requested_block_id="stable_project_facts")
    index.add_edge(
        MemoryEdgeRecord(
            src_id=derived.memory_id,
            dst_id=first.memory_id,
            edge_type=EdgeType.DERIVED_FROM,
            workspace_id="alpha",
        )
    )
    first_commit = index._conn.execute(
        "SELECT committed_at FROM blocks WHERE block_id=?", (first.memory_id,)
    ).fetchone()[0]
    time.sleep(0.003)
    corrected = client.correction(
        first.memory_id, "Corrected value.", requested_block_id="stable_project_facts"
    )
    successor = index.get_block(str(corrected["block_id"]))
    assert successor is not None
    boundary = datetime.fromisoformat(successor.valid_from.replace("Z", "+00:00"))
    index.close()
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        query = dict(
            query="",
            workspace_id="alpha",
            scope_chain=["workspace:alpha"],
            record_ids=[first.memory_id, successor.block_id],
            include_candidates=True,
            temporal_mode="historical",
        )
        assert [
            r.block_id
            for r in index.recall(
                {**query, "as_of": (boundary - timedelta(microseconds=1)).isoformat()}
            )
        ] == [first.memory_id]
        assert [r.block_id for r in index.recall({**query, "as_of": successor.valid_from})] == [
            successor.block_id
        ]
        assert [
            r.block_id
            for r in index.recall(
                {**query, "as_of": successor.valid_from, "known_as_of": first_commit}
            )
        ] == [first.memory_id]
        assert not index.get_block(first.memory_id).valid_from
        assert (
            index.validate_dependencies(
                derived.memory_id, as_of=(boundary - timedelta(microseconds=1)).isoformat()
            ).status
            == "valid"
        )
        assert (
            index.validate_dependencies(derived.memory_id, as_of=successor.valid_from).status
            == "invalidated"
        )
    finally:
        index.close()


def test_known_as_of_uses_trusted_commit_time_before_event_time(tmp_path, monkeypatch) -> None:
    path = tmp_path / "memory.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        first = client.remember(
            "The service used Flask.",
            title="Service framework",
            requested_block_id="stable_project_facts",
            valid_from="2020-01-01T00:00:00Z",
            _recorded_at="2001-01-01T00:00:00Z",
        )
        first_commit = index._conn.execute(
            "SELECT committed_at FROM canonical_mutations WHERE successor_id=?",
            (first.memory_id,),
        ).fetchone()[0]
        assert datetime.fromisoformat(first_commit).year == datetime.now(UTC).year
        time.sleep(0.003)
        second = client.correction(
            first.memory_id,
            "The service uses FastAPI.",
            requested_block_id="stable_project_facts",
        )
        second_id = str(second["block_id"])
        second_commit = index._conn.execute(
            "SELECT committed_at FROM canonical_mutations WHERE successor_id=?",
            (second_id,),
        ).fetchone()[0]
        assert second_commit > first_commit
        query = {
            "query": "",
            "workspace_id": "alpha",
            "include_candidates": True,
            "scope_chain": ["workspace:alpha"],
            "record_ids": [first.memory_id, second_id],
            "limit": 10,
        }
        assert [row.block_id for row in index.recall({**query, "known_as_of": first_commit})] == [
            first.memory_id
        ]
        assert [
            hit.metadata["block_id"]
            for hit in client.recall(
                "",
                known_as_of=first_commit,
                logical_block_ids=["stable_project_facts"],
                record_ids=[first.memory_id, second_id],
                include_candidates=True,
            )
        ] == [first.memory_id]
        monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
        monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)
        mcp_result = mcp_facade.lethyx_recall(
            "",
            workspace_id="alpha",
            repo_root=str(tmp_path),
            logical_block_ids=["stable_project_facts"],
            record_ids=[first.memory_id, second_id],
            include_candidates=True,
            known_as_of=first_commit,
        )
        assert mcp_result["ok"] is True
        assert [hit["metadata"]["block_id"] for hit in mcp_result["hits"]] == [first.memory_id]
        assert not mcp_result["hits"][0]["metadata"].get("valid_to")
        assert [row.block_id for row in index.recall({**query, "known_as_of": second_commit})] == [
            second_id
        ]
        assert [
            row.block_id
            for row in index.recall(
                {
                    **query,
                    "known_as_of": first_commit,
                    "as_of": "2020-06-01T00:00:00Z",
                }
            )
        ] == [first.memory_id]
        assert (
            index.recall({**query, "known_as_of": first_commit, "as_of": "2019-01-01T00:00:00Z"})
            == []
        )
    finally:
        index.close()

    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        assert [
            row.block_id for row in reopened.recall({**query, "known_as_of": first_commit})
        ] == [first.memory_id]
    finally:
        reopened.close()


def test_previous_sqlite_store_requires_explicit_new_store(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    index.close()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE schema_metadata SET value=15 WHERE key='schema_version'")
    with pytest.raises(ValueError, match="create a fresh store"):
        BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")


def test_explicit_creation_operation_replays_and_rejects_changed_arguments(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        arguments = {
            "title": "Measurement",
            "requested_block_id": "stable_project_facts",
            "operation_id": "measurement-write-1",
        }
        first = client.remember("delta = +5", **arguments)
        replay = client.remember("delta = +5", **arguments)
        assert first.memory_id == replay.memory_id
        assert index._conn.execute("SELECT COUNT(*) FROM canonical_mutations").fetchone()[0] == 1
        with pytest.raises(ValueError, match="receipt conflict"):
            client.remember("delta = -5", **arguments)
        assert index.get_block(first.memory_id).content == "delta = +5"
    finally:
        index.close()


def test_exact_semantic_retry_returns_the_existing_immutable_revision(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)

        first = client.remember(
            "The package uses FastAPI.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
            operation_id="package-framework-create",
        )
        second = client.remember(
            "The package uses FastAPI.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
            operation_id="package-framework-create",
        )

        assert first.memory_id == second.memory_id
        assert len(index.export_blocks()) == 1
        block = index.get_block(first.memory_id)
        assert block is not None
        assert block.canonical_key.startswith("alpha:fact:")
        assert block.revision_id.startswith("rev_")
        assert block.recorded_at.endswith("Z")
        assert index._conn.execute("SELECT COUNT(*) FROM canonical_mutations").fetchone()[0] == 1
    finally:
        index.close()


def test_catalog_revision_change_does_not_create_a_semantic_revision(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        first = client.remember(
            "The package uses FastAPI.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        current = index.get_block(first.memory_id)
        assert current is not None
        assert current.block_membership is not None
        rerouted = replace(
            current,
            block_id="",
            canonical_key="",
            revision_id="",
            semantic_catalog_version="catalog:next",
            block_membership=RecordBlockMembership(
                primary_block_id=current.block_membership.primary_block_id,
                secondary_block_ids=current.block_membership.secondary_block_ids,
                catalog_revision_id="catalog:next",
            ),
        )

        decision = plan_memory_mutation(rerouted, existing=current)

        assert decision.action == "retry"
        assert decision.record == current
    finally:
        index.close()


def test_correction_appends_immutable_successor_and_historical_recall(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        first = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
            _recorded_at="2026-01-01T00:00:00Z",
            valid_from="2026-01-01T00:00:00Z",
        )
        corrected = client.correction(
            first.memory_id,
            "The package uses FastAPI.",
            reason="README corrected",
            requested_block_id="stable_project_facts",
        )

        original = index.get_block(first.memory_id)
        successor = index.get_block(str(corrected["block_id"]))
        assert original is not None and original.content == "The package uses Flask."
        assert successor is not None and successor.content == "The package uses FastAPI."
        assert successor.revision_of == first.memory_id
        assert successor.supersedes_ids == (first.memory_id,)
        assert successor.recorded_at != original.recorded_at
        assert successor.valid_from == successor.recorded_at
        assert successor.valid_to == ""
        assert original.valid_to == successor.valid_from
        stored_original = index._conn.execute(
            "SELECT payload_json FROM blocks WHERE block_id=?", (first.memory_id,)
        ).fetchone()[0]
        stored_mutation = index._conn.execute(
            """
            SELECT record_payload_id, record_payload_sha256
            FROM canonical_mutations WHERE successor_id=?
            """,
            (first.memory_id,),
        ).fetchone()
        assert json.loads(stored_original)["valid_to"] == ""
        assert stored_mutation is not None
        original_payload = index._load_payload(
            str(stored_mutation[0]), str(stored_mutation[1]), payload_kind="record"
        )
        assert original_payload["valid_to"] == ""
        raw_mutations = index._conn.execute(
            "SELECT * FROM canonical_mutations ORDER BY commit_revision"
        ).fetchall()
        raw_original_revision_id = json.loads(stored_original)["revision_id"]
        index.close()
        index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        reopened_original = index.get_block(first.memory_id)
        reopened_successor = index.get_block(str(corrected["block_id"]))
        assert reopened_original is not None
        assert reopened_successor is not None
        assert reopened_original.valid_to == reopened_successor.valid_from
        assert reopened_original.revision_id == raw_original_revision_id
        assert (
            index._conn.execute(
                "SELECT payload_json FROM blocks WHERE block_id=?", (first.memory_id,)
            ).fetchone()[0]
            == stored_original
        )
        assert (
            index._conn.execute(
                "SELECT record_payload_id, record_payload_sha256 FROM canonical_mutations "
                "WHERE successor_id=?",
                (first.memory_id,),
            ).fetchone()
            == stored_mutation
        )
        assert (
            index._conn.execute(
                "SELECT * FROM canonical_mutations ORDER BY commit_revision"
            ).fetchall()
            == raw_mutations
        )
        original = reopened_original
        successor = reopened_successor
        assert index._conn.execute("SELECT COUNT(*) FROM canonical_mutations").fetchone()[0] == 2
        assert index.canonical_mutation_identity(
            successor_id=successor.block_id,
            workspace_id="alpha",
        ) == CanonicalMutationIdentity(
            action="correction",
            predecessor_id=first.memory_id,
            successor_id=successor.block_id,
        )
        assert [
            row.block_id
            for row in index.recall(
                {
                    "query": "package uses",
                    "workspace_id": "alpha",
                    "include_candidates": True,
                    "source_horizon": "durable",
                    "scope_chain": ["workspace:alpha"],
                    "limit": 10,
                }
            )
        ] == [successor.block_id]
        assert [
            row.block_id
            for row in index.recall(
                {
                    "query": "package uses",
                    "workspace_id": "alpha",
                    "include_candidates": True,
                    "source_horizon": "durable",
                    "scope_chain": ["workspace:alpha"],
                    "temporal_mode": "historical",
                    "as_of": "2026-01-15T00:00:00Z",
                    "limit": 10,
                }
            )
        ] == [first.memory_id]
        assert [
            row.block_id
            for row in index.recall(
                {
                    "query": "package uses",
                    "workspace_id": "alpha",
                    "include_candidates": True,
                    "source_horizon": "durable",
                    "scope_chain": ["workspace:alpha"],
                    "temporal_mode": "historical",
                    "as_of": successor.valid_from,
                    "limit": 10,
                }
            )
        ] == [successor.block_id]
        historical = index.recall(
            {
                "query": "package uses",
                "workspace_id": "alpha",
                "include_candidates": True,
                "source_horizon": "durable",
                "scope_chain": ["workspace:alpha"],
                "temporal_mode": "historical",
                "limit": 10,
            }
        )
        assert {row.block_id for row in historical} == {first.memory_id, successor.block_id}
        retried = client.correction(
            first.memory_id,
            "The package uses FastAPI.",
            reason="README corrected",
            requested_block_id="stable_project_facts",
        )
        assert retried["block_id"] == successor.block_id
        assert index._conn.execute("SELECT COUNT(*) FROM canonical_mutations").fetchone()[0] == 2

        rolled_back = client.rollback(
            successor.block_id,
            revision_id=original.revision_id,
            reason="rollback to verified prior fact",
        )
        rollback_record = index.get_block(str(rolled_back["block_id"]))
        assert rollback_record is not None
        assert rollback_record.content == original.content
        assert rollback_record.recorded_at != successor.recorded_at
        rollback_routes = client.export_graph()["provenance_routes"][rollback_record.block_id]
        assert any(
            route["edge_type"] == EdgeType.SUPERSEDES and route["target_id"] == successor.block_id
            for route in rollback_routes
        )
        assert index._conn.execute(
            "SELECT mutation_action, rollback_target_id FROM canonical_mutations "
            "ORDER BY commit_revision DESC LIMIT 1"
        ).fetchone() == ("rollback", first.memory_id)
        rollback_retry = client.rollback(
            successor.block_id,
            revision_id=original.revision_id,
            reason="rollback to verified prior fact",
        )
        assert rollback_retry["block_id"] == rollback_record.block_id
        assert index._conn.execute("SELECT COUNT(*) FROM canonical_mutations").fetchone()[0] == 3

        deprecated = client.deprecate(
            rollback_record.block_id,
            reason="superseded source retired",
        )
        deprecated_retry = client.deprecate(
            str(deprecated["block_id"]),
            reason="superseded source retired",
        )
        assert deprecated_retry["block_id"] == deprecated["block_id"]
        assert index._conn.execute("SELECT COUNT(*) FROM canonical_mutations").fetchone()[0] == 4

        bounded = client.remember(
            "Bounded historical fact.",
            title="Bounded historical fact",
            kind="fact",
            valid_from="2026-01-01T00:00:00Z",
            valid_to="2026-01-15T00:00:00Z",
            requested_block_id="stable_project_facts",
        )
        bounded_successor = client.correction(
            bounded.memory_id,
            "Bounded historical correction.",
            reason="bounded interval test",
            requested_block_id="stable_project_facts",
        )
        bounded_record = index.get_block(bounded.memory_id)
        assert bounded_record is not None
        assert bounded_record.valid_to == "2026-01-15T00:00:00Z"
        gap_rows = index.recall(
            {
                "query": "Bounded historical fact",
                "workspace_id": "alpha",
                "include_candidates": True,
                "source_horizon": "durable",
                "scope_chain": ["workspace:alpha"],
                "temporal_mode": "historical",
                "as_of": "2026-02-01T00:00:00Z",
                "record_ids": [bounded.memory_id, bounded_successor["block_id"]],
                "limit": 10,
            }
        )
        assert gap_rows == []
        current_gap_rows = index.recall(
            {
                "query": "Bounded historical fact",
                "workspace_id": "alpha",
                "include_candidates": True,
                "source_horizon": "durable",
                "scope_chain": ["workspace:alpha"],
                "as_of": "2026-02-01T00:00:00Z",
                "record_ids": [bounded.memory_id, bounded_successor["block_id"]],
                "limit": 10,
            }
        )
        assert current_gap_rows == []
        assert bounded_successor["block_id"] != bounded.memory_id
    finally:
        index.close()


def test_mutation_envelope_rejects_invalid_action_cross_key_rollback_and_uncommitted_receipt(
    tmp_path,
) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        first = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        current = index.get_block(first.memory_id)
        assert current is not None
        replacement = replace(current, content="The package uses FastAPI.")

        with pytest.raises(ValueError, match="unsupported Order15 mutation action"):
            plan_memory_mutation(replacement, existing=current, action="lifecycle")

        provisional = plan_memory_mutation(replacement, existing=current, action="correction")
        repeated_provisional = plan_memory_mutation(
            replacement,
            existing=current,
            action="correction",
        )
        assert repeated_provisional.record.revision_id == provisional.record.revision_id
        assert repeated_provisional.record.block_id == provisional.record.block_id
        evidence = EvidenceRef(
            ref_type="test_run",
            ref_id="consolidation:receipt",
            produced_by="verifier",
            producer_id="verifier:mutation-policy",
        )

        def mutation_state() -> dict[str, list[tuple[object, ...]]]:
            return {
                "heads": index._conn.execute(
                    "SELECT * FROM canonical_heads ORDER BY workspace_id, canonical_key_sha256"
                ).fetchall(),
                "block_statuses": index._conn.execute(
                    "SELECT block_id, status FROM blocks ORDER BY block_id"
                ).fetchall(),
                "mutations": index._conn.execute(
                    "SELECT * FROM canonical_mutations ORDER BY commit_revision"
                ).fetchall(),
                "invalidations": index._conn.execute(
                    "SELECT * FROM canonical_mutation_invalidations ORDER BY commit_revision, view_id"
                ).fetchall(),
                "receipts": index._conn.execute(
                    "SELECT * FROM canonical_mutation_receipts ORDER BY commit_revision, receipt_id"
                ).fetchall(),
            }

        before_rejection = mutation_state()
        with pytest.raises(ValueError, match="must commit the mutation successor"):
            plan_memory_mutation(
                replacement,
                existing=current,
                action="correction",
                consolidation_receipt=ConsolidationCommitReceipt(
                    receipt_id="receipt-missing-successor",
                    evidence_refs=(evidence,),
                    committed_ids=(current.block_id,),
                ),
            )
        assert mutation_state() == before_rejection
        accepted = plan_memory_mutation(
            replacement,
            existing=current,
            action="correction",
            consolidation_receipt=ConsolidationCommitReceipt(
                receipt_id="receipt-successor",
                evidence_refs=(evidence,),
                committed_ids=(provisional.record.block_id,),
            ),
        )
        assert accepted.consolidation_receipt is not None

        action_mismatch = plan_memory_mutation(
            current,
            existing=current,
            action="correction",
            current_identity=CanonicalMutationIdentity(
                action="deprecate",
                predecessor_id=current.revision_of,
                successor_id=current.block_id,
            ),
        )
        target_mismatch = plan_memory_mutation(
            current,
            existing=current,
            action="rollback",
            rollback_target_id="different-rollback-target",
            current_identity=CanonicalMutationIdentity(
                action="rollback",
                predecessor_id=current.revision_of,
                successor_id=current.block_id,
                rollback_target_id="original-rollback-target",
            ),
        )
        assert action_mismatch.action == "correction"
        assert target_mismatch.action == "rollback"

        with pytest.raises(ValueError, match="different canonical key"):
            plan_lifecycle_mutation(
                current,
                action="rollback",
                status=current.status,
                rollback_target=replace(
                    current,
                    block_id="different-canonical-target",
                    canonical_key="alpha:fact:different",
                ),
            )
    finally:
        index.close()


def test_remember_keeps_conflicting_independent_facts(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)

        old = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        new = client.remember(
            "The package uses FastAPI.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )

        assert new.memory_id != old.memory_id
        old_block = index.get_block(old.memory_id)
        new_block = index.get_block(new.memory_id)
        assert old_block is not None and old_block.status == "candidate"
        assert old_block.revision_of == ""
        assert old_block.supersedes_ids == ()
        assert new_block is not None
        assert new_block.revision_of == ""
        assert new_block.supersedes_ids == ()
        assert index.edges_for(old.memory_id) == []
    finally:
        index.close()


def test_append_only_host_and_codex_events_keep_distinct_canonical_records(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        hermes_first = client.remember(
            "Hermes turn evidence.",
            title="Conversation turn event",
            kind="episode",
            source_id="hermes-session:alpha",
            requested_block_id="workspace_episodes",
            metadata={"host_event_id": "hermes-turn-1"},
        )
        hermes_second = client.remember(
            "Hermes turn evidence.",
            title="Conversation turn event",
            kind="episode",
            source_id="hermes-session:alpha",
            requested_block_id="workspace_episodes",
            metadata={"host_event_id": "hermes-turn-2"},
        )
        codex_first = client.remember(
            "Observed Codex lifecycle event.",
            title="Codex UserPromptSubmit",
            source_id="codex-hook:user-prompt:aaaa",
            requested_block_id="workspace_episodes",
        )
        codex_second = client.remember(
            "Observed Codex lifecycle event.",
            title="Codex UserPromptSubmit",
            source_id="codex-hook:user-prompt:bbbb",
            requested_block_id="workspace_episodes",
        )

        records = [
            index.get_block(result.memory_id)
            for result in (hermes_first, hermes_second, codex_first, codex_second)
        ]
        assert all(record is not None and record.status == "candidate" for record in records)
        canonical_keys = {record.canonical_key for record in records if record is not None}
        assert len(canonical_keys) == 4
    finally:
        index.close()


def test_stable_source_reference_keeps_a_mutable_successor_chain(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        first = client.remember(
            "Runtime profile selects concise recalls.",
            title="Runtime profile",
            source_id="runtime-request:first",
            requested_block_id="workspace_episodes",
            metadata={"source_ref": "hermes-runtime:alpha:profile"},
        )
        second = client.amend(
            first.memory_id,
            expected_revision=index.get_block(first.memory_id).revision_id,
            reason="explicit source update",
            content="Runtime profile selects evidence-backed concise recalls.",
        )

        prior = index.get_block(first.memory_id)
        current = index.get_block(second["memory_id"])
        assert prior is not None and prior.status == "candidate"
        assert current is not None
        assert current.canonical_key == prior.canonical_key
        assert current.revision_of == first.memory_id
        assert current.supersedes_ids == (first.memory_id,)
    finally:
        index.close()


def test_remember_rejects_transport_identity_metadata_at_the_public_boundary(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        for key in (
            "block_id",
            "canonical_id",
            "memory_id",
            "record_id",
            "native_id",
            "native_reference",
            "commit_revision",
            "idempotency_key",
            "mutation_correlation_id",
            "service_receipt_sha256",
        ):
            result = client.remember(
                "Transport metadata must not route.",
                requested_block_id="workspace_episodes",
                metadata={key: "forged"},
            )
            assert result.saved is False
            assert f"reserved_metadata:{key}" in result.raw["policy_issues"]
        assert index.export_blocks() == []
    finally:
        index.close()


def test_correction_to_prior_content_has_new_lineage_revision_and_retry_is_stable(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        first = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        client.remember(
            "The package uses FastAPI.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        corrected = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
            operation_id="return-to-flask",
        )
        retried = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
            operation_id="return-to-flask",
        )

        first_block = index.get_block(first.memory_id)
        corrected_block = index.get_block(corrected.memory_id)
        retried_block = index.get_block(retried.memory_id)
        assert first_block is not None
        assert corrected_block is not None
        assert retried_block is not None
        assert corrected_block.revision_id != first_block.revision_id
        assert retried_block.revision_id == corrected_block.revision_id
        assert retried_block.recorded_at == corrected_block.recorded_at
    finally:
        index.close()


def test_remember_supersedes_current_canonical_head_after_reopen(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)

        first = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        second = client.amend(
            first.memory_id,
            expected_revision=index.get_block(first.memory_id).revision_id,
            reason="explicit source update",
            content="The package uses FastAPI.",
        )
        third = client.amend(
            second["memory_id"],
            expected_revision=index.get_block(second["memory_id"]).revision_id,
            reason="explicit source update",
            content="The package uses Starlette.",
        )
    finally:
        index.close()

    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        first_block = reopened.get_block(first.memory_id)
        second_block = reopened.get_block(second["memory_id"])
        third_block = reopened.get_block(third["memory_id"])

        assert first_block is not None
        assert second_block is not None
        assert third_block is not None
        assert first_block.status == "candidate"
        assert first_block.revision_of == ""
        assert first_block.supersedes_ids == ()
        assert second_block.status == "candidate"
        assert second_block.revision_of == first.memory_id
        assert second_block.supersedes_ids == (first.memory_id,)
        assert third_block.status == "candidate"
        assert third_block.revision_of == second["memory_id"]
        assert third_block.supersedes_ids == (second["memory_id"],)
        assert {block.block_id for block in reopened.export_blocks()} == {
            first.memory_id,
            second["memory_id"],
            third["memory_id"],
        }
        assert [
            block.block_id
            for block in reopened.recall(
                {
                    "query": "package framework",
                    "workspace_id": "alpha",
                    "include_candidates": True,
                    "source_horizon": "durable",
                    "scope_chain": ["workspace:alpha"],
                    "limit": 8,
                }
            )
        ] == [third["memory_id"]]
    finally:
        reopened.close()


def test_writer_rejects_stale_successor_before_second_canonical_head(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        root = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        committed = client.amend(
            root.memory_id,
            expected_revision=index.get_block(root.memory_id).revision_id,
            reason="explicit source update",
            content="The package uses FastAPI.",
        )
    finally:
        index.close()

    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        committed_block = reopened.get_block(committed["memory_id"])
        assert committed_block is not None
        stale = _write_memory_event(
            tmp_path,
            MemoryEvent(
                content="The package uses Django.",
                block_membership=committed_block.block_membership,
                title="Package framework",
                memory_kind="fact",
                status="candidate",
                trust_level="agent_reported",
                scope="workspace",
                delivery_policy="on_demand",
                workspace_id="alpha",
                revision_of=root.memory_id,
                supersedes_ids=(root.memory_id,),
                canonical_key=committed_block.canonical_key,
                block_id="stale-package-framework-successor",
            ),
            workspace_id="alpha",
            backend_instance=reopened,
        )

        assert stale.saved is False
        assert stale.error == "canonical_head_conflict"
        assert reopened.get_block("stale-package-framework-successor") is None
        assert {
            block.block_id for block in reopened.export_blocks() if block.status == "candidate"
        } == {root.memory_id, committed["memory_id"]}
        assert [
            block.block_id
            for block in reopened.recall(
                {
                    "query": "package framework",
                    "workspace_id": "alpha",
                    "include_candidates": True,
                    "source_horizon": "durable",
                    "scope_chain": ["workspace:alpha"],
                    "limit": 8,
                }
            )
        ] == [committed["memory_id"]]
        assert [
            (edge.src_id, edge.dst_id)
            for edge in reopened.edges_for(root.memory_id)
            if edge.edge_type == EdgeType.SUPERSEDES
        ] == [(committed["memory_id"], root.memory_id)]
    finally:
        reopened.close()


@pytest.mark.parametrize("trigger", ["expired", "cap"])
def test_expiry_or_cap_preserves_canonical_head_after_uncommitted_crash(
    monkeypatch,
    tmp_path,
    trigger: str,
) -> None:
    if trigger == "cap":
        monkeypatch.setenv("LETHYX_BACKEND_INDEX_MAX_ROWS", "1")
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        root = _write_planned_record(
            tmp_path,
            index,
            content="The package uses Flask.",
            title="Package framework",
            expires_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat()
            if trigger == "expired"
            else "",
        )
        pending = plan_memory_mutation(
            replace(root, content="The package uses FastAPI."),
            existing=root,
            action="correction",
        )

        class SimulatedCrash(BaseException):
            pass

        def crash_before_completion(*_args) -> None:
            raise SimulatedCrash

        with monkeypatch.context() as patch:
            patch.setattr(index, "complete_canonical_head", crash_before_completion)
            with pytest.raises(SimulatedCrash):
                _write_memory_event(
                    tmp_path,
                    block_record_to_event(pending.record),
                    workspace_id="alpha",
                    backend_instance=index,
                    mutation=pending,
                )

        assert index.get_block(pending.record.block_id) is None
        if trigger == "expired":
            assert index.expire_scope("", "") == 0
        else:
            _write_planned_record(
                tmp_path,
                index,
                content="Unrelated bounded record.",
                title="Unrelated record",
            )

        assert index.get_block(root.block_id) is not None
        assert (
            index._conn.execute(
                """
                SELECT COUNT(*) FROM canonical_heads AS heads
                LEFT JOIN blocks ON blocks.block_id = heads.block_id
                WHERE blocks.block_id IS NULL
                """
            ).fetchone()[0]
            == 0
        )
        assert (
            index._conn.execute(
                "SELECT 1 FROM canonical_head_outbox WHERE successor_id=?",
                (pending.record.block_id,),
            ).fetchone()
            is None
        )
        assert index.write_outbox_status(pending.record.block_id) is None
    finally:
        index.close()


def test_revision_journal_keeps_content_free_head_linkage_after_reopen(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        first = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        second = client.amend(
            first.memory_id,
            expected_revision=index.get_block(first.memory_id).revision_id,
            reason="explicit source update",
            content="The package uses FastAPI.",
        )
        third = client.amend(
            second["memory_id"],
            expected_revision=index.get_block(second["memory_id"]).revision_id,
            reason="explicit source update",
            content="The package uses Starlette.",
        )
    finally:
        index.close()

    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        current = reopened.get_block(third["memory_id"])
        assert current is not None
        assert current.status == "candidate"
        canonical_key = current.canonical_key
    finally:
        reopened.close()

    journal = [
        event
        for event in read_memory_events(tmp_path)
        if event["event_type"] == "memory_revision_committed"
    ]
    assert len(journal) == 3
    assert [event["operation"] for event in journal] == ["revision_commit"] * 3
    assert [event["decision"] for event in journal] == ["canonical_mutation"] * 3
    assert [event["outcome"] for event in journal] == ["insert", "correction", "correction"]
    assert [event["data"]["predecessor_count"] for event in journal] == [0, 1, 1]
    assert [event["data"]["canonical_key_sha256"] for event in journal] == [
        hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:16],
        hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:16],
        hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:16],
    ]
    assert journal[0]["selected_id_hashes"] == [
        hashlib.sha256(first.memory_id.encode("utf-8")).hexdigest()[:16],
    ]
    assert journal[1]["selected_id_hashes"] == [
        hashlib.sha256(second["memory_id"].encode("utf-8")).hexdigest()[:16],
        hashlib.sha256(first.memory_id.encode("utf-8")).hexdigest()[:16],
    ]
    assert journal[2]["selected_id_hashes"] == [
        hashlib.sha256(third["memory_id"].encode("utf-8")).hexdigest()[:16],
        hashlib.sha256(second["memory_id"].encode("utf-8")).hexdigest()[:16],
    ]
    serialized = json.dumps(journal)
    for value in (
        first.memory_id,
        second["memory_id"],
        third["memory_id"],
        canonical_key,
        "The package uses Flask.",
        "The package uses FastAPI.",
        "The package uses Starlette.",
    ):
        assert value not in serialized


def test_pending_creation_recovers_original_after_amend(tmp_path, monkeypatch) -> None:
    from lethyx import api

    index = BackendSQLiteIndex(
        tmp_path / ".Lethyx" / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        complete = api.complete_remember_receipt

        def lost_response(*args, **kwargs):
            raise OSError("lost response")

        monkeypatch.setattr(api, "complete_remember_receipt", lost_response)
        with pytest.raises(OSError, match="lost response"):
            client.remember(
                "Original.", requested_block_id="stable_project_facts", operation_id="create-one"
            )
        status = client.operation_status("create-one")
        assert status["durable_status"] == "committed"
        assert status["receipt_recovery"] == "canonical_creation_receipt"
        assert status["record_id"] == index.export_blocks()[0].block_id
        receipt_reader = api.read_mutation_receipt_status
        monkeypatch.setattr(api, "read_mutation_receipt_status", lambda *_a, **_k: None)
        assert client.operation_status("create-one")["record_id"] == status["record_id"]
        monkeypatch.setattr(api, "read_mutation_receipt_status", receipt_reader)
        original = index.export_blocks()[0]
        updated = client.amend(
            original.block_id,
            expected_revision=original.revision_id,
            content="Changed.",
            reason="update",
        )
        monkeypatch.setattr(api, "complete_remember_receipt", complete)
        replay = client.remember(
            "Original.", requested_block_id="stable_project_facts", operation_id="create-one"
        )
        assert replay.saved and replay.memory_id == original.block_id
        assert replay.raw["replayed"] is True
        assert index.get_block(updated["memory_id"]).content == "Changed."
        assert len(index.export_blocks()) == 2
    finally:
        index.close()


def test_creation_status_recovers_after_reopen_when_caller_receipt_was_lost(
    tmp_path, monkeypatch
) -> None:
    from lethyx import api

    path = tmp_path / "index.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    complete = api.complete_remember_receipt

    def lost_response(*_args, **_kwargs):
        raise OSError("lost response")

    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        monkeypatch.setattr(api, "complete_remember_receipt", lost_response)
        with pytest.raises(OSError, match="lost response"):
            client.remember(
                "Exact original.",
                requested_block_id="stable_project_facts",
                operation_id="reopen-creation",
            )
    finally:
        index.close()
        monkeypatch.setattr(api, "complete_remember_receipt", complete)

    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=reopened)
        status = client.operation_status("reopen-creation")
        assert status["durable_status"] == "committed"
        assert status["receipt_recovery"] == "canonical_creation_receipt"
        other_profile = MemoryClient(
            tmp_path, workspace_id="alpha", profile_id="other", backend_instance=reopened
        )
        assert other_profile.operation_status("reopen-creation")["durable_status"] == "not_found"
        assert client.read_exact_record(record_id=status["record_id"]).content == "Exact original."
        replay = client.remember(
            "Exact original.",
            requested_block_id="stable_project_facts",
            operation_id="reopen-creation",
        )
        assert replay.saved and replay.memory_id == status["record_id"]
        assert len(reopened.export_blocks()) == 1
    finally:
        reopened.close()


def test_creation_receipt_replays_after_leaf_archival(tmp_path) -> None:
    from lethyx.memory.blocks.registry import load_workspace_topology, create_workspace_block

    index = BackendSQLiteIndex(
        tmp_path / ".Lethyx" / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        root = next(
            state.block.block_id
            for state in load_workspace_topology(tmp_path, "alpha")
            if not state.block.parent_id
        )
        leaf = create_workspace_block(
            tmp_path,
            "alpha",
            name="Findings",
            synopsis="Findings",
            parent_id=root,
            reason="model choice",
        )
        first = client.remember("Finding.", requested_block_id=leaf["block_id"], operation_id="one")
        client.forget(first.memory_id, reason="retired")
        client.archive_block(leaf["block_id"], expected_revision=leaf["revision"], reason="retired")
        replay = client.remember(
            "Finding.", requested_block_id=leaf["block_id"], operation_id="one"
        )
        assert replay.saved and replay.memory_id == first.memory_id and replay.raw["replayed"]
        assert len(index.export_blocks()) == 2
    finally:
        index.close()
