from dataclasses import replace
import json
import sqlite3

import pytest

from lethyx import MemoryClient
from lethyx.contracts import ConsolidationCommitReceipt, EvidenceRef
from lethyx.memory.backends.contracts import CanonicalHeadConflict, EdgeType
from lethyx.memory.backends.mempalace import MemPalaceBackend, _record_partition_identity
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.mutation import plan_memory_mutation


def test_prepare_stages_payload_without_publishing_successor(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        root = client.remember(
            "The parser uses an unchecked writer.",
            title="Parser writer",
            kind="fact",
            metadata={"source_path": "parser.py"},
            requested_block_id="stable_project_facts",
        )
        current = index.get_block(root.memory_id)
        assert current is not None
        mutation = plan_memory_mutation(
            replace(current, content="The parser uses a checked writer."),
            existing=current,
            action="correction",
        )
        reservation = index.reserve_canonical_head(
            workspace_id="alpha",
            canonical_key=mutation.record.canonical_key,
            predecessor_id=current.block_id,
            successor_id=mutation.record.block_id,
            successor_status=mutation.record.status,
            successor_authority=mutation.record.authority,
            mutation_action=mutation.action,
            mutation=mutation,
        )

        prepared = index.prepare_canonical_write(mutation.record)

        assert index.get_block(mutation.record.block_id) is None
        assert index._conn.execute(
            "SELECT COUNT(*) FROM blocks WHERE block_id=?", (mutation.record.block_id,)
        ).fetchone() == (0,)
        assert index.write_outbox_status(mutation.record.block_id) == "pending"

        index.complete_canonical_head(reservation, mutation)

        assert index.get_block(mutation.record.block_id) == mutation.record
        assert index._conn.execute(
            "SELECT COUNT(*) FROM canonical_mutations WHERE successor_id=?",
            (mutation.record.block_id,),
        ).fetchone() == (1,)
        index.complete_write_outbox(int(prepared["commit_revision"]))
    finally:
        index.close()


def test_writer_commits_successor_edge_and_head_in_one_transaction(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        original = client.remember(
            "The parser uses an unchecked writer.",
            title="Parser writer",
            kind="fact",
            metadata={"source_path": "parser.py"},
            requested_block_id="stable_project_facts",
        )
        corrected = client.correction(
            original.memory_id,
            "The parser uses a checked writer.",
            reason="writer contract corrected",
            requested_block_id="stable_project_facts",
        )

        successor_id = str(corrected["block_id"])
        [event] = index._conn.execute(
            """
            SELECT predecessor_id, successor_id, mutation_action
            FROM canonical_mutations ORDER BY commit_revision DESC LIMIT 1
            """
        ).fetchall()
        assert event == (original.memory_id, successor_id, "correction")
        assert [
            (edge.src_id, edge.dst_id, edge.edge_type)
            for edge in index.edges_for(original.memory_id)
        ] == [(successor_id, original.memory_id, EdgeType.SUPERSEDES)]
        head = index._conn.execute("SELECT block_id FROM canonical_heads").fetchone()
        assert head == (successor_id,)
    finally:
        index.close()


def test_conflicting_head_reservation_does_not_append_a_partial_mutation(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        root = client.remember(
            "The parser uses an unchecked writer.",
            title="Parser writer",
            kind="fact",
            metadata={"source_path": "parser.py"},
            requested_block_id="stable_project_facts",
        )
        client.correction(
            root.memory_id,
            "The parser uses a checked writer.",
            requested_block_id="stable_project_facts",
        )

        root_record = index.get_block(root.memory_id)
        assert root_record is not None
        stale = plan_memory_mutation(
            replace(root_record, content="The parser uses an unsafe writer."),
            existing=root_record,
            action="correction",
        )
        with pytest.raises(CanonicalHeadConflict, match="canonical_head_conflict"):
            index.reserve_canonical_head(
                workspace_id="alpha",
                canonical_key=stale.record.canonical_key,
                predecessor_id=stale.superseded_block_id,
                successor_id=stale.record.block_id,
                successor_status=stale.record.status,
                successor_authority=stale.record.authority,
                mutation_action=stale.action,
                rollback_target_id=stale.rollback_target_id,
                mutation=stale,
            )
        assert index._conn.execute("SELECT COUNT(*) FROM canonical_mutations").fetchone()[0] == 2
        assert index.get_block(stale.record.block_id) is None
    finally:
        index.close()


def test_restart_replays_the_persisted_typed_mutation_and_receipt(tmp_path) -> None:
    backend = MemPalaceBackend(tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        root = client.remember(
            "The parser uses an unchecked writer.",
            title="Parser writer",
            kind="episode",
            metadata={"source_path": "parser.py"},
            requested_block_id="workspace_episodes",
        )
        current = backend.get_block(root.memory_id)
        assert current is not None
        replacement = replace(current, content="The parser uses a checked writer.")
        provisional = plan_memory_mutation(replacement, existing=current, action="correction")
        evidence = EvidenceRef(
            ref_type="test_run",
            ref_id="consolidation:restart",
            produced_by="verifier",
            producer_id="verifier:restart",
        )
        mutation = plan_memory_mutation(
            replacement,
            existing=current,
            action="correction",
            consolidation_receipt=ConsolidationCommitReceipt(
                receipt_id="receipt-restart",
                evidence_refs=(evidence,),
                committed_ids=(provisional.record.block_id,),
            ),
        )
        reservation = backend.reserve_canonical_head(
            workspace_id="alpha",
            canonical_key=mutation.record.canonical_key,
            predecessor_id=mutation.superseded_block_id,
            successor_id=mutation.record.block_id,
            successor_status=mutation.record.status,
            successor_authority=mutation.record.authority,
            mutation_action=mutation.action,
            rollback_target_id=mutation.rollback_target_id,
            mutation=mutation,
        )
        prepared = backend.prepare_canonical_write(mutation.record)
        assert prepared["ok"] is True
        assert backend.get_block(mutation.record.block_id) is None
        assert [
            record.block_id
            for record in backend.recall(
                {
                    "query": "parser writer",
                    "workspace_id": "alpha",
                    "include_candidates": True,
                    "source_horizon": "archive",
                    "logical_block_ids": ["workspace_episodes"],
                    "record_ids": [root.memory_id],
                    "scope_chain": ["workspace:alpha"],
                    "limit": 8,
                }
            )
        ] == [root.memory_id]
    finally:
        backend.close()

    reopened = MemPalaceBackend(tmp_path, workspace_id="alpha")
    try:
        recovered = reopened.get_block(mutation.record.block_id)
        original = reopened.get_block(root.memory_id)
        receipt_json = reopened._index._conn.execute(
            "SELECT payload_json FROM canonical_mutation_receipts"
        ).fetchone()
        assert recovered == mutation.record
        assert original is not None and original.content == current.content
        assert [
            (edge.src_id, edge.dst_id, edge.edge_type)
            for edge in reopened.edges_for(root.memory_id)
        ] == [(mutation.record.block_id, root.memory_id, EdgeType.SUPERSEDES)]
        assert receipt_json is not None
        assert '"receipt_id": "receipt-restart"' in receipt_json[0]
        assert '"ref_id": "consolidation:restart"' in receipt_json[0]
        assert reopened._index.canonical_head_status(mutation.record.block_id) == "committed"
        assert reservation.successor_id == mutation.record.block_id
    finally:
        reopened.close()


@pytest.mark.parametrize("corruption", ["outbox_identity", "workspace"])
def test_restart_fails_closed_for_a_mismatched_pending_native_projection(
    tmp_path,
    corruption: str,
) -> None:
    backend = MemPalaceBackend(tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        root = client.remember(
            "The parser uses an unchecked writer.",
            title="Parser writer",
            kind="fact",
            metadata={"source_path": "parser.py"},
            requested_block_id="stable_project_facts",
        )
        current = backend.get_block(root.memory_id)
        assert current is not None
        mutation = plan_memory_mutation(
            replace(current, content="The parser uses a checked writer."),
            existing=current,
            action="correction",
        )
        backend.reserve_canonical_head(
            workspace_id="alpha",
            canonical_key=mutation.record.canonical_key,
            predecessor_id=mutation.superseded_block_id,
            successor_id=mutation.record.block_id,
            successor_status=mutation.record.status,
            successor_authority=mutation.record.authority,
            mutation_action=mutation.action,
            rollback_target_id=mutation.rollback_target_id,
            mutation=mutation,
        )
        assert backend.retain_block(mutation.record)["ok"] is True
        collection = backend._native.collection(
            "palace.idea",
            partition=_record_partition_identity(mutation.record),
        )
        native = collection.get(ids=[mutation.record.block_id])
        metadata = dict(native.metadatas[0])
        if corruption == "outbox_identity":
            metadata["commit_revision"] = "0"
        else:
            canonical_record = json.loads(metadata["canonical_record_json"])
            canonical_record["workspace_id"] = "beta"
            metadata["canonical_record_json"] = json.dumps(canonical_record, sort_keys=True)
        collection.update(ids=[mutation.record.block_id], metadatas=[metadata])
        index_path = backend._index.path
    finally:
        backend.close()

    reopened = MemPalaceBackend(tmp_path, workspace_id="alpha")
    try:
        # Canonical recovery is independent of a damaged derived index.
        assert reopened.get_block(mutation.record.block_id).content == mutation.record.content
        with pytest.raises(RuntimeError, match="native_projection_rebuild_required"):
            reopened.recall(
                {
                    "query": "parser",
                    "logical_block_ids": ["stable_project_facts"],
                    "include_candidates": True,
                    "scope_chain": ["workspace:alpha"],
                }
            )
    finally:
        reopened.close()

    with sqlite3.connect(index_path) as connection:
        head_status = connection.execute(
            "SELECT status FROM canonical_head_outbox WHERE successor_id=?",
            (mutation.record.block_id,),
        ).fetchone()[0]
        write_status = connection.execute(
            "SELECT status FROM write_outbox WHERE block_id=? ORDER BY commit_revision DESC LIMIT 1",
            (mutation.record.block_id,),
        ).fetchone()[0]
    assert head_status == "committed"
    assert write_status == "committed"
