from dataclasses import replace

import pytest

from um_mem import MemoryClient
from um_mem.contracts import ConsolidationCommitReceipt, EvidenceRef
from um_mem.memory.backends.contracts import EdgeType, MemoryEdgeRecord, MemoryIndexNode
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.blocks.catalog import (
    block_descriptor_node_id,
    compile_semantic_catalog,
    sync_block_descriptor_index,
)
from um_mem.memory.blocks.models import MemoryBlock
from um_mem.memory.blocks.registry import (
    load_memory_blocks,
    validate_leaf_block_contracts,
    validate_memory_block_tree,
)
from um_mem.memory.edge_graph import score_memory_edges
from um_mem.memory.kernel.models import block_record_to_event
from um_mem.memory.kernel.writer import _write_memory_event
from um_mem.memory.mutation import plan_memory_mutation


def _evidence(name: str) -> EvidenceRef:
    return EvidenceRef(
        ref_type="test_run",
        ref_id=name,
        produced_by="verifier",
        producer_id=f"verifier:graph-index:{name}",
    )


def test_writer_keeps_external_evidence_on_record_without_dangling_graph_node(
    tmp_path,
) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    drive_root = tmp_path / "legacy-drive"
    try:
        result = client.remember(
            "The release graph uses the selected backend edge contract.",
            title="Release graph",
            kind="fact",
            requested_block_id="stable_project_facts",
            evidence_refs=[_evidence("source-event")],
            metadata={"drive_root": str(drive_root)},
        )

        assert index.edges_for(result.memory_id) == []
        assert not (drive_root / "graph").exists()
        graph = client.export_graph()
        assert graph["edges"] == []
        assert graph["audit"]["dangling_edges"] == []
        assert graph["audit"]["isolated_durable_semantic_record_ids"] == [result.memory_id]
    finally:
        index.close()


def test_edge_graph_scores_typed_product_edges_without_a_parallel_store() -> None:
    score = score_memory_edges(
        "candidate",
        (
            MemoryEdgeRecord("proof", "candidate", EdgeType.VERIFIED_BY),
            MemoryEdgeRecord("candidate", "conflict", EdgeType.CONTRADICTS),
        ),
    )

    assert score.verified_support_paths == 1
    assert score.contradiction_paths == 1
    assert score.hard_contradiction is True
    assert score.details["support_paths"] == [["proof", "candidate"]]


def test_edge_replay_is_idempotent_and_export_is_workspace_scoped(tmp_path) -> None:
    database = tmp_path / "index.sqlite"
    alpha = BackendSQLiteIndex(database, backend_id="test", workspace_id="alpha")
    beta = BackendSQLiteIndex(database, backend_id="test", workspace_id="beta")
    try:
        for node_id in ("proof", "candidate"):
            alpha.retain_index_node(
                MemoryIndexNode(
                    node_id=node_id,
                    title=node_id,
                    node_type="test",
                    workspace_id="alpha",
                )
            )
        for node_id in ("other-proof", "other-candidate"):
            beta.retain_index_node(
                MemoryIndexNode(
                    node_id=node_id,
                    title=node_id,
                    node_type="test",
                    workspace_id="beta",
                )
            )
        edge = MemoryEdgeRecord(
            "proof",
            "candidate",
            EdgeType.VERIFIED_BY,
            workspace_id="alpha",
            reason="verified release proof",
            created_at="2026-07-14T00:00:00+00:00",
        )
        first = alpha.add_edge(edge)
        assert "replayed" not in first
        replay = alpha.add_edge(replace(edge, created_at="2026-07-14T00:00:01+00:00"))
        assert replay["replayed"] is True

        beta.add_edge(
            MemoryEdgeRecord(
                "other-proof",
                "other-candidate",
                EdgeType.SUPPORTS,
                workspace_id="beta",
            )
        )

        assert [(item.src_id, item.dst_id) for item in alpha.export_edges()] == [
            ("proof", "candidate")
        ]
        with pytest.raises(ValueError, match="workspace"):
            alpha.add_edge(
                MemoryEdgeRecord(
                    "leak",
                    "candidate",
                    EdgeType.SUPPORTS,
                    workspace_id="beta",
                )
            )
        with pytest.raises(ValueError, match="immutable"):
            alpha.add_edge(replace(edge, reason="changed proof"))
    finally:
        beta.close()
        alpha.close()


def test_descriptor_synopsis_coverage_and_revision_update_incrementally(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    evidence = _evidence("consolidation")
    try:
        first = client.remember(
            "The release writer starts in legacy mode.",
            title="Release writer state",
            kind="project_fact",
            requested_block_id="stable_project_facts",
            evidence_refs=[_evidence("initial")],
            operation_id="initial-release-state",
        )
        descriptor_id = block_descriptor_node_id("ws", "stable_project_facts")
        initial = index.get_index_node(descriptor_id, workspace_id="ws")
        assert initial is not None
        assert initial.metadata["index_revision"] == 1
        assert initial.metadata["coverage"]["canonical_revision_count"] == 1
        assert initial.metadata["latest_record_title"] == "Release writer state"
        descriptor = initial.metadata["descriptor"]
        assert descriptor["producer"] == "canonical_writer"
        assert descriptor["consumers"]
        assert descriptor["recall_path"] == "hybrid_recall"
        assert descriptor["lifecycle_policy"]["initial_status"] == "candidate"
        assert descriptor["isolation_policy"] == {
            "cross_workspace": False,
            "dimension": "workspace",
        }

        client.remember(
            "The release writer starts in legacy mode.",
            title="Release writer state",
            kind="project_fact",
            requested_block_id="stable_project_facts",
            evidence_refs=[_evidence("initial")],
            operation_id="initial-release-state",
        )
        replayed = index.get_index_node(descriptor_id, workspace_id="ws")
        assert replayed is not None
        assert replayed.metadata["index_revision"] == 1

        correction = client.correction(
            first.memory_id,
            "The release writer uses the verified path.",
            reason="verified source",
            requested_block_id="stable_project_facts",
        )
        corrected = index.get_block(str(correction["block_id"]))
        assert corrected is not None
        after_correction = index.get_index_node(descriptor_id, workspace_id="ws")
        assert after_correction is not None
        assert after_correction.metadata["index_revision"] == 2
        assert after_correction.metadata["last_update_reason"] == "correction"

        replacement = replace(
            corrected,
            content="The consolidated release writer uses the verified path.",
        )
        provisional = plan_memory_mutation(
            replacement,
            existing=corrected,
            action="correction",
        )
        receipt = ConsolidationCommitReceipt(
            receipt_id="consolidation-release-writer",
            evidence_refs=(evidence,),
            committed_ids=(provisional.record.block_id,),
        )
        mutation = plan_memory_mutation(
            replacement,
            existing=corrected,
            action="correction",
            consolidation_receipt=receipt,
        )
        written = _write_memory_event(
            tmp_path,
            block_record_to_event(mutation.record),
            workspace_id="ws",
            backend_instance=index,
            mutation=mutation,
        )
        assert written.saved is True

        consolidated = index.get_index_node(descriptor_id, workspace_id="ws")
        assert consolidated is not None
        assert consolidated.metadata["index_revision"] == 3
        assert consolidated.metadata["last_update_reason"] == "consolidation"
        # Descriptor coverage is current-head coverage; historical revisions remain
        # in canonical history but cannot re-enter current recall/index membership.
        assert consolidated.metadata["coverage"]["canonical_revision_count"] == 1
        assert consolidated.metadata["source_revision_ids"] == [mutation.record.revision_id]
        assert {
            record.revision_id
            for record in index.canonical_history(
                canonical_key=mutation.record.canonical_key,
                workspace_id="ws",
            )
        } == {
            index.get_block(first.memory_id).revision_id,
            corrected.revision_id,
            mutation.record.revision_id,
        }

        catalog = load_memory_blocks(tmp_path, "ws")
        sync_block_descriptor_index(
            index,
            catalog=compile_semantic_catalog(catalog),
            record=mutation.record,
            workspace_id="ws",
            update_reason="consolidation",
        )
        idempotent = index.get_index_node(descriptor_id, workspace_id="ws")
        assert idempotent is not None
        assert idempotent.metadata["index_revision"] == 3
    finally:
        index.close()


def test_every_leaf_block_has_operational_contract(tmp_path) -> None:
    blocks = load_memory_blocks(tmp_path, "ws", strict=True)
    child_ids = {child for block in blocks for child in block.child_ids}
    parent_ids = {block.parent_id for block in blocks if block.parent_id}
    parent_ids.update(block.block_id for block in blocks if block.child_ids)
    leaves = [block for block in blocks if block.block_id not in parent_ids]

    assert leaves
    assert child_ids
    for block in leaves:
        assert block.producer
        assert block.consumers
        assert block.recall_path
        assert block.lifecycle_policy
        assert block.isolation_policy

    incomplete = MemoryBlock(
        block_id="incomplete_leaf",
        block_type="fact",
        temperature="warm",
        scope="workspace",
        authority="advisory",
        delivery_policy="on_demand",
        retention="durable",
    )
    validate_memory_block_tree([incomplete])
    with pytest.raises(ValueError, match="leaf block operational contract"):
        validate_leaf_block_contracts([incomplete])


def test_unknown_logical_block_field_is_rejected(tmp_path) -> None:
    blocks_path = tmp_path / ".UM-MEM" / "memory" / "blocks.yaml"
    blocks_path.parent.mkdir(parents=True)
    blocks_path.write_text(
        """
blocks:
  - block_id: invalid_field
    block_type: fact
    unsupported_field: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid block") as error:
        load_memory_blocks(tmp_path, "ws", strict=True)
    assert "unknown memory block fields" in str(error.value.__cause__)
