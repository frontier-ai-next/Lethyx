import json
from dataclasses import replace
from pathlib import Path

import pytest

from um_mem.api import MemoryClient
from um_mem.contracts import MemoryBlockRecord, RecordBlockMembership
from um_mem.memory.backends.contracts import MemoryIndexNode
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.blocks.catalog import (
    block_descriptor_node_id,
    compile_semantic_catalog,
    sync_block_descriptor_index,
)
from um_mem.memory.blocks.models import MemoryBlock
from um_mem.memory.blocks.registry import load_workspace_topology, validate_memory_block_tree


def _block(
    block_id: str,
    *,
    parent_id: str = "",
    child_ids: tuple[str, ...] = (),
    index_only: bool = False,
    related_block_ids: tuple[str, ...] = (),
) -> MemoryBlock:
    return MemoryBlock(
        block_id=block_id,
        block_type="catalog_test",
        temperature="warm",
        scope="workspace",
        authority="advisory",
        delivery_policy="on_demand",
        retention="durable",
        parent_id=parent_id,
        child_ids=child_ids,
        index_only=index_only,
        graph_policy={"related_block_ids": related_block_ids},
    )


@pytest.mark.parametrize("axis", ["temperature", "scope", "delivery_policy", "retention"])
def test_block_policy_rejects_unknown_memory_axes(axis: str) -> None:
    with pytest.raises(ValueError, match=axis):
        replace(_block("leaf"), **{axis: "unsupported"})


def test_expand_block_navigates_deep_catalog_and_exposes_explicit_relations(
    tmp_path: Path,
) -> None:
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=object())
    root_id = client.block_map()[0]["block_id"]
    semantic = client.create_index_block(
        name="Semantic areas",
        synopsis="Agent-authored semantic areas.",
        parent_id=root_id,
        reason="create semantic branch",
    )
    resources = client.create_index_block(
        name="Resources",
        synopsis="Resources selected by the agent.",
        parent_id=semantic["block_id"],
        reason="create resources branch",
    )
    code = client.create_leaf_block(
        name="Code",
        synopsis="Code resources.",
        parent_id=resources["block_id"],
        reason="create code leaf",
    )
    decisions = client.create_leaf_block(
        name="Decisions",
        synopsis="Decisions and contracts.",
        parent_id=semantic["block_id"],
        reason="create decisions leaf",
    )
    client.add_block_edge(
        resources["block_id"],
        decisions["block_id"],
        edge_type="related_to",
        reason="explicit relation",
    )

    root_map = client.block_map()

    assert [item["block_id"] for item in root_map] == [root_id]
    assert root_map[0]["child_ids"] == [semantic["block_id"]]

    first_level = client.expand_block(semantic["block_id"])
    assert [item["block_id"] for item in first_level["children"]] == [
        resources["block_id"],
        decisions["block_id"],
    ]

    second_level = client.expand_block(resources["block_id"])
    assert second_level["path_ids"] == [
        root_id,
        semantic["block_id"],
        resources["block_id"],
    ]
    assert second_level["root"]["links"]["related_block_ids"] == [decisions["block_id"]]
    assert [item["block_id"] for item in second_level["children"]] == [code["block_id"]]

    leaf = client.expand_block(code["block_id"])
    assert leaf["path_ids"] == [
        root_id,
        semantic["block_id"],
        resources["block_id"],
        code["block_id"],
    ]
    assert leaf["root"]["expandable"] is False
    assert leaf["children"] == []
    assert client.expand_block(root_id)["root"]["protected"] is True


def test_catalog_rejects_relation_to_an_unknown_block() -> None:
    blocks = [
        _block(
            "semantic_memory_index",
            related_block_ids=("missing_block",),
        )
    ]

    with pytest.raises(ValueError, match="missing related block"):
        validate_memory_block_tree(blocks)


def test_expanded_catalog_uses_descriptor_summary_without_fake_importance_or_record_ids(
    tmp_path: Path,
) -> None:
    setup = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=object())
    root_id = setup.block_map()[0]["block_id"]
    branch = setup.create_index_block(
        name="Areas",
        synopsis="Agent-authored areas.",
        parent_id=root_id,
        reason="test",
    )
    leaf = setup.create_leaf_block(
        name="Decisions",
        synopsis="Decision records.",
        parent_id=branch["block_id"],
        reason="test",
    )
    blocks = [state.block for state in load_workspace_topology(tmp_path, "alpha")]
    descriptor = compile_semantic_catalog(blocks).descriptor(leaf["block_id"])
    node = MemoryIndexNode(
        node_id=block_descriptor_node_id("alpha", leaf["block_id"]),
        title="Logical block: decisions",
        node_type="block_descriptor",
        workspace_id="alpha",
        metadata={
            "descriptor": descriptor.to_mapping(),
            "synopsis": "Decision records. Coverage: 2 canonical revisions; latest: API contract.",
            "coverage": {
                "canonical_record_count": 2,
                "exact_dimensions": ["paths", "symbols"],
            },
            "latest_record_title": "API contract",
            "related_block_ids": ["must-not-leak-record-id"],
        },
        updated_at="2026-07-22T10:00:00+00:00",
    )

    class Backend:
        def get_index_node(self, node_id: str, *, workspace_id: str):
            assert workspace_id == "alpha"
            return node if node_id == node.node_id else None

    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=Backend())

    expanded = client.expand_block(branch["block_id"])
    child = expanded["children"][0]
    direct = client.expand_block(leaf["block_id"])["root"]

    assert child["block_id"] == leaf["block_id"]
    assert child["path"] == [root_id, branch["block_id"], leaf["block_id"]]
    assert child["description"]
    assert child["purpose"]
    assert child["synopsis"].endswith("Coverage: 2 canonical record(s).")
    assert "API contract" not in child["synopsis"]
    assert child["canonical_record_count"] == 2
    assert child["latest_title"] == ""
    assert child["exact_dimensions"] == ["paths", "symbols"]
    assert child["freshness"] == "2026-07-22T10:00:00+00:00"
    for field_name in (
        "canonical_record_count",
        "latest_title",
        "exact_dimensions",
        "freshness",
    ):
        assert direct[field_name] == child[field_name]
    assert "importance" not in child
    assert "must-not-leak-record-id" not in json.dumps(child)


def test_descriptor_live_state_counts_only_current_deliverable_canonical_records() -> None:
    blocks = [_block("decisions_and_contracts")]
    catalog = compile_semantic_catalog(blocks)

    def record(record_id: str, *, title: str, status: str = "candidate") -> MemoryBlockRecord:
        return MemoryBlockRecord(
            block_id=record_id,
            content=f"content for {record_id}",
            title=title,
            status=status,
            retention="archive" if status == "retracted" else "durable",
            delivery_policy="hidden" if status == "retracted" else "on_demand",
            workspace_id="alpha",
            recorded_at="2026-07-22T10:00:00+00:00",
            updated_at="2026-07-22T10:00:00+00:00",
            block_membership=RecordBlockMembership(
                primary_block_id="decisions_and_contracts",
                catalog_revision_id=catalog.revision_id,
            ),
            metadata={"paths": [f"src/{record_id}.py"]},
        )

    current = record("current", title="Current contract")
    superseded = record("old", title="Old contract", status="superseded")
    retracted = record("retracted", title="Retracted contract", status="retracted")

    class Backend:
        def __init__(self) -> None:
            self.node: MemoryIndexNode | None = None

        def get_index_node(self, _node_id: str, *, workspace_id: str):
            assert workspace_id == "alpha"
            return self.node

        def record_ids_for_logical_blocks(
            self,
            block_ids: tuple[str, ...],
            *,
            workspace_id: str,
        ) -> tuple[str, ...]:
            assert block_ids == ("decisions_and_contracts",)
            assert workspace_id == "alpha"
            return (current.block_id, superseded.block_id, retracted.block_id)

        def get_block(self, record_id: str, *, workspace_id: str = ""):
            assert workspace_id == "alpha"
            return {item.block_id: item for item in (current, superseded, retracted)}[record_id]

        def retain_index_node(self, node: MemoryIndexNode) -> dict[str, object]:
            self.node = node
            return {"ok": True}

    backend = Backend()
    sync_block_descriptor_index(
        backend,
        catalog=catalog,
        record=current,
        workspace_id="alpha",
    )

    assert backend.node is not None
    assert backend.node.metadata["coverage"]["canonical_record_count"] == 1
    assert backend.node.metadata["latest_record_title"] == "Current contract"
    assert backend.node.metadata["related_block_ids"] == ["current"]


def test_expand_block_falls_back_to_canonical_records_when_descriptor_is_missing(
    tmp_path: Path,
) -> None:
    setup = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=object())
    leaf = setup.create_leaf_block(
        name="Decisions",
        synopsis="Decision records.",
        parent_id=setup.block_map()[0]["block_id"],
        reason="test",
    )
    blocks = [state.block for state in load_workspace_topology(tmp_path, "alpha")]
    catalog = compile_semantic_catalog(blocks)
    record = MemoryBlockRecord(
        block_id="decision-current",
        content="Use opaque value N7Q.",
        title="Current deployment decision",
        status="candidate",
        workspace_id="alpha",
        recorded_at="2026-07-22T10:00:00+00:00",
        updated_at="2026-07-22T10:00:00+00:00",
        block_membership=RecordBlockMembership(
            primary_block_id=leaf["block_id"],
            catalog_revision_id=catalog.revision_id,
        ),
        metadata={"config_keys": ["deployment.channel"]},
    )

    class Backend:
        def get_index_node(self, _node_id: str, *, workspace_id: str):
            assert workspace_id == "alpha"

        def record_ids_for_logical_blocks(
            self,
            block_ids: tuple[str, ...],
            *,
            workspace_id: str,
        ) -> tuple[str, ...]:
            assert block_ids == (leaf["block_id"],)
            assert workspace_id == "alpha"
            return (record.block_id,)

        def get_block(self, record_id: str, *, workspace_id: str = ""):
            assert record_id == record.block_id
            assert workspace_id == "alpha"
            return record

    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=Backend())

    expanded = client.expand_block(leaf["block_id"])["root"]

    assert expanded["canonical_record_count"] == 1
    assert expanded["latest_title"] == ""
    assert expanded["freshness"] == "2026-07-22T10:00:00+00:00"
    assert expanded["exact_dimensions"] == ["config_keys"]
    assert "Coverage: 1 canonical record" in expanded["synopsis"]


def test_directive_leaf_rejection_returns_actionable_typed_route(
    tmp_path: Path,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        rejected = client.remember(
            "Use the verified deployment contract.",
            kind="decision",
            requested_block_id="workspace_core_directives",
        )
        suggested = rejected.raw["suggested_route"]
        accepted = client.remember(
            "Use the verified deployment contract.",
            kind="decision",
            requested_block_id=suggested["requested_block_id"],
        )
    finally:
        backend.close()

    assert rejected.saved is False
    assert suggested == {
        "action": "retry_remember",
        "requested_block_id": "decisions_and_contracts",
    }
    assert rejected.raw["route_decision"]["suggested_block_id"] == ("decisions_and_contracts")
    assert accepted.saved is True


def test_expand_leaf_exposes_content_free_operational_routing_metadata(
    tmp_path: Path,
) -> None:
    class Backend:
        block_id = ""

        def health(self) -> dict[str, object]:
            partition = {
                "backend_id": "mempalace",
                "workspace_id": "alpha",
                "primary_block_id": self.block_id,
                "catalog_revision_id": "catalog:test",
                "partition_id": "partition:test",
                "isolation_class": "physical",
                "partition_mode": "native_collection_family",
            }
            return {
                "native_partitions": [partition],
                "block_metrics": [
                    {
                        **partition,
                        "record_count": 3,
                        "recall_count": 4,
                        "hit_count": 2,
                        "latency_p50_ms": 8.0,
                        "latency_p95_ms": 13.0,
                        "health": "ready",
                    }
                ],
            }

    backend = Backend()
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        backend_instance=backend,
    )
    leaf = client.create_leaf_block(
        name="Decisions",
        synopsis="Decision records.",
        parent_id=client.block_map()[0]["block_id"],
        reason="test",
    )
    backend.block_id = leaf["block_id"]

    root_map = client.block_map()
    expanded = client.expand_block(leaf["block_id"])["root"]

    assert "native_partitions" not in root_map[0]
    assert expanded["retention"] == "durable"
    assert expanded["lifecycle"]["initial_status"] == "candidate"
    assert expanded["retrieval_mode"] == "hybrid_recall"
    assert expanded["writable"] is True
    assert expanded["writable_reason"] == "agent_writable_leaf"
    assert "remember" in expanded["allowed_operations"]
    assert expanded["isolation"]["class"] == "physical"
    assert expanded["isolation"]["partition_mode"] == "native_collection_family"
    assert expanded["metrics"]["record_count"] == 3
    assert expanded["metrics"]["latency_p95_ms"] == 13.0


def test_catalog_snapshot_returns_all_blocks_and_edges_in_one_view(tmp_path: Path) -> None:
    client = MemoryClient(tmp_path, workspace_id="alpha")
    root_id = client.block_map()[0]["block_id"]
    first = client.create_leaf_block(
        name="First",
        synopsis="First leaf.",
        parent_id=root_id,
        reason="test",
    )
    second = client.create_leaf_block(
        name="Second",
        synopsis="Second leaf.",
        parent_id=root_id,
        reason="test",
    )
    edge = client.add_block_edge(
        first["block_id"],
        second["block_id"],
        edge_type="related_to",
        reason="test",
    )

    snapshot = client.catalog_snapshot()

    block_ids = {item["block_id"] for item in snapshot["blocks"]}
    assert {root_id, first["block_id"], second["block_id"]}.issubset(block_ids)
    assert snapshot["edges"] == [edge]
    assert snapshot["catalog_revision"] == client.catalog_revision()
