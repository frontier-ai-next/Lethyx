import sqlite3
from hashlib import sha256
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import lethyx.memory.backends.sqlite_index as sqlite_index_module

from lethyx.api import MemoryClient
from lethyx.contracts import EvidenceRef
from lethyx.memory.backends.contracts import (
    ALLOWED_EDGE_TYPES,
    EdgeType,
    MemoryEdgeRecord,
    MemoryIndexNode,
)
from lethyx.memory.backends.sqlite_index import SQLITE_INDEX_SCHEMA_VERSION, BackendSQLiteIndex
from lethyx.memory.blocks.catalog import (
    DynamicLabel,
    RefinementCase,
    admit_dynamic_label,
    block_descriptor_node_id,
    compile_semantic_catalog,
    descriptor_from_index_node,
)
from lethyx.memory.blocks.registry import load_memory_blocks


def test_new_sqlite_index_installs_busy_timeout_before_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []

    class RecordingConnection:
        def execute(self, statement: str, *_args: object) -> "RecordingConnection":
            statements.append(statement)
            return self

        def executescript(self, statement: str) -> None:
            statements.append(statement)

        def commit(self) -> None:
            return None

        def close(self) -> None:
            return None

    def connect(*_args: object, **_kwargs: object) -> RecordingConnection:
        return RecordingConnection()

    monkeypatch.setattr(sqlite_index_module.sqlite3, "connect", connect)

    index = BackendSQLiteIndex(
        tmp_path / "new-index.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    index.close()

    assert statements[:3] == [
        "PRAGMA busy_timeout=10000",
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=FULL",
    ]


def test_memory_index_node_contract_and_sqlite_roundtrip(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        node = MemoryIndexNode(
            node_id="topic-python",
            title="Python project facts",
            node_type="topic",
            workspace_id="alpha",
            tags=("python", "project"),
            metadata={"path": "src"},
        )

        result = index.retain_index_node(node)
        loaded = index.get_index_node("topic-python", workspace_id="alpha")

        assert result == {"ok": True, "node_id": "topic-python", "backend": "test"}
        assert loaded == node
        assert [
            row.node_id for row in index.select_index_nodes(workspace_id="alpha", tags=["python"])
        ] == ["topic-python"]
    finally:
        index.close()


def test_index_tag_filter_applies_before_recent_node_limit(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        index.retain_index_node(
            MemoryIndexNode(
                node_id="old-parser",
                title="Parser evidence",
                node_type="topic",
                workspace_id="alpha",
                tags=("parser",),
                updated_at="2025-01-01T00:00:00+00:00",
            )
        )
        for offset in range(130):
            index.retain_index_node(
                MemoryIndexNode(
                    node_id=f"new-decoy-{offset:03d}",
                    title=f"Unrelated node {offset}",
                    node_type="topic",
                    workspace_id="alpha",
                    tags=("unrelated",),
                    updated_at=f"2026-01-01T00:{offset // 60:02d}:{offset % 60:02d}+00:00",
                )
            )

        selected = index.select_index_nodes(
            workspace_id="alpha",
            tags=["parser"],
            limit=1,
        )

        assert [node.node_id for node in selected] == ["old-parser"]
    finally:
        index.close()


def test_sqlite_schema_is_current_and_rejects_legacy_stores(tmp_path) -> None:
    current_path = tmp_path / "current.sqlite"
    index = BackendSQLiteIndex(current_path, backend_id="test", workspace_id="alpha")
    index.close()
    reopened = BackendSQLiteIndex(current_path, backend_id="test", workspace_id="alpha")
    reopened.close()

    connection = sqlite3.connect(current_path)
    try:
        assert connection.execute("SELECT key, value FROM schema_metadata").fetchall() == [
            ("schema_version", SQLITE_INDEX_SCHEMA_VERSION)
        ]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(blocks)")}
        assert {"status", "authority", "stage_id"}.issubset(columns)
        assert {"lifecycle", "surface"}.isdisjoint(columns)
    finally:
        connection.close()

    for name, version in (("unversioned", None), ("v1", 1), ("v13", 13), ("unknown", 3)):
        legacy_path = tmp_path / f"{name}.sqlite"
        connection = sqlite3.connect(legacy_path)
        try:
            if version is not None:
                connection.execute(
                    "CREATE TABLE schema_metadata(key TEXT PRIMARY KEY, value INTEGER NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO schema_metadata(key, value) VALUES(?, ?)",
                    ("schema_version", version),
                )
            connection.commit()
        finally:
            connection.close()
        original_sha256 = sha256(legacy_path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="fresh store"):
            BackendSQLiteIndex(legacy_path, backend_id="test", workspace_id="alpha")
        assert sha256(legacy_path.read_bytes()).hexdigest() == original_sha256


def test_index_node_edges_are_first_class_edge_types(tmp_path) -> None:
    assert EdgeType.CONTAINS in ALLOWED_EDGE_TYPES
    assert EdgeType.SUPPORTS in ALLOWED_EDGE_TYPES
    assert EdgeType.MENTIONS_SYMBOL in ALLOWED_EDGE_TYPES

    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        index.retain_index_node(
            MemoryIndexNode(
                node_id="source-readme",
                title="README",
                node_type="source",
                workspace_id="alpha",
            )
        )
        index.retain_index_node(
            MemoryIndexNode(
                node_id="mem-1",
                title="Memory target",
                node_type="memory",
                workspace_id="alpha",
            )
        )
        result = index.add_edge(
            MemoryEdgeRecord(
                src_id="source-readme",
                dst_id="mem-1",
                edge_type=EdgeType.CONTAINS,
                workspace_id="alpha",
            )
        )

        assert result["ok"] is True
        assert [edge.edge_type for edge in index.edges_for("source-readme")] == [EdgeType.CONTAINS]
    finally:
        index.close()


def test_semantic_catalog_compiles_one_descriptor_with_nested_guidance(tmp_path) -> None:
    blocks = load_memory_blocks(tmp_path, workspace_id="alpha")

    catalog = compile_semantic_catalog(blocks)
    rebuilt = compile_semantic_catalog(list(reversed(blocks)))
    descriptor = catalog.descriptor("stable_project_facts")

    assert catalog.revision_id == rebuilt.revision_id
    assert descriptor.semantic_memberships == ("semantic", "project")
    assert descriptor.guidance.block_id == descriptor.block_id
    assert descriptor.guidance.synopsis
    assert not hasattr(descriptor, "lexical_terms")
    assert not hasattr(descriptor, "dense_routing_text")
    assert not hasattr(catalog, "lexical_view")
    assert not hasattr(catalog, "dense_view")


def test_memory_block_map_exposes_roots_and_expands_only_the_selected_branch(tmp_path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        [root] = client.block_map()
        created = client.create_leaf_block(
            name="Stable facts",
            synopsis="Durable facts selected by the agent.",
            parent_id=root["block_id"],
            reason="exercise free-topology navigation",
        )
        leaf_id = created["block_id"]
        saved = client.remember(
            "Opaque private record body that must never appear in the catalog map.",
            kind="project_fact",
            requested_block_id=leaf_id,
        )

        root_map = client.block_map()
        expanded = client.expand_block(root["block_id"])
        stable = next(item for item in expanded["children"] if item["block_id"] == leaf_id)

        assert saved.saved is True
        assert [item["block_id"] for item in root_map] == [root["block_id"]]
        assert root_map[0]["protected"] is True
        assert root_map[0]["child_ids"] == [leaf_id]
        assert root_map[0]["description"]
        assert root_map[0]["purpose"]
        assert stable["record_count"] == 1
        assert stable["freshness"]
        assert stable["writable"] is True
        assert stable["parent_id"] == root["block_id"]
        assert "Opaque private record body" not in str(root_map)
        assert "Opaque private record body" not in str(expanded)
        assert saved.memory_id not in str(root_map)
        assert saved.memory_id not in str(expanded)
    finally:
        backend.close()


def test_normal_write_automatically_persists_and_updates_block_descriptor_node(
    tmp_path,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    evidence = [
        {
            "ref_type": "artifact",
            "ref_id": "pyproject.toml",
            "produced_by": "harness",
            "producer_id": "harness:catalog-index",
        }
    ]
    try:
        first = client.remember(
            "Python 3.12 is required.",
            kind="project_fact",
            requested_block_id="stable_project_facts",
            evidence_refs=evidence,
        )
        second = client.remember(
            "Ruff is the linter.",
            kind="project_fact",
            requested_block_id="stable_project_facts",
            evidence_refs=evidence,
        )
        node = backend.get_index_node(
            block_descriptor_node_id("alpha", "stable_project_facts"),
            workspace_id="alpha",
        )

        assert first.saved and second.saved
        assert node is not None
        assert node.node_type == "block_descriptor"
        assert node.metadata["block_id"] == "stable_project_facts"
        assert node.metadata["descriptor"]["block_id"] == "stable_project_facts"
        assert node.metadata["catalog_revision_id"].startswith("catalog:")
        persisted_descriptor = descriptor_from_index_node(node)
        assert persisted_descriptor.block_id == "stable_project_facts"
        assert (
            persisted_descriptor.descriptor_revision_id
            == node.metadata["descriptor"]["descriptor_revision_id"]
        )
        assert node.metadata["related_block_ids"] == sorted([first.memory_id, second.memory_id])
    finally:
        backend.close()


def test_descriptor_write_and_catalog_snapshot_do_not_decode_whole_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="alpha")
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        leaf_id = client.create_leaf_block(
            name="Facts",
            synopsis="Agent-selected facts",
            parent_id=client.block_map()[0]["block_id"],
            reason="bounded descriptor test",
        )["block_id"]
        for ordinal in range(70):
            assert client.remember(f"Fact {ordinal}", requested_block_id=leaf_id).saved
        original = backend.get_block
        reads = 0

        def counted(record_id: str, *, workspace_id: str = ""):
            nonlocal reads
            reads += 1
            return original(record_id, workspace_id=workspace_id)

        monkeypatch.setattr(backend, "get_block", counted)
        assert client.remember("Fact 70", requested_block_id=leaf_id).saved
        assert reads < 10
        before_catalog = reads
        views = client.catalog_snapshot()["blocks"]
        assert reads - before_catalog < 10
        leaf = next(view for view in views if view["block_id"] == leaf_id)
        assert leaf["canonical_record_count"] == 71
        node = backend.get_index_node(
            block_descriptor_node_id("alpha", leaf_id), workspace_id="alpha"
        )
        assert node is not None
        assert node.metadata["coverage"]["canonical_record_count"] == 71
        assert node.metadata["coverage"]["record_ids_truncated"] is True
        assert len(node.metadata["related_block_ids"]) == 64
    finally:
        backend.close()


def test_dynamic_labels_enforce_immediate_local_scope_workspace_ttl_and_global_gate(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 14, 19, 45, tzinfo=timezone.utc)
    task_label = DynamicLabel(
        label_id="label:parser-repair",
        label="parser repair",
        semantic_memberships=("semantic", "task-local"),
        scope="task",
        workspace_id="alpha",
        task_id="task-1",
        created_at=now.isoformat(),
    )

    admitted = admit_dynamic_label(task_label, existing=())

    assert admitted.status == "active"
    assert admitted.authority == "advisory"
    catalog = compile_semantic_catalog(load_memory_blocks(tmp_path, workspace_id="alpha"))
    extended = catalog.with_dynamic_label(task_label)
    assert extended.revision_id != catalog.revision_id
    assert extended.dynamic_labels == (task_label,)

    workspace_label = DynamicLabel(
        label_id="label:experimental-parser",
        label="experimental parser",
        semantic_memberships=("semantic",),
        scope="workspace",
        workspace_id="alpha",
        status="experimental",
        created_at=now.isoformat(),
        expires_at=(now + timedelta(hours=1)).isoformat(),
    )
    assert admit_dynamic_label(workspace_label, existing=(), workspace_quota=1) == workspace_label
    with pytest.raises(ValueError, match="workspace dynamic-label quota"):
        admit_dynamic_label(workspace_label, existing=(workspace_label,), workspace_quota=1)

    with pytest.raises(ValueError, match="shadow or canary"):
        DynamicLabel(
            label_id="label:shared-active",
            label="shared active",
            semantic_memberships=("semantic",),
            scope="shared",
            workspace_id="shared",
            status="active",
            created_at=now.isoformat(),
        )

    canary = DynamicLabel(
        label_id="label:global-canary",
        label="global canary",
        semantic_memberships=("reflection",),
        scope="global",
        workspace_id="global",
        status="canary",
        created_at=now.isoformat(),
    )
    assert admit_dynamic_label(canary, existing=()).status == "canary"


@pytest.mark.parametrize(
    ("case_type", "failure_stage"),
    [
        ("capture_miss", "storage"),
        ("selection_miss", "selection"),
        ("overretrieval", "retrieval"),
        ("reader_misuse", "comprehension"),
        ("novel_concept", "selection"),
        ("label_disagreement", "selection"),
        ("stale_guidance", "action"),
        ("weak_model_failure", "verifier"),
    ],
)
def test_refinement_case_preserves_closed_failure_type_and_typed_evidence(
    case_type: str,
    failure_stage: str,
) -> None:
    evidence = EvidenceRef(
        ref_type="test_run",
        ref_id=f"refinement:{case_type}",
        produced_by="verifier",
        producer_id=f"verifier:{case_type}",
    )
    case = RefinementCase(
        case_id=f"refinement:{case_type}",
        case_type=case_type,
        failure_stage=failure_stage,
        query="Where should this memory route?",
        decision_id="decision-refinement",
        task_signature="routing:memory",
        action_trace_id="action-refinement",
        selected_descriptor_ids=("stable_project_facts",),
        expected_descriptor_ids=("mutable_project_state",),
        evidence_refs=(evidence,),
        reason="typed routing failure",
    )

    assert case.case_type == case_type
    assert case.status == "candidate"
    assert case.evidence_refs == (evidence,)

    with pytest.raises(ValueError, match="unsupported refinement case type"):
        RefinementCase(
            case_id="refinement:unknown",
            case_type="other",
            failure_stage="selection",
            query="query",
            decision_id="decision-refinement",
            task_signature="routing:memory",
            action_trace_id="action-refinement",
            evidence_refs=(evidence,),
            reason="unknown type",
        )
