import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from filelock import FileLock

from um_mem import api
from um_mem.api import MemoryClient
from um_mem.contracts import MemoryBlockRecord, RecordBlockMembership
from um_mem.llm_budget import estimate_tokens
from um_mem.memory.authoring.models import MemoryRouteDecision, MemoryWriteProposal
from um_mem.memory.authoring.policy import route_by_taxonomy
from um_mem.memory.backends.contracts import CanonicalHeadReservation, MemoryIndexNode
from um_mem.memory.backends.hindsight import HindsightBackend
from um_mem.memory.backends.mempalace import MemPalaceBackend
from um_mem.memory.backends.mnemosyne import MnemosyneBackend
from um_mem.memory.backends.query_policy import record_allowed_for_query
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.blocks.catalog import compile_semantic_catalog
from um_mem.memory.blocks.models import MemoryBlock
from um_mem.memory.blocks.registry import load_memory_blocks


def _logical_block(block_id: str, **overrides: Any) -> MemoryBlock:
    values: dict[str, Any] = {
        "block_id": block_id,
        "block_type": "lesson",
        "temperature": "warm",
        "scope": "workspace",
        "authority": "advisory",
        "delivery_policy": "on_demand",
        "retention": "durable",
        "tags": (),
    }
    values.update(overrides)
    return MemoryBlock(**values)


def _record(block_id: str, content: str = "parser evidence", **overrides: Any) -> MemoryBlockRecord:
    values: dict[str, Any] = {
        "block_id": block_id,
        "revision_id": f"revision-{block_id}",
        "content": content,
        "title": block_id,
        "status": "candidate",
        "retention": "durable",
        "delivery_policy": "on_demand",
        "temperature": "warm",
        "authority": "advisory",
        "trust_level": "workspace_verified",
        "verification_status": "unverified",
        "scope": "workspace",
        "workspace_id": "ws",
        "block_membership": RecordBlockMembership(
            primary_block_id="project_ideas",
            catalog_revision_id="catalog:test",
        ),
        "metadata": {},
    }
    values.update(overrides)
    return MemoryBlockRecord(**values)


def _write(**overrides: Any) -> MemoryWriteProposal:
    values: dict[str, Any] = {
        "title": "Parser lesson",
        "content": "Parser evidence",
        "kind": "lesson",
        "tags": ("parser",),
        "workspace_id": "ws",
        "evidence_kind": "test_run",
        "evidence_refs": (
            {
                "ref_type": "test_run",
                "ref_id": "test-parser",
                "produced_by": "verifier",
                "producer_id": "verifier:hierarchical-routing",
            },
        ),
    }
    values.update(overrides)
    return MemoryWriteProposal(**values)


def test_registry_rejects_missing_hierarchy_references(tmp_path: Path) -> None:
    config = tmp_path / ".UM-MEM" / "memory" / "blocks.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """
blocks:
  - block_id: orphan
    block_type: lesson
    temperature: warm
    scope: workspace
    authority: advisory
    delivery_policy: on_demand
    retention: durable
    parent_id: absent-parent
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent-parent"):
        load_memory_blocks(tmp_path, "ws", strict=True)


def test_registry_rejects_hierarchy_cycles(tmp_path: Path) -> None:
    config = tmp_path / ".UM-MEM" / "memory" / "blocks.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """
blocks:
  - block_id: cycle-a
    block_type: memory_index
    temperature: warm
    scope: workspace
    authority: advisory
    delivery_policy: on_demand
    retention: durable
    parent_id: cycle-b
    child_ids: [cycle-b]
    index_only: true
  - block_id: cycle-b
    block_type: memory_index
    temperature: warm
    scope: workspace
    authority: advisory
    delivery_policy: on_demand
    retention: durable
    parent_id: cycle-a
    child_ids: [cycle-a]
    index_only: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cycle"):
        load_memory_blocks(tmp_path, "ws", strict=True)


def test_index_only_block_is_neither_write_target_nor_evidence() -> None:
    index = _logical_block(
        "parser-index",
        block_type="memory_index",
        index_only=True,
    )

    decision = route_by_taxonomy(
        _write(requested_block_id="parser-index"),
        [index],
    )
    record = _record(
        "bad-index-record",
        block_membership=RecordBlockMembership(
            primary_block_id="parser-index",
            catalog_revision_id="catalog:test",
        ),
        metadata={"index_only": True},
    )

    assert decision.accepted is False
    assert decision.reason == "index_only_block_cannot_store_records"
    assert record_allowed_for_query(record, {"workspace_id": "ws"}) is False


def test_non_index_parent_with_children_is_not_a_writable_leaf(
    tmp_path: Path,
) -> None:
    parent = _logical_block("custom-parent", child_ids=("custom-leaf",))
    leaf = _logical_block("custom-leaf", parent_id="custom-parent")

    decision = route_by_taxonomy(
        _write(requested_block_id="custom-parent"),
        [parent, leaf],
    )
    catalog = MemoryClient(tmp_path, workspace_id="ws", backend_instance=object()).block_map()

    assert decision.accepted is False
    assert decision.reason == "non_leaf_block_cannot_store_records"
    assert catalog[0]["protected"] is True
    assert catalog[0]["writable"] is False


def test_route_decision_is_semantic_not_physical() -> None:
    block = _logical_block("decisions_and_contracts")

    decision = route_by_taxonomy(_write(requested_block_id=block.block_id), [block])

    assert decision.accepted is True
    assert decision.block_id == block.block_id
    assert set(decision.__dataclass_fields__) == {
        "accepted",
        "reason",
        "block_id",
        "catalog_revision_id",
        "retention",
        "delivery_policy",
        "temperature",
        "authority",
        "scope",
        "trust_level",
        "verification_status",
        "verification_methods",
        "promotion_decision",
        "status",
        "tags",
        "policy_issues",
        "suggested_block_id",
        "suggested_action",
    }


@pytest.mark.parametrize(
    ("overrides", "accepted", "issue"),
    (
        ({"kind": "observation"}, False, "requested_block_id_required"),
        (
            {
                "requested_block_id": "workspace_episodes",
                "metadata": {"promotion_gate": "tribunal_accept"},
            },
            False,
            "reserved_metadata",
        ),
        (
            {"requested_block_id": "workspace_core_directives"},
            False,
            "typed_promotion_route_required",
        ),
    ),
)
def test_agent_authored_routes_remain_candidates(
    overrides: dict[str, object], accepted: bool, issue: str
) -> None:
    decision = route_by_taxonomy(
        _write(**overrides),
        [
            _logical_block("project_ideas"),
            _logical_block("workspace_episodes"),
            _logical_block("workspace_core_directives", authority="directive"),
        ],
    )

    assert decision.accepted is accepted
    assert decision.status == "candidate"
    assert decision.authority == "advisory"
    assert decision.verification_status == "unverified"
    assert decision.promotion_decision == "not_requested"
    if issue:
        assert any(issue in policy_issue for policy_issue in decision.policy_issues)


def test_requested_leaf_controls_route_without_kind_semantic_rerouting() -> None:
    selected = _logical_block(
        "decisions_and_contracts",
        temperature="hot",
        scope="project",
        retention="session",
        delivery_policy="working_set",
    )

    decision = route_by_taxonomy(
        _write(
            kind="user_preference",
            requested_block_id="decisions_and_contracts",
            evidence_refs=(),
        ),
        [
            selected,
            _logical_block("user_global_preferences", scope="user_global"),
            _logical_block("project_ideas"),
        ],
    )

    assert decision.accepted is True
    assert decision.block_id == "decisions_and_contracts"
    assert decision.reason == "requested_block_policy"
    assert decision.scope == "project"
    assert decision.retention == "session"
    assert decision.delivery_policy == "working_set"
    assert decision.temperature == "hot"
    assert decision.status == "candidate"
    assert decision.authority == "advisory"
    assert decision.trust_level == "agent_reported"
    assert decision.verification_status == "unverified"


def test_memory_client_routed_write_materializes_semantic_route_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _RecordingBackend()
    decision = MemoryRouteDecision(
        accepted=True,
        reason="test_route",
        block_id="verified_lessons",
        catalog_revision_id=compile_semantic_catalog(
            load_memory_blocks(tmp_path, workspace_id="ws", strict=True)
        ).revision_id,
        retention="durable",
        delivery_policy="on_demand",
        temperature="warm",
        scope="workspace",
        authority="advisory",
        trust_level="agent_reported",
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
        status="candidate",
        tags=("parser",),
    )
    monkeypatch.setattr(api, "route_memory_write", lambda *_args, **_kwargs: decision)
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)

    result = client.remember("parser routed evidence", requested_block_id="verified_lessons")

    assert result.saved is True
    assert len(backend.writes) == 1
    assert backend.writes[0].block_membership is not None
    assert backend.writes[0].block_membership.primary_block_id == "verified_lessons"
    assert backend.writes[0].metadata["route_decision_reason"] == "test_route"


class _RecordingBackend:
    def __init__(self, rows: list[MemoryBlockRecord] | None = None) -> None:
        self.rows = list(rows or [])
        self.writes: list[MemoryBlockRecord] = []
        self.recall_calls = 0
        self.commit_revision = 0

    def reserve_canonical_head(self, **kwargs: Any) -> CanonicalHeadReservation:
        self.commit_revision += 1
        return CanonicalHeadReservation(
            commit_revision=self.commit_revision,
            workspace_id=str(kwargs["workspace_id"]),
            canonical_key_sha256="test",
            predecessor_id=str(kwargs["predecessor_id"]),
            successor_id=str(kwargs["successor_id"]),
            successor_status=str(kwargs["successor_status"]),
            successor_authority=str(kwargs["successor_authority"]),
        )

    def complete_canonical_head(self, reservation: Any, mutation: Any) -> dict[str, Any]:
        del reservation, mutation
        return {"ok": True}

    def fail_canonical_head(self, reservation: Any, *, error: str) -> None:
        del reservation, error

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        self.writes.append(record)
        return {"ok": True, "block_id": record.block_id}

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        return next((row for row in self.rows if row.block_id == block_id), None)

    def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
        del query
        self.recall_calls += 1
        return list(self.rows)


def test_logical_block_ids_filter_records_strictly(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    try:
        index.retain_block(
            _record(
                "first",
                block_membership=RecordBlockMembership(
                    primary_block_id="other",
                    catalog_revision_id="catalog:test",
                ),
            )
        )
        index.retain_block(
            _record(
                "preferred",
                block_membership=RecordBlockMembership(
                    primary_block_id="wanted",
                    catalog_revision_id="catalog:test",
                ),
            )
        )

        strict = index.recall(
            {
                "query": "parser evidence",
                "workspace_id": "ws",
                "logical_block_ids": ["wanted"],
                "scope_chain": ["workspace:ws"],
                "include_candidates": True,
            }
        )
    finally:
        index.close()

    assert [row.block_id for row in strict] == ["preferred"]


@pytest.mark.parametrize(
    ("backend_class", "workspace_attribute"),
    [
        (MemPalaceBackend, "_workspace_id"),
        (HindsightBackend, "_workspace_id"),
        (MnemosyneBackend, "workspace_id"),
    ],
)
def test_real_backend_adapters_proxy_index_nodes(
    tmp_path: Path, backend_class: type[Any], workspace_attribute: str
) -> None:
    sidecar = BackendSQLiteIndex(
        tmp_path / f"{backend_class.__name__}.sqlite",
        backend_id="test",
        workspace_id="ws",
    )
    backend = backend_class.__new__(backend_class)
    backend._index = sidecar
    setattr(backend, workspace_attribute, "ws")
    if backend_class is MemPalaceBackend:
        backend._native = SimpleNamespace(release=lambda: None)
        backend._process_operation_lock = threading.RLock()
        backend._interprocess_operation_lock = FileLock(f"{backend._index.path}.operation.lock")
        backend._operation_lock_timeout_seconds = 30.0
        backend._operation_state = threading.local()
    node = MemoryIndexNode(
        node_id="parser-index",
        title="Parser index",
        node_type="topic",
        workspace_id="ws",
        tags=("parser",),
    )
    try:
        result = backend.retain_index_node(node)
        loaded = backend.get_index_node(node.node_id, workspace_id="ws")
        selected = backend.select_index_nodes(workspace_id="ws", tags=["parser"])
        searched = backend.search_index_nodes(workspace_id="ws", terms=("parser",))
        exported = backend.export_index_nodes()
    finally:
        sidecar.close()

    assert result["ok"] is True
    assert loaded == node
    assert selected == [node]
    assert searched == [node]
    assert exported == [node]


class _EmptyHindsightClient:
    def __init__(self) -> None:
        self.recall_calls = 0

    def recall(self, **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        self.recall_calls += 1
        return []


class _DistractingMnemosyne:
    def __init__(self) -> None:
        self.recall_calls = 0

    def recall(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        del query, top_k
        self.recall_calls += 1
        return [{"id": "distractor", "content": "irrelevant", "score": 1.0}]

    def get(self, native_id: str) -> dict[str, Any] | None:
        del native_id
        return None


@pytest.mark.parametrize("backend_kind", ["hindsight", "mnemosyne"])
def test_native_backends_exact_fetch_selected_ids_from_sidecar(
    tmp_path: Path, backend_kind: str
) -> None:
    if backend_kind == "hindsight":
        from um_mem.memory.hindsight.config import HindsightConfig

        backend: Any = HindsightBackend(
            repo_root=tmp_path,
            workspace_id="ws",
            config=HindsightConfig(enabled=True),
            client=_EmptyHindsightClient(),
        )
    else:
        backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
        backend.available = True
        backend._native_memory = _DistractingMnemosyne()
    backend._index.retain_block(_record("selected", "the exact retained parser detail"))
    try:
        rows = backend.recall(
            {
                "query": "unrelated weak query",
                "workspace_id": "ws",
                "record_ids": ["selected"],
                "scope_chain": ["workspace:ws"],
                "limit": 1,
                "include_candidates": True,
            }
        )
    finally:
        backend.close()

    assert [row.block_id for row in rows] == ["selected"]


def test_sqlite_truncates_oversized_top_record_instead_of_returning_zero(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    try:
        index.retain_block(_record("huge", "parser " + "A" * 5000))

        rows = index.recall(
            {
                "query": "parser",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "limit": 1,
                "max_tokens": 40,
                "include_candidates": True,
            }
        )
    finally:
        index.close()

    assert [row.block_id for row in rows] == ["huge"]
    assert rows[0].content
    assert rows[0].metadata["retrieval_truncated"] is True
    assert estimate_tokens(f"{rows[0].title}\n{rows[0].content}") <= 40
