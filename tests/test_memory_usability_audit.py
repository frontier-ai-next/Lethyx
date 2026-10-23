"""Regression coverage for the memory usability audit fixes."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from um_mem.api import MemoryClient
from um_mem.contracts.memory import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from um_mem.memory import mcp_server
from um_mem.memory.backends.mnemosyne import MnemosyneBackend
from um_mem.memory.mcp_compact import compact_response


def _native_record(block_id: str) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        revision_id=f"revision:{block_id}",
        content=f"content:{block_id}",
        title="Native record",
        block_membership=RecordBlockMembership(
            primary_block_id="workspace_episodes",
            catalog_revision_id="test-catalog",
        ),
        kind="observation",
        status="accepted",
        retention="durable",
        delivery_policy="on_demand",
        temperature="warm",
        authority="advisory",
        trust_level="workspace_verified",
        scope="workspace",
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
        workspace_id="ws",
        run_id="run",
        stage_id="stage",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id=block_id,
                produced_by="verifier",
                producer_id="verifier:audit",
            ),
        ),
    )


class _MemoriaNativeFake:
    """Native surface that mixes real rows with MEMORIA supplement rows."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def remember(
        self, content: str, *, source: str, metadata: dict[str, Any], **_kwargs: Any
    ) -> str:
        native_id = f"native-{len(self.rows) + 1}"
        self.rows[native_id] = {
            "id": native_id,
            "content": content,
            "source": source,
            "metadata": dict(metadata),
        }
        return native_id

    def get_all_memories(self) -> list[dict[str, Any]]:
        return []

    def get(self, native_id: str) -> dict[str, Any] | None:
        row = self.rows.get(native_id)
        return dict(row) if row is not None else None

    def recall(self, _query: str, *, top_k: int, source: str = "") -> list[dict[str, Any]]:
        memoria = [
            {
                "id": "memoria_source_native-1",
                "content": "duplicate view of a stored row",
                "source": source,
                "tier": "memoria_source",
                "source_memory_id": "native-1",
                "score": 0.55,
            },
            {
                "id": "memoria_source",
                "content": "[MEMORIA source] extracted context",
                "source": "memoria_source",
                "tier": "memoria",
                "score": 0.6,
            },
        ]
        real = [
            {
                "id": row["id"],
                "content": row["content"][:500],
                "source": row["source"],
                "score": 0.9 - index / 10,
            }
            for index, row in enumerate(list(self.rows.values())[:top_k])
        ]
        return memoria + real


def test_mnemosyne_recall_ignores_memoria_supplement_rows(tmp_path) -> None:
    """Mnemosyne adds MEMORIA rows to its own recall; they are not records."""

    native = _MemoriaNativeFake()
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    backend._native_memory = native
    record = _native_record("mnemo-memoria")
    backend.retain_block(record)
    recalled = backend.recall(
        {
            "query": "content",
            "workspace_id": "ws",
            "scope_chain": ["workspace:ws"],
            "source_horizon": "durable",
            "logical_block_ids": ["workspace_episodes"],
            "limit": 5,
        }
    )
    assert {row.block_id for row in recalled} == {record.block_id}


_RECORDS = [
    {
        "block_id": "mem_config",
        "revision_id": "rev_config",
        "title": "OpenCode config location",
        "content": "OpenCode config is C:/Users/example/.config/opencode/opencode.json",
        "tags": ["opencode", "config"],
        "kind": "fact",
        "recorded_at": "2026-10-20T00:00:00Z",
        "status": "candidate",
        "sensitivity": "internal",
        "content_mode": "full",
        "operational_risk": "low",
        "block_membership": {"primary_block_id": "leaf_config"},
    },
    {
        "block_id": "mem_python",
        "revision_id": "rev_python",
        "title": "Python environment",
        "content": "The Python environment uses uv and pinned dependencies.",
        "tags": ["python", "environment"],
        "kind": "decision",
        "recorded_at": "2026-10-19T00:00:00Z",
        "status": "candidate",
        "sensitivity": "internal",
        "content_mode": "full",
        "operational_risk": "low",
        "block_membership": {"primary_block_id": "leaf_python"},
    },
]


def _fake_recent(*_args: object, **_kwargs: object) -> dict[str, object]:
    return {"ok": True, "records": _RECORDS, "hits": [{"source": r["block_id"]} for r in _RECORDS]}


def test_recall_fallback_recovers_from_native_backend_error(monkeypatch) -> None:
    monkeypatch.setattr(mcp_server, "_service_call", _fake_recent)
    result = mcp_server._apply_recall_fallback(
        object(),
        object(),
        "recall",
        {
            "query": "where is the opencode config",
            "limit": 4,
            "include_candidates": True,
            "source_horizon": "durable",
        },
        {
            "ok": False,
            "error": "ValueError: mnemosyne native recall detail is missing",
            "hits": [],
        },
    )

    assert result["ok"] is True
    assert result["degraded"] is True
    assert result["backend_error"].startswith("ValueError:")
    assert result["fallback"] == "unicode_ngram"
    assert result["relaxed_tried"] is True
    assert result["hits"][0]["metadata"]["record_id"] == "mem_config"


def test_build_context_falls_back_when_the_backend_returns_no_items(monkeypatch) -> None:
    monkeypatch.setattr(mcp_server, "_service_call", _fake_recent)
    result = mcp_server._apply_recall_fallback(
        object(),
        object(),
        "build_context",
        {
            "query": "pinned decisions about the python environment",
            "limit": 4,
            "include_candidates": True,
            "source_horizon": "durable",
        },
        {"ok": True, "items": []},
    )

    assert result["ok"] is True
    assert result["fallback"] == "unicode_ngram"
    assert result["items"][0]["metadata"]["record_id"] == "mem_python"


def test_build_context_fallback_can_render_packed_context(monkeypatch) -> None:
    monkeypatch.setattr(mcp_server, "_service_call", _fake_recent)
    result = mcp_server._apply_recall_fallback(
        object(),
        object(),
        "build_context",
        {
            "query": "pinned decisions about the python environment",
            "limit": 4,
            "include_candidates": True,
            "source_horizon": "durable",
            "render_context": True,
            "max_tokens": 2048,
        },
        {"ok": True, "items": []},
    )

    rendered = result["rendered_context"]
    assert "UM-MEM recalled context" in rendered
    assert "The Python environment uses uv" in rendered


def test_fallback_respects_explicit_tags_and_skips_nondurable_horizons(monkeypatch) -> None:
    monkeypatch.setattr(mcp_server, "_service_call", _fake_recent)
    tagged = mcp_server._apply_recall_fallback(
        object(),
        object(),
        "recall",
        {
            "query": "opencode config",
            "limit": 4,
            "include_candidates": True,
            "source_horizon": "durable",
            "tags": ["does-not-exist"],
        },
        {"ok": True, "hits": []},
    )
    assert tagged["hits"] == []
    assert tagged["relaxed_tried"] is True

    working = mcp_server._apply_recall_fallback(
        object(),
        object(),
        "recall",
        {
            "query": "opencode config",
            "limit": 4,
            "include_candidates": True,
            "source_horizon": "working",
        },
        {"ok": True, "hits": []},
    )
    assert "relaxed_tried" not in working


def test_recall_honors_tags_even_when_the_native_backend_ignores_them(
    tmp_path: Path,
) -> None:
    from um_mem.api import _hit_allowed
    from um_mem.memory.backends.contracts import block_record_to_memory_hit
    from um_mem.memory.recall_plan import RecallPlan

    client = MemoryClient(tmp_path, workspace_id="tags-audit", backend="mnemosyne")
    root = client.block_map()[0]
    leaf = client.create_leaf_block(
        name="Tagged facts",
        synopsis="Tag filtering regression",
        parent_id=root["block_id"],
        reason="test",
    )
    saved = client.remember(
        "Alpha config fact",
        title="Alpha config",
        tags=["alpha", "config"],
        requested_block_id=leaf["block_id"],
    )
    assert saved.saved
    record = client.read_exact_record(record_id=saved.memory_id)
    hit = block_record_to_memory_hit(record)
    base_plan = {
        "query": "config",
        "workspace_id": "tags-audit",
        "profile_id": client.profile_id,
        "scope_chain": ("workspace:tags-audit",),
        "include_candidates": True,
        "search_all": True,
        "source_horizon": "durable",
    }

    assert _hit_allowed(hit, RecallPlan(tags=("alpha",), **base_plan)) is True
    assert _hit_allowed(hit, RecallPlan(tags=("beta",), **base_plan)) is False
    assert _hit_allowed(hit, RecallPlan(tags=("missing",), **base_plan)) is False


def test_compact_failed_remember_keeps_route_repair_instructions(monkeypatch) -> None:
    monkeypatch.setenv("UM_MEM_MCP_RESPONSE_MODE", "compact")
    result = compact_response(
        "remember",
        {
            "ok": False,
            "saved": False,
            "workspace_id": "opencode",
            "backend": "mnemosyne",
            "reason": "index_only_block_cannot_store_records",
            "requested_block_id": "memory-root",
            "route_rejected": True,
            "policy_issues": ["index-only blocks cannot hold records"],
            "route_decision": {
                "accepted": False,
                "block_id": "memory-root",
                "status": "index_only",
            },
            "suggested_route": {
                "action": "retry_remember",
                "requested_block_id": "decisions_and_contracts",
            },
        },
    )

    assert result["reason"] == "index_only_block_cannot_store_records"
    assert result["suggested_route"]["requested_block_id"] == "decisions_and_contracts"
    assert result["route_decision"]["accepted"] is False
    assert result["policy_issues"] == ["index-only blocks cannot hold records"]


def test_compact_recall_keeps_the_score_scale(monkeypatch) -> None:
    monkeypatch.setenv("UM_MEM_MCP_RESPONSE_MODE", "compact")
    result = compact_response(
        "recall",
        {
            "ok": True,
            "hits": [
                {
                    "record_id": "mem_lexical",
                    "title": "Lexical",
                    "text": "lexical hit",
                    "score": 1,
                    "score_mode": "relative_lexical_ngram",
                },
                {
                    "record_id": "mem_native",
                    "title": "Native",
                    "text": "native hit",
                    "score": 0.71,
                    "metadata": {"native_score_mode": "cosine_similarity"},
                },
            ],
        },
    )

    assert [item["score_mode"] for item in result["items"]] == [
        "relative_lexical_ngram",
        "cosine_similarity",
    ]


def test_packed_compact_recall_preserves_kind_attribution_and_time(monkeypatch) -> None:
    from um_mem.memory.mcp_facade import _packed_payload_from_records

    monkeypatch.setenv("UM_MEM_MCP_RESPONSE_MODE", "compact")
    record = dataclasses.replace(
        _native_record("attributed"),
        kind="user_preference",
        metadata={"source_speakers": ["user"], "source_dates": ["2026-10-22"]},
        valid_from="2026-10-22T00:00:00+00:00",
        valid_to="2026-10-23T00:00:00+00:00",
    )
    packed = _packed_payload_from_records([record], max_tokens=1000, query="preference")
    assert packed.items
    result = compact_response("recall", {"ok": True, "hits": list(packed.items)})
    item = result["items"][0]
    assert item["record_id"] == record.block_id
    assert item["kind"] == "user_preference"
    assert item["source_speakers"] == ["user"]
    assert item["source_dates"] == ["2026-10-22"]
    assert item["valid_to"] == record.valid_to
    assert item["verification_status"] == packed.items[0]["metadata"]["verification_status"]
    assert item["authority"] == "advisory"


def test_compact_active_resolution_preserves_actionable_status_and_budget(
    tmp_path, monkeypatch
) -> None:
    from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex

    monkeypatch.setenv("UM_MEM_MCP_RESPONSE_MODE", "compact")
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(
        tmp_path, workspace_id="ws", session_id="session", task_id="task", backend_instance=index
    )
    try:
        absent = compact_response(
            "resolve_active_mask",
            {"ok": True, "lease": dataclasses.asdict(client.resolve_active_mask())},
        )["lease"]
        assert absent["trace"]["resolution_status"] == "not_found"
        leaf = client.create_leaf_block(
            name="Focus",
            synopsis="Current constraints",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        mask = client.set_active_mask(
            selected_block_ids=(leaf["block_id"],), token_budget=200, activation_reason="test"
        )
        for expected in (mask.revision, mask.revision + 1):
            lease = client.resolve_active_mask(expected_revision=expected, max_tokens=80)
            compact = compact_response(
                "resolve_active_mask", {"ok": True, "lease": dataclasses.asdict(lease)}
            )["lease"]
            assert compact["trace"]["resolution_status"] == lease.trace.resolution_status
            assert compact["trace"]["mask_id"] == mask.mask_id
            assert compact["trace"]["mask_revision"] == lease.trace.mask_revision
            assert (
                compact["trace"]["scope_validation_result"] == lease.trace.scope_validation_result
            )
            assert compact["token_budget"] == lease.token_budget
            assert compact["estimated_tokens"] == lease.estimated_tokens
            assert "visibility" not in compact["trace"]
        assert compact_response("get_active_mask", {"ok": True, "mask": None})["status"] == "absent"
    finally:
        index.close()


def test_local_service_route_receipt_keeps_route_decision() -> None:
    from um_mem.memory.mcp_call import _compact_remember_receipt

    receipt = _compact_remember_receipt(
        {
            "ok": False,
            "saved": False,
            "route_rejected": True,
            "reason": "index_only_block_cannot_store_records",
            "backend": "mnemosyne",
            "workspace_id": "opencode",
            "route_decision": {
                "accepted": False,
                "reason": "index_only_block_cannot_store_records",
                "block_id": "memory-root",
                "policy_issues": ["index_only_not_record_storage"],
                "suggested_block_id": "decisions_and_contracts",
                "suggested_action": "retry_remember",
            },
            "suggested_route": {
                "action": "retry_remember",
                "requested_block_id": "decisions_and_contracts",
            },
        }
    )

    assert receipt["route_decision"]["policy_issues"] == ["index_only_not_record_storage"]
    assert receipt["route_decision"]["suggested_block_id"] == "decisions_and_contracts"
    assert receipt["suggested_route"]["requested_block_id"] == "decisions_and_contracts"


def test_compact_exact_read_derives_record_and_logical_block_ids(monkeypatch) -> None:
    monkeypatch.setenv("UM_MEM_MCP_RESPONSE_MODE", "compact")
    result = compact_response(
        "read_exact_record",
        {
            "ok": True,
            "record": {
                "block_id": "mem_exact",
                "revision_id": "rev_exact",
                "title": "Exact",
                "content": "Exact content",
                "block_membership": {"primary_block_id": "leaf_exact"},
            },
        },
    )

    assert result["record"]["record_id"] == "mem_exact"
    assert result["record"]["logical_block_id"] == "leaf_exact"


def test_navigation_reads_only_visible_branch_and_keeps_paths_links(tmp_path, monkeypatch) -> None:
    from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
    from um_mem.memory.catalog_store import WorkspaceCatalogStore

    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    try:
        root_id = client.block_map()[0]["block_id"]
        branch = client.create_index_block(
            name="Branch", synopsis="Branch", parent_id=root_id, reason="test"
        )
        leaf = client.create_leaf_block(
            name="Leaf", synopsis="Leaf", parent_id=branch["block_id"], reason="test"
        )
        unrelated = client.create_index_block(
            name="Unrelated", synopsis="Unrelated", parent_id=root_id, reason="test"
        )
        other = client.create_leaf_block(
            name="Other", synopsis="Other", parent_id=unrelated["block_id"], reason="test"
        )
        client.add_block_edge(
            src_id=leaf["block_id"], dst_id=other["block_id"], edge_type="related_to", reason="test"
        )

        def forbidden_scan(*args, **kwargs):
            raise AssertionError("navigation must not enumerate the entire topology")

        monkeypatch.setattr(WorkspaceCatalogStore, "list_blocks", forbidden_scan)
        seen_edges = []
        edges_for = WorkspaceCatalogStore.edges_for

        def tracked_edges(self, block_id):
            seen_edges.append(block_id)
            return edges_for(self, block_id)

        monkeypatch.setattr(WorkspaceCatalogStore, "edges_for", tracked_edges)
        overview = client.block_map(depth=1)
        assert set(overview[0]["child_ids"]) == {branch["block_id"], unrelated["block_id"]}
        assert "children" not in overview[0]
        expanded = client.expand_block(branch["block_id"])
        assert expanded["path_ids"] == [root_id, branch["block_id"]]
        assert [child["block_id"] for child in expanded["children"]] == [leaf["block_id"]]
        assert expanded["children"][0]["links"]["related_block_ids"] == [other["block_id"]]
        assert seen_edges == []
        assert expanded["children"][0]["writable"] is True
    finally:
        index.close()


def test_navigation_envelope_projection_drops_raw_provider_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("UM_MEM_MCP_RESPONSE_MODE", "compact")
    raw = {
        "ok": True,
        "workspace_id": "opencode",
        "profile_id": "default",
        "catalog_revision": 7,
        "tree_path": ["root", "leaf"],
        "block": {
            "block_id": "leaf",
            "name": "Leaf",
            "synopsis": "Leaf synopsis",
            "record_count": 2,
            "native_partitions": [{"large": "diagnostic"}],
            "metrics": {"large": "diagnostic"},
        },
        "children": [],
        "related_blocks": [
            {
                "block_id": "other",
                "native_partitions": [{"large": "diagnostic"}],
            }
        ],
        "related_count": 3,
        "related_next_cursor": "next-graph-page",
        "health_hints": {"freshness": "2026-10-20T00:00:00Z"},
        "maintenance_observations": [],
        "proposal_options": [],
        "mutation_performed": False,
    }

    compact = compact_response("navigation_envelope", raw)

    assert "native_partitions" not in compact["block"]
    assert "metrics" not in compact["block"]
    assert compact["related_block_ids"] == ["other"]
    assert compact["related_count"] == 3
    assert compact["related_next_cursor"] == "next-graph-page"
    assert len(json.dumps(compact, ensure_ascii=False)) < len(json.dumps(raw, ensure_ascii=False))


def test_block_map_can_expose_the_protected_meta_tree(tmp_path: Path) -> None:
    client = MemoryClient(tmp_path, workspace_id="meta-audit", backend="mnemosyne")

    plain = client.block_map()
    assert len(plain) == 1
    assert plain[0]["block_id"] != "memory_root_index"

    blocks: list[dict[str, object]] = []

    def walk(items: list[dict[str, object]]) -> None:
        for item in items:
            blocks.append(item)
            walk(item.get("children") or [])

    walk(client.block_map(depth=3, include_meta=True))
    meta_root = next(item for item in blocks if item["block_id"] == "memory_root_index")
    assert meta_root["meta"] is True
    assert meta_root["node_kind"] == "meta"
    assert any(
        item["block_id"] == "decisions_and_contracts" and item["meta"] is True for item in blocks
    )

    expanded = client.expand_block("decisions_and_contracts")
    assert expanded["root"]["meta"] is True
    assert expanded["root"]["record_count"] == 0


def test_delete_block_reports_records_then_cascade_removes_them(tmp_path: Path) -> None:
    client = MemoryClient(tmp_path, workspace_id="cascade-audit", backend="mnemosyne")
    root = client.block_map()[0]
    leaf = client.create_leaf_block(
        name="Cascade leaf",
        synopsis="Cascade delete regression",
        parent_id=root["block_id"],
        reason="test",
    )
    saved = client.remember(
        "Cascade probe",
        title="Cascade probe",
        requested_block_id=leaf["block_id"],
    )
    assert saved.saved
    archived = client.archive_block(
        leaf["block_id"],
        expected_revision=leaf["revision"],
        reason="test",
    )

    first = client.delete_block(
        leaf["block_id"],
        expected_revision=archived["revision"],
        reason="cleanup",
    )
    assert first["ok"] is True
    assert first["deleted"] is False
    assert first["archived"] is True
    assert first["records_remaining"] == 1
    assert "cascade=true" in first["hint"]

    second = client.delete_block(
        leaf["block_id"],
        expected_revision=archived["revision"],
        reason="cleanup",
        cascade=True,
    )
    assert second["ok"] is True
    assert second["deleted"] is True
    assert second["records_remaining"] == 0


def test_amend_and_exact_read_expose_stable_record_ids(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UM_MEM_MCP_RESPONSE_MODE", "compact")
    client = MemoryClient(tmp_path, workspace_id="amend-audit", backend="mnemosyne")
    root = client.block_map()[0]
    leaf = client.create_leaf_block(
        name="Amend leaf",
        synopsis="Amend identity regression",
        parent_id=root["block_id"],
        reason="test",
    )
    saved = client.remember(
        "Original content",
        title="Original",
        requested_block_id=leaf["block_id"],
    )
    amended = client.amend(
        saved.memory_id,
        expected_revision=saved.raw["revision_id"],
        content="Corrected content",
        reason="test correction",
    )
    assert amended["ok"] is True
    compact_amend = compact_response("amend", amended)
    assert compact_amend["record_id"] == amended["new_id"]
    assert compact_amend["memory_id"] == amended["new_id"]

    original = client.read_exact_record(record_id=saved.memory_id)
    compact_read = compact_response(
        "read_exact_record",
        {"ok": True, "record": dataclasses.asdict(original)},
    )
    assert compact_read["record"]["record_id"] == saved.memory_id
    assert compact_read["record"]["logical_block_id"] == leaf["block_id"]


def test_mcp_fallback_cannot_admit_unvalidated_recent_records(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("UM_MEM_SEMANTIC", "0")
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *args, **kwargs: {
            "ok": True,
            "records": [
                {
                    "record_id": "mem_helper",
                    "title": "OpenCode config",
                    "content": "OpenCode config is in the global config directory.",
                    "status": "accepted",
                    "expires_at": "2000-01-01T00:00:00Z",
                    "tags": ["opencode", "config"],
                }
            ],
            "hits": [],
        },
    )
    result = mcp_server._apply_recall_fallback(
        SimpleNamespace(repo_root=tmp_path, workspace_id="ws"),
        None,
        "recall",
        {"query": "where is the opencode config", "limit": 3, "include_candidates": True},
        {"ok": False, "error": "native recall detail is missing", "hits": []},
    )
    assert result["hits"] == []
    assert result["ok"] is False
