"""Regression coverage for the memory usability audit fixes."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lethyx.api import MemoryClient
from lethyx.contracts.memory import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from lethyx.memory import mcp_server
from lethyx.memory.backends.mnemosyne import MnemosyneBackend
from lethyx.memory.mcp_compact import compact_response


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
    return {"ok": True, "records": _RECORDS}


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
    assert "Lethyx recalled context" in rendered
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
    from lethyx.api import _hit_allowed
    from lethyx.memory.backends.contracts import block_record_to_memory_hit
    from lethyx.memory.recall_plan import RecallPlan

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
    monkeypatch.setenv("LETHYX_MCP_RESPONSE_MODE", "compact")
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


def test_local_service_route_receipt_keeps_route_decision() -> None:
    from lethyx.memory.mcp_call import _compact_remember_receipt

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

    assert receipt["route_decision"]["policy_issues"] == [
        "index_only_not_record_storage"
    ]
    assert receipt["route_decision"]["suggested_block_id"] == "decisions_and_contracts"
    assert receipt["suggested_route"]["requested_block_id"] == "decisions_and_contracts"


def test_compact_exact_read_derives_record_and_logical_block_ids(monkeypatch) -> None:
    monkeypatch.setenv("LETHYX_MCP_RESPONSE_MODE", "compact")
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


def test_navigation_envelope_projection_drops_raw_provider_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("LETHYX_MCP_RESPONSE_MODE", "compact")
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
        "health_hints": {"freshness": "2026-10-20T00:00:00Z"},
        "maintenance_observations": [],
        "proposal_options": [],
        "mutation_performed": False,
    }

    compact = compact_response("navigation_envelope", raw)

    assert "native_partitions" not in compact["block"]
    assert "metrics" not in compact["block"]
    assert compact["related_block_ids"] == ["other"]
    assert len(json.dumps(compact, ensure_ascii=False)) < len(
        json.dumps(raw, ensure_ascii=False)
    )


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
        item["block_id"] == "decisions_and_contracts" and item["meta"] is True
        for item in blocks
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


def test_amend_and_exact_read_expose_stable_record_ids(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LETHYX_MCP_RESPONSE_MODE", "compact")
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


def test_active_recall_helper_uses_universal_fallback_when_native_recall_raises() -> None:
    path = Path(__file__).resolve().parents[2] / "bin" / "lethyx-context.py"
    spec = importlib.util.spec_from_file_location("lethyx_context_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Client:
        def recall(self, *_args: object, **_kwargs: object) -> list[object]:
            raise ValueError("mnemosyne native recall detail is missing")

        def recent_records(self, **_kwargs: object) -> list[object]:
            return [
                SimpleNamespace(
                    block_id="mem_helper",
                    title="OpenCode config",
                    content="OpenCode config is in the global config directory.",
                    tags=["opencode", "config"],
                    recorded_at="2026-10-20T00:00:00Z",
                )
            ]

    evidence = module._recall(Client(), "where is the opencode config", 3, "run", "opencode")
    assert evidence[0]["record_id"] == "mem_helper"
    assert "OpenCode config" in evidence[0]["text"]
