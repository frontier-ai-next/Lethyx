"""Regressions found by exercising the public MCP memory workflow."""

from types import SimpleNamespace

import pytest

from um_mem.api import MemoryClient
from um_mem.memory import mcp_facade, mcp_server
from um_mem.memory.selection import MemorySelectionPolicy, SelectionContext


@pytest.mark.parametrize("evidence_kind", ["agent_tool_claim", "user_statement", "user_stated"])
def test_default_mcp_remember_is_recallable_in_another_run(
    tmp_path, monkeypatch, evidence_kind
) -> None:
    monkeypatch.setenv("UM_MEM_SEMANTIC", "0")
    writer = MemoryClient(tmp_path, workspace_id="shared", backend="mnemosyne")
    root = writer.block_map()[0]
    leaf = writer.create_leaf_block(
        name="Shared facts",
        synopsis="Facts written through the MCP facade",
        parent_id=root["block_id"],
        reason="test",
    )
    saved = mcp_facade.um_mem_remember(
        "Vega's maintenance window begins at 03:25 UTC.",
        title="Vega maintenance window",
        requested_block_id=leaf["block_id"],
        repo_root=str(tmp_path),
        workspace_id="shared",
        backend="mnemosyne",
        run_id="codex-run",
        evidence_kind=evidence_kind,
    )

    assert saved["saved"] is True
    reader = MemoryClient(tmp_path, workspace_id="shared", backend="mnemosyne")
    record = reader.read_exact_record(record_id=saved["memory_id"])
    assert record.metadata["evidence_kind"] == evidence_kind
    assert MemorySelectionPolicy().eligible_for_recall(
        record,
        SelectionContext(
            query="Vega maintenance window",
            scope_chain=("workspace:shared",),
            run_id="opencode-run",
            include_candidates=True,
        ),
    )
    assert not MemorySelectionPolicy().eligible_for_directive(
        record,
        SelectionContext(
            query="Vega maintenance window",
            scope_chain=("workspace:shared",),
            run_id="opencode-run",
            include_candidates=True,
        ),
    )
    assert not MemorySelectionPolicy().eligible_for_recall(
        record,
        SelectionContext(
            query="Vega maintenance window",
            scope_chain=("workspace:shared",),
            run_id="opencode-run",
            include_candidates=False,
        ),
    )

    original_local_tier = mcp_facade._apply_local_tier
    local_arguments = {}

    def capture_local_tier(result, *, arguments, **kwargs):
        local_arguments.update(arguments)
        return original_local_tier(result, arguments=arguments, **kwargs)

    monkeypatch.setattr(mcp_facade, "_apply_local_tier", capture_local_tier)
    recalled = mcp_facade.um_mem_recall(
        "Vega maintenance window",
        repo_root=str(tmp_path),
        workspace_id="shared",
        backend="mnemosyne",
        run_id="opencode-run",
        include_candidates=True,
        search_all=True,
    )
    assert local_arguments["run_id"] == "opencode-run"
    assert saved["memory_id"] in [hit["metadata"]["block_id"] for hit in recalled["hits"]]


@pytest.mark.parametrize("name", ["recall", "build_context"])
@pytest.mark.parametrize("backend", ["mnemosyne", "hindsight"])
def test_unsupported_hybrid_fails_before_backend_fallback(monkeypatch, name, backend) -> None:
    identity = SimpleNamespace(
        repo_root="store", workspace_id="ws", profile_id="default", backend=backend
    )
    monkeypatch.setattr(mcp_server, "build_local_service_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("service called")),
    )

    result = mcp_server._call_tool(name, {"query": "fact", "retrieval_mode": "hybrid"})

    assert result["reason_code"] == "unsupported_retrieval_mode"
    assert result["supported_retrieval_modes"] == ["semantic"]


def test_duplicate_warning_requires_content_overlap(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *_args, **_kwargs: {
            "ok": True,
            "hits": [
                {
                    "source": "unrelated",
                    "title": "Backup policy",
                    "text": "Backups must stay in the local vault.",
                    "score": 0.92,
                },
                {
                    "source": "duplicate",
                    "title": "Launch date",
                    "text": "The North project launches on 19 November 2026.",
                    "score": 0.84,
                },
            ],
        },
    )

    matches = mcp_server._similar_records(
        None, None, "The North project launches on 19 November 2026.", "Launch date", "new"
    )

    assert [match["record_id"] for match in matches] == ["duplicate"]


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        (
            "The North project launches on 19 November 2026.",
            "The North project launches on 20 November 2026.",
        ),
        (
            "Backups must stay in the local vault.",
            "Backups must not stay in the local vault.",
        ),
        (
            "The project manager Anna prefers written weekly reports and no calls.",
            "The project manager Boris prefers written weekly reports and no calls.",
        ),
    ],
)
def test_changed_claim_anchor_is_not_a_duplicate(monkeypatch, earlier, later) -> None:
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *_args, **_kwargs: {
            "ok": True,
            "hits": [{"source": "old", "title": "Same topic", "text": earlier, "score": 0.95}],
        },
    )

    assert mcp_server._similar_records(None, None, later, "Same topic", "new") == []


def test_identical_generic_content_with_different_subject_titles_is_not_duplicate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *_args, **_kwargs: {
            "ok": True,
            "hits": [
                {
                    "source": "anna",
                    "title": "Anna communication preference",
                    "text": "Prefers written reports.",
                    "score": 0.99,
                }
            ],
        },
    )

    assert (
        mcp_server._similar_records(
            None, None, "Prefers written reports.", "Boris communication preference", "new"
        )
        == []
    )


def test_replayed_remember_reports_original_memory_id(monkeypatch) -> None:
    identity = SimpleNamespace(
        repo_root="store", workspace_id="ws", profile_id="default", backend="mnemosyne"
    )
    monkeypatch.setattr(mcp_server, "build_local_service_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(mcp_server, "local_service_scope_lineage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *_args, **_kwargs: {
            "ok": True,
            "saved": True,
            "replayed": True,
            "canonical_reference": "mem_original",
            "revision_id": "rev_original",
        },
    )
    monkeypatch.setattr(mcp_server, "_await_projection", lambda result, *_args: result)
    monkeypatch.setattr(
        mcp_server,
        "_similar_records",
        lambda *_args: (_ for _ in ()).throw(AssertionError("replay must not check duplicates")),
    )

    reply = mcp_server._call_tool(
        "remember", {"content": "fact", "requested_block_id": "leaf", "operation_id": "replay"}
    )

    assert reply["memory_id"] == "mem_original"
    assert reply["replayed"] is True
    assert "duplicate_warning" not in reply


def test_everyday_operation_errors_point_to_the_right_next_tool(monkeypatch) -> None:
    identity = SimpleNamespace(
        repo_root="store", workspace_id="ws", profile_id="default", backend="mnemosyne"
    )
    monkeypatch.setattr(mcp_server, "build_local_service_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(mcp_server, "local_service_scope_lineage", lambda *_args, **_kwargs: None)
    responses = {
        "set_active_mask": {
            "ok": False,
            "error": "ValueError: an active memory mask already exists for this exact scope",
        },
        "operation_status": {"ok": False, "reason_code": "operation_not_found"},
    }
    monkeypatch.setattr(
        mcp_server, "_service_call", lambda _identity, _lineage, name, _args: responses[name]
    )

    focus = mcp_server._call_tool("set_active_mask", {"selected_block_ids": ["leaf"]})
    topology = mcp_server._call_tool("operation_status", {"operation_id": "split-1"})
    responses["set_active_mask"] = {
        "ok": False,
        "error": "ValueError: active memory mask requires selected_block_ids",
    }
    clear = mcp_server._call_tool("set_active_mask", {"selected_block_ids": []})

    assert "amend_active_mask" in focus["hint"]
    assert "pending_topology_evolutions" in topology["hint"]
    assert "transition='deactivate'" in clear["hint"]
