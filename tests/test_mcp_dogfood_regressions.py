"""Regressions found by exercising the public MCP memory workflow."""

import re
from types import SimpleNamespace

import pytest

from um_mem.api import MemoryClient
from um_mem.memory import mcp_facade, mcp_server
from um_mem.memory.selection import MemorySelectionPolicy, SelectionContext


def test_compact_server_instructions_preserve_safety_rules_with_bounded_size(monkeypatch) -> None:
    monkeypatch.setenv("UM_MEM_MCP_INSTRUCTIONS", "compact")
    instructions = mcp_server._server_instructions()
    words = re.findall(r"\b[\w'-]+\b", instructions)

    assert 120 <= len(words) <= 170
    for required in (
        "exact authorized scope",
        "unverified candidates",
        "hypotheses unverified",
        "retrieved evidence",
        "stored mask is not automatic delivery",
        "remove unrelated focus",
        "exact current revision",
        "similarity warning alone is not grounds",
    ):
        assert required in instructions


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
    calls = []
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda _identity, _lineage, name, arguments: (
            calls.append((name, arguments))
            or {
                "ok": True,
                "scope_chain": ["project:project-a"],
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
            }
        ),
    )

    matches = mcp_server._similar_records(
        None,
        None,
        "The North project launches on 19 November 2026.",
        "Launch date",
        "new",
        scope_chain=("project:project-a",),
    )

    assert calls[0][0] == "recall"
    assert calls[0][1]["search_all"] is True
    assert calls[0][1]["scope_chain"] == ["project:project-a"]
    assert [match["record_id"] for match in matches] == ["duplicate"]


def test_duplicate_warning_fails_closed_without_exact_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unscoped duplicate recall must not run")
        ),
    )

    assert mcp_server._similar_records(None, None, "claim", "title", "new", scope_chain=()) == []


def test_duplicate_warning_fails_closed_on_mismatched_response_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *_args, **_kwargs: {
            "ok": True,
            "scope_chain": ["project:project-b"],
            "hits": [
                {
                    "source": "foreign-project-record",
                    "title": "Same title",
                    "text": "The exact project claim is stable.",
                }
            ],
        },
    )
    assert (
        mcp_server._similar_records(
            None,
            None,
            "The exact project claim is stable.",
            "Same title",
            "new",
            scope_chain=("project:project-a",),
        )
        == []
    )


def test_duplicate_recall_isolated_by_live_project_and_legacy_workspace_scopes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("UM_MEM_SEMANTIC", "0")
    content = "The COPPER-OWL manifest is deploy/copper-owl.yaml."
    for project_id, scope, block_id in (
        ("project-a", "project", "project_scope_stable_project_facts"),
        ("project-b", "project", "project_scope_stable_project_facts"),
        ("", "workspace", "stable_project_facts"),
    ):
        writer = MemoryClient(
            tmp_path,
            workspace_id="shared",
            project_id=project_id,
            backend="mnemosyne",
        )
        writer.remember(
            content,
            title="COPPER-OWL manifest",
            requested_block_id=block_id,
        )

    identity = SimpleNamespace(
        repo_root=str(tmp_path), workspace_id="shared", profile_id="default", backend="mnemosyne"
    )
    calls = []
    monkeypatch.setattr(
        mcp_facade,
        "_client",
        lambda **_kwargs: MemoryClient(
            tmp_path, workspace_id="shared", project_id="project-a", backend="mnemosyne"
        ),
    )

    def real_facade_recall(_identity, _lineage, name, arguments):
        assert name == "recall"
        calls.append(dict(arguments))
        return mcp_facade.um_mem_recall(
            query=arguments["query"],
            repo_root=str(tmp_path),
            workspace_id="shared",
            backend="mnemosyne",
            scope_chain=arguments["scope_chain"],
            include_candidates=True,
            search_all=True,
            limit=arguments["limit"],
        )

    monkeypatch.setattr(mcp_server, "_service_call", real_facade_recall)
    matches = mcp_server._similar_records(
        identity,
        None,
        content,
        "COPPER-OWL manifest",
        "new-record",
        scope_chain=("project:project-a",),
    )

    assert calls[0]["scope_chain"] == ["project:project-a"]
    assert matches, mcp_facade.um_mem_recall(
        content,
        repo_root=str(tmp_path),
        workspace_id="shared",
        backend="mnemosyne",
        scope_chain=["project:project-a"],
        include_candidates=True,
        search_all=True,
    )
    assert all(match["title"] == "COPPER-OWL manifest" for match in matches)
    assert len(matches) == 1


def test_recall_fallback_keeps_exact_scope_chain(monkeypatch) -> None:
    calls = []

    def service_call(_identity, _lineage, name, arguments):
        calls.append((name, dict(arguments)))
        if name == "recent":
            return {
                "ok": True,
                "records": [
                    {
                        "record_id": "project-record",
                        "content": "project content",
                        "scope": "project",
                        "project_id": "project-a",
                        "status": "candidate",
                    },
                    {
                        "record_id": "workspace-record",
                        "content": "workspace content",
                        "scope": "workspace",
                        "workspace_id": "ws",
                        "status": "candidate",
                    },
                ],
            }
        assert name == "recall"
        return {
            "ok": True,
            "scope_chain": ["project:project-a"],
            "hits": [{"source": "project-record"}],
        }

    monkeypatch.setattr(mcp_server, "_service_call", service_call)
    records = mcp_server._fallback_records(
        None, None, {"scope_chain": ["project:project-a"], "include_candidates": True}
    )

    assert [record["record_id"] for record in records] == ["project-record"]
    assert calls[1][1]["scope_chain"] == ["project:project-a"]
    assert calls[1][1]["record_ids"] == ["project-record"]


def test_remember_duplicate_compact_response_uses_confirmed_scope(monkeypatch) -> None:
    identity = SimpleNamespace(
        repo_root="store", workspace_id="ws", profile_id="default", backend="mnemosyne"
    )
    calls = []
    monkeypatch.delenv("UM_MEM_WRITE_SCOPE_PROJECT_ONLY", raising=False)
    monkeypatch.delenv("UM_MEM_ACTIVE_SCOPE_PROJECT_ONLY", raising=False)
    monkeypatch.setenv("UM_MEM_MCP_RESPONSE_MODE", "compact")
    monkeypatch.setattr(mcp_server, "build_local_service_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(
        mcp_server,
        "local_service_scope_lineage",
        lambda *_args, **_kwargs: None,
    )

    def service_call(_identity, _lineage, name, arguments):
        if name == "remember":
            return {
                "ok": True,
                "saved": True,
                "memory_id": "new-project-record",
                "scope": "project",
                "scope_id": "project-a",
            }
        calls.append(dict(arguments))
        return {
            "ok": True,
            "scope_chain": ["project:project-a"],
            "hits": [
                {
                    "source": "workspace-legacy",
                    "title": "Same title",
                    "text": "The exact project claim is stable.",
                    "scope_key": "workspace:ws",
                },
                {
                    "source": "project-duplicate",
                    "title": "Same title",
                    "text": "The exact project claim is stable.",
                    "scope_key": "project:project-a",
                },
            ],
        }

    monkeypatch.setattr(mcp_server, "_service_call", service_call)
    monkeypatch.setattr(mcp_server, "_await_projection", lambda result, *_args: result)

    reply = mcp_server._call_tool(
        "remember",
        {
            "content": "The exact project claim is stable.",
            "title": "Same title",
            "requested_block_id": "project_scope_stable_project_facts",
            "operation_id": "project-write",
        },
    )

    assert calls[0]["scope_chain"] == ["project:project-a"]
    assert [hit["record_id"] for hit in reply["similar_records"]] == ["project-duplicate"]
    assert "workspace-legacy" not in repr(reply)
    from um_mem.memory.mcp_compact import compact_response

    compact = compact_response("remember", reply)
    assert [hit["record_id"] for hit in compact["similar_records"]] == ["project-duplicate"]
    assert "workspace-legacy" not in repr(compact)


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
            "scope_chain": ["project:project-a"],
            "hits": [{"source": "old", "title": "Same topic", "text": earlier, "score": 0.95}],
        },
    )

    assert (
        mcp_server._similar_records(
            None, None, later, "Same topic", "new", scope_chain=("project:project-a",)
        )
        == []
    )


def test_identical_generic_content_with_different_subject_titles_is_not_duplicate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *_args, **_kwargs: {
            "ok": True,
            "scope_chain": ["project:project-a"],
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
            None,
            None,
            "Prefers written reports.",
            "Boris communication preference",
            "new",
            scope_chain=("project:project-a",),
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("replay must not check duplicates")
        ),
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
