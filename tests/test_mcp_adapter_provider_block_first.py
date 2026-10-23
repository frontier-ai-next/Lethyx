import json
from pathlib import Path
from typing import Any

from um_mem.api import MemoryClient
from um_mem.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from um_mem.integrations.hermes_provider import UmMemMemoryProvider
from um_mem.memory import mcp_facade
from um_mem.memory.backends.base import MemoryHit
from um_mem.memory.backends.contracts import block_record_to_memory_hit
from um_mem.memory.mcp_facade import (
    um_mem_block_map,
    um_mem_build_context,
    um_mem_expand_block,
)
from um_mem.memory.recall_plan import RecallPlan


class FacadeBackend:
    def __init__(self, record: MemoryBlockRecord) -> None:
        self.record = record

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        return self.record if block_id == self.record.block_id else None

    def export_blocks(self) -> list[MemoryBlockRecord]:
        return []


class FacadeClient:
    workspace_id = "alpha"
    user_id = ""
    repo_root = Path("unused")

    def __init__(self) -> None:
        self.record = MemoryBlockRecord(
            block_id="record-1",
            content="parser routed context",
            title="Parser",
            status="accepted",
            retention="durable",
            delivery_policy="on_demand",
            temperature="warm",
            authority="advisory",
            trust_level="workspace_verified",
            scope="workspace",
            workspace_id="alpha",
            verification_status="verified",
            verification_methods=("test_run",),
            promotion_decision="accepted",
            evidence_refs=(
                EvidenceRef(
                    ref_type="test_run",
                    ref_id="mcp-facade-record-1",
                    produced_by="verifier",
                    producer_id="mcp-facade-test",
                ),
            ),
            block_membership=RecordBlockMembership(
                primary_block_id="code_resources",
                catalog_revision_id="catalog:test",
            ),
            metadata={"retrieval_score": 0.9},
        )
        self.backend = FacadeBackend(self.record)
        self.calls: list[dict[str, Any]] = []

    def recall(self, query: str, **kwargs: Any) -> list[MemoryHit]:
        self.calls.append({"query": query, **kwargs})
        return [
            MemoryHit(
                text=self.record.content,
                source="record-1",
                score=0.9,
                metadata={"block_id": "record-1", "workspace_id": "alpha"},
            )
        ]

    def compile_recall_plan(self, query: str, **kwargs: Any) -> RecallPlan | None:
        planner = MemoryClient(self.repo_root, workspace_id=self.workspace_id)
        return planner.compile_recall_plan(query, **kwargs)


def test_mcp_facade_build_context_forwards_semantic_block_parameters(monkeypatch) -> None:
    client = FacadeClient()
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = um_mem_build_context(
        query="parser",
        workspace_id="alpha",
        repo_root="unused",
        backend="mempalace",
        source_horizon="archive",
        scope_chain=["workspace:alpha"],
        include_candidates=True,
        logical_block_ids=["code_resources"],
        record_ids=["record-1"],
    )

    assert payload["ok"] is True
    assert payload["items"][0]["metadata"]["block_id"] == "record-1"
    call = client.calls[-1]
    plan = call["recall_plan"]
    assert isinstance(plan, RecallPlan)
    assert plan.source_horizon == "archive"
    assert plan.scope_chain == ("workspace:alpha",)
    assert plan.include_candidates is True
    assert plan.logical_block_ids == ("code_resources",)
    assert plan.record_ids == ("record-1",)


def test_mcp_and_hermes_expose_root_map_and_selected_branch_expansion(monkeypatch) -> None:
    class CatalogClient:
        workspace_id = "alpha"
        repo_root = Path("unused")

        def block_map(self, *, depth=1, include_meta=False, child_limit=50) -> list[dict[str, Any]]:
            return [{"block_id": "semantic_memory_index", "child_ids": ["project_facts"]}]

        def expand_block(self, block_id: str, *, limit=50, cursor="") -> dict[str, Any]:
            assert block_id == "semantic_memory_index"
            return {
                "root": {
                    "block_id": block_id,
                    "native_partitions": [{"large": "diagnostic"}],
                    "metrics": {"large": "diagnostic"},
                },
                "children": [
                    {
                        "block_id": "project_facts",
                        "record_count": 2,
                        "block_metrics": [{"large": "diagnostic"}],
                    }
                ],
            }

    client = CatalogClient()
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    root_map = um_mem_block_map(workspace_id="alpha", repo_root="unused")
    expanded = um_mem_expand_block(
        "semantic_memory_index",
        workspace_id="alpha",
        repo_root="unused",
    )

    assert root_map["roots"] == [
        {"block_id": "semantic_memory_index", "child_ids": ["project_facts"]}
    ]
    assert expanded["children"] == [{"block_id": "project_facts", "record_count": 2}]
    assert expanded["root"] == {"block_id": "semantic_memory_index"}

    provider = UmMemMemoryProvider()
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        provider,
        "_call",
        lambda tool, args: calls.append((tool, args)) or {"ok": True},
    )
    schema_names = {schema["name"] for schema in provider.get_tool_schemas()}

    provider.handle_tool_call("um_mem_provider_block_map", {})
    provider.handle_tool_call(
        "um_mem_provider_expand_block",
        {"block_id": "semantic_memory_index"},
    )

    assert {"um_mem_provider_block_map", "um_mem_provider_expand_block"} <= schema_names
    assert calls == [
        ("block_map", {}),
        ("expand_block", {"block_id": "semantic_memory_index"}),
    ]


class BlockFirstBackend:
    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []
        self.record = MemoryBlockRecord(
            block_id="episodic-record",
            content="mnemosyne episodic workspace memory says parser state is retained",
            title="Episodic record",
            status="accepted",
            retention="durable",
            delivery_policy="on_demand",
            temperature="warm",
            authority="advisory",
            trust_level="workspace_verified",
            scope="workspace",
            workspace_id="alpha",
            verification_status="verified",
            verification_methods=("test_run",),
            promotion_decision="accepted",
            evidence_refs=(
                EvidenceRef(
                    ref_type="test_run",
                    ref_id="mcp-block-first-episodic",
                    produced_by="verifier",
                    producer_id="mcp-block-first-test",
                ),
            ),
            block_membership=RecordBlockMembership(
                primary_block_id="workspace_episodes",
                catalog_revision_id="catalog:test",
            ),
        )

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        return self.record if block_id == self.record.block_id else None

    def export_blocks(self) -> list[MemoryBlockRecord]:
        return []

    def recall_evidence(self, query: dict[str, Any]) -> list[MemoryHit]:
        self.queries.append(dict(query))
        return [block_record_to_memory_hit(self.record)]


def test_mcp_facade_build_context_keeps_explicit_block_scope(monkeypatch, tmp_path) -> None:
    backend = BlockFirstBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = um_mem_build_context(
        query="mnemosyne episodic workspace memory",
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        source_horizon="durable",
        max_tokens=512,
        logical_block_ids=["workspace_episodes"],
    )

    assert payload["ok"] is True
    assert payload["items"][0]["metadata"]["block_id"] == "episodic-record"
    query = backend.queries[-1]
    assert query["logical_block_ids"] == ["workspace_episodes"]
    diagnostics = payload["diagnostics"]
    assert diagnostics["recall_plan_reason"] == "explicit_block_first_recall"
    assert "workspace_episodes" in diagnostics["logical_block_ids"]
    assert diagnostics["source_horizon"] == "durable"


def test_hermes_provider_tool_schema_and_call_support_semantic_block_hints(monkeypatch) -> None:
    provider = UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "user_id": "user-1",
        "backend": "mempalace",
        "repo_root": "unused",
    }
    provider._session_id = "session-1"
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append((tool, args))
        return {"ok": True, "items": []}

    monkeypatch.setattr(provider, "_call", fake_call)
    schema = next(
        tool
        for tool in provider.get_tool_schemas()
        if tool["name"] == "um_mem_provider_recall_blocks"
    )
    properties = schema["parameters"]["properties"]

    assert "logical_block_ids" in properties
    assert "record_ids" in properties
    assert "source_horizon" in properties
    assert "scope_chain" not in properties
    assert "include_candidates" not in properties
    assert "selected_block_ids" not in properties
    assert "include_passive" not in properties
    assert "include_global" not in properties

    provider.handle_tool_call(
        "um_mem_provider_recall_blocks",
        {
            "query": "parser",
            "logical_block_ids": ["code_resources"],
            "record_ids": ["record-1"],
            "source_horizon": "archive",
        },
    )

    assert calls[-1][0] == "build_context"
    payload = calls[-1][1]
    assert payload["logical_block_ids"] == ["code_resources"]
    assert payload["record_ids"] == ["record-1"]
    assert payload["source_horizon"] == "archive"
    assert payload["scope_chain"] == [
        "session:session-1",
        "workspace:alpha",
        "user_global:user-1",
    ]
    assert payload["include_candidates"] is True


def test_hermes_provider_rejects_unknown_recall_argument_before_bridge_dispatch(
    monkeypatch,
) -> None:
    provider = UmMemMemoryProvider()
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append((tool, args))
        return {"ok": True, "items": []}

    monkeypatch.setattr(provider, "_call", fake_call)

    payload = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_recall_blocks",
            {
                "query": "parser",
                "logical_block_ids": ["code_resources"],
                "unexpected_argument": [],
            },
        )
    )

    assert payload == {
        "ok": False,
        "error": "unknown Hermes provider recall_blocks argument: unexpected_argument",
    }
    assert calls == []


def test_hermes_provider_rejects_model_claimed_scope_before_dispatch(monkeypatch) -> None:
    provider = UmMemMemoryProvider()
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(provider, "_call", lambda tool, args: calls.append((tool, args)))

    payload = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_recall_blocks",
            {
                "query": "parser",
                "logical_block_ids": ["code_resources"],
                "scope_chain": ["workspace:foreign"],
            },
        )
    )

    assert payload["ok"] is False
    assert payload["error"] == "unknown Hermes provider recall_blocks argument: scope_chain"
    assert calls == []


def test_hermes_write_tool_requires_selected_block_and_supports_correction_by_id(
    monkeypatch,
) -> None:
    provider = UmMemMemoryProvider()
    provider._session_id = "session-1"
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        provider,
        "_remember_and_export",
        lambda payload, **kwargs: (
            calls.append((kwargs["tool"], payload)) or {"ok": True, "saved": True}
        ),
    )
    monkeypatch.setattr(
        provider,
        "_observe_many_and_export",
        lambda *_args, **_kwargs: {
            "ok": True,
            "results": [
                {
                    "witness_id": "witness-1",
                    "evidence_refs": [
                        {
                            "ref_type": "artifact",
                            "ref_id": "witness-1",
                            "produced_by": "watcher",
                            "producer_id": "hermes:test",
                        }
                    ],
                }
            ],
        },
    )

    schema = next(
        tool for tool in provider.get_tool_schemas() if tool["name"] == "um_mem_provider_remember"
    )
    assert set(schema["parameters"]["required"]) == {
        "content",
        "operation",
        "requested_block_id",
    }
    rejected = json.loads(
        provider.handle_tool_call("um_mem_provider_remember", {"content": "missing scope"})
    )
    assert rejected["error"] == "requested_block_id is required"

    accepted = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_remember",
            {
                "content": "corrected value",
                "operation": "correct",
                "memory_id": "memory-1",
                "requested_block_id": "stable_project_facts",
            },
        )
    )
    assert accepted["ok"] is True
    assert calls[0][1]["operation"] == "correct"
    assert calls[0][1]["memory_id"] == "memory-1"
    assert calls[0][1]["requested_block_id"] == "stable_project_facts"
