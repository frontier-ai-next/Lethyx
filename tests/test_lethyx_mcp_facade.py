import asyncio
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.fastmcp.exceptions import ToolError
from lethyx.api import AmbiguousEvidenceRefError, MemoryClient
from lethyx.contracts import (
    EvidenceRef,
    MemoryBlockRecord,
    RecordBlockMembership,
)
from lethyx.memory import local_service, mcp_facade, mcp_server
from lethyx.memory.backends.base import MemoryBusyError, MemoryHit
from lethyx.memory.backends.contracts import CanonicalHeadConflict, block_record_to_memory_hit
from lethyx.memory.kernel.telemetry import read_memory_events
from lethyx.memory.local_service import (
    LOCAL_SERVICE_API_VERSION,
    LOCAL_SERVICE_SCHEMA_VERSION,
    LocalServiceClientPool,
    build_local_service_identity,
    local_service_bootstrap_secret,
    shutdown_local_service_listener,
)
from lethyx.memory.mcp_facade import (
    _safe_error,
    lethyx_build_context,
    lethyx_build_granted_context,
    lethyx_export_graph,
    lethyx_export_obsidian,
    lethyx_forget,
    lethyx_promote,
    lethyx_recall,
    lethyx_remember,
    lethyx_read_exact_record,
    lethyx_read_exact_records,
)
from lethyx.memory.mcp_server import TOOL_NAMES, create_mcp_server


def test_mcp_remember_operation_id_can_be_queried(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LETHYX_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("LETHYX_WORKSPACE", "alpha")
    monkeypatch.setenv("LETHYX_BACKEND", "mempalace")
    monkeypatch.setenv("LETHYX_SESSION_ID", "mcp-status-session")
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="operator"
    )
    try:
        saved = mcp_server._call_tool(
            "remember",
            {"content": "MCP status receipt.", "requested_block_id": "workspace_episodes"},
        )
        status = mcp_server._call_tool("operation_status", {"operation_id": saved["operation_id"]})
    finally:
        mcp_server.close_local_service_clients()
        shutdown_local_service_listener(identity)

    assert saved["ok"] is True
    assert status["ok"] is True
    assert status["record_id"] == saved["memory_id"]
    assert status["original_receipt"]["canonical_reference"] == saved["memory_id"]


def test_facade_exact_reads_preserve_batch_order_and_ambiguity(monkeypatch, tmp_path) -> None:
    from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex

    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="alpha")
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)
    try:
        first = client.remember("first", requested_block_id="stable_project_facts")
        second = client.remember("second", requested_block_id="stable_project_facts")
        one = lethyx_read_exact_record(record_id=first.memory_id)
        batch = lethyx_read_exact_records(record_ids=[second.memory_id, first.memory_id])
        assert one["ok"] is True
        assert one["record"]["content"] == "first"
        assert [record["content"] for record in batch["records"]] == ["second", "first"]
    finally:
        index.close()


def test_facade_remember_exposes_durable_revision_without_unpacking_raw(monkeypatch, tmp_path):
    from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex

    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="alpha")
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)
    try:
        receipt = lethyx_remember(
            "durable source",
            requested_block_id="stable_project_facts",
            operation_id="stable-operation",
        )
        assert receipt["saved"] is True
        for key in (
            "operation_id",
            "commit_revision",
            "durable_status",
            "projection_status",
            "projection_revision",
            "retryable",
            "source_coverage",
        ):
            assert receipt[key] == receipt["raw"][key]
        assert receipt["actual_block_id"] == "stable_project_facts"
    finally:
        index.close()


def test_facade_recall_and_context_share_tag_and_speaker_filters(monkeypatch, tmp_path) -> None:
    from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex

    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="alpha")
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)
    try:
        for speaker in ("Ada", "Bob"):
            client.remember(
                f"{speaker} reported the date.",
                requested_block_id="workspace_episodes",
                write_mode="lossless_atomic",
                tags=["report"],
                metadata={
                    "source_segments": [
                        {
                            "source_id": f"turn-{speaker}",
                            "text": f"{speaker} reported the date.",
                            "speaker": speaker,
                        }
                    ]
                },
            )
        args = {
            "query": "reported",
            "logical_block_ids": ["workspace_episodes"],
            "include_candidates": True,
            "tags": ["report"],
            "source_speakers": ["Ada"],
        }
        recalled = lethyx_recall(**args)
        context = lethyx_build_context(**args)
        assert recalled["ok"] is True
        assert context["ok"] is True
        assert [hit["text"] for hit in recalled["hits"]] == ["Ada reported the date."]
        assert [item["text"] for item in context["items"]] == ["Ada reported the date."]
        assert recalled["diagnostics"]["source_speaker_filter_complete"] is False
        assert context["diagnostics"]["source_speaker_filter_complete"] is False
    finally:
        index.close()


def _repo(tmp_path: Path) -> str:
    return tmp_path.as_posix()


@pytest.mark.parametrize(
    ("error", "reason_code"),
    [
        (ConnectionError("native transport failed"), "native_transport_failure"),
        (TimeoutError("native transport timed out"), "transport_timeout"),
        (PermissionError("store access denied"), "access_denied"),
        (BlockingIOError("store lock busy"), "store_lock_unavailable"),
        (MemoryBusyError("store lock busy"), "store_lock_unavailable"),
        (CanonicalHeadConflict("stale"), "canonical_revision_conflict"),
        (AmbiguousEvidenceRefError("ambiguous_evidence_ref"), "ambiguous_evidence_ref"),
        (OSError("local IO failed"), "memory_operation_failed"),
        (
            RuntimeError("native_projection_rebuild_required"),
            "native_projection_rebuild_required",
        ),
        (
            RuntimeError("native_block_scoping_unsupported"),
            "native_block_scoping_unsupported",
        ),
    ],
)
def test_safe_error_preserves_typed_native_failure_reason(
    error: BaseException,
    reason_code: str,
) -> None:
    payload = _safe_error(error)

    assert payload["ok"] is False
    assert payload["error_type"] == type(error).__name__
    assert payload["reason_code"] == reason_code
    assert payload["retryable"] is (
        reason_code in {"native_transport_failure", "transport_timeout", "store_lock_unavailable"}
    )


def _record(
    block_id: str,
    content: str,
    workspace_id: str = "alpha",
    **overrides: Any,
) -> MemoryBlockRecord:
    record = MemoryBlockRecord(
        block_id=block_id,
        content=content,
        title="Record",
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
        workspace_id=workspace_id,
        run_id="",
        producer="test",
        source_backend="mempalace",
        tags=("parser",),
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"mcp-{block_id}",
                produced_by="verifier",
                producer_id=f"verifier:{block_id}",
            ),
        ),
    )
    return replace(record, **overrides)


def test_recall_preserves_independent_records_even_with_same_diagnostic_cluster() -> None:
    records = [
        _record("first", "Same fact.", duplicate_cluster_id="dup_exact"),
        _record("second", "Same fact.", duplicate_cluster_id="dup_exact"),
        _record("third", "Distinct run receipt."),
        _record("fourth", "Distinct run receipt."),
    ]

    recalled = mcp_facade._dedup_records(records)

    assert [record.block_id for record in recalled] == ["first", "second", "third", "fourth"]


def test_recall_does_not_guess_that_lexically_similar_claims_are_duplicates() -> None:
    membership = RecordBlockMembership(
        primary_block_id="experiment-metrics",
        catalog_revision_id="catalog:test",
    )
    records = [
        _record(
            "latest",
            (
                "Measured final window metrics: r=2.8 mean=0.642857 and std=0; "
                "r=3.7 mean=0.665344 and std=0.209623. The run used 100 values."
            ),
            block_membership=membership,
        ),
        _record(
            "repeat",
            (
                "For the measured final window, r=2.8 has mean 0.642857 and std 0, "
                "while r=3.7 has mean 0.665344 and std 0.209623 across 100 values."
            ),
            block_membership=membership,
        ),
        _record(
            "changed-value",
            (
                "For the measured final window, r=2.8 has mean 0.642857 and std 0, "
                "while r=3.7 has mean 0.665344 and std 0.210000 across 100 values."
            ),
            block_membership=membership,
        ),
        _record(
            "other-leaf",
            (
                "For the measured final window, r=2.8 has mean 0.642857 and std 0, "
                "while r=3.7 has mean 0.665344 and std 0.209623 across 100 values."
            ),
            block_membership=RecordBlockMembership(
                primary_block_id="independent-experiment",
                catalog_revision_id="catalog:test",
            ),
        ),
    ]

    recalled = mcp_facade._dedup_records(records)

    assert [record.block_id for record in recalled] == [
        "latest",
        "repeat",
        "changed-value",
        "other-leaf",
    ]


@pytest.mark.parametrize("use_active_mask", [False, True])
def test_build_granted_context_uses_only_exact_indexed_records_without_recall(
    monkeypatch,
    tmp_path: Path,
    use_active_mask: bool,
) -> None:
    granted = replace(
        _record("granted-record", "Accepted exact project context."),
        block_membership=RecordBlockMembership(
            primary_block_id="project-knowledge",
            catalog_revision_id="catalog:test",
        ),
    )
    unrequested = replace(
        _record("unrequested-record", "Same block but not pinned by the Start grant."),
        block_membership=RecordBlockMembership(
            primary_block_id="project-knowledge",
            catalog_revision_id="catalog:test",
        ),
    )
    outside = replace(
        _record("outside-record", "Must never cross the granted block boundary."),
        block_membership=RecordBlockMembership(
            primary_block_id="unrelated-knowledge",
            catalog_revision_id="catalog:test",
        ),
    )

    class Backend:
        def __init__(self) -> None:
            self.records = {record.block_id: record for record in (granted, unrequested, outside)}
            self.resolutions: list[tuple[tuple[str, ...], str, str, bool]] = []

        def record_ids_for_logical_blocks(
            self,
            logical_block_ids: tuple[str, ...],
            *,
            workspace_id: str,
            profile_id: str,
            historical: bool,
        ) -> tuple[str, ...]:
            self.resolutions.append((logical_block_ids, workspace_id, profile_id, historical))
            # Include a malformed backend result to prove the normal logical
            # block policy is still enforced after deterministic resolution.
            return ("granted-record", "unrequested-record", "outside-record")

        def get_block(self, block_id: str) -> MemoryBlockRecord | None:
            return self.records.get(block_id)

        def recall(self, *_args: object, **_kwargs: object) -> list[MemoryHit]:
            raise AssertionError("exact granted context must not invoke semantic recall")

    backend = Backend()
    client = SimpleNamespace(
        backend=backend,
        workspace_id="alpha",
        profile_id="default",
        repo_root=tmp_path,
    )

    def resolve_active_mask(**kwargs):
        assert kwargs == {"mask_id": "mask-1", "expected_revision": 3, "max_tokens": 256}
        return SimpleNamespace(
            trace=SimpleNamespace(resolution_status="resolved", semantic_recall_performed=False),
            items=tuple(SimpleNamespace(block_id=key) for key in backend.records),
        )

    client.resolve_active_mask = resolve_active_mask
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)
    monkeypatch.setattr(mcp_facade, "record_memory_event", lambda *_args, **_kwargs: None)

    payload = lethyx_build_granted_context(
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        scope_chain=["workspace:alpha"],
        logical_block_ids=["project-knowledge"],
        record_ids=["granted-record", "outside-record"],
        max_tokens=256,
        render_context=True,
        active_mask_id="mask-1" if use_active_mask else "",
        active_mask_revision=3 if use_active_mask else None,
    )

    assert payload["ok"] is True
    assert "Accepted exact project context." in payload["rendered_context"]
    assert payload["rendered_context"].count("Accepted exact project context.") == 1
    assert len(payload["rendered_context"]) <= 256 * 4
    assert "Same block but not pinned" not in payload["rendered_context"]
    assert "Must never cross" not in payload["rendered_context"]
    assert payload["diagnostics"]["native_path"] == "exact_grant_index"
    assert payload["diagnostics"]["semantic_recall_performed"] is False
    assert payload["diagnostics"]["rendered_record_ids"] == ["granted-record"]
    assert backend.resolutions == [(("project-knowledge",), "alpha", "default", False)]


@pytest.mark.parametrize("limit", [0, 2])
def test_build_granted_context_stops_loading_after_eligible_limit(
    monkeypatch,
    tmp_path: Path,
    limit: int,
) -> None:
    loaded: list[str] = []
    membership = RecordBlockMembership(
        primary_block_id="project-knowledge", catalog_revision_id="catalog:test"
    )

    def get_block(record_id: str) -> MemoryBlockRecord:
        loaded.append(record_id)
        return _record(
            record_id,
            f"Evidence {record_id}",
            workspace_id="foreign" if record_id == "0" else "alpha",
            block_membership=membership,
        )

    client = SimpleNamespace(
        backend=SimpleNamespace(
            record_ids_for_logical_blocks=lambda *_args, **_kwargs: tuple(
                str(index) for index in range(1000)
            ),
            get_block=get_block,
        ),
        workspace_id="alpha",
        profile_id="default",
        repo_root=tmp_path,
    )
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)
    monkeypatch.setattr(mcp_facade, "record_memory_event", lambda *_args, **_kwargs: None)

    payload = lethyx_build_granted_context(
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        scope_chain=["workspace:alpha"],
        logical_block_ids=["project-knowledge"],
        limit=limit,
        max_tokens=512,
    )

    assert payload["ok"] is True
    assert loaded == (["0", "1", "2"] if limit else [])
    assert payload["diagnostics"]["selected_record_count"] == limit
    assert payload["diagnostics"]["assigned_record_count"] == 1000


def test_build_granted_context_rejects_an_unbounded_record_only_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend = SimpleNamespace()
    client = SimpleNamespace(
        backend=backend,
        workspace_id="alpha",
        profile_id="default",
        repo_root=tmp_path,
    )
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)

    payload = lethyx_build_granted_context(
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        scope_chain=["workspace:alpha"],
        logical_block_ids=[],
        record_ids=["record-without-a-block-grant"],
        render_context=True,
    )

    assert payload["ok"] is False
    assert "requires exact logical_block_ids" in payload["error"]
    assert payload["rendered_context"] == ""


class _CanonicalRecallBackend:
    def __init__(self, records: tuple[MemoryBlockRecord, ...]) -> None:
        self._records = {record.block_id: record for record in records}

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        return self._records.get(block_id)


class _CanonicalRecallClient:
    workspace_id = "alpha"

    def __init__(self, records: tuple[MemoryBlockRecord, ...], repo_root: Path) -> None:
        self.repo_root = repo_root
        self.backend = _CanonicalRecallBackend(records)
        self._records = records

    def recall(self, query: str, **_kwargs: object) -> list[MemoryHit]:
        del query
        return [block_record_to_memory_hit(record) for record in self._records]

    def compile_recall_plan(self, query: str, **kwargs: object):
        planner = MemoryClient(self.repo_root, workspace_id=self.workspace_id)
        return planner.compile_recall_plan(query, **kwargs)


def test_mcp_server_exposes_only_canonical_recall_scope_fields() -> None:
    async def mcp_contract() -> dict[str, dict[str, object]]:
        server = create_mcp_server()
        tools = await server.list_tools()
        return {
            tool.name: dict(tool.inputSchema)
            for tool in tools
            if tool.name in {"recall", "build_context"}
        }

    schemas = asyncio.run(mcp_contract())
    assert schemas["recall"]["properties"]["limit"]["default"] == 4
    assert schemas["build_context"]["properties"]["limit"]["default"] == 4
    for schema in schemas.values():
        properties = schema["properties"]
        assert "logical_block_ids" in properties
        assert "record_ids" in properties
        assert "source_horizon" in properties
        for identity_field in (
            "workspace_id",
            "repo_root",
            "backend",
            "user_id",
            "project_id",
            "session_id",
            "task_id",
            "subtask_id",
            "run_id",
            "scope_chain",
        ):
            assert identity_field not in properties
        assert "selected_block_ids" not in properties
        assert "include_passive" not in properties
        assert "include_global" not in properties
        assert schema["additionalProperties"] is False


def test_evolution_tools_publish_complete_mapping_schemas_on_both_hosts() -> None:
    from lethyx.integrations.hermes_provider import UmMemMemoryProvider

    async def mcp_schemas():
        return {tool.name: tool.inputSchema for tool in await create_mcp_server().list_tools()}

    schemas = asyncio.run(mcp_schemas())
    provider = UmMemMemoryProvider()
    provider._config = {"tool_access": "full"}
    hermes = {
        tool["name"].removeprefix("lethyx_provider_"): tool["parameters"]
        for tool in provider.get_tool_schemas()
    }
    for surface in (schemas, hermes):
        split = surface["split_block"]["properties"]["targets"]["items"]
        merge = surface["merge_blocks"]["properties"]
        for target in (split, merge["target"]):
            assert {"name", "synopsis", "record_ids"} <= set(target["required"])
            assert target["additionalProperties"] is False
            assert target["properties"]["record_ids"]["uniqueItems"] is True
        assert set(merge["sources"]["items"]["required"]) == {"block_id", "expected_revision"}


def test_mcp_server_binds_identity_outside_every_tool_schema() -> None:
    async def schemas() -> dict[str, dict[str, object]]:
        tools = await create_mcp_server().list_tools()
        return {tool.name: dict(tool.inputSchema) for tool in tools}

    identity_fields = {
        "workspace_id",
        "repo_root",
        "backend",
        "user_id",
        "project_id",
        "session_id",
        "task_id",
        "subtask_id",
        "run_id",
        "scope_chain",
    }
    for schema in asyncio.run(schemas()).values():
        assert identity_fields.isdisjoint(schema["properties"])
        assert schema["additionalProperties"] is False


def test_mcp_server_remember_routes_child_claim_transport(monkeypatch) -> None:
    dispatched: list[tuple[str, dict[str, object]]] = []
    transport = {
        "claim_type": "user_intent",
        "source_role": "user",
        "content": "Use the routed provider path.",
        "parent_session_id": "session-alpha",
        "origin_segments": [
            {
                "claim_id": "session-alpha:user-intent",
                "origin": {
                    "channel": "host_event",
                    "principal_id": "hermes",
                    "integrity_class": "reported",
                },
                "integrity_ceiling": "reported",
                "independence_domain": "hermes:session-alpha",
                "evidence_refs": [
                    {
                        "ref_type": "artifact",
                        "ref_id": "hermes-session:session-alpha",
                        "produced_by": "agent",
                        "producer_id": "hermes:session-alpha",
                    }
                ],
            }
        ],
        "temperature": "hot",
        "tags": ["intent"],
    }

    def fake_call_tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
        dispatched.append((name, arguments))
        return {"ok": True}

    monkeypatch.setattr(mcp_server, "_call_tool", fake_call_tool)

    async def mcp_contract() -> dict[str, object]:
        server = create_mcp_server()
        tools = await server.list_tools()
        remember = next(tool for tool in tools if tool.name == "remember")
        await server.call_tool(
            "remember",
            {
                "content": "turn",
                "child_claims": [transport],
                "requested_block_id": "workspace_episodes",
            },
        )
        return dict(remember.inputSchema)

    schema = asyncio.run(mcp_contract())

    assert "child_claims" in schema["properties"]
    assert [name for name, _arguments in dispatched] == ["remember"]
    assert dispatched[0][1]["content"] == "turn"
    assert dispatched[0][1]["child_claims"] == [transport]


def test_mcp_server_exposes_one_routed_remember_without_authority_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched: list[tuple[str, dict[str, object]]] = []

    def fake_call_tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
        dispatched.append((name, arguments))
        return {"ok": True, "saved": True}

    monkeypatch.setattr(mcp_server, "_call_tool", fake_call_tool)

    async def mcp_contract() -> dict[str, object]:
        server = create_mcp_server()
        tools = await server.list_tools()
        remember = next(tool for tool in tools if tool.name == "remember")
        assert all(tool.name != "remember_routed" for tool in tools)
        await server.call_tool(
            "remember",
            {
                "content": "A scoped candidate should use the taxonomy route.",
                "kind": "project_fact",
                "tags": ["candidate"],
                "requested_block_id": "stable_project_facts",
            },
        )
        with pytest.raises(ToolError, match="extra_forbidden"):
            await server.call_tool(
                "remember",
                {
                    "content": "Do not permit callers to choose authority.",
                    "scope": "manager",
                },
            )
        return dict(remember.inputSchema)

    schema = asyncio.run(mcp_contract())

    assert schema["additionalProperties"] is False
    for prohibited in (
        "scope",
        "status",
        "retention",
        "delivery_policy",
        "temperature",
        "authority",
        "trust_level",
        "verification_status",
        "verification_methods",
        "promotion_decision",
        "requested_store",
        "record_id",
        "workspace_id",
        "repo_root",
        "backend",
        "user_id",
        "project_id",
        "session_id",
        "task_id",
        "subtask_id",
        "run_id",
    ):
        assert prohibited not in schema["properties"]
    assert dispatched[0][0] == "remember"
    assert dispatched[0][1]["content"] == "A scoped candidate should use the taxonomy route."


def test_mcp_facade_remember_applies_route_and_marks_policy_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")

        def remember(self, content: str, **kwargs: object) -> SimpleNamespace:
            captured["content"] = content
            captured.update(kwargs)
            return SimpleNamespace(
                saved=True,
                memory_id="routed-candidate",
                backend="mempalace",
                raw={
                    "scope": "workspace",
                    "scope_id": "alpha",
                    "project_id": "project-alpha",
                    "user_id": "user-alpha",
                    "route_decision": {
                        "accepted": True,
                        "reason": "default_passive_candidate",
                    },
                },
            )

        def correction(self, memory_id: str, content: str, **kwargs: object) -> dict[str, object]:
            captured["correction_memory_id"] = memory_id
            captured["correction_content"] = content
            captured.update(kwargs)
            return {
                "ok": True,
                "saved": True,
                "block_id": "correction-candidate",
                "pending_host_approval": True,
            }

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: Client())
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    rejected = lethyx_remember(
        "Routed MCP write",
        workspace_id="alpha",
        kind="project_fact",
        evidence_kind="test_run",
        evidence_refs=[{"ref_type": "pytest", "ref_id": "route", "produced_by": "test"}],
    )
    assert rejected["ok"] is False
    assert rejected["error"].startswith("ValueError:")

    payload = lethyx_remember(
        "Routed MCP write",
        workspace_id="alpha",
        kind="project_fact",
        evidence_kind="test_run",
        evidence_refs=["mcp-routed-remember:route"],
        requested_block_id="stable_project_facts",
        source_id="hermes:session-alpha",
        run_id="run-alpha",
        metadata={"session_id": "session-alpha"},
    )

    assert payload["ok"] is True
    assert payload["saved"] is True
    assert payload["route_rejected"] is False
    assert payload["scope"] == "workspace"
    assert payload["scope_id"] == "alpha"
    assert payload["project_id"] == "project-alpha"
    assert payload["user_id"] == "user-alpha"
    assert captured["content"] == "Routed MCP write"
    assert captured["requested_block_id"] == "stable_project_facts"
    assert captured["source_id"] == "hermes:session-alpha"
    evidence = captured["evidence_refs"]
    assert isinstance(evidence, list)
    assert evidence[0].ref_type == "session_event"
    assert evidence[0].ref_id == "mcp-routed-remember:route"
    assert evidence[0].produced_by == "harness"

    correction = lethyx_remember(
        "Corrected MCP value",
        operation="correct",
        memory_id="trusted-memory",
        requested_block_id="stable_project_facts",
        workspace_id="alpha",
        metadata={"working_set_owner": {"owner_id": "host", "generation_id": "attempt"}},
    )
    assert correction["ok"] is True
    assert captured["metadata"] == {
        "working_set_owner": {"owner_id": "host", "generation_id": "attempt"}
    }
    assert correction["pending_host_approval"] is True
    assert captured["correction_memory_id"] == "trusted-memory"
    assert captured["requested_block_id"] == "stable_project_facts"

    repair = lethyx_remember(
        "Routed MCP write",
        operation="repair",
        repair_of="raw-witness-1",
        evidence_refs=[
            {
                "ref_type": "artifact",
                "ref_id": "raw-witness-1",
                "produced_by": "watcher",
                "producer_id": "provider-test",
            }
        ],
        requested_block_id="stable_project_facts",
        workspace_id="alpha",
    )
    assert repair["ok"] is True
    assert captured["evidence_refs"][0].ref_id == "raw-witness-1"
    assert captured["metadata"]["repair_of"] == "raw-witness-1"


def test_mcp_facade_promote_forwards_typed_verification_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Client:
        workspace_id = "alpha"

        def promote(self, memory_id: str, **kwargs: object) -> dict[str, object]:
            captured["memory_id"] = memory_id
            captured.update(kwargs)
            return {"block_id": memory_id, "backend": "mempalace"}

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: Client())
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_promote(
        "promote-1",
        reason="reviewed",
        evidence_refs=[
            {
                "report_id": "mcp-promotion:promote-1",
                "report_hash": "report-sha256",
                "workspace_hash": "workspace-sha256",
                "diff_hash": "diff-sha256",
                "produced_after_event_id": "event-1",
                "verifier_id": "verifier:mcp-promotion",
                "passed": True,
            }
        ],
        workspace_id="alpha",
    )

    assert payload["ok"] is True
    assert captured["memory_id"] == "promote-1"
    evidence = captured["evidence_refs"]
    assert isinstance(evidence, list)
    assert evidence[0]["report_id"] == "mcp-promotion:promote-1"
    assert evidence[0]["verifier_id"] == "verifier:mcp-promotion"


def test_mcp_health_truthfully_marks_tools_only_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    class Backend:
        def health(self) -> dict[str, object]:
            return {"ok": True, "backend": "mempalace"}

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")
        backend = Backend()

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: Client())
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = mcp_facade.lethyx_health(workspace_id="alpha", backend="mempalace")

    assert payload["integration_mode"] == "mcp_tools_only"
    assert payload["deep_agent_support"] is False
    assert payload["degraded_reasons"] == ["mcp_transport_has_no_host_lifecycle_capture"]
    assert payload["runtime"]["effective_profile"] == "mcp_tools_only"
    assert payload["runtime"]["health"] is True


def test_mcp_health_keeps_tools_only_degradation_when_backend_health_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backend:
        def health(self) -> dict[str, object]:
            raise RuntimeError("backend health unavailable")

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")
        backend = Backend()

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: Client())
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = mcp_facade.lethyx_health(workspace_id="alpha", backend="mempalace")

    assert payload["ok"] is False
    assert payload["integration_mode"] == "mcp_tools_only"
    assert payload["deep_agent_support"] is False
    assert payload["degraded_reasons"] == ["mcp_transport_has_no_host_lifecycle_capture"]
    assert payload["runtime"]["effective_profile"] == "mcp_tools_only"
    assert payload["runtime"]["health"] is False


def test_mcp_facade_normalizes_child_claim_transport_before_write(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")

        def remember(self, content: str, **kwargs: object) -> SimpleNamespace:
            captured["content"] = content
            captured["child_claims"] = kwargs["child_claims"]
            return SimpleNamespace(
                saved=True,
                memory_id="canonical-turn",
                backend="mempalace",
                raw={},
            )

    transport = {
        "claim_type": "assistant_outcome",
        "source_role": "assistant",
        "content": "The routed write completed.",
        "parent_session_id": "session-alpha",
        "origin_segments": [
            {
                "claim_id": "session-alpha:assistant-outcome",
                "origin": {
                    "channel": "host_event",
                    "principal_id": "hermes",
                    "integrity_class": "reported",
                },
                "integrity_ceiling": "reported",
                "independence_domain": "hermes:session-alpha",
                "evidence_refs": [
                    {
                        "ref_type": "artifact",
                        "ref_id": "hermes-session:session-alpha",
                        "produced_by": "agent",
                        "producer_id": "hermes:session-alpha",
                    }
                ],
            }
        ],
        "temperature": "warm",
        "tags": ["outcome"],
    }
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: Client())
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_remember(
        "turn",
        workspace_id="alpha",
        requested_block_id="workspace_episodes",
        child_claims=[transport],
    )

    assert payload["ok"] is True
    claims = captured["child_claims"]
    (claim,) = claims
    assert claim.__class__.__name__ == "TurnChildClaim"
    assert claim.parent_session_id == "session-alpha"
    assert claim.temperature == "warm"


def test_mcp_facade_native_recall_at_limit_avoids_full_export_scan() -> None:
    native = [_record(f"native-{index}", f"parser native fact {index}") for index in range(3)]

    class Backend:
        export_calls = 0

        def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
            return native

        def export_blocks(self) -> list[MemoryBlockRecord]:
            self.export_calls += 1
            raise AssertionError("a complete native recall must not trigger an O(N) export scan")

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")
        backend = Backend()

    records, diagnostics = mcp_facade._recall_records(
        Client(),
        query="parser native fact",
        limit=3,
        tags=[],
        source_horizon="archive",
        scope_chain=("workspace:alpha",),
        max_tokens=512,
    )

    assert [record.block_id for record in records] == [
        "native-0",
        "native-1",
        "native-2",
    ]
    assert Client.backend.export_calls == 0
    assert "index_used" not in diagnostics
    assert "index_scan_reason" not in diagnostics


def test_mcp_facade_payload_budget_includes_metadata_and_transport_fields() -> None:
    record = replace(
        _record("bounded", "parser " + ("important fact " * 200)),
        metadata={
            "large_backend_debug_blob": "x" * 20_000,
        },
    )

    packed = mcp_facade._packed_payload_from_records([record], max_tokens=180)
    items = list(packed.items)

    assert items
    assert items[0]["metadata"]["block_id"] == "bounded"
    assert items[0]["truncated"] is True
    assert packed.estimated_tokens <= 180


def test_mcp_facade_block_plan_retrieves_routed_native_memory(monkeypatch, tmp_path: Path) -> None:
    record = replace(
        _record("plain-memory", "Plain remembered parser detail."),
        block_membership=RecordBlockMembership(
            primary_block_id="stable_project_facts",
            catalog_revision_id="catalog:test",
        ),
    )

    class Backend:
        def __init__(self) -> None:
            self.queries: list[dict[str, Any]] = []

        def get_block(self, block_id: str) -> MemoryBlockRecord | None:
            return record if block_id == record.block_id else None

        def export_blocks(self) -> list[MemoryBlockRecord]:
            return []

        def recall_evidence(self, query: dict[str, Any]) -> list[MemoryHit]:
            payload = dict(query)
            self.queries.append(payload)
            if payload.get("logical_block_ids") != ["stable_project_facts"] or payload.get("tags"):
                return []
            return [
                MemoryHit(
                    text=record.content,
                    source=record.block_id,
                    score=0.9,
                    metadata=block_record_to_memory_hit(record).metadata,
                )
            ]

    backend = Backend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_build_context(
        query="plain parser detail",
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        include_candidates=True,
        logical_block_ids=["stable_project_facts"],
        max_tokens=256,
    )

    assert payload["ok"] is True
    assert [item["metadata"]["block_id"] for item in payload["items"]] == ["plain-memory"]


def test_mcp_facade_scope_chain_excludes_stale_session_context() -> None:
    old_session = replace(
        _record("old-session", "shared session marker from old session"),
        scope="session",
        session_id="session-old",
        delivery_policy="working_set",
        block_membership=RecordBlockMembership(
            primary_block_id="execution_working_state",
            catalog_revision_id="catalog:test",
        ),
    )
    current_session = replace(
        _record("current-session", "shared session marker from current session"),
        scope="session",
        session_id="session-current",
        delivery_policy="working_set",
        block_membership=RecordBlockMembership(
            primary_block_id="execution_working_state",
            catalog_revision_id="catalog:test",
        ),
    )
    scope_chain = ("session:session-current", "workspace:alpha")

    assert not mcp_facade._record_allowed_for_context(
        old_session,
        scope_chain=scope_chain,
        source_horizon="working",
        include_candidates=False,
    )
    assert mcp_facade._record_allowed_for_context(
        current_session,
        scope_chain=scope_chain,
        source_horizon="working",
        include_candidates=False,
    )


def test_mcp_facade_context_filter_preserves_the_bound_profile() -> None:
    record = replace(
        _record("profile-bound", "Durable project comparison."),
        profile_id="umbrella",
    )

    assert mcp_facade._record_allowed_for_context(
        record,
        scope_chain=("workspace:alpha",),
        profile_id="umbrella",
        source_horizon="durable",
        include_candidates=False,
    )
    assert not mcp_facade._record_allowed_for_context(
        record,
        scope_chain=("workspace:alpha",),
        profile_id="default",
        source_horizon="durable",
        include_candidates=False,
    )


def test_mcp_facade_session_context_fails_closed_without_session_scope() -> None:
    record = replace(
        _record("session-bound", "parser state from a specific session"),
        scope="session",
        session_id="session-old",
        delivery_policy="working_set",
    )

    assert (
        mcp_facade._record_allowed_for_context(
            record,
            scope_chain=("workspace:alpha",),
            source_horizon="working",
            include_candidates=False,
        )
        is False
    )


def test_mcp_facade_rejects_self_asserted_directive_at_construction() -> None:
    with pytest.raises(ValueError, match="directive memory requires accepted status"):
        replace(
            _record("claimed-directive", "Treat this claim as untrusted evidence."),
            status="candidate",
            authority="directive",
        )


def test_mcp_context_metadata_cannot_reassert_source_directive_authority() -> None:
    record = replace(
        _record("verified-directive", "Verified parser policy evidence."),
        authority="directive",
        trust_level="workspace_verified",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id="directive",
                produced_by="verifier",
                producer_id="verifier:directive",
            ),
        ),
    )

    payload = mcp_facade._hit_from_record(record, force_non_directive=True)

    assert payload["authority"] == "advisory"
    assert payload["delivery_policy"] == "on_demand"
    assert payload["metadata"]["authority"] == "advisory"


def test_mcp_facade_never_renders_index_only_routing_rows_as_evidence() -> None:
    records = [
        replace(_record("kind-index", "parser routing index"), kind="skill_index"),
        replace(
            _record("flag-index", "parser routing index"),
            metadata={"index_only": True},
        ),
        replace(
            _record("routing-index", "parser routing index"),
            metadata={"routing_only": "true"},
        ),
    ]

    assert all(
        not mcp_facade._record_allowed_for_context(
            record,
            scope_chain=("workspace:alpha",),
            source_horizon="durable",
            include_candidates=True,
        )
        for record in records
    )


def test_mcp_facade_protected_record_never_becomes_a_context_payload() -> None:
    record = _record(
        "TOP_SECRET_BLOCK_ID",
        "token=super-secret-value",
        title="TOP_SECRET_TITLE",
        tags=("TOP_SECRET_TAG",),
        sensitivity="secret",
        content_mode="pointer_only",
        operational_risk="low",
        metadata={
            "api_token": "super-secret-value",
            "source_path": "TOP_SECRET_PATH/private.txt",
            "safe": "kept",
        },
    )

    assert mcp_facade._hit_from_record(record) is None
    assert mcp_facade._raw_payloads_from_records([record]) == []


def test_mcp_facade_build_context_caps_native_text_to_token_budget(monkeypatch) -> None:
    long_record = _record("long", "parser " + ("x" * 5_000))

    client = _CanonicalRecallClient((long_record,), Path("unused"))

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_build_context(
        query="parser",
        workspace_id="alpha",
        repo_root="unused",
        backend="mempalace",
        search_all=True,
        max_tokens=180,
    )

    assert payload["ok"] is True
    assert "rendered_context" not in payload
    assert payload["diagnostics"]["context_payload_estimated_tokens"] <= 180
    assert payload["items"][0]["truncated"] is True
    assert payload["diagnostics"]["truncated_item_count"] == 1
    assert payload["diagnostics"]["context_token_budget"] == 180


def test_mcp_facade_build_context_spreads_budget_across_records(monkeypatch) -> None:
    first = _record("zfirst", "parser first " + ("x" * 200))
    second = _record("asecond", "parser second-important " + ("y" * 20))

    client = _CanonicalRecallClient((first, second), Path("unused"))

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_build_context(
        query="parser",
        workspace_id="alpha",
        repo_root="unused",
        backend="mempalace",
        search_all=True,
        max_tokens=260,
    )

    assert payload["ok"] is True
    assert len(payload["items"]) == 2
    assert any("second-important" in item["text"] for item in payload["items"])


def test_build_context_filters_typed_source_speaker_after_scoped_native_recall(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assistant_record = replace(
        _record("assistant-rec", "Assistant suggested a five-hour game"),
        metadata={"source_segments": [{"speaker": "assistant"}]},
    )
    user_record = replace(
        _record("user-rec", "I played the game for five hours"),
        metadata={"source_segments": [{"speaker": "user"}]},
    )
    client = SimpleNamespace(workspace_id="alpha", repo_root=tmp_path)
    captured = {}
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)

    def scoped(*_args, **kwargs):
        captured["limit"] = kwargs["limit"]
        return [assistant_record, user_record], {"allocated_context_token_budget": 256}

    monkeypatch.setattr(mcp_facade, "_recall_records_scoped", scoped)
    monkeypatch.setattr(mcp_facade, "_record_packed_decision", lambda *_args, **_kwargs: None)
    payload = lethyx_build_context(
        query="game hours",
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        search_all=True,
        source_speakers=["user"],
        limit=1,
        max_tokens=256,
    )

    assert payload["ok"] is True
    assert captured["limit"] == 64
    assert len(payload["items"]) == 1
    assert payload["items"][0]["source"].endswith("user-rec")
    assert payload["diagnostics"]["source_speaker_filter_stage"] == "after_native_candidate_limit"


def test_mcp_facade_preserves_native_hit_score_when_refetching_block(monkeypatch) -> None:
    from lethyx.memory.backends.base import MemoryHit

    record = _record("ranked", "parser ranked native fact")

    class Backend:
        def get_block(self, block_id: str):
            return record if block_id == "ranked" else None

        def export_blocks(self):
            return []

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")
        backend = Backend()

        def recall(self, query: str, *, recall_plan):
            assert recall_plan.query == query
            return [
                MemoryHit(
                    text="parser ranked native fact",
                    source="native:ranked",
                    score=0.42,
                    metadata={
                        **block_record_to_memory_hit(record).metadata,
                        "native_score": 0.42,
                    },
                )
            ]

        def compile_recall_plan(self, query: str, **kwargs: object):
            planner = MemoryClient(self.repo_root, workspace_id=self.workspace_id)
            return planner.compile_recall_plan(query, **kwargs)

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: Client())
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_build_context(
        query="parser",
        workspace_id="alpha",
        repo_root="unused",
        backend="mempalace",
        search_all=True,
    )

    assert payload["items"][0]["score"] == 0.42
    assert payload["items"][0]["metadata"]["retrieval_score"] == 0.42


def test_mcp_recall_and_context_fail_closed_on_native_exception(monkeypatch) -> None:
    class Backend:
        @staticmethod
        def get_block(_block_id: str):
            return None

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")
        backend = Backend()

        @staticmethod
        def recall(_query: str, *, recall_plan):
            del recall_plan
            raise OSError("native transport unavailable")

        def compile_recall_plan(self, query: str, **kwargs: object):
            planner = MemoryClient(self.repo_root, workspace_id=self.workspace_id)
            return planner.compile_recall_plan(query, **kwargs)

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: Client())
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    recalled = mcp_facade.lethyx_recall(
        query="native failure",
        workspace_id="alpha",
        repo_root="unused",
        backend="mempalace",
        search_all=True,
    )
    context = lethyx_build_context(
        query="native failure",
        workspace_id="alpha",
        repo_root="unused",
        backend="mempalace",
        search_all=True,
        render_context=True,
    )

    assert recalled["ok"] is False
    assert recalled["hits"] == []
    assert "OSError: native transport unavailable" in recalled["error"]
    assert context["ok"] is False
    assert context["items"] == []
    assert context["rendered_context"] == ""
    assert "OSError: native transport unavailable" in context["error"]


def test_mcp_facade_preserves_backend_order_for_negative_question() -> None:
    subject = _record(
        "api_tasks",
        "The synthetic_fastapi_tasks repository is a FastAPI task tracker.",
    )
    wrong_report = _record(
        "typer_reports",
        "The synthetic_typer_reports repository uses reportkit.",
    )
    wrong_dashboard = _record(
        "vite_dashboard",
        "The synthetic_vite_dashboard repository uses Zustand.",
    )

    class Backend:
        def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
            return [wrong_report, wrong_dashboard, subject]

        def export_blocks(self) -> list[MemoryBlockRecord]:
            return []

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")
        backend = Backend()

    records, diagnostics = mcp_facade._recall_records(
        Client(),
        query="Does synthetic_fastapi_tasks use reportkit or Zustand?",
        limit=5,
        tags=[],
        source_horizon="archive",
        scope_chain=("workspace:alpha",),
        max_tokens=4096,
    )

    assert diagnostics["native_count"] == 3
    assert [record.block_id for record in records] == [
        "typer_reports",
        "vite_dashboard",
        "api_tasks",
    ]


def test_mcp_facade_returns_explicit_selected_record_id() -> None:
    selected = _record(
        "selected-evidence",
        "Explicitly selected evidence whose wording does not repeat the question subject.",
    )

    class Backend:
        def get_block(self, block_id: str) -> MemoryBlockRecord | None:
            return selected if block_id == selected.block_id else None

        def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
            del query
            return []

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")
        backend = Backend()

    records, diagnostics = mcp_facade._recall_records(
        Client(),
        query="Does synthetic_fastapi_tasks use reportkit or Zustand?",
        limit=1,
        tags=[],
        source_horizon="archive",
        scope_chain=("workspace:alpha",),
        record_ids=[selected.block_id],
    )

    assert [record.block_id for record in records] == [selected.block_id]
    assert "index_scan_reason" not in diagnostics


def test_mcp_facade_preserves_native_hit_without_post_backend_filtering() -> None:
    native = _record("native-vector-hit", "Native backend returned this exact vector hit.")

    class Backend:
        def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
            return [native]

        def export_blocks(self) -> list[MemoryBlockRecord]:
            return []

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")
        backend = Backend()

    records, diagnostics = mcp_facade._recall_records(
        Client(),
        query="source-card target symbols absent from lexical text but native recall found it",
        limit=5,
        tags=[],
        source_horizon="archive",
        scope_chain=("workspace:alpha",),
        max_tokens=4096,
    )

    assert [record.block_id for record in records] == ["native-vector-hit"]
    assert diagnostics["native_count_before_filter"] == 1
    assert diagnostics["native_count_after_safety"] == 1


def test_mcp_facade_keeps_core_selected_order_without_secondary_relevance_cut(monkeypatch) -> None:
    query = (
        "Final synthesis: audit langchain-core from subject evidence only, including "
        "version, Python range, runtime deps, and whether that evidence says it directly "
        "depends on OpenAI or Anthropic."
    )
    core = replace(
        _record(
            "pyproject-core",
            "Project name: langchain-core. Version: 1.4.8. Runtime dependencies: "
            "langsmith, pydantic, uuid-utils.",
        ),
        title="LangChain core project manifest",
        block_membership=RecordBlockMembership(
            primary_block_id="stable_project_facts",
            catalog_revision_id="catalog:test",
        ),
    )
    openai = replace(
        _record(
            "pyproject-openai",
            "Project name: langchain-openai. Runtime dependencies: langchain-core, openai.",
        ),
        title="OpenAI partner project manifest",
        block_membership=RecordBlockMembership(
            primary_block_id="stable_project_facts",
            catalog_revision_id="catalog:test",
        ),
    )
    anthropic = replace(
        _record(
            "pyproject-anthropic",
            "Project name: langchain-anthropic. Runtime dependencies: langchain-core, anthropic.",
        ),
        title="Anthropic partner project manifest",
        block_membership=RecordBlockMembership(
            primary_block_id="stable_project_facts",
            catalog_revision_id="catalog:test",
        ),
    )

    client = _CanonicalRecallClient((openai, anthropic, core), Path("unused"))

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_build_context(
        query=query,
        workspace_id="alpha",
        repo_root="unused",
        backend="mempalace",
        limit=3,
        max_tokens=512,
        logical_block_ids=["stable_project_facts"],
        render_context=True,
    )

    assert payload["ok"] is True
    assert payload["diagnostics"]["candidate_record_ids"] == [
        "pyproject-openai",
        "pyproject-anthropic",
        "pyproject-core",
    ]
    assert payload["diagnostics"]["selected_record_ids"] == [
        "pyproject-openai",
        "pyproject-anthropic",
        "pyproject-core",
    ]
    assert payload["diagnostics"]["rendered_block_ids"] == [
        "pyproject-openai",
        "pyproject-anthropic",
        "pyproject-core",
    ]
    assert "Project name: langchain-core" in payload["rendered_context"]
    assert "Project name: langchain-openai" in payload["rendered_context"]
    assert "Project name: langchain-anthropic" in payload["rendered_context"]


def test_mcp_facade_compact_source_card_header_survives_tiny_budget() -> None:
    record = _record(
        "source_agents",
        "LangChain source file card for libs/core/langchain_core/agents.py.\n"
        "Classes: AgentAction, AgentStep.\n"
        "Functions: format_log_to_str, parse_agent_output.\n"
        "Imports/modules: langchain_core.messages, pydantic.\n"
        "Excerpt:\n" + ("very long body " * 80),
        metadata={
            "source_path": "libs/core/langchain_core/agents.py",
            "classes": ["AgentAction", "AgentStep"],
            "functions": ["format_log_to_str", "parse_agent_output"],
            "imports": ["langchain_core.messages", "pydantic"],
        },
    )

    payload = mcp_facade._packed_payload_from_records([record], max_tokens=180).items[0]

    assert "Path: libs/core/langchain_core/agents.py" in payload["text"]
    assert "Classes: AgentAction, AgentStep" in payload["text"]


def test_mcp_facade_rendered_context_emits_delivery_trace(monkeypatch, tmp_path: Path) -> None:
    record = replace(
        _record("delivery-block", "Parser delivery evidence survives final packing."),
        metadata={
            "native_logical_block_id": "workspace_episodes",
            "native_rank": 1,
            "native_score": 0.7,
            "candidate_pool_ids": ["delivery-block", "raw-native-candidate-id"],
            "selected_record_ids": ["delivery-block"],
            "selection_mode": "block_rank_round",
        },
    )

    client = _CanonicalRecallClient((record,), tmp_path)

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    unrendered_payload = lethyx_build_context(
        query="parser delivery evidence",
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        search_all=True,
        max_tokens=256,
    )
    unrendered_rows = read_memory_events(tmp_path)
    payload = lethyx_build_context(
        query="parser delivery evidence",
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        search_all=True,
        max_tokens=256,
        render_context=True,
    )

    rows = read_memory_events(tmp_path)

    assert payload["ok"] is True
    assert "rendered_context" not in unrendered_payload
    assert unrendered_rows == []
    assert payload["diagnostics"]["rendered_block_ids"] == ["delivery-block"]
    assert payload["diagnostics"]["native_query_trace"] == {
        "entry_count": 1,
        "candidate_record_count": 2,
        "selected_record_count": 1,
    }
    assert [(row["event_type"], row["decision"], row["outcome"]) for row in rows] == [
        ("context_delivery", "delivery", "rendered")
    ]
    assert rows[0]["candidate_count"] == 1
    assert rows[0]["selected_id_hashes"]
    trace_text = (tmp_path / ".Lethyx" / "memory" / "telemetry" / "memory_events.jsonl").read_text(
        encoding="utf-8"
    )
    assert "delivery-block" not in trace_text
    assert "raw-native-candidate-id" not in trace_text
    assert "candidate_id_hashes" in trace_text


def test_mcp_facade_remember_emits_content_free_capture_trace(monkeypatch, tmp_path: Path) -> None:
    content = "raw-capture-content-must-not-reach-telemetry"
    title = "Private capture title"
    captured: dict[str, object] = {}

    class Client:
        workspace_id = "alpha"
        repo_root = tmp_path

        def remember(self, received_content: str, **kwargs):
            assert received_content == content
            assert kwargs["title"] == title
            captured["evidence_refs"] = kwargs["evidence_refs"]
            return SimpleNamespace(
                saved=True,
                memory_id="capture-memory-1",
                backend="mempalace",
                raw={"backend_receipt": "ok"},
            )

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: Client())
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_remember(
        content=content,
        title=title,
        workspace_id="alpha",
        backend="mempalace",
        repo_root=str(tmp_path),
        requested_block_id="workspace_episodes",
        run_id="run-1",
        evidence_refs=["mcp-remember:capture"],
    )

    rows = read_memory_events(tmp_path)
    trace_text = (tmp_path / ".Lethyx" / "memory" / "telemetry" / "memory_events.jsonl").read_text(
        encoding="utf-8"
    )

    assert payload["saved"] is True
    assert [(row["event_type"], row["decision"], row["outcome"]) for row in rows] == [
        ("memory_capture", "write", "committed")
    ]
    assert rows[0]["selected_id_hashes"]
    assert content not in trace_text
    assert title not in trace_text
    evidence = captured["evidence_refs"]
    assert isinstance(evidence, list)
    assert [(ref.ref_type, ref.ref_id, ref.produced_by) for ref in evidence] == [
        ("session_event", "mcp-remember:capture", "harness")
    ]


def test_mcp_facade_recall_emits_content_free_event(monkeypatch, tmp_path: Path) -> None:
    record = _record("recall-block", "Explicit recall evidence for parser.")

    client = _CanonicalRecallClient((record,), tmp_path)

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_recall(
        query="parser recall evidence",
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        search_all=True,
        max_tokens=256,
    )

    rows = read_memory_events(tmp_path)
    trace_text = (tmp_path / ".Lethyx" / "memory" / "telemetry" / "memory_events.jsonl").read_text(
        encoding="utf-8"
    )

    assert payload["ok"] is True
    assert [item["metadata"]["block_id"] for item in payload["hits"]] == ["recall-block"]
    assert [(row["event_type"], row["decision"], row["outcome"]) for row in rows] == [
        ("memory_recall", "call", "returned")
    ]
    assert rows[0]["candidate_count"] == 1
    assert rows[0]["selected_id_hashes"]
    assert "recall-block" not in trace_text


def test_mcp_recall_compacts_internal_native_trace_before_returning_to_agent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    record = _record("selected-record", "Useful bounded project memory.")
    internal_ids = [f"internal-record-{index}" for index in range(200)]
    diagnostics = {
        "allocated_context_token_budget": 256,
        "native_query_trace": [
            {
                "record_id": record.block_id,
                "native_logical_block_id": "project-memory",
                "native_rank": 1,
                "native_score": 0.9,
                "candidate_pool_ids": internal_ids,
                "selected_record_ids": [record.block_id],
                "selection_mode": "native",
            }
            for _ in range(20)
        ],
    }
    client = SimpleNamespace(workspace_id="alpha", repo_root=tmp_path)
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)
    monkeypatch.setattr(
        mcp_facade,
        "_recall_records_scoped",
        lambda *_args, **_kwargs: ([record], dict(diagnostics)),
    )
    monkeypatch.setattr(mcp_facade, "_record_packed_decision", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mcp_facade, "record_memory_event", lambda *_args, **_kwargs: None)

    payload = lethyx_recall(
        query="project memory",
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        logical_block_ids=["project-memory"],
        max_tokens=256,
    )
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["ok"] is True
    assert payload["diagnostics"]["native_query_trace"] == {
        "entry_count": 20,
        "candidate_record_count": 200,
        "selected_record_count": 1,
    }
    assert "internal-record-199" not in serialized
    assert len(serialized) < 12_000


def test_mcp_server_exposes_expected_lethyx_tools() -> None:
    server = create_mcp_server()

    assert server is not None
    assert set(TOOL_NAMES) == {
        "health",
        "block_map",
        "expand_block",
        "navigation_envelope",
        "create_block",
        "amend_block",
        "move_block",
        "archive_block",
        "revive_block",
        "delete_block",
        "split_block",
        "merge_blocks",
        "pending_topology_evolutions",
        "resume_topology_evolution",
        "abort_topology_evolution",
        "add_block_edge",
        "remove_block_edge",
        "get_active_mask",
        "set_active_mask",
        "amend_active_mask",
        "transition_active_mask",
        "resolve_active_mask",
        "observe_maintenance",
        "consolidate_observations",
        "remember",
        "amend",
        "recall",
        "procedure",
        "memory_map",
        "recent",
        "read_exact_record",
        "read_exact_records",
        "list_records",
        "operation_status",
        "projection_barrier",
        "read_working_set",
        "build_context",
        "forget",
        "add_edge",
        "export_graph",
        "export_obsidian",
    }


def test_mcp_facade_exports_scoped_graph(tmp_path: Path) -> None:
    repo_root = _repo(tmp_path)
    saved = lethyx_remember(
        content="Graph export preserves this scoped evidence.",
        title="Graph Export Evidence",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        requested_block_id="stable_project_facts",
    )

    graph = lethyx_export_graph(
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
    )

    assert saved["ok"] is True
    assert graph["ok"] is True
    assert graph["workspace_id"] == "alpha"
    assert any(block["block_id"] == saved["memory_id"] for block in graph["blocks"])


def test_mcp_module_stdio_matches_python_and_authenticated_loopback_transport(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    identity = build_local_service_identity(
        repo_root=repo_root,
        workspace_id="stdio-alpha",
        backend="mempalace",
        caller_role="operator",
    )
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment.update(
        {
            "LETHYX_REPO_ROOT": str(repo_root),
            "LETHYX_WORKSPACE": identity.workspace_id,
            "LETHYX_BACKEND": identity.backend,
            "LETHYX_SESSION_ID": "mcp-stdio-session",
            "PYTHONPATH": os.pathsep.join(
                value for value in (str(source_root), environment.get("PYTHONPATH", "")) if value
            ),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    pool = LocalServiceClientPool()
    try:
        python_remember = pool.call(
            identity,
            "remember",
            {
                "content": "Python transport evidence for the shared listener.",
                "title": "Python transport",
                "requested_block_id": "workspace_episodes",
            },
            timeout_seconds=20,
            session_id="transport-session",
        )
        endpoint = local_service.local_service_registry(identity)
        assert endpoint is not None
        lease_response = local_service._local_service_http_request(
            endpoint,
            "/lease",
            {
                "protocol_version": identity.protocol_version,
                "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
                "api_version": LOCAL_SERVICE_API_VERSION,
                "request_id": "raw-lease",
                "caller_role": identity.caller_role,
                "session_id": "transport-session",
                "capabilities": sorted(
                    local_service._LOCAL_SERVICE_ROLE_TOOLS[identity.caller_role]
                ),
                "bootstrap_secret": local_service_bootstrap_secret(identity),
            },
            timeout_seconds=10,
        )
        lease = lease_response.get("lease")
        assert isinstance(lease, str)
        raw_remember = local_service._local_service_http_request(
            endpoint,
            "/call",
            {
                "protocol_version": identity.protocol_version,
                "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
                "api_version": LOCAL_SERVICE_API_VERSION,
                "request_id": "raw-remember",
                "lease": lease,
                "caller_role": identity.caller_role,
                "session_id": "transport-session",
                "tool": "remember",
                "arguments": {
                    "content": "Loopback HTTP transport evidence for the shared listener.",
                    "title": "Raw transport",
                    "requested_block_id": "workspace_episodes",
                },
            },
            timeout_seconds=20,
        )

        async def stdio_calls() -> tuple[set[str], dict[str, object], dict[str, object]]:
            parameters = StdioServerParameters(
                command=sys.executable,
                args=["-m", "lethyx.memory.mcp_server"],
                cwd=repo_root,
                env=environment,
            )
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    remembered = await session.call_tool(
                        "remember",
                        {
                            "content": "MCP stdio transport evidence for the shared listener.",
                            "title": "MCP transport",
                            "requested_block_id": "workspace_episodes",
                        },
                    )
                    recalled = await session.call_tool(
                        "recall",
                        {
                            "query": "transport evidence",
                            "logical_block_ids": ["workspace_episodes"],
                            "limit": 10,
                        },
                    )
            return (
                {tool.name for tool in tools.tools},
                _mcp_result_payload(remembered),
                _mcp_result_payload(recalled),
            )

        tool_names, mcp_remember, mcp_recall = asyncio.run(stdio_calls())
        python_recall = pool.call(
            identity,
            "recall",
            {
                "query": "transport evidence",
                "logical_block_ids": ["workspace_episodes"],
                "limit": 10,
            },
            timeout_seconds=20,
            session_id="transport-session",
        )
        raw_recall = local_service._local_service_http_request(
            endpoint,
            "/call",
            {
                "protocol_version": identity.protocol_version,
                "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
                "api_version": LOCAL_SERVICE_API_VERSION,
                "request_id": "raw-recall",
                "lease": lease,
                "caller_role": identity.caller_role,
                "session_id": "transport-session",
                "tool": "recall",
                "arguments": {
                    "query": "transport evidence",
                    "logical_block_ids": ["workspace_episodes"],
                    "limit": 10,
                },
            },
            timeout_seconds=20,
        )
        service = pool.service_status(identity)
    finally:
        pool.close()
        shutdown_local_service_listener(identity)

    assert set(TOOL_NAMES) <= tool_names
    payloads = {
        "python_remember": python_remember,
        "raw_remember": raw_remember,
        "mcp_remember": mcp_remember,
        "python_recall": python_recall,
        "raw_recall": raw_recall,
        "mcp_recall": mcp_recall,
    }
    assert all(payload["ok"] is True for payload in payloads.values()), payloads
    assert all(
        payload["workspace_id"] == identity.workspace_id and payload["backend"] == identity.backend
        for payload in payloads.values()
    )
    assert service["transport"] == "loopback_http"
    assert service["listener_pid"] > 0
    graph = lethyx_export_graph(
        workspace_id=identity.workspace_id,
        backend=identity.backend,
        repo_root=str(identity.repo_root),
    )
    mcp_record = next(
        block for block in graph["blocks"] if block["block_id"] == mcp_remember["memory_id"]
    )
    assert mcp_record["session_id"] == "mcp-stdio-session"


def _mcp_result_payload(result: object) -> dict[str, object]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return dict(structured)
    content = getattr(result, "content", ())
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
    raise AssertionError("MCP result does not contain an object payload")


def test_mcp_facade_remember_recall_and_workspace_isolation(tmp_path: Path) -> None:
    repo_root = _repo(tmp_path)

    first = lethyx_remember(
        content="Alpha project keeps parser facts in Lethyx only.",
        title="Alpha Parser Fact",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        kind="project_fact",
        requested_block_id="stable_project_facts",
        tags=["parser"],
        evidence_refs=[
            {
                "ref_type": "test_run",
                "ref_id": "alpha-parser-fact",
                "produced_by": "verifier",
                "producer_id": "pytest:alpha-parser-fact",
            }
        ],
    )
    second = lethyx_remember(
        content="Beta project must not leak into alpha recall.",
        title="Beta Isolation Fact",
        workspace_id="beta",
        backend="mempalace",
        repo_root=repo_root,
        kind="project_fact",
        requested_block_id="stable_project_facts",
        tags=["parser"],
        evidence_refs=[
            {
                "ref_type": "test_run",
                "ref_id": "beta-parser-fact",
                "produced_by": "verifier",
                "producer_id": "pytest:beta-parser-fact",
            }
        ],
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert first["workspace_id"] == "alpha"

    alpha_hits = lethyx_recall(
        query="",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        limit=10,
        include_candidates=True,
        logical_block_ids=["stable_project_facts"],
    )

    rendered = "\n".join(hit["text"] for hit in alpha_hits["hits"])
    assert "Alpha project keeps parser facts" in rendered
    assert "Beta project must not leak" not in rendered
    assert all(hit["authority"] == "advisory" for hit in alpha_hits["hits"])


def test_mcp_facade_scope_chain_never_opens_named_global_workspaces(tmp_path: Path) -> None:
    repo_root = _repo(tmp_path)
    project = lethyx_remember(
        content="Project-only GMAS fact about RoleGraph.",
        title="Project Fact",
        workspace_id="gmas_demo",
        backend="mempalace",
        repo_root=repo_root,
        kind="project_fact",
        requested_block_id="stable_project_facts",
        evidence_refs=[
            {
                "ref_type": "test_run",
                "ref_id": "project-fact",
                "produced_by": "verifier",
                "producer_id": "pytest:project-fact",
            }
        ],
    )
    global_memory = lethyx_remember(
        content="Global operator fact about concise status.",
        title="Global Fact",
        workspace_id="global",
        backend="mempalace",
        repo_root=repo_root,
        kind="project_fact",
        requested_block_id="stable_project_facts",
        evidence_refs=[
            {
                "ref_type": "test_run",
                "ref_id": "global-fact",
                "produced_by": "verifier",
                "producer_id": "pytest:global-fact",
            }
        ],
    )
    shared_memory = lethyx_remember(
        content="Shared operator fact about concise status.",
        title="Shared Fact",
        workspace_id="shared",
        backend="mempalace",
        repo_root=repo_root,
        kind="project_fact",
        requested_block_id="stable_project_facts",
        evidence_refs=[
            {
                "ref_type": "test_run",
                "ref_id": "shared-fact",
                "produced_by": "verifier",
                "producer_id": "pytest:shared-fact",
            }
        ],
    )
    scoped_hits = lethyx_recall(
        query="fact",
        workspace_id="gmas_demo",
        user_id="user-1",
        backend="mempalace",
        repo_root=repo_root,
        scope_chain=["workspace:gmas_demo", "user_global:user-1"],
        search_all=True,
        record_ids=[
            project["memory_id"],
            global_memory["memory_id"],
            shared_memory["memory_id"],
        ],
        limit=10,
        include_candidates=True,
    )
    api_client = MemoryClient(
        repo_root,
        workspace_id="gmas_demo",
        user_id="user-1",
        backend="mempalace",
    )
    try:
        api_hits = api_client.recall(
            "fact",
            search_all=True,
            record_ids=[
                project["memory_id"],
                global_memory["memory_id"],
                shared_memory["memory_id"],
            ],
            scope_chain=["workspace:gmas_demo", "user_global:user-1"],
            include_candidates=True,
            limit=10,
        )
    finally:
        mcp_facade._close_client(api_client)

    scoped_ids = {hit["metadata"]["block_id"] for hit in scoped_hits["hits"]}
    api_ids = {hit.metadata["block_id"] for hit in api_hits}
    foreign_ids = {global_memory["memory_id"], shared_memory["memory_id"]}
    assert scoped_hits["ok"] is True
    assert scoped_hits["scope_chain"] == ["workspace:gmas_demo", "user_global:user-1"]
    assert scoped_ids == {project["memory_id"]}
    assert api_ids == {project["memory_id"]}
    assert foreign_ids.isdisjoint(scoped_ids | api_ids)

    wildcard = lethyx_recall(
        query="fact",
        workspace_id="gmas_demo",
        user_id="user-1",
        backend="mempalace",
        repo_root=repo_root,
        scope_chain=["workspace:gmas_demo", "user_global:*"],
    )
    assert wildcard["ok"] is False
    assert "wildcard identity" in wildcard["error"]


def test_mcp_facade_source_horizon_never_renders_reflection_only_memory(tmp_path: Path) -> None:
    repo_root = _repo(tmp_path)
    needle = "active-passive-filter-proof"
    active = lethyx_remember(
        content=f"{needle} active memory should be returned by default.",
        title="Active Filter Fact",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        kind="project_fact",
        requested_block_id="stable_project_facts",
        evidence_refs=[
            {
                "ref_type": "test_run",
                "ref_id": "active-filter",
                "produced_by": "verifier",
                "producer_id": "pytest:active-filter",
            }
        ],
    )
    passive = lethyx_remember(
        content=f"{needle} reflection-only memory must never be rendered.",
        title="Passive Filter Fact",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        kind="idea",
        requested_block_id="project_ideas",
    )
    assert active["ok"] is True
    assert passive["ok"] is True

    default_context = lethyx_build_context(
        query=needle,
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        limit=10,
        include_candidates=True,
        logical_block_ids=["stable_project_facts", "project_ideas"],
    )
    default_text = "\n".join(item["text"] for item in default_context["items"])
    assert "active memory should be returned" in default_text
    assert "reflection-only memory must never" not in default_text

    archive_context = lethyx_build_context(
        query=needle,
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        limit=10,
        source_horizon="archive",
        include_candidates=True,
        logical_block_ids=["stable_project_facts", "project_ideas"],
    )
    archive_text = "\n".join(item["text"] for item in archive_context["items"])
    assert archive_context["diagnostics"]["source_horizon"] == "archive"
    assert "active memory should be returned" in archive_text
    assert "reflection-only memory must never" not in archive_text


def test_mcp_facade_build_context_filters_retracted_memory(tmp_path: Path) -> None:
    repo_root = _repo(tmp_path)
    saved = lethyx_remember(
        content="Retracted content should not be returned to an agent.",
        title="Retracted Fact",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        requested_block_id="stable_project_facts",
    )

    forgotten = lethyx_forget(
        memory_id=saved["memory_id"],
        reason="test retraction",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
    )
    assert forgotten["ok"] is True

    context = lethyx_build_context(
        query="",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        limit=5,
        search_all=True,
    )

    assert context["delivery_policy"] == "on_demand"
    assert context["authority"] == "advisory"
    assert "Retracted content" not in "\n".join(item["text"] for item in context["items"])


def test_mcp_facade_promote_requires_evidence_and_can_export_obsidian(
    tmp_path: Path, candidate_semantic_report
) -> None:
    repo_root = _repo(tmp_path)
    saved = lethyx_remember(
        content="Verified memory can be promoted and exported.",
        title="Verified Export Fact",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        requested_block_id="stable_project_facts",
    )

    client = MemoryClient(repo_root, workspace_id="alpha", backend="mempalace")
    report = candidate_semantic_report(client, saved["memory_id"])
    revision = client.backend.get_block(saved["memory_id"]).revision_id
    client.backend.close()
    promoted = lethyx_promote(
        memory_id=saved["memory_id"],
        expected_source_revision_id=revision,
        reason="test promotion",
        evidence_refs=[asdict(report)],
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
    )
    assert promoted["ok"] is True

    vault = tmp_path / "vault"
    exported = lethyx_export_obsidian(
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        vault_dir=str(vault),
        project_title="Alpha Memory",
    )

    assert exported["ok"] is True
    assert (vault / "Lethyx" / "Projects" / "alpha" / "Index.md").exists()
    assert (vault / "Lethyx" / "Projects" / "alpha" / "Memory Overview.md").exists()


@pytest.mark.parametrize("selection", ["empty", "pinned", "active", "stale"])
def test_selected_grant_never_prefetches_an_unselected_execution_record(
    monkeypatch,
    tmp_path: Path,
    selection: str,
) -> None:
    records = {
        key: replace(
            _record(key, text),
            block_membership=RecordBlockMembership(
                primary_block_id="execution",
                catalog_revision_id="catalog:test",
            ),
        )
        for key, text in (
            ("useful", "Measured accepted input 1200."),
            ("unrelated", "Lorenz discussion unrelated to the arithmetic task."),
        )
    }
    backend = SimpleNamespace(
        record_ids_for_logical_blocks=lambda *_args, **_kwargs: tuple(records),
        get_block=lambda key: records[key],
    )
    client = SimpleNamespace(
        backend=backend,
        workspace_id="alpha",
        profile_id="default",
        repo_root=tmp_path,
        resolve_active_mask=lambda **_kwargs: SimpleNamespace(
            trace=SimpleNamespace(
                resolution_status="stale" if selection == "stale" else "resolved",
                semantic_recall_performed=False,
            ),
            items=(SimpleNamespace(block_id="useful"), SimpleNamespace(block_id="foreign")),
        ),
    )
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)
    monkeypatch.setattr(mcp_facade, "record_memory_event", lambda *_args, **_kwargs: None)
    payload = lethyx_build_granted_context(
        workspace_id="alpha",
        repo_root=str(tmp_path),
        backend="mempalace",
        scope_chain=["workspace:alpha"],
        logical_block_ids=["execution"],
        record_ids=["useful"] if selection == "pinned" else [],
        active_mask_id="mask-1" if selection in {"active", "stale"} else "",
        active_mask_revision=3 if selection in {"active", "stale"} else None,
        max_tokens=256,
        render_context=True,
        selected_records_only=True,
    )
    assert payload["ok"] is True
    assert payload["diagnostics"]["rendered_record_ids"] == (
        ["useful"] if selection in {"pinned", "active"} else []
    )
    assert "Lorenz" not in payload.get("rendered_context", "")


@pytest.mark.parametrize("target", ["directive", "verified_lessons"])
def test_mcp_semantic_scope_preserves_existing_typed_promotion_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    candidate_semantic_report,
) -> None:
    from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex

    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda value: None)
    try:
        saved = client.remember(
            "Use the inspected parser entry point.",
            title="Reviewed parser boundary",
            kind="lesson" if target == "verified_lessons" else "fact",
            requested_block_id="reflection_candidates"
            if target == "verified_lessons"
            else "stable_project_facts",
        )
        assert saved.saved is True
        candidate = backend.get_block(saved.memory_id)
        assert candidate is not None
        report = candidate_semantic_report(
            client,
            candidate.block_id,
            authority="directive" if target == "directive" else "advisory",
            target_block_id="verified_lessons" if target == "verified_lessons" else "",
        )
        result = lethyx_promote(
            saved.memory_id,
            reason="Independent exact review accepted",
            evidence_refs=[asdict(report)],
            expected_source_revision_id=candidate.revision_id,
            authority="directive" if target == "directive" else "advisory",
            target_block_id="verified_lessons" if target == "verified_lessons" else "",
            workspace_id="alpha",
        )
        assert result["ok"] is True
        accepted = backend.get_block(result["block_id"])
        assert accepted is not None
        assert accepted.verification_status == "verified"
        assert accepted.authority == ("directive" if target == "directive" else "advisory")
        assert accepted.metadata["candidate_semantic_verification"]["report"]["passed"]
    finally:
        backend.close()
