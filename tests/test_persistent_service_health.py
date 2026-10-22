import json
from pathlib import Path

from um_mem.integrations import hermes_provider
from um_mem.memory import mcp_call
from um_mem.memory.backends.contracts import MemoryIndexNode
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.local_service import build_local_service_identity
from um_mem.memory.service_policy import projection_revision_id_for_snapshot


def test_backend_index_revision_changes_without_exposing_index_content(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "index.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    try:
        before = index.operational_revisions()
        index.retain_index_node(
            MemoryIndexNode(
                node_id="descriptor:release",
                title="Release descriptor",
                node_type="block_descriptor",
                workspace_id="alpha",
            )
        )
        after = index.operational_revisions()
    finally:
        index.close()

    assert before["schema_revision"] == after["schema_revision"]
    assert before["index_revision"] != after["index_revision"]
    assert set(after) == {
        "schema_revision",
        "index_revision",
        "canonical_snapshot_sha256",
    }
    assert after["canonical_snapshot_sha256"] == after["index_revision"]
    assert "release" not in after["index_revision"]


def test_listener_health_separates_state_and_all_revision_domains(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )

    class Backend:
        def health(self) -> dict[str, object]:
            return {
                "ok": True,
                "native_ok": True,
                "strict_ok": True,
                "schema_revision": "schema:7",
                "index_revision": "sha256:" + "a" * 64,
            }

    class Client:
        backend = Backend()

    runtime = mcp_call._LocalServiceListenerRuntime(identity, Client())  # type: ignore[arg-type]
    runtime.start()
    try:
        status = runtime.health()
    finally:
        runtime.close()

    assert status["liveness"] == "live"
    assert status["readiness"] == "ready"
    assert status["native_health"] == "healthy"
    assert status["degraded_state"] is None
    assert status["service_schema_revision"] == "local-service-schema:2"
    assert status["schema_revision"] == "schema:7"
    assert status["index_revision"] == "sha256:" + "a" * 64
    assert str(status["catalog_revision"]).startswith("catalog:")
    assert str(status["projection_revision"]).startswith("projection:")
    assert status["catalog_revision"] != status["projection_revision"]


def test_root_map_hook_reuses_one_client_pool_and_never_spawns_per_query(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pools: list[object] = []
    calls: list[str] = []

    class Pool:
        def __init__(self, **_kwargs: object) -> None:
            pools.append(self)

        def call(self, _identity: object, tool: str, _payload: object, **_kwargs: object):
            calls.append(tool)
            return {
                "ok": True,
                "backend": "mempalace",
                "roots": [
                    {
                        "block_id": "semantic_memory_index",
                        "description": "durable semantic memory",
                        "tags": ["semantic"],
                        "scope": "workspace",
                        "temperature": "warm",
                        "trust": "advisory",
                        "child_ids": ["stable_project_facts"],
                    }
                ],
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(hermes_provider, "LocalServiceClientPool", Pool)
    provider = hermes_provider.UmMemMemoryProvider()
    provider._hermes_home = tmp_path
    provider._session_id = "session-1"
    provider._config = {
        "repo_root": str(tmp_path),
        "workspace_id": "alpha",
        "backend": "mempalace",
        "timeout_seconds": 20,
    }
    try:
        first = provider.prefetch("first persistent query", session_id="session-1")
        second = provider.prefetch("second persistent query", session_id="session-1")
    finally:
        provider.shutdown()

    assert json.loads(first) == json.loads(second)
    assert json.loads(first)["um_mem_navigation"]["roots"][0]["block_id"] == (
        "semantic_memory_index"
    )
    assert len(pools) == 1
    assert calls == [
        "block_map",
        "resolve_active_mask",
        "block_map",
        "resolve_active_mask",
    ]


def test_projection_revision_is_stable_and_changes_with_snapshot() -> None:
    first = ({"block_id": "facts", "tags": ["fact"]},)
    second = ({"block_id": "facts", "tags": ["decision"]},)

    assert projection_revision_id_for_snapshot(first) == projection_revision_id_for_snapshot(first)
    assert projection_revision_id_for_snapshot(first) != projection_revision_id_for_snapshot(second)
