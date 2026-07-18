from pathlib import Path
from typing import Any

from lethyx.memory import mcp_facade
from lethyx.memory.backends.contracts import MemoryBlockRecord
from lethyx.memory.mcp_facade import (
    lethyx_build_context,
    lethyx_export_obsidian,
    lethyx_forget,
    lethyx_promote,
    lethyx_recall,
    lethyx_remember,
)
from lethyx.memory.mcp_server import TOOL_NAMES, create_mcp_server


def _repo(tmp_path: Path) -> str:
    return tmp_path.as_posix()


def _record(block_id: str, content: str, workspace_id: str = "alpha") -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        content=content,
        title="Record",
        kind="observation",
        lifecycle="active",
        trust_level="workspace_verified",
        scope="workspace",
        tier="warm",
        surface="supplemental_evidence",
        workspace_id=workspace_id,
        run_id="",
        phase_id="",
        producer="test",
        source_backend="mempalace",
        tags=("parser",),
        metadata={"memory_activity": "active"},
        verified=True,
    )


class _NativeFailIndexBackend:
    def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
        raise RuntimeError("native recall down")

    def export_blocks(self) -> list[MemoryBlockRecord]:
        return [_record("indexed", "Indexed fallback fact about parser.")]


class _DummyClient:
    workspace_id = "alpha"
    repo_root = Path("unused")
    backend = _NativeFailIndexBackend()

    def recall(self, query: str, limit: int = 10, tags=None, max_tokens: int = 2048):
        raise RuntimeError("client native recall down")


def test_mcp_facade_recall_reports_native_error_when_index_fallback_used(monkeypatch) -> None:
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: _DummyClient())
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_recall(
        query="parser",
        workspace_id="alpha",
        backend="mempalace",
        repo_root="unused",
        include_passive=True,
    )

    assert payload["ok"] is True
    assert [hit["metadata"]["block_id"] for hit in payload["hits"]] == ["indexed"]
    assert payload["diagnostics"]["native_error"] == "native recall down"
    assert payload["diagnostics"]["index_used"] is True
    assert payload["diagnostics"]["index_count"] == 1


class _TerseNativeRichIndexBackend:
    def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
        return [
            _record(
                "bench_api_tasks",
                "The synthetic_fastapi_tasks repository is a FastAPI task tracker located at /tmp/synthetic_fastapi_tasks.",
            )
        ]

    def export_blocks(self) -> list[MemoryBlockRecord]:
        return [
            _record(
                "bench_api_tasks",
                "Synthetic repo synthetic_fastapi_tasks: FastAPI task tracker. App package is app/. "
                "Routes live in app/routes/tasks.py, persistence uses SQLite file data/tasks.db, "
                "migrations are managed by alembic under migrations/.",
            )
        ]


class _TerseNativeRichIndexClient:
    workspace_id = "alpha"
    repo_root = Path("unused")
    backend = _TerseNativeRichIndexBackend()

    def recall(self, query: str, limit: int = 10, tags=None, max_tokens: int = 2048):
        return []


def test_mcp_facade_prefers_richer_index_record_over_terse_native_duplicate() -> None:
    records, diagnostics = mcp_facade._recall_records(
        _TerseNativeRichIndexClient(),
        query="What is the package, route file, database file, and migration location for synthetic_fastapi_tasks?",
        limit=5,
        tags=[],
        include_global=False,
        include_passive=True,
        max_tokens=4096,
    )

    assert diagnostics["native_count"] == 1
    assert diagnostics["index_count"] == 1
    assert [record.block_id for record in records] == ["bench_api_tasks"]
    assert "data/tasks.db" in records[0].content


def test_mcp_facade_index_fallback_matches_natural_architecture_queries() -> None:
    record = _record(
        "api_tasks",
        "Synthetic repo synthetic_fastapi_tasks: FastAPI task tracker. App package is app/. "
        "Routes live in app/routes/tasks.py, persistence uses SQLite file data/tasks.db, "
        "migrations are managed by alembic under migrations/.",
    )

    hits = mcp_facade._safe_records_from_backend(
        type("Backend", (), {"export_blocks": lambda self: [record]})(),
        workspace_id="alpha",
        include_global=False,
        include_passive=True,
        query="What is the package, route file, database file, and migration location for synthetic_fastapi_tasks?",
        limit=5,
    )

    assert [hit.block_id for hit in hits] == ["api_tasks"]
    assert "data/tasks.db" in hits[0].content


def test_mcp_facade_export_fallback_prefers_active_records_over_newer_candidates() -> None:
    active = MemoryBlockRecord(
        block_id="active",
        content="parser active fact",
        lifecycle="active",
        workspace_id="alpha",
        scope="workspace",
        updated_at="2024-01-01T00:00:00Z",
    )
    candidate = MemoryBlockRecord(
        block_id="candidate",
        content="parser candidate fact",
        lifecycle="candidate",
        workspace_id="alpha",
        scope="workspace",
        updated_at="2026-01-01T00:00:00Z",
    )

    hits = mcp_facade._safe_records_from_backend(
        type("Backend", (), {"export_blocks": lambda self: [candidate, active]})(),
        workspace_id="alpha",
        include_global=False,
        include_passive=True,
        query="parser",
        limit=2,
    )

    assert [hit.block_id for hit in hits] == ["active", "candidate"]


def test_mcp_facade_negative_query_matches_subject_not_forbidden_alternatives() -> None:
    record = _record(
        "api_tasks",
        "The synthetic_fastapi_tasks repository is a FastAPI task tracker. "
        "Routes live in app/routes/tasks.py.",
    )

    hits = mcp_facade._safe_records_from_backend(
        type("Backend", (), {"export_blocks": lambda self: [record]})(),
        workspace_id="alpha",
        include_global=False,
        include_passive=True,
        query="Does synthetic_fastapi_tasks use reportkit or Zustand?",
        limit=5,
    )

    assert [hit.block_id for hit in hits] == ["api_tasks"]


def test_mcp_facade_secret_record_redacts_text_and_sensitive_metadata() -> None:
    record = MemoryBlockRecord(
        block_id="secret_block",
        content="token=super-secret-value",
        lifecycle="active",
        workspace_id="alpha",
        scope="workspace",
        privacy_class="secret",
        risk_level="secret",
        metadata={
            "privacy_class": "normal",
            "risk_level": "normal",
            "api_token": "super-secret-value",
            "safe": "kept",
        },
    )

    hit = mcp_facade._hit_from_record(record)

    assert hit["text"] == "[redacted non-exportable memory: secret_block]"
    assert hit["metadata"]["privacy_class"] == "secret"
    assert hit["metadata"]["risk_level"] == "secret"
    assert "api_token" not in hit["metadata"]
    assert hit["metadata"]["safe"] == "kept"


def test_mcp_facade_build_context_caps_fallback_text_to_token_budget(monkeypatch) -> None:
    long_record = _record("long", "parser " + ("x" * 500))

    class Client:
        workspace_id = "alpha"
        repo_root = Path("unused")
        backend = type("Backend", (), {"export_blocks": lambda self: [long_record]})()

        def recall(self, query: str, limit: int = 10, tags=None, max_tokens: int = 2048):
            return []

    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: Client())
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)

    payload = lethyx_build_context(
        query="parser",
        workspace_id="alpha",
        repo_root="unused",
        backend="mempalace",
        max_tokens=10,
    )

    assert payload["ok"] is True
    assert len(payload["items"][0]["text"]) <= 40


def test_mcp_facade_general_query_matches_partial_semantic_subject() -> None:
    record = _record(
        "api_tasks",
        "The synthetic_fastapi_tasks repository is a FastAPI task tracker. Persistence uses SQLite.",
    )

    hits = mcp_facade._safe_records_from_backend(
        type("Backend", (), {"export_blocks": lambda self: [record]})(),
        workspace_id="alpha",
        include_global=False,
        include_passive=True,
        query="Which repo is the task tracker and what persistence technology does it use?",
        limit=5,
    )

    assert [hit.block_id for hit in hits] == ["api_tasks"]


def test_mcp_facade_native_hits_filter_negative_query_to_subject_record() -> None:
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
        include_global=False,
        include_passive=True,
        max_tokens=4096,
    )

    assert diagnostics["native_count"] == 1
    assert [record.block_id for record in records] == ["api_tasks"]


def test_mcp_server_exposes_expected_lethyx_tools() -> None:
    server = create_mcp_server()

    assert server is not None
    assert set(TOOL_NAMES) == {
        "health",
        "remember",
        "recall",
        "build_context",
        "promote",
        "forget",
        "add_edge",
        "export_obsidian",
    }


def test_mcp_facade_remember_recall_and_workspace_isolation(tmp_path: Path) -> None:
    repo_root = _repo(tmp_path)

    first = lethyx_remember(
        content="Alpha project keeps parser facts in Lethyx only.",
        title="Alpha Parser Fact",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        tags=["parser"],
    )
    second = lethyx_remember(
        content="Beta project must not leak into alpha recall.",
        title="Beta Isolation Fact",
        workspace_id="beta",
        backend="mempalace",
        repo_root=repo_root,
        tags=["parser"],
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
    )

    rendered = "\n".join(hit["text"] for hit in alpha_hits["hits"])
    assert "Alpha project keeps parser facts" in rendered
    assert "Beta project must not leak" not in rendered
    assert all(hit["authority"] == "non_directive" for hit in alpha_hits["hits"])


def test_mcp_facade_include_global_opens_global_workspace_store(tmp_path: Path) -> None:
    repo_root = _repo(tmp_path)
    lethyx_remember(
        content="Project-only GMAS fact about RoleGraph.",
        title="Project Fact",
        workspace_id="gmas_demo",
        backend="mempalace",
        repo_root=repo_root,
        memory_id="project_fact",
        kind="project_fact",
        trust_level="workspace_verified",
        evidence_refs=[{"ref_type": "test", "ref_id": "project", "produced_by": "pytest"}],
    )
    lethyx_remember(
        content="Global operator fact about concise status.",
        title="Global Fact",
        workspace_id="global",
        backend="mempalace",
        repo_root=repo_root,
        memory_id="global_fact",
        kind="preference",
        scope="manager",
        trust_level="workspace_verified",
        evidence_refs=[{"ref_type": "test", "ref_id": "global", "produced_by": "pytest"}],
    )

    default_hits = lethyx_recall(
        query="fact",
        workspace_id="gmas_demo",
        backend="mempalace",
        repo_root=repo_root,
        limit=10,
    )
    global_hits = lethyx_recall(
        query="fact",
        workspace_id="gmas_demo",
        backend="mempalace",
        repo_root=repo_root,
        include_global=True,
        limit=10,
    )

    assert [hit["metadata"]["block_id"] for hit in default_hits["hits"]] == ["project_fact"]
    assert {hit["metadata"]["block_id"] for hit in global_hits["hits"]} == {
        "project_fact",
        "global_fact",
    }


def test_mcp_facade_build_context_filters_passive_memory_by_default(tmp_path: Path) -> None:
    repo_root = _repo(tmp_path)
    needle = "active-passive-filter-proof"
    active = lethyx_remember(
        content=f"{needle} active memory should be returned by default.",
        title="Active Filter Fact",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        metadata={"memory_activity": "active"},
    )
    passive = lethyx_remember(
        content=f"{needle} passive memory requires explicit include_passive.",
        title="Passive Filter Fact",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        metadata={"memory_activity": "passive"},
    )
    assert active["ok"] is True
    assert passive["ok"] is True

    default_context = lethyx_build_context(
        query=needle,
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        limit=10,
    )
    default_text = "\n".join(item["text"] for item in default_context["items"])
    assert "active memory should be returned" in default_text
    assert "passive memory requires explicit" not in default_text

    passive_context = lethyx_build_context(
        query=needle,
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        limit=10,
        include_passive=True,
    )
    passive_text = "\n".join(item["text"] for item in passive_context["items"])
    assert "active memory should be returned" in passive_text
    assert "passive memory requires explicit" in passive_text


def test_mcp_facade_build_context_recent_session_fallback_when_semantic_query_misses(
    tmp_path: Path,
) -> None:
    repo_root = _repo(tmp_path)
    saved = lethyx_remember(
        content="Session summary\n\nUser intent: Study gMAS.\n\nFinal outcome: gMAS has RoleGraph and MACPRunner.",
        title="Session summary 20260627_demo",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        kind="session_summary",
        tags=["session-summary", "workspace:alpha"],
        metadata={"retention": "auto_session_summary", "memory_activity": "active"},
        lifecycle="active",
        surface="supplemental_evidence",
    )
    diagnostic = lethyx_remember(
        content="Session summary\n\nUser intent: Check previous remembered session context.\n\nFinal outcome: no recalled context was visible.",
        title="Session summary 20260627_diagnostic",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        kind="session_summary",
        tags=["session-summary", "workspace:alpha"],
        metadata={"retention": "auto_session_summary", "memory_activity": "active"},
        lifecycle="active",
        surface="supplemental_evidence",
    )
    assert saved["ok"] is True
    assert diagnostic["ok"] is True

    context = lethyx_build_context(
        query="previous remembered session context",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        limit=5,
    )

    rendered = "\n".join(item["text"] for item in context["items"])
    assert "RoleGraph and MACPRunner" in rendered
    assert all(item["authority"] == "non_directive" for item in context["items"])


def test_mcp_facade_build_context_filters_retracted_memory(tmp_path: Path) -> None:
    repo_root = _repo(tmp_path)
    saved = lethyx_remember(
        content="Retracted content should not be returned to an agent.",
        title="Retracted Fact",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
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
    )

    assert context["surface"] == "supplemental_evidence"
    assert context["authority"] == "non_directive"
    assert "Retracted content" not in "\n".join(item["text"] for item in context["items"])


def test_mcp_facade_promote_requires_evidence_and_can_export_obsidian(tmp_path: Path) -> None:
    repo_root = _repo(tmp_path)
    saved = lethyx_remember(
        content="Verified memory can be promoted and exported.",
        title="Verified Export Fact",
        workspace_id="alpha",
        backend="mempalace",
        repo_root=repo_root,
        trust_level="workspace_verified",
        evidence_refs=[{"ref_type": "test", "ref_id": "mcp-facade", "produced_by": "pytest"}],
    )

    promoted = lethyx_promote(
        memory_id=saved["memory_id"],
        reason="test promotion",
        evidence_refs=[{"ref_type": "test", "ref_id": "mcp-facade", "produced_by": "pytest"}],
        target_surface="supplemental_evidence",
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
