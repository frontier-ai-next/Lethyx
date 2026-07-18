import sqlite3
from pathlib import Path

from lethyx.integrations import hermes_smart_migration as smart
from lethyx.integrations.hermes_smart_migration import (
    classify_session_importance,
    detect_workspace,
    import_builtin_memories,
    index_project_docs,
    redact_sensitive_text,
    summarize_session,
)


def test_redacts_secret_like_values():
    text = "token sk-abc...7890 and Bearer ghp_ab...3456"

    redacted = redact_sensitive_text(text)

    assert "sk-abcdef" not in redacted
    assert "ghp_" not in redacted
    assert "[redacted" in redacted


def test_detect_workspace_from_repo_and_title():
    assert (
        detect_workspace(
            "default", "C:/Users/example/Documents/project-alpha", "Настройка Project Alpha"
        )
        == "project_alpha"
    )
    assert (
        detect_workspace("default", "C:/Users/example/Documents/Lethyx", "memory benchmark")
        == "lethyx"
    )
    assert detect_workspace("research", "", "arxiv literature review") == "research"


def test_classify_tool_heavy_session_as_archive_hint():
    row = {
        "id": "s1",
        "title": "temporary debug log dump",
        "source": "cli",
        "cwd": "C:/Users/example",
        "git_repo_root": "",
        "message_count": 120,
        "tool_call_count": 110,
        "input_tokens": 200000,
        "output_tokens": 10000,
        "started_at": 1780000000.0,
    }

    classified = classify_session_importance(
        row, user_texts=["check process output"], final_assistant_text="log tail only"
    )

    assert classified["tier"] == "cold"
    assert classified["lifecycle"] == "archived"
    assert classified["surface"] == "archive_hint"
    assert classified["memory_activity"] == "passive"


def test_summarize_session_uses_transcript_metadata_without_tool_dump(tmp_path: Path):
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute(
        "create table sessions (id text primary key, source text, user_id text, model text, model_config text, system_prompt text, parent_session_id text, started_at real, ended_at real, end_reason text, message_count integer, tool_call_count integer, input_tokens integer, output_tokens integer, cache_read_tokens integer, cache_write_tokens integer, reasoning_tokens integer, cwd text, billing_provider text, billing_base_url text, billing_mode text, estimated_cost_usd real, actual_cost_usd real, cost_status text, cost_source text, pricing_version text, title text, api_call_count integer, handoff_state text, handoff_platform text, handoff_error text, rewind_count integer, archived integer, git_branch text, git_repo_root text)"
    )
    con.execute(
        "create table messages (id integer primary key autoincrement, session_id text, role text, content text, tool_call_id text, tool_calls text, tool_name text, timestamp real, token_count integer, finish_reason text, reasoning text, reasoning_content text, reasoning_details text, codex_reasoning_items text, codex_message_items text, platform_message_id text, observed integer, active integer, compacted integer)"
    )
    con.execute(
        "insert into sessions (id, source, started_at, message_count, tool_call_count, input_tokens, output_tokens, cwd, title, archived, git_branch, git_repo_root) values (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "s2",
            "tui",
            1780000000.0,
            4,
            1,
            1000,
            200,
            "C:/Users/example/Documents/Lethyx",
            "Lethyx setup",
            0,
            "main",
            "C:/Users/example/Documents/Lethyx",
        ),
    )
    con.executemany(
        "insert into messages (session_id, role, content, tool_name, timestamp, active, compacted) values (?,?,?,?,?,?,?)",
        [
            ("s2", "user", "please finish Lethyx setup", None, 1.0, 1, 0),
            ("s2", "tool", "very long raw log that should not appear", "terminal", 2.0, 1, 0),
            ("s2", "assistant", "Готово: setup wizard works", None, 3.0, 1, 0),
        ],
    )
    con.commit()
    con.close()

    summary = summarize_session(db, "s2", profile="default")

    assert summary is not None
    assert summary.workspace_id == "lethyx"
    assert "very long raw log" not in summary.content
    assert "please finish Lethyx setup" in summary.content
    assert "Готово: setup wizard works" in summary.content
    assert summary.memory_id == "session_default_s2"


def test_index_project_docs_includes_codex_memory_without_raw_dump(monkeypatch, tmp_path: Path):
    project = tmp_path / "gmas-demo"
    (project / ".codex").mkdir(parents=True)
    (project / "vendor" / "docs").mkdir(parents=True)
    (project / "AGENTS.md").write_text("# Rules\nDo not edit vendor/gmas.", encoding="utf-8")
    (project / ".codex" / "memory.md").write_text(
        "# Stable Project Facts\nReact + FastAPI.", encoding="utf-8"
    )
    (project / "vendor" / "docs" / "ignored.md").write_text("# Vendor docs", encoding="utf-8")
    calls = []

    def fake_remember(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "memory_id": kwargs["memory_id"], "workspace": kwargs["workspace"]}

    monkeypatch.setattr(smart, "_remember_block", fake_remember)

    result = index_project_docs(
        repo_root=tmp_path / "store",
        backend="hindsight",
        workspace="gmas_demo",
        project_path=project,
        limit=20,
    )

    paths = {str(call["metadata"]["relative_path"]).replace("\\", "/") for call in calls}
    assert result.ok
    assert "AGENTS.md" in paths
    assert ".codex/memory.md" in paths
    assert all("vendor" not in path for path in paths)
    assert all(call["kind"] == "doc_pointer" for call in calls)


def test_import_builtin_memories_mirrors_user_profile_to_workspace_and_global(
    monkeypatch, tmp_path: Path
):
    hermes_home = tmp_path / "hermes"
    memdir = hermes_home / "memories"
    memdir.mkdir(parents=True)
    (memdir / "MEMORY.md").write_text(
        "Project convention A\n§\nProject convention B", encoding="utf-8"
    )
    (memdir / "USER.md").write_text("User prefers concise verdicts", encoding="utf-8")
    calls = []

    def fake_remember(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "memory_id": kwargs["memory_id"], "workspace": kwargs["workspace"]}

    monkeypatch.setattr(smart, "_remember_block", fake_remember)

    result = import_builtin_memories(
        hermes_home=hermes_home,
        repo_root=tmp_path / "store",
        backend="hindsight",
        workspace="gmas_demo",
        profiles=["default"],
    )

    assert result.ok
    assert result.count == 4
    workspaces = [call["workspace"] for call in calls]
    assert workspaces.count("gmas_demo") == 3
    assert workspaces.count("global") == 1
    kinds = {call["kind"] for call in calls}
    assert {"profile_memory", "user_profile"}.issubset(kinds)
