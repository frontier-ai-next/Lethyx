from pathlib import Path


def _load_plugin():
    """Load the packaged Hermes provider implementation.

    Release tests must not depend on a developer-local plugin copy. The
    installed Hermes plugin can be a thin wrapper, but the implementation belongs
    in the `lethyx` package.
    """
    from lethyx.integrations import hermes_provider as mod

    return mod


def test_packaged_provider_can_install_thin_hermes_wrapper(tmp_path: Path) -> None:
    mod = _load_plugin()

    wrapper = mod.install_hermes_plugin(tmp_path, overwrite=True)

    assert wrapper == tmp_path / "plugins" / "lethyx" / "__init__.py"
    text = wrapper.read_text(encoding="utf-8")
    assert "lethyx.integrations.hermes_provider" in text
    assert "LethyxMemoryProvider" in text


def test_provider_defaults_are_release_safe(monkeypatch, tmp_path: Path) -> None:
    mod = _load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("LETHYX_CODE_ROOT", raising=False)
    monkeypatch.delenv("LETHYX_REPO_ROOT", raising=False)
    monkeypatch.delenv("LETHYX_OBSIDIAN_VAULT", raising=False)

    cfg = mod._load_config()
    joined = " ".join(str(value) for value in cfg.values())

    assert cfg["code_root"] == ""
    assert str(tmp_path / "hermes" / "lethyx_store") == cfg["repo_root"]
    assert str(tmp_path / "hermes" / "lethyx_vault") == cfg["vault_dir"]
    assert "Users/admin" not in joined
    assert "LOCAL_APP_DATA_MARKER" not in joined


def test_provider_semantic_turn_payloads_split_candidate_memory() -> None:
    mod = _load_plugin()

    payloads = mod._semantic_turn_payloads(
        session="session-alpha",
        digest="abc123",
        user_text="Я хочу, чтобы память не была cron. Почему это важно? Решение: provider path.",
        assistant_text="Решили использовать provider write path and no cron.",
        tags=["conversation-turn"],
        metadata={"session_id": "session-alpha"},
    )

    kinds = {payload["kind"] for payload in payloads}
    assert {
        "session_turn",
        "user_intent",
        "assistant_outcome",
        "decision",
        "preference",
        "open_question",
    }.issubset(kinds)
    assert all(
        payload["surface"] in {"supplemental_evidence", "reflection_candidate"}
        for payload in payloads
    )
    assert all(payload["scope"] == "workspace" for payload in payloads)
    assert all("semantic-extract" in payload["tags"] for payload in payloads)


def test_provider_sync_turn_saves_many_then_exports_once(monkeypatch) -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()
    provider._config = {
        "workspace_id": "eval",
        "backend": "composite",
        "repo_root": "unused",
        "auto_retain": True,
        "auto_export_obsidian": True,
        "vault_dir": "unused",
    }
    provider._session_id = "session-alpha"
    calls: list[tuple[str, dict]] = []

    def fake_call(tool: str, payload: dict) -> dict:
        calls.append((tool, payload))
        return {"ok": True, "saved": True, "tool": tool}

    monkeypatch.setattr(provider, "_call", fake_call)
    provider.sync_turn(
        "Я хочу, чтобы память не была cron. Почему это важно? Решение: provider path.",
        "Решили использовать provider write path and no cron.",
        session_id="session-alpha",
        messages=[{"role": "user"}, {"role": "assistant"}],
    )

    remember_calls = [payload for tool, payload in calls if tool == "remember"]
    export_calls = [payload for tool, payload in calls if tool == "export_obsidian"]
    assert len(remember_calls) >= 6
    assert len(export_calls) == 1
    assert {payload["kind"] for payload in remember_calls} >= {
        "session_turn",
        "user_intent",
        "assistant_outcome",
        "decision",
        "preference",
        "open_question",
    }


def test_provider_tool_remember_marks_explicit_memory_verified_with_evidence(monkeypatch) -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()
    provider._config = {
        "workspace_id": "eval",
        "backend": "composite",
        "repo_root": "unused",
        "auto_retain": True,
        "auto_export_obsidian": True,
        "vault_dir": "unused",
    }
    provider._session_id = "session-tool"
    calls: list[tuple[str, dict]] = []

    def fake_call(tool: str, payload: dict) -> dict:
        calls.append((tool, payload))
        return {"ok": True, "saved": True, "tool": tool}

    monkeypatch.setattr(provider, "_call", fake_call)
    provider.handle_tool_call(
        "lethyx_provider_remember",
        {"content": "Project fact", "title": "Fact", "kind": "project_fact", "tier": "hot"},
    )

    remember_payload = calls[0][1]
    assert remember_payload["trust_level"] == "workspace_verified"
    assert remember_payload["evidence_refs"] == [
        {
            "ref_type": "lethyx_provider_tool",
            "ref_id": "session-tool",
            "produced_by": "lethyx_provider",
        }
    ]
    assert [tool for tool, _ in calls] == ["remember", "export_obsidian"]


def test_provider_memory_write_is_transactional_remember_then_export(monkeypatch) -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()
    provider._config = {
        "workspace_id": "eval",
        "backend": "composite",
        "repo_root": "unused",
        "auto_retain": True,
        "auto_export_obsidian": True,
        "vault_dir": "unused",
    }
    calls: list[tuple[str, dict]] = []

    def fake_call(tool: str, payload: dict) -> dict:
        calls.append((tool, payload))
        return {"ok": True, "saved": True, "tool": tool}

    monkeypatch.setattr(provider, "_call", fake_call)
    provider.on_memory_write(
        "add", "user", "User prefers scoped project memory.", {"session_id": "s"}
    )

    assert [tool for tool, _ in calls] == ["remember", "export_obsidian"]
    remember_payload = calls[0][1]
    assert remember_payload["kind"] == "user_profile"
    assert remember_payload["lifecycle"] == "active"
    assert remember_payload["trust_level"] == "workspace_verified"


def test_provider_memory_replace_reuses_existing_runtime_block_id(monkeypatch) -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()
    provider._config = {
        "workspace_id": "eval",
        "backend": "hindsight",
        "repo_root": "unused",
        "auto_retain": True,
        "auto_export_obsidian": True,
        "vault_dir": "unused",
    }
    calls: list[tuple[str, dict]] = []

    def fake_call(tool: str, payload: dict) -> dict:
        calls.append((tool, payload))
        if tool == "recall":
            return {
                "ok": True,
                "hits": [
                    {
                        "tags": ["runtime-sync", "target:memory", "workspace:eval"],
                        "metadata": {
                            "block_id": "runtime_memory_eval_existing",
                            "builtin_memory_target": "memory",
                            "workspace_id": "eval",
                        },
                    }
                ],
            }
        return {"ok": True, "saved": True, "tool": tool}

    monkeypatch.setattr(provider, "_call", fake_call)
    provider.on_memory_write(
        "replace",
        "memory",
        "Project feature now uses WebSocket streaming.",
        {"session_id": "s", "old_text": "Project feature used polling."},
    )

    assert [tool for tool, _ in calls] == ["recall", "remember", "export_obsidian"]
    remember_payload = calls[1][1]
    assert remember_payload["memory_id"] == "runtime_memory_eval_existing"
    assert remember_payload["content"] == "Project feature now uses WebSocket streaming."


def test_provider_records_failed_context_payload_in_status() -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()
    provider._config = {"workspace_id": "eval", "backend": "hindsight", "repo_root": "unused"}

    rendered = provider._format_context({"ok": False, "error": "native recall unavailable"})

    assert rendered == ""
    status = provider.status_payload()
    assert status["last_errors"][-1]["kind"] == "context_format"
    assert status["last_errors"][-1]["error"] == "native recall unavailable"


def test_provider_status_redacts_paths_and_streams_by_default(tmp_path: Path) -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()
    project = tmp_path / "project-secret-path"
    provider._config = {
        "workspace_id": "eval",
        "backend": "hindsight",
        "repo_root": str(tmp_path / "repo-secret-path"),
        "diagnostics_debug": False,
    }
    provider._hermes_home = tmp_path / "hermes"

    provider._record_error(
        "tool:recall",
        {
            "error": "failed",
            "tool": "recall",
            "stdout": "stdout token=abc123",
            "stderr": "stderr secret=def456",
        },
    )
    provider._mark_bootstrap_failure(
        project_root=project,
        workspace="eval",
        error="boom",
        returncode=7,
        stdout="stdout token=abc123",
        stderr="stderr secret=def456",
    )

    status = provider.status_payload()
    dumped = str(status)
    assert "repo_root" not in status
    assert status["repo_root_configured"] is True
    assert "project-secret-path" not in dumped
    assert "repo-secret-path" not in dumped
    assert "abc123" not in dumped
    assert "def456" not in dumped
    assert "stdout" not in status["last_errors"][-1]
    assert "stderr" not in status["last_errors"][-1]
    assert "path" not in status["bootstrap"]
    assert "project_path" not in status["bootstrap"]
    assert status["bootstrap"]["returncode"] == 7


def test_provider_shutdown_joins_prefetch_threads() -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()

    class FakeThread:
        joined = False

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            self.joined = True

    thread = FakeThread()
    provider._prefetch_threads["s"] = thread  # type: ignore[assignment]

    provider.shutdown()

    assert thread.joined is True


def test_provider_prefetch_builds_context_synchronously_when_cache_empty(monkeypatch) -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()
    provider._config = {
        "workspace_id": "eval",
        "backend": "hindsight",
        "repo_root": "unused",
        "auto_prefetch": True,
        "include_passive": False,
    }
    calls: list[tuple[str, dict]] = []

    def fake_call(tool: str, payload: dict) -> dict:
        calls.append((tool, payload))
        return {
            "ok": True,
            "items": [
                {
                    "text": "Routes live in app/routes/tasks.py and persistence uses data/tasks.db.",
                    "source": "test",
                    "authority": "non_directive",
                }
            ],
        }

    monkeypatch.setattr(provider, "_call", fake_call)

    rendered = provider.prefetch("Where is the database file?", session_id="s")

    assert "data/tasks.db" in rendered
    assert calls == [
        (
            "build_context",
            {
                "query": "Where is the database file?",
                "limit": 8,
                "include_global": False,
                "include_passive": False,
                "max_tokens": 2048,
            },
        )
    ]


def test_provider_records_tool_timing_in_status(monkeypatch) -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()
    provider._config = {
        "workspace_id": "eval",
        "backend": "hindsight",
        "repo_root": "unused",
        "timeout_seconds": 1,
    }

    class SuccessProcess:
        returncode = 0
        stdout = '{"ok": true, "items": []}'
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: SuccessProcess())

    result = provider._call("build_context", {"query": "latency"})

    assert result["ok"] is True
    status = provider.status_payload()
    assert status["last_timings"][-1]["tool"] == "build_context"
    assert status["last_timings"][-1]["ok"] is True
    assert status["last_timings"][-1]["elapsed_ms"] >= 0


def test_provider_auto_bootstrap_failure_writes_marker(monkeypatch, tmp_path: Path) -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()
    project = tmp_path / "gmas-demo"
    (project / ".git").mkdir(parents=True)
    provider._config = {
        "workspace_id": "gmas_demo",
        "backend": "hindsight",
        "repo_root": "unused",
        "vault_dir": "unused",
        "auto_bootstrap": True,
        "bootstrap_session_limit": 3,
        "bootstrap_doc_limit": 5,
        "timeout_seconds": 1,
    }
    provider._hermes_home = tmp_path / "hermes"

    class FailedProcess:
        returncode = 7
        stdout = "partial stdout"
        stderr = "boom stderr"

    monkeypatch.setattr(provider, "_bootstrap_due", lambda **kwargs: True)
    monkeypatch.setattr(provider, "_start_sync_thread", lambda target, name: target())
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: FailedProcess())

    provider._maybe_start_bootstrap({"cwd": str(project)})

    marker = provider.status_payload()["bootstrap"]
    assert marker["ok"] is False
    assert marker["error"] == "bootstrap_subprocess_failed"
    assert marker["returncode"] == 7
    assert "stderr" not in marker


def test_provider_auto_bootstrap_schedules_for_project_workspace(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_plugin()
    provider = mod.LethyxMemoryProvider()
    project = tmp_path / "gmas-demo"
    (project / ".git").mkdir(parents=True)
    provider._config = {
        "workspace_id": "gmas_demo",
        "backend": "hindsight",
        "repo_root": "unused",
        "vault_dir": "unused",
        "auto_bootstrap": True,
        "bootstrap_session_limit": 3,
        "bootstrap_doc_limit": 5,
        "timeout_seconds": 1,
    }
    provider._hermes_home = tmp_path / "hermes"
    scheduled = []

    monkeypatch.setattr(provider, "_bootstrap_due", lambda **kwargs: True)
    monkeypatch.setattr(
        provider, "_start_sync_thread", lambda target, name: scheduled.append((target, name))
    )

    provider._maybe_start_bootstrap({"cwd": str(project)})

    assert len(scheduled) == 1
    assert scheduled[0][1] == "lethyx-bootstrap"
