import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from um_mem.api import AmbiguousEvidenceRefError
from um_mem.contracts import CandidateVerificationSubject, VerificationReportRef
from um_mem.integrations import hermes_provider
from um_mem.memory.backends.base import MemoryBusyError
from um_mem.memory.backends.contracts import CanonicalHeadConflict


def test_hermes_provider_reads_caller_scoped_receipt_and_exact_record(tmp_path) -> None:
    from um_mem.memory.local_service import (
        build_local_service_identity,
        shutdown_local_service_listener,
    )

    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": str(tmp_path),
        "workspace_id": "alpha",
        "backend": "mempalace",
        "profile_id": "default",
        "timeout_seconds": 20,
    }
    provider._session_id = "hermes-canary"
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="parent"
    )
    try:
        saved = provider._call(
            "remember",
            {
                "content": "Hermes exact canary.",
                "requested_block_id": "workspace_episodes",
                "operation_id": "hermes-exact-canary-1",
            },
        )
        status = provider._call("operation_status", {"operation_id": "hermes-exact-canary-1"})
        exact = provider._call("read_exact_record", {"record_id": saved.get("memory_id", "")})
        listing = provider._call("list_records", {"scope": "workspace", "scope_id": "alpha"})
    finally:
        provider._close_local_service_clients()
        shutdown_local_service_listener(identity)

    assert saved["ok"] is True
    assert status["ok"] is True
    assert status["original_receipt"]["canonical_reference"] == saved["memory_id"]
    assert exact["ok"] is True and exact["record"]["content"] == "Hermes exact canary."
    assert saved["memory_id"] in {row["record_id"] for row in listing["records"]}


def test_system_prompt_is_an_operational_authoring_contract() -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
    }

    prompt = provider.system_prompt_block()

    for expected in (
        "free semantic topology",
        "no required branch taxonomy or project level",
        "navigation envelope before writing",
        "writable leaf",
        "service generates block IDs",
        "typed graph links",
        "Machine observations are facts only",
        "active mask supplies bounded context",
        "passive recall is a separate temporary operation",
        "You author semantic choices",
    ):
        assert expected in prompt


def test_provider_import_does_not_eagerly_load_backend_runtime() -> None:
    script = """
import sys
import um_mem.integrations.hermes_provider

assert "um_mem.memory.working_continuity" not in sys.modules
assert "um_mem.memory.backends.mempalace" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_recall_mode_and_aggregate_budget_are_per_host_turn(monkeypatch, tmp_path) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": str(tmp_path),
        "workspace_id": "alpha",
        "profile_id": "default",
        "backend": "mempalace",
        "retrieval_mode": "hybrid",
        "recall_total_tokens": 4096,
        "recall_max_calls": 2,
        "recall_max_tokens": 4096,
    }
    provider._session_id = "s"
    provider._config["scope_session_id"] = "logical-session"
    calls = []

    def call(tool, args):
        calls.append((tool, args))
        return {"ok": True, "backend": "mempalace", "rendered_context": "факт 2023-04-12 " * 1000}

    monkeypatch.setattr(provider, "_call", call)
    monkeypatch.setattr(provider, "_submit_host_state", lambda **kwargs: None)
    provider.on_turn_start(1, "question")
    first = provider._handle_recall_tool(
        {"query": "date", "logical_block_ids": ["leaf"]}, scoped=True
    )
    second = provider._handle_recall_tool(
        {"query": "date again", "logical_block_ids": ["leaf"]}, scoped=True
    )
    assert len(first.encode()) + len(second.encode()) <= 4096
    assert calls[0][1]["retrieval_mode"] == "hybrid"
    third = json.loads(
        provider._handle_recall_tool({"query": "again", "logical_block_ids": ["leaf"]}, scoped=True)
    )
    assert not third["ok"]
    assert len(calls) <= 2
    assert (
        len(first.encode())
        + len(second.encode())
        + len(json.dumps(third, separators=(",", ":")).encode())
        <= 4096
    )
    provider.on_turn_start(2, "next")
    provider._handle_recall_tool({"query": "date", "logical_block_ids": ["leaf"]}, scoped=True)
    assert len(calls) >= 2


def test_budget_bounds_invalid_arguments_and_repeated_denials(monkeypatch, tmp_path) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._session_id = "s"
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "recall_total_tokens": 256,
        "recall_max_calls": 2,
    }
    monkeypatch.setattr(provider, "_submit_host_state", lambda **kwargs: None)
    provider.on_turn_start(1, "question")
    outputs = [provider._handle_recall_tool({"x" * 1000: 1}, scoped=True) for _ in range(20)]
    assert sum(len(output.encode()) for output in outputs) <= 256


def test_hybrid_profile_rejects_unsupported_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="MemPalace"):
        hermes_provider._load_config(
            tmp_path, overrides={"backend": "mnemosyne", "retrieval_mode": "hybrid"}
        )
    assert (
        hermes_provider._load_config(
            tmp_path, overrides={"backend": "zvec", "retrieval_mode": "hybrid"}
        )["retrieval_mode"]
        == "hybrid"
    )


def test_active_prefetch_and_recalls_share_the_host_allowance(monkeypatch, tmp_path) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._session_id = "native-session"
    provider._config = {
        "repo_root": str(tmp_path),
        "workspace_id": "alpha",
        "scope_session_id": "logical-session",
        "backend": "mempalace",
        "active_mask_id": "mask",
        "recall_total_tokens": 4096,
        "recall_max_calls": 2,
        "recall_max_tokens": 4096,
    }
    calls = []

    def call(tool, args):
        calls.append(tool)
        if tool == "block_map":
            return {"ok": True, "roots": [{"block_id": "root", "child_ids": ["leaf"]}]}
        if tool == "resolve_active_mask":
            return {
                "ok": True,
                "lease": {
                    "rendered_context": "факт 2023-04-12 " * 1000,
                    "trace": {"resolution_status": "active", "mask_id": "mask", "mask_revision": 1},
                },
            }
        return {"ok": True, "backend": "mempalace", "rendered_context": "more facts " * 1000}

    monkeypatch.setattr(provider, "_call", call)
    monkeypatch.setattr(provider, "_submit_host_state", lambda **kwargs: None)
    assert provider.prefetch("question", session_id="native-session") == ""
    assert not calls
    provider.on_turn_start(1, "question")
    active = provider.prefetch("question", session_id="native-session")
    assert json.loads(active)["um_mem_active"]["truncated"]
    assert json.loads(active)["um_mem_active"]["mask_id"] == "mask"
    outputs = [active]
    for _ in range(20):
        outputs.append(
            provider._handle_recall_tool(
                {"query": "date", "logical_block_ids": ["leaf"]}, scoped=True
            )
        )
    assert sum(len(output.encode()) for output in outputs) + 256 <= 4096
    assert calls.count("build_context") <= 2
    assert provider.prefetch("question", session_id="native-session") == ""
    assert set(calls) <= {"block_map", "resolve_active_mask", "build_context"}


def test_prefetch_only_grant_is_charged_to_the_same_allowance(monkeypatch, tmp_path) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._session_id = "s"
    provider._config = {
        "tool_access": "prefetch_only",
        "recall_total_tokens": 4096,
        "repo_root": str(tmp_path),
    }
    monkeypatch.setattr(provider, "_submit_host_state", lambda **kwargs: None)
    monkeypatch.setattr(
        provider, "_prefetch_context", lambda *args, **kwargs: "advisory факт " * 1000
    )
    provider.on_turn_start(1, "question")
    outputs = [provider.prefetch("question", session_id="s") for _ in range(20)]
    assert sum(len(output.encode()) + (256 if output else 0) for output in outputs) <= 4096


def test_late_prefetch_does_not_charge_or_deliver_to_a_new_host_turn(monkeypatch, tmp_path) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._session_id = "s"
    provider._config = {"recall_total_tokens": 4096, "repo_root": str(tmp_path)}
    monkeypatch.setattr(provider, "_submit_host_state", lambda **kwargs: None)

    def late(*args, **kwargs):
        provider.on_turn_start(2, "next")
        return "previous turn context"

    monkeypatch.setattr(provider, "_prefetch_context", late)
    provider.on_turn_start(1, "question")
    assert provider.prefetch("question", session_id="s") == ""
    assert provider._recall_budget_by_session["s"] == (2, 0, 0)


def test_packaged_provider_installs_thin_hermes_wrapper(tmp_path: Path) -> None:
    wrapper = hermes_provider.install_hermes_plugin(tmp_path, overwrite=True)

    assert wrapper == tmp_path / "plugins" / "um_mem" / "__init__.py"
    source = wrapper.read_text(encoding="utf-8")
    assert "um_mem.integrations.hermes_provider" in source
    assert "UmMemMemoryProvider" in source


def test_provider_defaults_have_no_query_prefetch_or_cache_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("UM_MEM_CODE_ROOT", raising=False)
    monkeypatch.delenv("UM_MEM_REPO_ROOT", raising=False)
    monkeypatch.delenv("UM_MEM_OBSIDIAN_VAULT", raising=False)

    config = hermes_provider._load_config()

    assert config["auto_retain"] is False
    assert config["tool_access"] == "full"
    assert config["read_block_ids"] == []
    assert config["read_record_ids"] == []
    assert config["active_mask_id"] == ""
    assert config["query_max_chars"] > 0
    assert not {
        "auto_prefetch",
        "background_prefetch_enabled",
        "prefetch_trace_enabled",
        "prefetch_trace_max_bytes",
        "prefetch_trace_backups",
    }.intersection(config)


def test_prefetch_only_provider_exposes_no_parallel_memory_tool_path() -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "tool_access": "prefetch_only",
    }

    assert provider.get_tool_schemas() == []
    denied = json.loads(provider.handle_tool_call("um_mem_provider_search_all", {"query": "x"}))
    assert denied == {
        "ok": False,
        "error": "UM-MEM provider tools are disabled by the managed session grant",
    }


def test_provider_preserves_virtual_environment_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    try:
        interpreter.symlink_to(sys.executable)
    except OSError as exc:
        if os.name == "nt" and exc.winerror == 1314:
            pytest.skip("Creating symlinks requires Windows developer mode")
        raise
    config_dir = tmp_path / "um_mem"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"python_executable": "venv/bin/python"}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UM_MEM_ALLOW_ENV_CONFIG_OVERRIDE", raising=False)
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = hermes_provider._load_config(tmp_path)

    clients = provider._local_service_client_pool()
    try:
        assert clients.python_executable == str(interpreter)
        assert Path(clients.python_executable).is_symlink()
    finally:
        provider.shutdown()


def test_managed_provider_prefetch_uses_the_umbrella_scope_lineage(monkeypatch, tmp_path) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": str(tmp_path),
        "workspace_id": "workspace",
        "profile_id": "umbrella",
        "project_id": "project",
        "user_id": "user",
        "scope_session_id": "memory-session",
        "scope_task_id": "memory-task",
        "scope_subtask_id": "memory-subtask",
        "backend": "mempalace",
        "timeout_seconds": 10,
    }
    provider._session_id = "hermes-session"
    lineage_calls: list[dict[str, object]] = []
    service_calls: list[dict[str, object]] = []

    class Pool:
        def call(self, _identity, _tool, _payload, **kwargs):
            service_calls.append(kwargs)
            return {"ok": True}

    monkeypatch.setattr(provider, "_local_service_client_pool", lambda: Pool())
    monkeypatch.setattr(
        hermes_provider,
        "build_local_service_identity",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        hermes_provider,
        "local_service_scope_lineage",
        lambda _identity, **kwargs: lineage_calls.append(kwargs) or ("scope",),
    )

    assert provider._call("resolve_active_mask", {}) == {"ok": True}
    assert service_calls[0]["session_id"] == "memory-session"
    assert lineage_calls == [
        {
            "session_id": "memory-session",
            "project_id": "project",
            "task_id": "memory-task",
            "subtask_id": "memory-subtask",
            "user_id": "user",
        }
    ]


@pytest.mark.parametrize(
    ("failure", "reason_code", "retryable"),
    [
        (PermissionError("store access denied"), "access_denied", False),
        (BlockingIOError("store lock busy"), "store_lock_unavailable", True),
        (MemoryBusyError("store lock busy"), "store_lock_unavailable", True),
        (CanonicalHeadConflict("stale"), "canonical_revision_conflict", False),
        (AmbiguousEvidenceRefError("ambiguous"), "ambiguous_evidence_ref", False),
        (OSError("local IO failed"), "memory_operation_failed", False),
        (ConnectionError("listener disconnected"), "native_transport_failure", True),
        (TimeoutError("listener timed out"), "transport_timeout", True),
    ],
)
def test_provider_classifies_local_service_exceptions_by_type(
    monkeypatch, tmp_path, failure: Exception, reason_code: str, retryable: bool
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": str(tmp_path),
        "workspace_id": "workspace",
        "backend": "mempalace",
        "timeout_seconds": 10,
    }
    provider._session_id = "session-1"

    class Pool:
        def call(self, *_args, **_kwargs):
            raise failure

    monkeypatch.setattr(provider, "_local_service_client_pool", lambda: Pool())
    monkeypatch.setattr(hermes_provider, "build_local_service_identity", lambda **_kwargs: object())
    monkeypatch.setattr(
        hermes_provider, "local_service_scope_lineage", lambda *_args, **_kwargs: ()
    )

    result = provider._call("resolve_active_mask", {})

    assert result["reason_code"] == reason_code
    assert result["retryable"] is retryable
    assert result["error_type"] == type(failure).__name__


def test_compact_recall_keeps_actionable_temporal_validation_error(monkeypatch):
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {"workspace_id": "ws", "backend": "mempalace"}
    monkeypatch.setattr(
        provider,
        "_call",
        lambda *a, **k: {
            "ok": False,
            "reason_code": "memory_operation_failed",
            "error_type": "ValueError",
            "retryable": False,
            "error": "ValueError: RecallPlan known_as_of requires a timezone",
        },
    )
    result = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_recall_blocks",
            {
                "query": "evidence",
                "logical_block_ids": ["leaf"],
                "known_as_of": "2023-01-20",
            },
        )
    )
    assert result["ok"] is False
    assert "known_as_of requires a timezone" in result["diagnostic"]
    assert result["retryable"] is False


def test_managed_prefetch_without_grants_does_not_build_the_full_catalog(
    monkeypatch,
    tmp_path,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": str(tmp_path),
        "workspace_id": "alpha",
        "backend": "mempalace",
        "tool_access": "prefetch_only",
    }
    provider._session_id = "session-a"
    calls: list[tuple[str, dict[str, object]]] = []
    telemetry: list[dict[str, object]] = []
    monkeypatch.setattr(
        hermes_provider,
        "record_memory_event",
        lambda repo_root, **kwargs: telemetry.append({"repo_root": repo_root, **kwargs}),
    )
    monkeypatch.setattr(
        provider,
        "_call",
        lambda tool, args: (
            calls.append((tool, args))
            or {
                "ok": True,
                "roots": [
                    {
                        "block_id": "semantic_memory_index",
                        "description": "Durable workspace knowledge",
                        "tags": ["semantic"],
                        "scope": "workspace",
                        "temperature": "warm",
                        "trust": "advisory",
                        "child_ids": ["stable_project_facts"],
                        "purpose": {"read": "Choose for durable facts"},
                        "record_ids": ["must-not-leak"],
                        "content": "must-not-leak",
                    }
                ],
            }
        ),
    )

    first = provider.prefetch("secret query", session_id="session-a")
    second = provider.prefetch("different query", session_id="session-b")
    provider.queue_prefetch("ignored", session_id="session-a")

    assert first == second
    assert calls == []
    assert first == ""
    assert [row["event_type"] for row in telemetry] == [
        "context_prefetch",
        "context_prefetch",
    ]
    assert [row["run_id"] for row in telemetry] == ["session-a", "session-b"]
    assert all(row["candidate_count"] == 0 for row in telemetry)
    assert all(row["outcome"] == "empty" for row in telemetry)
    assert "secret query" not in json.dumps(telemetry, default=str)


def test_managed_prefetch_injects_only_exact_granted_durable_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": str(tmp_path),
        "workspace_id": "workspace",
        "backend": "mempalace",
        "tool_access": "prefetch_only",
        "source_horizon": "durable",
        "read_block_ids": ["project-knowledge"],
        "read_record_ids": ["record-1"],
        "active_mask_id": "mask-1",
        "active_mask_revision": 3,
        "recall_max_tokens": 17,
    }
    calls: list[tuple[str, dict[str, object]]] = []

    def call(tool: str, args: dict[str, object]) -> dict[str, object]:
        calls.append((tool, args))
        if tool == "block_map":
            return {"ok": True, "roots": []}
        if tool == "resolve_active_mask":
            return {"ok": True, "lease": {}}
        if tool == "build_granted_context":
            return {
                "ok": True,
                "items": [{"record_id": "record-1"}],
                "rendered_context": "Accepted project context.",
            }
        raise AssertionError(tool)

    monkeypatch.setattr(provider, "_call", call)
    provider._config["scope_session_id"] = "memory-session"
    monkeypatch.setattr(provider, "_memory_run_for_session", lambda _session_id="": "run-1")
    monkeypatch.setattr(hermes_provider, "record_memory_event", lambda *args, **kwargs: None)

    prefetched = provider.prefetch("Current phase objective", session_id="session-1")

    assert [tool for tool, _args in calls] == [
        "build_granted_context",
    ]
    build_call = next(args for tool, args in calls if tool == "build_granted_context")
    assert build_call["active_mask_id"] == "mask-1"
    assert build_call["active_mask_revision"] == 3
    assert build_call["logical_block_ids"] == ["project-knowledge"]
    assert build_call["record_ids"] == ["record-1"]
    assert build_call["max_tokens"] == 17
    assert build_call["include_candidates"] is False
    assert build_call["source_horizon"] == "durable"
    assert build_call["scope_chain"][0] == "session:memory-session"
    assert prefetched == "Accepted project context."


def test_managed_prefetch_without_mask_skips_catalog_and_mask_resolution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": str(tmp_path),
        "workspace_id": "workspace",
        "backend": "mempalace",
        "tool_access": "prefetch_only",
        "source_horizon": "durable",
        "read_block_ids": ["project-knowledge"],
        "read_record_ids": [],
        "active_mask_id": "",
    }
    calls: list[tuple[str, dict[str, object]]] = []

    def call(tool: str, args: dict[str, object]) -> dict[str, object]:
        calls.append((tool, args))
        if tool == "build_granted_context":
            return {"ok": True, "rendered_context": "Accepted exact context."}
        raise AssertionError(tool)

    monkeypatch.setattr(provider, "_call", call)
    provider._config["scope_session_id"] = "memory-session"
    monkeypatch.setattr(provider, "_memory_run_for_session", lambda _session_id="": "run-1")
    monkeypatch.setattr(hermes_provider, "record_memory_event", lambda *args, **kwargs: None)

    prefetched = provider.prefetch("Current phase objective", session_id="session-1")

    assert [tool for tool, _args in calls] == ["build_granted_context"]
    assert prefetched == "Accepted exact context."


def test_profile_recall_budget_is_not_inflated_by_defaults_or_ambient_environment(
    monkeypatch,
    tmp_path,
) -> None:
    config_dir = tmp_path / "um_mem"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"recall_max_tokens": 17}),
        encoding="utf-8",
    )
    monkeypatch.setenv("UM_MEM_PROVIDER_RECALL_MAX_TOKENS", "8000")
    monkeypatch.delenv("UM_MEM_ALLOW_ENV_CONFIG_OVERRIDE", raising=False)
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = hermes_provider._load_config(tmp_path)

    assert provider._recall_max_tokens() == 17
    assert provider._recall_max_tokens(1) == 1
    assert provider._recall_max_tokens(8_000) == 17


def test_resource_pointers_require_verified_host_context_and_hide_logs(tmp_path: Path) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._host_context = {
        "project_path": str(tmp_path),
        "loaded_skill_manifest": {"verified": False, "skills": ["must-not-claim-loaded"]},
        "task_id": "",
        "checkpoint_id": "",
        "log_pointers": [
            {
                "uri": "file:///secret/transcript.jsonl",
                "sha256": "sha256:abc",
                "time": "2026-07-22T10:00:00+00:00",
                "content": "must-not-ingest",
            }
        ],
    }

    pointers = provider._resource_pointer_summary()
    serialized = json.dumps(pointers)

    assert pointers["project_resources"]["root_label"] == tmp_path.name
    assert pointers["project_resources"]["path_sha256"].startswith("sha256:")
    assert pointers["skills_index"] == {
        "included": False,
        "exclusion": "verified_loaded_skill_manifest_unavailable",
    }
    assert pointers["task_state"] == {
        "included": False,
        "exclusion": "stable_task_or_checkpoint_id_unavailable",
    }
    assert pointers["logs_transcripts"] == {
        "included": False,
        "exclusion": "hidden_witness_only",
        "pointer_count": 1,
    }
    assert "must-not-claim-loaded" not in serialized
    assert "file:///secret" not in serialized
    assert "must-not-ingest" not in serialized


def test_cli_initialize_binds_observable_cwd_and_session_task_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        hermes_provider,
        "_load_config",
        lambda _home=None: {
            "workspace_id": "alpha",
            "backend": "mempalace",
            "repo_root": str(tmp_path / "memory"),
        },
    )

    provider.initialize(
        "session-1",
        platform="cli",
        hermes_home=tmp_path,
    )

    assert provider._host_context["project_path"] == str(tmp_path.resolve())
    assert provider._host_context["task_id"] == "hermes-session-task:session-1"


def test_managed_initialize_warms_transport_before_bounded_prefetch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        hermes_provider,
        "_load_config",
        lambda _home=None: {
            "workspace_id": "alpha",
            "backend": "mempalace",
            "repo_root": str(tmp_path / "memory"),
            "tool_access": "prefetch_only",
            "read_block_ids": ["project-knowledge"],
            "read_record_ids": [],
            "active_mask_id": "",
        },
    )
    monkeypatch.setattr(
        provider,
        "_call",
        lambda tool, args: calls.append((tool, args)) or {"ok": True},
    )

    provider.initialize("session-1", hermes_home=tmp_path)

    assert calls == [
        (
            "resolve_active_mask",
            {
                "mask_id": "",
                "max_tokens": 0,
                "expected_revision": None,
            },
        )
    ]


def test_verified_loaded_skill_and_stable_task_manifests_are_content_free() -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._host_context = {
        "project_path": "",
        "loaded_skill_manifest": {
            "verified": True,
            "skills": ["um-mem-project", "um-mem-python-discipline"],
        },
        "task_id": "task-7",
        "checkpoint_id": "checkpoint-2",
        "log_pointers": [],
    }

    pointers = provider._resource_pointer_summary()

    assert pointers["skills_index"]["loaded_skill_ids"] == [
        "um-mem-project",
        "um-mem-python-discipline",
    ]
    assert pointers["skills_index"]["manifest_sha256"].startswith("sha256:")
    assert pointers["task_state"]["task_id_sha256"].startswith("sha256:")
    assert pointers["task_state"]["checkpoint_id_sha256"].startswith("sha256:")


def test_agent_retention_correlation_is_bound_to_raw_witness_and_route() -> None:
    payload = {
        "operation": "add",
        "requested_block_id": "decisions_and_contracts",
        "metadata": {"raw_witness_id": "raw-witness-1"},
    }

    first = hermes_provider._host_mutation_correlation_id(payload)
    replay = hermes_provider._host_mutation_correlation_id(dict(payload))
    different_route = hermes_provider._host_mutation_correlation_id(
        {**payload, "requested_block_id": "workspace_episodes"}
    )

    assert first.startswith("hermes-agent-retain:")
    assert replay == first
    assert different_route != first


def test_provider_tools_enforce_block_first_recall_and_selected_block_write(
    monkeypatch,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "user_id": "user-1",
        "backend": "mempalace",
        "repo_root": "unused",
        "recall_max_tokens": 256,
        "recall_limit": 8,
        "query_max_chars": 4096,
    }
    provider._session_id = "session-1"
    calls: list[tuple[str, dict[str, object]]] = []

    def call(tool: str, args: dict[str, object]) -> dict[str, object]:
        calls.append((tool, args))
        return {
            "ok": True,
            "backend": "mempalace",
            "items": [],
            "rendered_context": "",
            "diagnostics": {},
        }

    monkeypatch.setattr(provider, "_call", call)
    monkeypatch.setattr(
        provider,
        "_remember_and_export",
        lambda payload, **kwargs: (
            calls.append((str(kwargs["tool"]), payload)) or {"ok": True, "saved": True}
        ),
    )
    monkeypatch.setattr(
        provider,
        "_observe_many_and_export",
        lambda payloads, **_kwargs: {
            "ok": True,
            "results": [
                {
                    "ok": True,
                    "witness_id": "raw-witness-1",
                    "evidence_refs": [
                        {
                            "ref_type": "artifact",
                            "ref_id": "raw-witness-1",
                            "produced_by": "watcher",
                            "producer_id": "hermes:provider-tool",
                        }
                    ],
                }
                for _payload in payloads
            ],
        },
    )

    schema_by_name = {schema["name"]: schema for schema in provider.get_tool_schemas()}
    assert {
        "um_mem_provider_block_map",
        "um_mem_provider_expand_block",
        "um_mem_provider_recall_blocks",
        "um_mem_provider_search_all",
        "um_mem_provider_read_exact_record",
        "um_mem_provider_read_exact_records",
        "um_mem_provider_operation_status",
        "um_mem_provider_projection_barrier",
        "um_mem_provider_list_records",
        "um_mem_provider_remember",
        "um_mem_provider_amend",
        "um_mem_provider_create_block",
        "um_mem_provider_amend_block",
        "um_mem_provider_archive_block",
        "um_mem_provider_propose_promotion",
        "um_mem_provider_recent_verification",
        "um_mem_provider_export_graph",
    } <= set(schema_by_name)
    assert "um_mem_provider_recall" not in schema_by_name
    assert set(schema_by_name["um_mem_provider_recall_blocks"]["parameters"]["properties"]) == {
        "query",
        "logical_block_ids",
        "limit",
        "source_horizon",
        "record_ids",
        "max_tokens",
        "source_speakers",
        "tags",
        "temporal_mode",
        "as_of",
        "known_as_of",
        "revision_id",
    }
    assert set(schema_by_name["um_mem_provider_search_all"]["parameters"]["properties"]) == {
        "query",
        "limit",
        "source_horizon",
        "max_tokens",
        "source_speakers",
        "tags",
        "temporal_mode",
        "as_of",
        "known_as_of",
        "revision_id",
    }
    assert set(schema_by_name["um_mem_provider_remember"]["parameters"]["required"]) == {
        "content",
        "operation",
        "requested_block_id",
    }
    assert set(schema_by_name["um_mem_provider_remember"]["parameters"]["properties"]) == {
        "content",
        "requested_block_id",
        "title",
        "tags",
        "operation",
        "memory_id",
        "reason",
        "repair_of",
        "evidence_refs",
        "child_claims",
        "write_mode",
        "source_segments",
        "operation_id",
    }
    remember_parameters = schema_by_name["um_mem_provider_remember"]["parameters"]
    assert "default" not in remember_parameters["properties"]["operation"]
    assert len(remember_parameters["oneOf"]) == 3

    exact = json.loads(
        provider.handle_tool_call("um_mem_provider_read_exact_record", {"record_id": "record-1"})
    )
    assert exact["ok"] is True
    assert calls[-1] == ("read_exact_record", {"record_id": "record-1"})
    invalid_batch = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_read_exact_records", {"record_ids": ["record-1", "record-1"]}
        )
    )
    assert invalid_batch["ok"] is False
    status = json.loads(
        provider.handle_tool_call("um_mem_provider_operation_status", {"operation_id": "op-1"})
    )
    assert status["ok"] is True
    assert calls[-1] == ("operation_status", {"operation_id": "op-1"})
    listed = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_list_records", {"scope": "workspace", "scope_id": "alpha", "limit": 2}
        )
    )
    assert listed["ok"] is True
    assert calls[-1] == ("list_records", {"scope": "workspace", "scope_id": "alpha", "limit": 2})
    calls.clear()

    unscoped = json.loads(
        provider.handle_tool_call("um_mem_provider_recall_blocks", {"query": "parser"})
    )
    assert unscoped["error"] == "logical_block_ids must be a nonempty array"
    assert calls == []

    recalled = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_recall_blocks",
            {"query": "parser", "logical_block_ids": ["code_resources"]},
        )
    )
    assert recalled["ok"] is True
    assert calls[-1][0] == "build_context"
    assert calls[-1][1]["logical_block_ids"] == ["code_resources"]
    assert calls[-1][1]["search_all"] is False

    temporal = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_recall_blocks",
            {
                "query": "parser",
                "logical_block_ids": ["code_resources"],
                "known_as_of": "2020-01-15T00:00:00Z",
            },
        )
    )
    assert temporal["ok"] is True
    assert calls[-1][1]["known_as_of"] == "2020-01-15T00:00:00Z"
    assert calls[-1][1]["retrieval_mode"] == "semantic"

    searched = json.loads(
        provider.handle_tool_call("um_mem_provider_search_all", {"query": "parser"})
    )
    assert searched["ok"] is True
    assert calls[-1][1]["search_all"] is True
    assert "logical_block_ids" not in calls[-1][1]

    mixed = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_search_all",
            {"query": "parser", "logical_block_ids": ["code_resources"]},
        )
    )
    assert mixed["error"] == "unknown Hermes provider search_all argument: logical_block_ids"

    missing_block = json.loads(
        provider.handle_tool_call("um_mem_provider_remember", {"content": "fact"})
    )
    assert missing_block["error"] == "requested_block_id is required"

    missing_operation = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_remember",
            {"content": "fact", "requested_block_id": "stable_project_facts"},
        )
    )
    assert missing_operation["error"] == "remember operation is required"
    assert calls[-1][0] == "build_context"

    ambiguous_add = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_remember",
            {
                "content": "must not become a second current fact",
                "operation": "add",
                "memory_id": "memory-1",
                "requested_block_id": "stable_project_facts",
            },
        )
    )
    assert ambiguous_add["error"] == "add operation forbids memory_id"
    assert calls[-1][0] == "build_context"

    added = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_remember",
            {
                "content": "new durable fact",
                "operation": "add",
                "requested_block_id": "stable_project_facts",
            },
        )
    )
    assert added["ok"] is True
    assert calls[-1][0] == "remember"
    assert calls[-1][1]["operation"] == "add"
    assert "memory_id" not in calls[-1][1]
    assert calls[-1][1]["metadata"]["raw_witness_id"] == "raw-witness-1"
    assert calls[-1][1]["evidence_refs"][0]["ref_id"] == "raw-witness-1"

    calls_before_repair = len(calls)
    repaired = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_remember",
            {
                "content": "new durable fact",
                "operation": "repair",
                "repair_of": "raw-witness-1",
                "evidence_refs": [
                    {
                        "ref_type": "artifact",
                        "ref_id": "raw-witness-1",
                        "produced_by": "watcher",
                        "producer_id": "hermes:provider-tool",
                    }
                ],
                "requested_block_id": "stable_project_facts",
                "child_claims": [],
            },
        )
    )
    assert repaired["ok"] is True
    assert calls[-1][0] == "remember"
    assert calls[-1][1]["operation"] == "repair"
    assert calls[-1][1]["repair_of"] == "raw-witness-1"
    assert calls[-1][1]["evidence_refs"][0]["ref_id"] == "raw-witness-1"
    assert len(calls) == calls_before_repair + 1

    corrected = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_remember",
            {
                "content": "corrected fact",
                "operation": "correct",
                "memory_id": "memory-1",
                "requested_block_id": "stable_project_facts",
            },
        )
    )
    assert corrected["ok"] is True
    assert calls[-1][0] == "remember"
    assert calls[-1][1]["operation"] == "correct"
    assert calls[-1][1]["memory_id"] == "memory-1"
    assert calls[-1][1]["requested_block_id"] == "stable_project_facts"
    assert calls[-1][1]["kind"] == "observation"

    amended = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_amend",
            {
                "memory_id": "memory-1",
                "expected_revision": "revision-1",
                "reason": "new source evidence",
                "content": "amended fact",
                "requested_block_id": "stable_project_facts",
            },
        )
    )
    assert amended["ok"] is True
    assert calls[-1][0] == "amend"
    assert calls[-1][1]["expected_revision"] == "revision-1"
    assert calls[-1][1]["evidence_refs"][0]["ref_id"] == "raw-witness-1"

    created_block = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_create_block",
            {
                "name": "Research findings",
                "synopsis": "Verified research evidence.",
                "parent_id": "semantic_memory_index",
                "reason": "new durable subject",
            },
        )
    )
    assert created_block["ok"] is True
    assert calls[-1] == (
        "create_block",
        {
            "name": "Research findings",
            "synopsis": "Verified research evidence.",
            "parent_id": "semantic_memory_index",
            "reason": "new durable subject",
        },
    )

    monkeypatch.setattr(
        provider,
        "_call_many",
        lambda tool, payloads, caller_role="parent": (
            [
                {
                    "ok": True,
                    "subject": asdict(
                        CandidateVerificationSubject(
                            workspace_id="alpha",
                            profile_id="default",
                            memory_id="reflection-1",
                            revision_id="reflection-revision-1",
                            payload_hash="payload",
                            source_evidence_hash="evidence",
                            target_block_id="verified_lessons",
                            authority="advisory",
                            scope="workspace",
                            scope_id="alpha",
                        )
                    ),
                    "candidate": {"content": "verified lesson"},
                    "sources": [{"content": "raw evidence"}],
                }
            ]
            if tool == "verification_subject"
            else [
                {
                    "ok": True,
                    "saved": True,
                    "source_memory_id": payloads[0]["memory_id"],
                    "result_memory_id": "verified-lesson-1",
                    "actual_block_id": "verified_lessons",
                    "status": "accepted",
                    "authority": "advisory",
                    "verification_status": "verified",
                    "promotion_decision": "accepted",
                    "tribunal_decision": "accept",
                    "evidence_refs": payloads[0]["evidence_refs"],
                    "backend": "mempalace",
                    "caller_role": caller_role,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        provider,
        "_semantic_review",
        lambda *_args, **_kwargs: VerificationReportRef(
            "semantic-1", "payload", "workspace", "diff", "", "reviewer", True, "ledger"
        ),
    )
    promoted = json.loads(
        provider.handle_tool_call(
            "um_mem_provider_propose_promotion",
            {
                "memory_id": "reflection-1",
                "expected_source_revision_id": "reflection-revision-1",
                "reason": "passed deterministic reuse",
                "target_block_id": "verified_lessons",
                "verification_reports": [{"report_id": "report-1"}],
            },
        )
    )
    assert promoted["ok"] is True
    assert promoted["status"] == "accepted"
    assert promoted["authority"] == "advisory"
    assert promoted["verification_status"] == "verified"
    recent = json.loads(provider.handle_tool_call("um_mem_provider_recent_verification", {}))
    assert recent == {
        "ok": True,
        "verification_reports": [],
        "passed_count": 0,
    }

    exported = json.loads(provider.handle_tool_call("um_mem_provider_export_graph", {}))
    assert exported["ok"] is True
    assert calls[-1] == ("export_graph", {})


def test_sync_turn_retains_one_raw_only_conversation_observation(
    monkeypatch,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "auto_retain": True,
    }
    provider._session_id = "session-1"
    observations: list[dict[str, object]] = []
    monkeypatch.setattr(provider, "_submit_observed_tool_outcomes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        provider,
        "_observe_many_and_export",
        lambda payloads, **_kwargs: observations.extend(payloads) or {"ok": True},
    )

    provider.sync_turn("User raw turn", "Assistant raw turn", session_id="session-1")

    assert len(observations) == 1
    assert observations[0]["event_kind"] == "conversation_turn"
    assert "User raw turn" in observations[0]["raw_evidence"]
    assert "Assistant raw turn" in observations[0]["raw_evidence"]
    assert "atomic_claim" not in observations[0]


def test_sync_turn_does_not_repeat_tool_retention_owned_by_live_observer(
    monkeypatch,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "auto_retain": True,
    }
    provider._session_id = "session-1"
    observations: list[dict[str, object]] = []
    outcomes: list[object] = []
    monkeypatch.setattr(
        provider,
        "_submit_observed_tool_outcomes",
        lambda messages, **_kwargs: outcomes.extend(messages or ()),
    )
    monkeypatch.setattr(
        provider,
        "_observe_many_and_export",
        lambda payloads, **_kwargs: observations.extend(payloads) or {"ok": True},
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "write-1",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":"src/a.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "write-1",
            "name": "write_file",
            "content": '{"ok":true}',
        },
    ]

    provider.sync_turn(
        "User turn",
        "Assistant turn",
        session_id="session-1",
        messages=messages,
    )

    assert outcomes == messages
    assert [observation["event_kind"] for observation in observations] == ["conversation_turn"]


def test_post_tool_call_captures_live_test_receipt_for_promotion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": str(tmp_path / "memory"),
        "workspace_id": "alpha",
        "backend": "mempalace",
        "auto_retain": True,
    }
    provider._session_id = "session-1"
    provider._host_context = {"project_path": str(tmp_path)}
    report = {
        "report_id": "ledger-report-1",
        "report_hash": "sha256:test",
        "workspace_hash": "workspace-hash",
        "diff_hash": "diff-hash",
        "produced_after_event_id": "",
        "verifier_id": "hermes:session:session-1",
        "passed": True,
        "ledger_hash": "ledger-hash",
    }
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "_call_many",
        lambda tool, payloads, caller_role="parent": (
            captured.extend(payloads)
            or [
                {
                    "ok": True,
                    "saved": True,
                    "witness_id": "witness-1",
                    "memory_id": "",
                    "requested_block_id": "",
                    "actual_block_id": "",
                    "route_outcome": "needs_agent_choice",
                    "reason": "semantic_placement_requires_explicit_agent_choice",
                    "verification_report": report,
                    "evidence_refs": [],
                }
            ]
        ),
    )
    monkeypatch.setattr(provider, "_export_obsidian_live", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(hermes_provider, "record_memory_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hermes_provider, "_workspace_changed_files", lambda _root: ("src/a.py",))

    provider.on_post_tool_call(
        tool_name="terminal",
        args={"command": "python -m pytest -q", "observation_kind": "test"},
        result="3 passed",
        session_id="session-1",
        tool_call_id="test-1",
        status="ok",
    )

    assert captured[0]["event_kind"] == "test_passed"
    assert captured[0]["workspace_root"] == str(tmp_path)
    assert captured[0]["changed_files"] == ["src/a.py"]
    recent = json.loads(provider.handle_tool_call("um_mem_provider_recent_verification", {}))
    assert recent["verification_reports"] == [report]
    assert recent["passed_count"] == 1


def test_pre_and_post_tool_hooks_preserve_created_file_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": str(tmp_path / "memory"),
        "workspace_id": "alpha",
        "backend": "mempalace",
        "auto_retain": True,
    }
    provider._session_id = "session-1"
    provider._host_context = {"project_path": str(tmp_path)}
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "_call_many",
        lambda _tool, payloads, caller_role="parent": (
            captured.extend(payloads)
            or [
                {
                    "ok": True,
                    "saved": True,
                    "witness_id": "witness-1",
                    "memory_id": "",
                    "requested_block_id": "",
                    "actual_block_id": "",
                    "route_outcome": "needs_agent_choice",
                    "reason": "semantic_placement_requires_explicit_agent_choice",
                    "evidence_refs": [],
                }
            ]
        ),
    )
    monkeypatch.setattr(provider, "_export_obsidian_live", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(hermes_provider, "record_memory_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hermes_provider, "_workspace_changed_files", lambda _root: ())

    provider.on_pre_tool_call(
        tool_name="write_file",
        args={"path": "src/new_module.py"},
        session_id="session-1",
        tool_call_id="write-1",
    )
    provider.on_post_tool_call(
        tool_name="write_file",
        args={"path": "src/new_module.py"},
        result='{"ok":true}',
        session_id="session-1",
        tool_call_id="write-1",
        status="ok",
    )

    assert captured[0]["event_kind"] == "file_created"


def test_workspace_changed_files_excludes_parent_repository_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed_root = tmp_path / "nested" / "workspace"
    observed_root.mkdir(parents=True)
    monkeypatch.setattr(hermes_provider, "_git_or_folder_root", lambda _path: tmp_path)
    monkeypatch.setattr(
        hermes_provider.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                " M unrelated.py\n?? nested/workspace/src/inside.py\n?? nested/other/outside.py\n"
            ),
        ),
    )

    changed = hermes_provider._workspace_changed_files(str(observed_root))

    assert changed == ("src/inside.py",)


def test_register_requires_and_wires_public_post_tool_hook() -> None:
    class Context:
        def __init__(self) -> None:
            self.provider = None
            self.hooks: list[tuple[str, object]] = []

        def register_memory_provider(self, provider) -> None:
            self.provider = provider

        def register_hook(self, name: str, callback) -> None:
            self.hooks.append((name, callback))

    context = Context()
    hermes_provider.register(context)

    assert isinstance(context.provider, hermes_provider.UmMemMemoryProvider)
    assert context.hooks == [
        ("pre_tool_call", context.provider.on_pre_tool_call),
        ("post_tool_call", context.provider.on_post_tool_call),
    ]


def test_sync_turn_hidden_witness_preserves_the_complete_raw_turn(monkeypatch) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "auto_retain": True,
    }
    provider._session_id = "session-1"
    observations: list[dict[str, object]] = []
    monkeypatch.setattr(provider, "_submit_observed_tool_outcomes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        provider,
        "_observe_many_and_export",
        lambda payloads, **_kwargs: observations.extend(payloads) or {"ok": True},
    )
    user_text = "user-line\n" + "u" * 1800 + "-user-tail"
    assistant_text = "assistant-line\n" + "a" * 1800 + "-assistant-tail"

    provider.sync_turn(user_text, assistant_text, session_id="session-1")

    content = str(observations[0]["raw_evidence"])
    assert user_text in content
    assert assistant_text in content
    assert content.endswith("-assistant-tail")


def test_session_end_submits_boundary_state_and_compact_episode(monkeypatch) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "auto_retain": True,
    }
    provider._session_id = "session-1"
    host_states: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "_submit_host_state",
        lambda **payload: host_states.append(payload),
    )
    monkeypatch.setattr(
        provider,
        "_observe_many_and_export",
        lambda payloads, **_kwargs: observations.extend(payloads) or {"ok": True},
    )

    provider.on_session_end(
        [
            {"role": "user", "content": "secret conversation"},
            {"role": "assistant", "content": "secret answer"},
        ]
    )

    assert len(host_states) == 1
    assert host_states[0]["event_type"] == "session_boundary"
    assert "secret conversation" not in json.dumps(host_states)
    assert "secret answer" not in json.dumps(host_states)
    assert len(observations) == 1
    assert observations[0]["event_kind"] == "session_episode"
    assert "secret conversation" not in json.dumps(observations)
    assert "secret answer" not in json.dumps(observations)


def test_session_end_boundary_identity_includes_host_turn_sequence(monkeypatch) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "auto_retain": True,
    }
    provider._session_id = "session-1"
    observations: list[dict[str, object]] = []
    monkeypatch.setattr(provider, "_submit_host_state", lambda **_payload: None)
    monkeypatch.setattr(
        provider,
        "_observe_many_and_export",
        lambda payloads, **_kwargs: observations.extend(payloads) or {"ok": True},
    )

    provider._host_turn_sequence_by_session["session-1"] = 1
    provider.on_session_end([])
    provider._host_turn_sequence_by_session["session-1"] = 2
    provider.on_session_end([])

    assert observations[0]["event_id"] != observations[1]["event_id"]
    assert observations[0]["raw_evidence"].endswith("boundary 1.")
    assert observations[1]["raw_evidence"].endswith("boundary 2.")


def test_repeated_session_end_reuses_exact_boundary_observation(monkeypatch) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "auto_retain": True,
    }
    provider._session_id = "session-1"
    provider._host_turn_sequence_by_session["session-1"] = 1
    observations: list[dict[str, object]] = []
    monkeypatch.setattr(provider, "_submit_host_state", lambda **_payload: None)
    monkeypatch.setattr(
        provider,
        "_observe_many_and_export",
        lambda payloads, **_kwargs: observations.extend(payloads) or {"ok": True},
    )
    timestamps = iter(("2026-07-30T10:00:00Z", "2026-07-30T10:01:00Z"))
    monkeypatch.setattr(
        hermes_provider.time,
        "strftime",
        lambda *_args, **_kwargs: next(timestamps),
    )

    provider.on_session_end([])
    provider.on_session_end([])

    assert observations[0] == observations[1]


def test_observe_batch_isolates_one_local_service_failure(monkeypatch) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": ".",
        "workspace_id": "alpha",
        "backend": "mempalace",
    }
    provider._session_id = "session-1"

    class Pool:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first event failed")
            return {"ok": True, "witness_id": "witness-2"}

    pool = Pool()
    monkeypatch.setattr(provider, "_local_service_client_pool", lambda: pool)
    monkeypatch.setattr(
        hermes_provider,
        "build_local_service_identity",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        hermes_provider,
        "local_service_scope_lineage",
        lambda *_args, **_kwargs: object(),
    )

    results = provider._call_many(
        "observe",
        [{"event_id": "event-1"}, {"event_id": "event-2"}],
        caller_role="host_hook",
    )

    assert results[0]["ok"] is False
    assert results[0]["reason"] == "local_service_exception"
    assert results[1] == {"ok": True, "witness_id": "witness-2"}


def test_failed_retention_receipt_includes_provider_error(monkeypatch, tmp_path: Path) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "repo_root": str(tmp_path),
        "workspace_id": "alpha",
        "backend": "mempalace",
    }
    provider._session_id = "session-1"
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "_call_many",
        lambda *_args, **_kwargs: [
            {
                "ok": False,
                "error": "idempotency conflict",
                "reason": "local_service_exception",
            }
        ],
    )
    monkeypatch.setattr(
        hermes_provider,
        "record_memory_event",
        lambda *_args, **kwargs: events.append(kwargs),
    )

    result = provider._observe_many_and_export(
        [{"event_id": "event-1", "event_kind": "source_opened"}],
        reason="post_tool_call",
    )

    assert result["ok"] is False
    assert events[0]["reason"] == "local_service_exception"
    assert events[0]["data"]["error"] == "idempotency conflict"


def test_provider_tool_outcomes_exclude_its_own_results() -> None:
    outcomes = hermes_provider._observed_tool_results(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "memory-call",
                        "function": {"name": "um_mem_provider_remember"},
                    },
                    {"id": "shell-call", "function": {"name": "shell"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "memory-call",
                "content": '{"ok":true}',
            },
            {
                "role": "tool",
                "tool_call_id": "shell-call",
                "content": '{"ok":true}',
            },
        ]
    )

    assert [outcome["tool_name"] for outcome in outcomes] == ["shell"]


def test_provider_types_file_test_source_and_ambiguous_tool_events() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "file",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":"src/example.py"}',
                    },
                },
                {
                    "id": "test",
                    "function": {
                        "name": "terminal",
                        "arguments": (
                            '{"command":"python -m pytest tests/test_example.py",'
                            '"observation_kind":"test"}'
                        ),
                    },
                },
                {
                    "id": "source",
                    "function": {
                        "name": "web_extract",
                        "arguments": '{"url":"https://example.test/source"}',
                    },
                },
                {
                    "id": "unknown",
                    "function": {
                        "name": "custom_tool",
                        "arguments": '{"value":"opaque"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "file", "content": '{"ok":true}'},
        {"role": "tool", "tool_call_id": "test", "content": "1 passed"},
        {"role": "tool", "tool_call_id": "source", "content": "primary source"},
        {"role": "tool", "tool_call_id": "unknown", "content": "opaque result"},
    ]

    observations = hermes_provider._typed_tool_observations(
        messages,
        session_id="session-1",
        producer_id="hermes:session-1",
    )

    assert [item["event_kind"] for item in observations] == [
        "file_changed",
        "test_passed",
        "source_opened",
        "ambiguous",
    ]
    assert observations[0]["source_path"] == "src/example.py"
    assert observations[2]["source_uri"] == "https://example.test/source"
    assert observations[3]["atomic_claim"] == ""
    assert all(item["raw_evidence"] for item in observations)


def test_shell_text_never_assigns_semantic_observation_type() -> None:
    observations = hermes_provider._typed_tool_observations(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "shell",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command":"python -m pytest -q"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "shell", "content": '{"exit_code":0}'},
        ],
        session_id="session-1",
        producer_id="hermes:session-1",
    )

    assert observations[0]["event_kind"] == "ambiguous"


def test_observer_types_created_file_report_todo_and_masked_test_receipt() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "created",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":"src/new_module.py"}',
                    },
                },
                {
                    "id": "report",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path":"reports/ab_results.json"}',
                    },
                },
                {
                    "id": "todo",
                    "function": {
                        "name": "todo",
                        "arguments": (
                            '{"todos":[{"id":"1","content":"verify","status":"completed"},'
                            '{"id":"2","content":"report","status":"in_progress"}]}'
                        ),
                    },
                },
                {
                    "id": "masked-test",
                    "function": {
                        "name": "terminal",
                        "arguments": (
                            '{"command":"cd /workspace && python -m pytest -q 2>&1 | tail -20",'
                            '"observation_kind":"run"}'
                        ),
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "created", "content": '{"ok":true}'},
        {
            "role": "tool",
            "tool_call_id": "report",
            "content": '{"ok":true,"artifact_kind":"report"}',
        },
        {"role": "tool", "tool_call_id": "todo", "content": '{"ok":true}'},
        {
            "role": "tool",
            "tool_call_id": "masked-test",
            "content": '{"output":"3 passed","exit_code":0,"error":null}',
        },
    ]

    observations = hermes_provider._typed_tool_observations(
        messages,
        session_id="session-1",
        producer_id="hermes:session-1",
        file_existence_by_call_id={"created": False, "report": False},
    )

    assert [item["event_kind"] for item in observations] == [
        "file_created",
        "report_created",
        "task_checkpoint",
        "run_receipt",
    ]
    assert observations[2]["atomic_claim"] == (
        "Hermes updated 2 host task checkpoints (completed:1, in_progress:1)."
    )


def test_builtin_memory_hook_fails_closed_without_mirroring(
    monkeypatch,
) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    provider._config = {
        "workspace_id": "alpha",
        "backend": "mempalace",
        "auto_retain": True,
    }
    provider._session_id = "session-1"
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "_remember_and_export",
        lambda payload, **_kwargs: writes.append(payload) or {"ok": True},
    )

    monkeypatch.setattr(hermes_provider, "record_memory_event", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="builtin memory hook is disabled"):
        provider.on_memory_write("add", "memory", "generic model memory")
    with pytest.raises(RuntimeError, match="builtin memory hook is disabled"):
        provider.on_memory_write("add", "user", "User prefers terse answers")

    assert writes == []


def test_provider_shutdown_only_closes_shared_clients(monkeypatch) -> None:
    provider = hermes_provider.UmMemMemoryProvider()
    closed: list[bool] = []
    monkeypatch.setattr(provider, "_close_local_service_clients", lambda: closed.append(True))

    provider.shutdown()

    assert closed == [True]


def test_native_remember_identity_uses_invocation_and_preserves_author(tmp_path, monkeypatch):
    import hashlib

    provider = hermes_provider.UmMemMemoryProvider()
    provider._session_id = "session-a"
    provider._host_context = {"main_model": "stale-session-model"}
    provider._config = {"repo_root": str(tmp_path), "workspace_id": "alpha"}
    witnesses = []
    mutations = []

    def observe(payloads, **kwargs):
        payload = payloads[0]
        witness = json.loads(payload["raw_evidence"])
        witnesses.append(witness)
        identifier = hashlib.sha256(payload["raw_evidence"].encode()).hexdigest()
        return {
            "ok": True,
            "results": [{"witness_id": identifier, "evidence_refs": [{"ref_id": identifier}]}],
        }

    def remember(payload, **kwargs):
        mutations.append(hermes_provider._host_mutation_correlation_id(payload))
        return {"ok": True, "saved": True}

    monkeypatch.setattr(provider, "_observe_many_and_export", observe)
    monkeypatch.setattr(provider, "_remember_and_export", remember)
    proposal = {"content": "same exact value", "operation": "add", "requested_block_id": "leaf-a"}
    for invocation in ("call-a", "call-b", "call-a"):
        result = json.loads(
            provider.handle_tool_call(
                "um_mem_provider_remember",
                proposal,
                tool_call_id=invocation,
                author_model="author-a",
            )
        )
        assert result["saved"] is True
    assert mutations[0] != mutations[1]
    assert mutations[0] == mutations[2]
    assert [row["native_tool_call_id"] for row in witnesses] == ["call-a", "call-b", "call-a"]
    assert all(row["author_model_version"] == "author-a" for row in witnesses)
    changed_request = {
        "metadata": {"native_tool_call_id": "call-a", "raw_witness_id": "another-witness"},
        "source_id": "hermes:session-a",
        "content": "changed after delivery",
    }
    assert hermes_provider._host_mutation_correlation_id(changed_request) == mutations[0]
