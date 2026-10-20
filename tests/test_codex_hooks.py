import json
from pathlib import Path

from lethyx.integrations.codex_hooks import (
    codex_host_capability_manifest,
    codex_hooks_status,
    dispatch_codex_hook,
    hook_control_response,
    install_codex_hooks,
    parse_hook_input,
)
from lethyx.memory.local_service import (
    build_local_service_identity,
    shutdown_local_service_listener,
)


class CapturingClientPool:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str, dict[str, object], dict[str, object]]] = []
        self.host_state_calls: list[tuple[object, dict[str, object], dict[str, object]]] = []

    def call(
        self,
        identity: object,
        tool: str,
        arguments: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append((identity, tool, arguments, kwargs))
        return {"ok": True, "saved": True, "memory_id": "captured-event"}

    def submit_host_state(
        self,
        identity: object,
        event: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        self.host_state_calls.append((identity, event, kwargs))
        return {"ok": True, "replayed": False}

    def close(self) -> None:
        raise AssertionError("an injected pool remains caller-owned")


class FailingClientPool(CapturingClientPool):
    def call(
        self,
        identity: object,
        tool: str,
        arguments: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del identity, tool, arguments, kwargs
        raise RuntimeError("listener unavailable")


def _payload(tmp_path: Path, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": "codex-session-123",
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
        "turn_id": "codex-turn-456",
        "prompt": "RAW_PROMPT_SECRET_MUST_NOT_REACH_MEMORY",
        "transcript_path": str(tmp_path / "private-transcript.jsonl"),
        "model": "gpt-test",
    }
    payload.update(overrides)
    return payload


def test_user_prompt_uses_host_hook_lease_without_forwarding_raw_host_content(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    pool = CapturingClientPool()

    result = dispatch_codex_hook(
        _payload(tmp_path),
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )

    assert result == {
        "ok": True,
        "captured": True,
        "event": "UserPromptSubmit",
        "memory_id": "captured-event",
    }
    [(identity, tool, arguments, kwargs)] = pool.calls
    assert identity.caller_role == "host_hook"
    assert identity.repo_root == tmp_path
    assert identity.workspace_id == "alpha"
    assert identity.backend == "mempalace"
    assert tool == "remember"
    assert kwargs["session_id"] == "codex-session-123"
    assert kwargs["timeout_seconds"] == 15
    assert isinstance(kwargs["mutation_correlation_id"], str)
    assert kwargs["mutation_correlation_id"].startswith("codex-hook:")
    serialized = json.dumps(arguments, sort_keys=True)
    assert "RAW_PROMPT_SECRET_MUST_NOT_REACH_MEMORY" not in serialized
    assert "private-transcript.jsonl" not in serialized
    assert "codex-session-123" not in serialized
    assert "codex-turn-456" not in serialized
    assert arguments["content"] == "Observed Codex lifecycle event: UserPromptSubmit."
    assert arguments["kind"] == "observation"
    assert arguments["metadata"] == {
        "host_adapter": "codex_hooks",
        "host_event_name": "UserPromptSubmit",
        "host_event_schema_version": 1,
        "host_identity_sha256": arguments["metadata"]["host_identity_sha256"],
    }
    [(host_state_identity, host_state, host_state_kwargs)] = pool.host_state_calls
    assert host_state_identity.caller_role == "host_state"
    assert host_state_kwargs["session_id"] == "codex-session-123"
    host_state_serialized = json.dumps(host_state, sort_keys=True)
    assert "RAW_PROMPT_SECRET_MUST_NOT_REACH_MEMORY" not in host_state_serialized
    assert "private-transcript.jsonl" not in host_state_serialized
    assert "codex-session-123" not in host_state_serialized
    assert "codex-turn-456" not in host_state_serialized
    assert host_state["event_type"] == "prospective_trigger"
    assert host_state["capability_codes"] == ("codex_hook", "typed_host_state")


def test_turn_scoped_hook_rejects_missing_turn_id_before_calling_listener(tmp_path: Path) -> None:
    pool = CapturingClientPool()

    result = dispatch_codex_hook(
        _payload(tmp_path, turn_id=""),
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )

    assert result == {"ok": False, "error": "codex hook turn_id is required"}
    assert pool.calls == []


def test_tool_event_has_stable_replay_key_and_drops_tool_input_and_output(tmp_path: Path) -> None:
    pool = CapturingClientPool()
    payload = _payload(
        tmp_path,
        hook_event_name="PostToolUse",
        tool_name="Bash",
        tool_use_id="tool-use-789",
        tool_input={"command": "echo RAW_TOOL_INPUT_SECRET"},
        tool_response={"output": "RAW_TOOL_OUTPUT_SECRET"},
    )

    first = dispatch_codex_hook(
        payload,
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )
    second = dispatch_codex_hook(
        payload,
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert (
        pool.calls[0][3]["mutation_correlation_id"] == pool.calls[1][3]["mutation_correlation_id"]
    )
    assert pool.host_state_calls[0][1]["event_id"] == pool.host_state_calls[1][1]["event_id"]
    serialized = json.dumps(pool.calls[0][2], sort_keys=True)
    assert "RAW_TOOL_INPUT_SECRET" not in serialized
    assert "RAW_TOOL_OUTPUT_SECRET" not in serialized


def test_subagent_hook_binds_the_hashed_subtask_to_the_service_lease(tmp_path: Path) -> None:
    pool = CapturingClientPool()

    result = dispatch_codex_hook(
        _payload(
            tmp_path,
            hook_event_name="SubagentStop",
            agent_id="agent-789",
            agent_type="worker",
        ),
        working_directory=tmp_path,
        pool=pool,
    )

    assert result["ok"] is True
    arguments = pool.calls[0][2]
    lineage = pool.calls[0][3]["scope_lineage"]
    assert arguments["subtask_id"].startswith("codex-subagent:")
    assert lineage.task_id.startswith("codex-session-task:")
    assert lineage.subtask_id == arguments["subtask_id"]


def test_distinct_codex_hook_events_emit_distinct_semantic_source_ids(tmp_path: Path) -> None:
    pool = CapturingClientPool()

    first = dispatch_codex_hook(
        _payload(tmp_path, turn_id="codex-turn-456"),
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )
    second = dispatch_codex_hook(
        _payload(tmp_path, turn_id="codex-turn-457"),
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )

    assert first["ok"] is True and second["ok"] is True
    assert pool.calls[0][2]["source_id"] != pool.calls[1][2]["source_id"]


def test_rejects_payload_cwd_that_does_not_match_hook_process_directory(tmp_path: Path) -> None:
    pool = CapturingClientPool()

    result = dispatch_codex_hook(
        _payload(tmp_path, cwd=str(tmp_path / "foreign")),
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )

    assert result == {"ok": False, "error": "codex hook cwd does not match process cwd"}
    assert pool.calls == []


def test_unknown_event_is_rejected_without_turning_into_a_memory_write(tmp_path: Path) -> None:
    pool = CapturingClientPool()

    result = dispatch_codex_hook(
        _payload(tmp_path, hook_event_name="UnknownFutureHook"),
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )

    assert result == {"ok": False, "error": "unsupported Codex hook event"}
    assert pool.calls == []


def test_listener_failure_is_nonblocking_and_does_not_report_a_false_capture(
    tmp_path: Path,
) -> None:
    pool = FailingClientPool()

    result = dispatch_codex_hook(
        _payload(tmp_path),
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )

    assert result == {"ok": False, "error": "local service host capture failed"}


def test_runner_uses_the_real_authenticated_listener_without_a_direct_backend(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    operator_identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    try:
        result = dispatch_codex_hook(
            _payload(tmp_path, hook_event_name="SessionStart", source="startup"),
            working_directory=tmp_path,
            workspace_id="alpha",
            backend="mempalace",
        )
    finally:
        shutdown_local_service_listener(operator_identity)

    assert result["ok"] is True
    assert result["captured"] is True
    assert result["event"] == "SessionStart"
    assert isinstance(result.get("memory_id"), str)


def test_hook_input_is_bounded_and_control_output_never_exposes_memory_result() -> None:
    oversized = b"x" * (1_048_577)

    assert parse_hook_input(oversized) == {
        "ok": False,
        "error": "codex hook input exceeds protocol limit",
    }
    assert hook_control_response("Stop") == {"continue": True}
    assert hook_control_response("SessionEnd") == {"continue": True}
    assert hook_control_response("PostToolUse") == {}


def test_installer_merges_all_supported_events_without_overwriting_user_hooks(
    tmp_path: Path,
) -> None:
    settings = tmp_path / ".codex" / "hooks.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps(
            {
                "custom": {"preserved": True},
                "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "user-stop-hook"}]}]},
            }
        ),
        encoding="utf-8",
    )

    installed = install_codex_hooks(tmp_path, python_executable="C:/Python312/python.exe")
    payload = json.loads(installed.read_text(encoding="utf-8"))

    assert installed == settings
    assert payload["custom"] == {"preserved": True}
    assert set(payload["hooks"]) == {
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PermissionRequest",
        "PostToolUse",
        "PreCompact",
        "PostCompact",
        "SubagentStart",
        "SubagentStop",
        "Stop",
        "SessionEnd",
    }
    stop_handlers = [handler for group in payload["hooks"]["Stop"] for handler in group["hooks"]]
    assert any(handler.get("command") == "user-stop-hook" for handler in stop_handlers)
    lethyx_handlers = [
        handler
        for groups in payload["hooks"].values()
        for group in groups
        for handler in group["hooks"]
        if "lethyx.integrations.codex_hooks" in handler.get("command", "")
    ]
    assert len(lethyx_handlers) == 11
    assert all(
        handler["timeout"] == (3 if event_name == "SessionEnd" else 20)
        for event_name, groups in payload["hooks"].items()
        for group in groups
        for handler in group["hooks"]
        if "lethyx.integrations.codex_hooks" in handler.get("command", "")
    )

    status = codex_hooks_status(tmp_path)
    assert status["installed"] is True
    assert status["installed_events"] == sorted(payload["hooks"])
    assert status["missing_events"] == []
    assert status["implementation"] == "packaged-native-hooks"


def test_installer_replaces_only_its_own_stale_command_and_is_idempotent(
    tmp_path: Path,
) -> None:
    first = install_codex_hooks(tmp_path, python_executable="C:/old/python.exe")
    second = install_codex_hooks(tmp_path, python_executable="C:/new/python.exe")

    assert first == second
    payload = json.loads(second.read_text(encoding="utf-8"))
    commands = [
        handler["command"]
        for groups in payload["hooks"].values()
        for group in groups
        for handler in group["hooks"]
        if "lethyx.integrations.codex_hooks" in handler.get("command", "")
    ]
    assert len(commands) == 11
    assert all("C:/new/python.exe" in command for command in commands)
    assert not any("C:/old/python.exe" in command for command in commands)


def test_stop_is_turn_completion_and_session_end_is_a_distinct_boundary(
    tmp_path: Path,
) -> None:
    pool = CapturingClientPool()
    stop = dispatch_codex_hook(
        _payload(tmp_path, hook_event_name="Stop"),
        working_directory=tmp_path,
        pool=pool,
    )
    end = dispatch_codex_hook(
        _payload(tmp_path, hook_event_name="SessionEnd", turn_id=None, reason="other"),
        working_directory=tmp_path,
        pool=pool,
    )

    assert stop["ok"] is True
    assert end["ok"] is True
    assert pool.host_state_calls[0][1]["event_type"] == "turn_completion"
    assert pool.host_state_calls[1][1]["event_type"] == "session_boundary"
    assert pool.calls[1][0].profile_id == "default"


def test_capability_manifest_remains_fail_closed_before_full_real_host_e2e() -> None:
    manifest = codex_host_capability_manifest()

    assert manifest.host_id == "codex"
    assert manifest.transport == "native_hooks+local_service+mcp"
    assert manifest.conformance_passed is False
    assert manifest.stable_turn_id is True
    assert manifest.pre_turn_delivery is False
    assert manifest.supports("stable_turn_id") is False


def test_pre_tool_and_permission_hooks_capture_no_tool_input_and_never_decide_authority(
    tmp_path: Path,
) -> None:
    pool = CapturingClientPool()
    for event_name in ("PreToolUse", "PermissionRequest"):
        payload = _payload(
            tmp_path,
            hook_event_name=event_name,
            tool_name="Bash",
            tool_use_id="tool-pre-123",
            tool_input={
                "command": "RAW_PRE_TOOL_COMMAND_SECRET",
                "description": "RAW_APPROVAL_REASON_SECRET",
            },
        )

        result = dispatch_codex_hook(
            payload,
            working_directory=tmp_path,
            pool=pool,
        )

        assert result["ok"] is True
        assert hook_control_response(event_name) == {}

    serialized = json.dumps([call[2] for call in pool.calls], sort_keys=True)
    assert "RAW_PRE_TOOL_COMMAND_SECRET" not in serialized
    assert "RAW_APPROVAL_REASON_SECRET" not in serialized
    assert pool.host_state_calls[0][1]["event_type"] == "prospective_trigger"
    assert pool.host_state_calls[1][1]["event_type"] == "high_risk_boundary"
