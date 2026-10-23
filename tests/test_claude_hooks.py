import json
from pathlib import Path

import pytest

from um_mem.integrations.claude_hooks import (
    claude_hooks_status,
    claude_host_capability_manifest,
    dispatch_claude_hook,
    install_claude_hooks,
    parse_hook_input,
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
        return {"ok": True, "saved": True, "memory_id": "claude-event"}

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


def _payload(tmp_path: Path, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": "claude-session-123",
        "cwd": str(tmp_path),
        "hook_event_name": "PostToolUse",
        "tool_name": "Write",
        "tool_use_id": "toolu-456",
        "tool_input": {
            "file_path": str(tmp_path / "secret.py"),
            "content": "RAW_CLAUDE_TOOL_INPUT_SECRET",
        },
        "tool_response": {"output": "RAW_CLAUDE_TOOL_OUTPUT_SECRET"},
        "transcript_path": str(tmp_path / "private-transcript.jsonl"),
        "permission_mode": "default",
    }
    payload.update(overrides)
    return payload


def test_tool_hook_captures_stable_content_free_lifecycle_identity(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    pool = CapturingClientPool()

    first = dispatch_claude_hook(
        _payload(tmp_path),
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )
    second = dispatch_claude_hook(
        _payload(tmp_path),
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )

    assert first == {
        "ok": True,
        "captured": True,
        "event": "PostToolUse",
    }
    assert second["ok"] is True
    assert pool.calls == []
    identity, host_state, kwargs = pool.host_state_calls[0]
    assert identity.caller_role == "host_state"
    assert identity.repo_root == tmp_path
    assert identity.workspace_id == "alpha"
    assert identity.backend == "mempalace"
    assert kwargs["session_id"] == "claude-session-123"
    serialized = json.dumps(host_state, sort_keys=True)
    assert "RAW_CLAUDE_TOOL_INPUT_SECRET" not in serialized
    assert "RAW_CLAUDE_TOOL_OUTPUT_SECRET" not in serialized
    assert "private-transcript.jsonl" not in serialized
    assert "claude-session-123" not in serialized
    assert "toolu-456" not in serialized
    assert host_state["event_type"] == "tool_result"
    assert host_state["capability_codes"] == ("claude_code_hook", "typed_host_state")
    assert host_state["event_id"] == pool.host_state_calls[1][1]["event_id"]


def test_task_and_subagent_events_keep_typed_lineage_without_raw_subjects(
    tmp_path: Path,
) -> None:
    pool = CapturingClientPool()

    task = dispatch_claude_hook(
        _payload(
            tmp_path,
            hook_event_name="TaskCompleted",
            task_id="task-789",
            task_subject="RAW_TASK_SUBJECT_SECRET",
        ),
        working_directory=tmp_path,
        pool=pool,
    )
    subagent = dispatch_claude_hook(
        _payload(
            tmp_path,
            hook_event_name="SubagentStop",
            agent_id="agent-789",
            agent_type="Explore",
            last_assistant_message="RAW_SUBAGENT_OUTPUT_SECRET",
        ),
        working_directory=tmp_path,
        pool=pool,
    )

    assert task["ok"] is True and subagent["ok"] is True
    assert pool.calls == []
    task_state = pool.host_state_calls[0][1]
    subagent_state = pool.host_state_calls[1][1]
    assert task_state["stage_id"].startswith("claude-task:")
    assert task_state["event_id"] != subagent_state["event_id"]
    serialized = json.dumps([task_state, subagent_state], sort_keys=True)
    assert "RAW_TASK_SUBJECT_SECRET" not in serialized
    assert "RAW_SUBAGENT_OUTPUT_SECRET" not in serialized
    assert "task-789" not in serialized
    assert "agent-789" not in serialized


def test_post_tool_batch_uses_only_tool_names_and_ids_for_replay_identity(
    tmp_path: Path,
) -> None:
    pool = CapturingClientPool()
    batch = [
        {
            "tool_name": "Read",
            "tool_use_id": "toolu-1",
            "tool_input": {"file_path": "RAW_FILE_PATH_SECRET"},
            "tool_response": "RAW_FILE_CONTENT_SECRET",
        },
        {
            "tool_name": "Bash",
            "tool_use_id": "toolu-2",
            "tool_input": {"command": "RAW_COMMAND_SECRET"},
            "tool_response": "RAW_COMMAND_OUTPUT_SECRET",
        },
    ]

    result = dispatch_claude_hook(
        _payload(tmp_path, hook_event_name="PostToolBatch", tool_calls=batch),
        working_directory=tmp_path,
        pool=pool,
    )

    assert result["ok"] is True
    assert pool.calls == []
    serialized = json.dumps(pool.host_state_calls[0][1], sort_keys=True)
    assert "RAW_FILE_PATH_SECRET" not in serialized
    assert "RAW_FILE_CONTENT_SECRET" not in serialized
    assert "RAW_COMMAND_SECRET" not in serialized
    assert "RAW_COMMAND_OUTPUT_SECRET" not in serialized


@pytest.mark.parametrize(
    "event_payload",
    [
        {"hook_event_name": "FileChanged", "file_path": "C:/repo/.env", "event": "change"},
        {"hook_event_name": "WorktreeCreate", "name": "feature-auth"},
        {"hook_event_name": "WorktreeRemove", "worktree_path": "C:/repo/worktree"},
        {"hook_event_name": "SessionEnd", "reason": "other"},
        {"hook_event_name": "StopFailure", "error": "rate_limit"},
        {"hook_event_name": "PreCompact", "trigger": "auto"},
    ],
)
def test_official_lifecycle_event_shapes_are_accepted_content_free(
    tmp_path: Path,
    event_payload: dict[str, object],
) -> None:
    pool = CapturingClientPool()

    result = dispatch_claude_hook(
        _payload(tmp_path, **event_payload),
        working_directory=tmp_path,
        pool=pool,
    )

    assert result["ok"] is True
    assert pool.calls == []
    serialized = json.dumps(pool.host_state_calls[0][1], sort_keys=True)
    for value in event_payload.values():
        if isinstance(value, str) and value not in {
            event_payload["hook_event_name"],
            "change",
            "other",
            "rate_limit",
            "auto",
        }:
            assert value not in serialized


def test_invalid_event_and_cwd_fail_before_service_calls(tmp_path: Path) -> None:
    pool = CapturingClientPool()

    unknown = dispatch_claude_hook(
        _payload(tmp_path, hook_event_name="UnknownFutureHook"),
        working_directory=tmp_path,
        pool=pool,
    )
    foreign = dispatch_claude_hook(
        _payload(tmp_path, cwd=str(tmp_path / "foreign")),
        working_directory=tmp_path,
        pool=pool,
    )

    assert unknown == {"ok": False, "error": "unsupported Claude Code hook event"}
    assert foreign == {
        "ok": False,
        "error": "Claude Code hook cwd does not match process cwd",
    }
    assert pool.calls == []


def test_input_is_bounded_and_rejects_raw_protocol_errors() -> None:
    assert parse_hook_input(b"x" * 1_048_577) == {
        "ok": False,
        "error": "Claude Code hook input exceeds protocol limit",
    }
    assert parse_hook_input(b"not-json") == {
        "ok": False,
        "error": "Claude Code hook input is not valid JSON",
    }


def test_installer_merges_safe_hooks_and_requires_explicit_file_watch_names(
    tmp_path: Path,
) -> None:
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Read"]},
                "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "user-stop-hook"}]}]},
            }
        ),
        encoding="utf-8",
    )

    installed = install_claude_hooks(
        tmp_path,
        python_executable="C:/Python312/python.exe",
        watch_files=(".env", ".envrc"),
    )
    payload = json.loads(installed.read_text(encoding="utf-8"))

    assert installed == settings
    assert payload["permissions"] == {"allow": ["Read"]}
    assert "WorktreeCreate" not in payload["hooks"]
    assert payload["hooks"]["FileChanged"][-1]["matcher"] == ".env|.envrc"
    assert {
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "PostToolBatch",
        "SubagentStart",
        "SubagentStop",
        "TaskCreated",
        "TaskCompleted",
        "Stop",
        "StopFailure",
        "PreCompact",
        "PostCompact",
        "SessionEnd",
        "WorktreeRemove",
        "FileChanged",
    }.issubset(payload["hooks"])
    stop_commands = [
        handler["command"] for group in payload["hooks"]["Stop"] for handler in group["hooks"]
    ]
    assert "user-stop-hook" in stop_commands
    assert any("um_mem.integrations.claude_hooks" in command for command in stop_commands)

    status = claude_hooks_status(tmp_path, watch_files=(".env", ".envrc"))
    assert status["installed"] is True
    assert status["missing_events"] == []
    assert status["unsafe_omissions"] == {
        "WorktreeCreate": "hook_replaces_default_worktree_creation"
    }


def test_default_claude_install_does_not_claim_unconfigured_file_watch(
    tmp_path: Path,
) -> None:
    installed = install_claude_hooks(tmp_path, python_executable="python")
    payload = json.loads(installed.read_text(encoding="utf-8"))

    assert "FileChanged" not in payload["hooks"]
    status = claude_hooks_status(tmp_path)
    assert status["installed"] is True
    assert "FileChanged" not in status["installed_events"]
    assert status["optional_unconfigured_events"] == ["FileChanged"]


def test_stop_is_turn_completion_while_session_end_closes_the_session(
    tmp_path: Path,
) -> None:
    pool = CapturingClientPool()
    stop = dispatch_claude_hook(
        _payload(tmp_path, hook_event_name="Stop"),
        working_directory=tmp_path,
        pool=pool,
    )
    end = dispatch_claude_hook(
        _payload(tmp_path, hook_event_name="SessionEnd", reason="other"),
        working_directory=tmp_path,
        pool=pool,
    )

    assert stop["ok"] is True
    assert end["ok"] is True
    assert pool.host_state_calls[0][1]["event_type"] == "turn_completion"
    assert pool.host_state_calls[1][1]["event_type"] == "session_boundary"


def test_capability_manifest_exposes_missing_turn_and_pre_turn_contracts_fail_closed() -> None:
    manifest = claude_host_capability_manifest()

    assert manifest.host_id == "claude_code"
    assert manifest.transport == "native_hooks+local_service+mcp"
    assert manifest.conformance_passed is False
    assert manifest.stable_session_id is True
    assert manifest.stable_turn_id is False
    assert manifest.pre_turn_delivery is False
    assert manifest.supports("stable_session_id") is False
