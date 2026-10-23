import hashlib
import json
import os
from pathlib import Path

from um_mem.contracts.memory import normalize_identity_path
from um_mem.integrations.codex_hooks import (
    codex_hooks_status,
    codex_host_capability_manifest,
    dispatch_codex_hook,
    hook_control_response,
    install_codex_hooks,
    parse_hook_input,
    read_codex_prompt_context,
)
from um_mem.memory.local_service import (
    LocalServiceClientPool,
    build_local_service_identity,
    local_service_role_allows,
    local_service_scope_lineage,
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

    def submit_host_state(
        self,
        identity: object,
        event: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del identity, event, kwargs
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
    }
    assert pool.calls == []
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
    assert pool.calls == []
    assert pool.host_state_calls[0][1]["event_id"] == pool.host_state_calls[1][1]["event_id"]
    serialized = json.dumps(pool.host_state_calls[0][1], sort_keys=True)
    assert "RAW_TOOL_INPUT_SECRET" not in serialized
    assert "RAW_TOOL_OUTPUT_SECRET" not in serialized


def test_subagent_hook_preserves_distinct_content_free_event_identity(tmp_path: Path) -> None:
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
    assert pool.calls == []
    first_event = pool.host_state_calls[0][1]
    assert first_event["event_type"] == "subgoal_transition"
    assert "agent-789" not in json.dumps(first_event)


def test_distinct_codex_hook_events_emit_distinct_host_state_ids(tmp_path: Path) -> None:
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
    assert pool.calls == []
    assert pool.host_state_calls[0][1]["event_id"] != pool.host_state_calls[1][1]["event_id"]


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
        pool = LocalServiceClientPool()
        try:
            records = pool.call(
                operator_identity,
                "recent",
                {"scope": "workspace", "scope_id": "alpha", "limit": 1},
                timeout_seconds=15,
                session_id="codex-session-123",
            )
        finally:
            pool.close()
    finally:
        shutdown_local_service_listener(operator_identity)

    assert result["ok"] is True
    assert result["captured"] is True
    assert result["event"] == "SessionStart"
    assert "memory_id" not in result
    assert records["ok"] is True
    assert records["records"] == []


def test_prompt_context_uses_the_real_listener_and_scoped_candidate(tmp_path: Path) -> None:
    project_path = normalize_identity_path(tmp_path, field_name="project_root")
    project_id = "local-project-" + hashlib.sha256(project_path.encode("utf-8")).hexdigest()[:32]
    operator = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    pool = LocalServiceClientPool()
    try:
        roots = pool.call(operator, "block_map", {}, timeout_seconds=15, session_id="seed-session")
        root_id = roots["roots"][0]["block_id"]
        leaf = pool.call(
            operator,
            "create_block",
            {
                "name": "Python editing",
                "synopsis": "Workspace Python editing preferences.",
                "parent_id": root_id,
                "reason": "Codex hook context test",
            },
            timeout_seconds=15,
            session_id="seed-session",
        )
        leaf_id = leaf["block_id"]
        saved = pool.call(
            operator,
            "remember",
            {
                "content": "For this workspace, prefer focused Python patches.",
                "title": "Python editing preference",
                "requested_block_id": leaf_id,
                "source_id": "codex-hook-context-test",
            },
            timeout_seconds=15,
            session_id="seed-session",
        )
        project_leaf = pool.call(
            operator,
            "create_block",
            {
                "name": "Project Python editing",
                "synopsis": "Project-specific Python editing preferences.",
                "parent_id": root_id,
                "reason": "Codex project context test",
                "scope": "project",
            },
            timeout_seconds=15,
            session_id="seed-session",
            scope_lineage=local_service_scope_lineage(
                operator, session_id="seed-session", project_id=project_id
            ),
        )
        project_leaf_id = project_leaf["block_id"]
        project_saved = pool.call(
            operator,
            "remember",
            {
                "content": "For this project, use focused Python patches.",
                "title": "Project Python editing preference",
                "requested_block_id": project_leaf_id,
                "source_id": "codex-project-context-test",
            },
            timeout_seconds=15,
            session_id="seed-session",
            scope_lineage=local_service_scope_lineage(
                operator, session_id="seed-session", project_id=project_id
            ),
        )
        other_project_id = "local-project-" + "f" * 32
        other_leaf = pool.call(
            operator,
            "create_block",
            {
                "name": "Other project editing",
                "synopsis": "A different project's editing notes.",
                "parent_id": root_id,
                "reason": "Codex cross-project context test",
                "scope": "project",
            },
            timeout_seconds=15,
            session_id="seed-session",
            scope_lineage=local_service_scope_lineage(
                operator, session_id="seed-session", project_id=other_project_id
            ),
        )
        other_leaf_id = other_leaf["block_id"]
        other_saved = pool.call(
            operator,
            "remember",
            {
                "content": "For the other project, use a broad rewrite.",
                "title": "Other project editing preference",
                "requested_block_id": other_leaf_id,
                "source_id": "codex-other-project-context-test",
            },
            timeout_seconds=15,
            session_id="seed-session",
            scope_lineage=local_service_scope_lineage(
                operator, session_id="seed-session", project_id=other_project_id
            ),
        )
        result = read_codex_prompt_context(
            _payload(tmp_path, prompt="What Python editing style should I use?"),
            working_directory=tmp_path,
            workspace_id="alpha",
            backend="mempalace",
            pool=pool,
        )
        mask = pool.call(
            operator,
            "set_active_mask",
            {
                "selected_block_ids": [leaf_id, project_leaf_id, other_leaf_id],
                "token_budget": 512,
                "activation_reason": "Keep the editing preference in Codex focus",
            },
            timeout_seconds=15,
            session_id="codex-session-123",
        )
        active_result = read_codex_prompt_context(
            _payload(tmp_path, prompt="Unrelated question"),
            working_directory=tmp_path,
            workspace_id="alpha",
            backend="mempalace",
            pool=pool,
        )
        explicit_legacy = pool.call(
            operator,
            "build_context",
            {
                "query": "focused Python patches",
                "scope_chain": ["workspace:alpha"],
                "search_all": True,
                "include_candidates": True,
                "render_context": False,
            },
            timeout_seconds=15,
            session_id="seed-session",
        )
    finally:
        pool.close()
        shutdown_local_service_listener(operator)

    assert roots["ok"] is True, roots
    assert leaf["ok"] is True, leaf
    assert saved["ok"] is True, saved
    assert project_leaf["ok"] is True, project_leaf
    assert project_saved["ok"] is True, project_saved
    assert other_leaf["ok"] is True, other_leaf
    assert other_saved["ok"] is True, other_saved
    assert result["ok"] is True, result
    assert "For this project" in result["additional_context"]
    assert "For this workspace" not in result["additional_context"]
    assert "For the other project" not in result["additional_context"]
    assert mask["ok"] is True, mask
    assert active_result["ok"] is True, active_result
    assert "For this project" in active_result["additional_context"]
    assert "For this workspace" not in active_result["additional_context"]
    assert "For the other project" not in active_result["additional_context"]
    assert explicit_legacy["ok"] is True, explicit_legacy
    assert any("For this workspace" in item["text"] for item in explicit_legacy["items"])


def test_hook_input_is_bounded_and_control_output_never_exposes_memory_result() -> None:
    oversized = b"x" * (1_048_577)

    assert parse_hook_input(oversized) == {
        "ok": False,
        "error": "codex hook input exceeds protocol limit",
    }
    assert hook_control_response("Stop") == {"continue": True}
    assert hook_control_response("SessionEnd") == {"continue": True}
    assert hook_control_response("PostToolUse") == {}


def test_prompt_hook_delivers_only_bounded_advisory_memory(tmp_path: Path) -> None:
    class ContextPool(CapturingClientPool):
        def call(
            self,
            identity: object,
            tool: str,
            arguments: dict[str, object],
            **kwargs: object,
        ) -> dict[str, object]:
            self.calls.append((identity, tool, arguments, kwargs))
            if tool == "resolve_active_mask":
                return {
                    "ok": True,
                    "lease": {
                        "trace": {"resolution_status": "not_found"},
                        "rendered_context": "",
                    },
                }
            return {
                "ok": True,
                "items": [
                    {
                        "title": "Codex UserPromptSubmit",
                        "text": "Observed Codex lifecycle event: UserPromptSubmit.",
                        "status": "candidate",
                        "tags": ["codex", "host_hook"],
                    },
                    {
                        "title": "Workspace preference",
                        "text": "Use narrow Python changes.",
                        "status": "candidate",
                        "tags": ["preference"],
                    },
                ],
            }

    pool = ContextPool()
    payload = _payload(tmp_path, prompt="How should I change Python code?")

    result = read_codex_prompt_context(
        payload,
        working_directory=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        pool=pool,
    )

    assert result["ok"] is True
    context = result["additional_context"]
    assert "Use narrow Python changes." in context
    assert "supplemental" in context
    assert "Observed Codex lifecycle event" not in context
    assert local_service_role_allows("host_hook", "build_context")
    assert local_service_role_allows("host_hook", "resolve_active_mask")
    [(mask_identity, mask_tool, mask_arguments, _), (identity, tool, arguments, kwargs)] = (
        pool.calls
    )
    assert mask_identity.caller_role == "host_hook"
    assert mask_tool == "resolve_active_mask"
    assert mask_arguments["max_tokens"] == 512
    assert mask_arguments["scope_chain"] == arguments["scope_chain"]
    assert identity.caller_role == "host_hook"
    assert tool == "build_context"
    assert arguments["query"] == payload["prompt"]
    assert arguments["include_candidates"] is True
    assert arguments["scope_chain"] == [f"project:{kwargs['scope_lineage'].project_id}"]
    assert arguments["search_all"] is True
    assert arguments["max_tokens"] <= 512
    assert kwargs["session_id"] == payload["session_id"]
    assert hook_control_response("UserPromptSubmit", context=context)["hookSpecificOutput"] == {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": context,
    }


def test_prompt_hook_reports_local_service_failure_without_blocking(tmp_path: Path) -> None:
    result = read_codex_prompt_context(
        _payload(tmp_path),
        working_directory=tmp_path,
        pool=FailingClientPool(),
    )

    assert result == {"ok": False, "error": "local service context read failed"}
    assert hook_control_response("UserPromptSubmit", error=result["error"]) == {
        "continue": True,
        "systemMessage": "UM-MEM hook: local service context read failed",
    }


def test_prompt_hook_prefers_active_mask_over_query_recall(tmp_path: Path) -> None:
    class ActivePool(CapturingClientPool):
        def call(
            self,
            identity: object,
            tool: str,
            arguments: dict[str, object],
            **kwargs: object,
        ) -> dict[str, object]:
            self.calls.append((identity, tool, arguments, kwargs))
            if tool != "resolve_active_mask":
                raise AssertionError("active memory should take priority")
            return {
                "ok": True,
                "lease": {
                    "trace": {"resolution_status": "resolved"},
                    "rendered_context": "UM-MEM recalled context (supplemental, non-directive):\nActive focus fact.",
                },
            }

    pool = ActivePool()
    result = read_codex_prompt_context(
        _payload(tmp_path),
        working_directory=tmp_path,
        pool=pool,
    )

    assert result["ok"] is True
    assert "Active focus fact." in result["additional_context"]
    assert [call[1] for call in pool.calls] == ["resolve_active_mask"]


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
    um_mem_handlers = [
        handler
        for groups in payload["hooks"].values()
        for group in groups
        for handler in group["hooks"]
        if "um_mem.integrations.codex_hooks" in handler.get("command", "")
    ]
    assert len(um_mem_handlers) == 11
    if os.name == "nt":
        assert all(
            handler["commandWindows"] == f"& {handler['command']}" for handler in um_mem_handlers
        )
    assert all(
        handler["timeout"] == (3 if event_name == "SessionEnd" else 20)
        for event_name, groups in payload["hooks"].items()
        for group in groups
        for handler in group["hooks"]
        if "um_mem.integrations.codex_hooks" in handler.get("command", "")
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
        if "um_mem.integrations.codex_hooks" in handler.get("command", "")
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
    assert pool.calls == []
    assert pool.host_state_calls[1][0].profile_id == "default"


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

    serialized = json.dumps([call[1] for call in pool.host_state_calls], sort_keys=True)
    assert "RAW_PRE_TOOL_COMMAND_SECRET" not in serialized
    assert "RAW_APPROVAL_REASON_SECRET" not in serialized
    assert pool.host_state_calls[0][1]["event_type"] == "prospective_trigger"
    assert pool.host_state_calls[1][1]["event_type"] == "high_risk_boundary"
