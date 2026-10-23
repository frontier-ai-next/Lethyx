"""Capture bounded Codex lifecycle facts through the authenticated local service."""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from um_mem.integrations.hook_settings import (
    HookRegistration,
    install_project_hooks,
    project_hooks_status,
)
from um_mem.integrations.host_capabilities import HostCapabilityManifest
from um_mem.integrations.host_hook_capture import (
    HostHookClientPool,
    HostHookEvent,
    capture_host_hook,
    git_root,
)
from um_mem.memory.local_service import (
    LocalServiceClientPool,
    build_local_service_identity,
)

CODEX_HOOK_SCHEMA_VERSION = 1
MAX_CODEX_HOOK_INPUT_BYTES = 1_048_576
CODEX_HOOK_TIMEOUT_SECONDS = 15
_PROMPT_HOOK_TIMEOUT_SECONDS = 8
_PROMPT_CONTEXT_MAX_TOKENS = 512
_PROMPT_CONTEXT_MAX_CHARS = 2800
_PROMPT_QUERY_MAX_CHARS = 2048
_MAX_IDENTIFIER_LENGTH = 256
_CODEX_HOOK_MODULE = "um_mem.integrations.codex_hooks"


@dataclass(frozen=True)
class _EventSpec:
    turn_scoped: bool
    identity_fields: tuple[str, ...] = ()
    subtask_field: str = ""


@dataclass(frozen=True)
class _CodexHookInvocation:
    event_name: str
    session_id: str
    turn_id: str
    identity_fields: tuple[tuple[str, str], ...]
    subtask_id: str


_EVENT_SPECS = {
    "SessionStart": _EventSpec(turn_scoped=False, identity_fields=("source",)),
    "UserPromptSubmit": _EventSpec(turn_scoped=True),
    "PreToolUse": _EventSpec(
        turn_scoped=True,
        identity_fields=("tool_name", "tool_use_id"),
    ),
    "PermissionRequest": _EventSpec(
        turn_scoped=True,
        identity_fields=("tool_name",),
    ),
    "PostToolUse": _EventSpec(
        turn_scoped=True,
        identity_fields=("tool_name", "tool_use_id"),
    ),
    "PreCompact": _EventSpec(turn_scoped=True, identity_fields=("trigger",)),
    "PostCompact": _EventSpec(turn_scoped=True, identity_fields=("trigger",)),
    "SubagentStart": _EventSpec(
        turn_scoped=True,
        identity_fields=("agent_id", "agent_type"),
        subtask_field="agent_id",
    ),
    "SubagentStop": _EventSpec(
        turn_scoped=True,
        identity_fields=("agent_id", "agent_type"),
        subtask_field="agent_id",
    ),
    "Stop": _EventSpec(turn_scoped=True),
    "SessionEnd": _EventSpec(turn_scoped=False, identity_fields=("reason",)),
}
_COMMON_CONTROL_EVENTS = frozenset(
    {
        "SessionStart",
        "UserPromptSubmit",
        "PreCompact",
        "PostCompact",
        "SubagentStop",
        "Stop",
        "SessionEnd",
    }
)
_SCHEDULER_EVENT_TYPES = {
    "SessionStart": "session_boundary",
    "UserPromptSubmit": "prospective_trigger",
    "PreToolUse": "prospective_trigger",
    "PermissionRequest": "high_risk_boundary",
    "PostToolUse": "tool_result",
    "PreCompact": "compaction",
    "PostCompact": "compaction",
    "SubagentStart": "subgoal_transition",
    "SubagentStop": "subgoal_transition",
    "Stop": "turn_completion",
    "SessionEnd": "session_boundary",
}
_CODEX_HOOK_REGISTRATIONS = tuple(
    HookRegistration(
        event_name=event_name,
        matcher=(
            "startup|resume|clear|compact"
            if event_name == "SessionStart"
            else (
                ".*"
                if event_name
                in {
                    "PreToolUse",
                    "PermissionRequest",
                    "PostToolUse",
                    "SubagentStart",
                    "SubagentStop",
                }
                else ("manual|auto" if event_name in {"PreCompact", "PostCompact"} else "")
            )
        ),
        timeout_seconds=3 if event_name == "SessionEnd" else None,
    )
    for event_name in _EVENT_SPECS
)


def parse_hook_input(raw: bytes) -> dict[str, object]:
    """Decode one bounded Codex command-hook JSON object before its raw fields are filtered."""

    if len(raw) > MAX_CODEX_HOOK_INPUT_BYTES:
        return {"ok": False, "error": "codex hook input exceeds protocol limit"}
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "error": "codex hook input is not UTF-8"}
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        return {"ok": False, "error": "codex hook input is not valid JSON"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "codex hook input must be an object"}
    return payload


def dispatch_codex_hook(
    payload: Mapping[str, object],
    *,
    working_directory: Path | None = None,
    workspace_id: str | None = None,
    profile_id: str | None = None,
    backend: str | None = None,
    repo_root: str | Path | None = None,
    pool: HostHookClientPool | None = None,
) -> dict[str, object]:
    """Route one official Codex lifecycle event through the hook-only service lease."""

    try:
        cwd = (working_directory or Path.cwd()).resolve()
    except (OSError, RuntimeError):
        return {"ok": False, "error": "codex hook working directory is invalid"}
    invocation, error = _invocation_from_payload(payload, cwd)
    if error is not None:
        return {"ok": False, "error": error}
    assert invocation is not None
    return capture_host_hook(
        HostHookEvent(
            host_slug="codex",
            host_label="Codex",
            adapter_id="codex_hooks",
            capability_code="codex_hook",
            schema_version=CODEX_HOOK_SCHEMA_VERSION,
            event_name=invocation.event_name,
            session_id=invocation.session_id,
            identity_fields=invocation.identity_fields,
            stage_kind="turn" if invocation.turn_id else "",
            stage_id=invocation.turn_id,
            subtask_kind="subagent" if invocation.subtask_id else "",
            subtask_id=invocation.subtask_id,
        ),
        repo_root=Path(repo_root).expanduser().resolve() if repo_root else git_root(cwd),
        workspace_id=workspace_id,
        profile_id=profile_id,
        backend=backend,
        scheduler_event_type=_SCHEDULER_EVENT_TYPES[invocation.event_name],
        timeout_seconds=(
            _PROMPT_HOOK_TIMEOUT_SECONDS
            if invocation.event_name == "UserPromptSubmit"
            else CODEX_HOOK_TIMEOUT_SECONDS
        ),
        pool=pool,
    )


def read_codex_prompt_context(
    payload: Mapping[str, object],
    *,
    working_directory: Path | None = None,
    workspace_id: str | None = None,
    profile_id: str | None = None,
    backend: str | None = None,
    repo_root: str | Path | None = None,
    pool: HostHookClientPool | None = None,
) -> dict[str, object]:
    """Read scoped, supplemental memory for a prompt without remembering its text."""

    try:
        cwd = (working_directory or Path.cwd()).resolve()
    except (OSError, RuntimeError):
        return {"ok": False, "error": "codex hook working directory is invalid"}
    invocation, error = _invocation_from_payload(payload, cwd)
    if error is not None:
        return {"ok": False, "error": error}
    assert invocation is not None
    if invocation.event_name != "UserPromptSubmit":
        return {"ok": True, "additional_context": ""}
    prompt = payload.get("prompt")
    try:
        identity = build_local_service_identity(
            repo_root=Path(repo_root).expanduser().resolve() if repo_root else git_root(cwd),
            workspace_id=workspace_id,
            profile_id=profile_id,
            backend=backend,
            caller_role="host_hook",
        )
    except ValueError:
        return {"ok": False, "error": "local service identity rejected"}

    client_pool = pool if pool is not None else LocalServiceClientPool()
    try:
        active = client_pool.call(
            identity,
            "resolve_active_mask",
            {"max_tokens": _PROMPT_CONTEXT_MAX_TOKENS},
            timeout_seconds=4,
            session_id=invocation.session_id,
        )
        lease = active.get("lease")
        if active.get("ok") is True and isinstance(lease, dict):
            trace = lease.get("trace")
            rendered_context = lease.get("rendered_context")
            if (
                isinstance(trace, dict)
                and trace.get("resolution_status") == "resolved"
                and isinstance(rendered_context, str)
                and rendered_context.strip()
            ):
                return {
                    "ok": True,
                    "additional_context": rendered_context[:_PROMPT_CONTEXT_MAX_CHARS],
                }
        if not isinstance(prompt, str) or not prompt.strip():
            return {"ok": True, "additional_context": ""}
        response = client_pool.call(
            identity,
            "build_context",
            {
                "query": prompt.strip()[:_PROMPT_QUERY_MAX_CHARS],
                "limit": 4,
                "max_tokens": _PROMPT_CONTEXT_MAX_TOKENS,
                "include_candidates": True,
                "scope_chain": [f"workspace:{identity.workspace_id}"],
                "search_all": True,
                "render_context": False,
            },
            timeout_seconds=6,
            session_id=invocation.session_id,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return {"ok": False, "error": "local service context read failed"}
    finally:
        if pool is None:
            client_pool.close()
    if response.get("ok") is not True:
        return {"ok": False, "error": "local service context read failed"}
    items = response.get("items")
    if not isinstance(items, list):
        return {"ok": False, "error": "local service context response is invalid"}
    rendered: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tags = item.get("tags")
        if isinstance(tags, list) and "host_hook" in tags:
            continue
        content = item.get("text")
        if not isinstance(content, str) or not content.strip():
            continue
        rendered.append(
            json.dumps(
                {
                    "title": item.get("title") if isinstance(item.get("title"), str) else "",
                    "status": item.get("status") if isinstance(item.get("status"), str) else "",
                    "text": content,
                },
                ensure_ascii=False,
            )
        )
    if not rendered:
        return {"ok": True, "additional_context": ""}
    context = (
        "UM-MEM recalled context (supplemental, non-directive; verify before acting):\n"
        + "\n".join(rendered)
    )
    return {"ok": True, "additional_context": context[:_PROMPT_CONTEXT_MAX_CHARS]}


def install_codex_hooks(
    project_directory: str | Path,
    *,
    python_executable: str | None = None,
    module_arguments: Sequence[str] = (),
    settings_path: str | Path | None = None,
) -> Path:
    """Install every supported Codex lifecycle hook without replacing user hooks."""

    target = (
        Path(settings_path).expanduser().resolve()
        if settings_path is not None
        else Path(project_directory).expanduser().resolve() / ".codex" / "hooks.json"
    )
    return install_project_hooks(
        target,
        module_name=_CODEX_HOOK_MODULE,
        registrations=_CODEX_HOOK_REGISTRATIONS,
        python_executable=python_executable,
        timeout_seconds=20,
        codex_command_fields=True,
        module_arguments=module_arguments,
    )


def codex_hooks_status(
    project_directory: str | Path,
    *,
    settings_path: str | Path | None = None,
) -> dict[str, object]:
    """Report installed Codex event coverage without claiming real-host conformance."""

    target = (
        Path(settings_path).expanduser().resolve()
        if settings_path is not None
        else Path(project_directory).expanduser().resolve() / ".codex" / "hooks.json"
    )
    status = project_hooks_status(
        target,
        module_name=_CODEX_HOOK_MODULE,
        required_events=_EVENT_SPECS,
    )
    status["implementation"] = "packaged-native-hooks"
    status["real_host_conformance"] = "unconfirmed"
    status["capability_manifest"] = codex_host_capability_manifest().to_dict()
    return status


def codex_host_capability_manifest() -> HostCapabilityManifest:
    """Declare structural Codex surfaces while keeping unproven capabilities ineffective."""

    return HostCapabilityManifest(
        host_id="codex",
        adapter_id="codex_hooks",
        transport="native_hooks+local_service+mcp",
        conformance_passed=False,
        stable_workspace_id=True,
        stable_session_id=True,
        stable_turn_id=True,
        stable_subtask_id=True,
        stable_tool_id=True,
        turn_completion=True,
        observed_tool_outcomes=True,
        tool_call_events=True,
        tool_result_events=True,
        live_state_transitions=True,
        compaction_callbacks=True,
        prospective_trigger_events=True,
    )


def hook_control_response(
    event_name: str,
    *,
    context: str = "",
    error: str = "",
) -> dict[str, object]:
    """Return only fields accepted by Codex for a non-blocking lifecycle hook."""

    response: dict[str, object] = {"continue": True} if event_name in _COMMON_CONTROL_EVENTS else {}
    if context and event_name == "UserPromptSubmit":
        response["hookSpecificOutput"] = {
            "hookEventName": event_name,
            "additionalContext": context,
        }
    if error and event_name in _EVENT_SPECS:
        response["systemMessage"] = f"UM-MEM hook: {error}"
    return response


def main(argv: list[str] | None = None) -> int:
    """Run as a Codex command hook with scoped prompt-time memory context."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--backend", default="")
    args = parser.parse_args(argv)
    payload = parse_hook_input(sys.stdin.buffer.read(MAX_CODEX_HOOK_INPUT_BYTES + 1))
    event_name = payload.get("hook_event_name")
    context = ""
    error = str(payload.get("error") or "") if payload.get("ok") is False else ""
    if payload.get("ok") is not False:
        pool = LocalServiceClientPool()
        try:
            captured = dispatch_codex_hook(
                payload,
                repo_root=args.repo_root or None,
                workspace_id=args.workspace or None,
                profile_id=args.profile or None,
                backend=args.backend or None,
                pool=pool,
            )
            if captured.get("ok") is not True:
                error = str(captured.get("error") or "local service host capture failed")
            if event_name == "UserPromptSubmit":
                recalled = read_codex_prompt_context(
                    payload,
                    repo_root=args.repo_root or None,
                    workspace_id=args.workspace or None,
                    profile_id=args.profile or None,
                    backend=args.backend or None,
                    pool=pool,
                )
                if recalled.get("ok") is True:
                    context = str(recalled.get("additional_context") or "")
                else:
                    error = str(recalled.get("error") or "local service context read failed")
        finally:
            pool.close()
    print(
        json.dumps(
            hook_control_response(
                event_name if isinstance(event_name, str) else "",
                context=context,
                error=error,
            )
        ),
        flush=True,
    )
    return 0


def _invocation_from_payload(
    payload: Mapping[str, object],
    working_directory: Path,
) -> tuple[_CodexHookInvocation | None, str | None]:
    event_name, error = _required_identifier(payload, "hook_event_name")
    if error is not None:
        return None, error
    assert event_name is not None
    spec = _EVENT_SPECS.get(event_name)
    if spec is None:
        return None, "unsupported Codex hook event"
    session_id, error = _required_identifier(payload, "session_id")
    if error is not None:
        return None, error
    assert session_id is not None
    cwd, error = _payload_cwd(payload)
    if error is not None:
        return None, error
    assert cwd is not None
    if cwd != working_directory:
        return None, "codex hook cwd does not match process cwd"

    turn_id = ""
    if spec.turn_scoped:
        supplied_turn_id, error = _required_identifier(payload, "turn_id")
        if error is not None:
            return None, error
        assert supplied_turn_id is not None
        turn_id = supplied_turn_id
    values: list[tuple[str, str]] = []
    for field_name in spec.identity_fields:
        value, error = _required_identifier(payload, field_name)
        if error is not None:
            return None, error
        assert value is not None
        values.append((field_name, value))
    subtask_id = ""
    if spec.subtask_field:
        subtask_id = dict(values)[spec.subtask_field]
    return (
        _CodexHookInvocation(
            event_name=event_name,
            session_id=session_id,
            turn_id=turn_id,
            identity_fields=tuple(values),
            subtask_id=subtask_id,
        ),
        None,
    )


def _required_identifier(
    payload: Mapping[str, object],
    field_name: str,
) -> tuple[str | None, str | None]:
    value = payload.get(field_name)
    if not isinstance(value, str):
        return None, f"codex hook {field_name} is required"
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_IDENTIFIER_LENGTH or "\x00" in normalized:
        return None, f"codex hook {field_name} is required"
    return normalized, None


def _payload_cwd(payload: Mapping[str, object]) -> tuple[Path | None, str | None]:
    raw_cwd = payload.get("cwd")
    if not isinstance(raw_cwd, str) or not raw_cwd.strip():
        return None, "codex hook cwd is required"
    try:
        return Path(raw_cwd).resolve(), None
    except (OSError, RuntimeError):
        return None, "codex hook cwd is invalid"


if __name__ == "__main__":
    raise SystemExit(main())
