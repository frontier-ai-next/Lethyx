"""Capture bounded Claude Code lifecycle facts through the local Lethyx service."""

import argparse
import json
import secrets
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lethyx.integrations.hook_settings import (
    HookRegistration,
    install_project_hooks,
    project_hooks_status,
)
from lethyx.integrations.host_hook_capture import (
    HostHookClientPool,
    HostHookEvent,
    capture_host_hook,
    git_root,
)
from lethyx.integrations.host_capabilities import HostCapabilityManifest

CLAUDE_HOOK_SCHEMA_VERSION = 1
MAX_CLAUDE_HOOK_INPUT_BYTES = 1_048_576
CLAUDE_HOOK_TIMEOUT_SECONDS = 15
_MAX_IDENTIFIER_LENGTH = 4096
_MAX_BATCH_TOOL_CALLS = 128
_CLAUDE_HOOK_MODULE = "lethyx.integrations.claude_hooks"


@dataclass(frozen=True)
class _EventSpec:
    scheduler_event_type: str
    identity_fields: tuple[str, ...] = ()
    stage_field: str = ""
    stage_kind: str = ""
    subtask_field: str = ""
    subtask_kind: str = ""
    replay_stable: bool = True


_EVENT_SPECS = {
    "SessionStart": _EventSpec("session_boundary", ("source",)),
    "UserPromptSubmit": _EventSpec("prospective_trigger", replay_stable=False),
    "PreToolUse": _EventSpec(
        "prospective_trigger",
        ("tool_name", "tool_use_id"),
        stage_field="tool_use_id",
        stage_kind="tool",
    ),
    "PostToolUse": _EventSpec(
        "tool_result",
        ("tool_name", "tool_use_id"),
        stage_field="tool_use_id",
        stage_kind="tool",
    ),
    "PostToolUseFailure": _EventSpec(
        "tool_result",
        ("tool_name", "tool_use_id"),
        stage_field="tool_use_id",
        stage_kind="tool",
    ),
    "PostToolBatch": _EventSpec("tool_result"),
    "SubagentStart": _EventSpec(
        "subgoal_transition",
        ("agent_id", "agent_type"),
        subtask_field="agent_id",
        subtask_kind="subagent",
    ),
    "SubagentStop": _EventSpec(
        "subgoal_transition",
        ("agent_id", "agent_type"),
        subtask_field="agent_id",
        subtask_kind="subagent",
    ),
    "TaskCreated": _EventSpec(
        "subgoal_transition",
        ("task_id",),
        stage_field="task_id",
        stage_kind="task",
    ),
    "TaskCompleted": _EventSpec(
        "subgoal_transition",
        ("task_id",),
        stage_field="task_id",
        stage_kind="task",
    ),
    "Stop": _EventSpec("turn_completion", replay_stable=False),
    "StopFailure": _EventSpec("turn_completion", ("error",), replay_stable=False),
    "FileChanged": _EventSpec(
        "source_change",
        ("file_path", "event"),
        replay_stable=False,
    ),
    "WorktreeCreate": _EventSpec("source_change", ("name",)),
    "WorktreeRemove": _EventSpec("source_change", ("worktree_path",)),
    "PreCompact": _EventSpec("compaction", ("trigger",), replay_stable=False),
    "PostCompact": _EventSpec("compaction", ("trigger",), replay_stable=False),
    "SessionEnd": _EventSpec("session_boundary", ("reason",)),
}

_SAFE_REGISTRATIONS = (
    HookRegistration("SessionStart", "startup|resume|clear|compact"),
    HookRegistration("UserPromptSubmit"),
    HookRegistration("PreToolUse", ".*"),
    HookRegistration("PostToolUse", ".*"),
    HookRegistration("PostToolUseFailure", ".*"),
    HookRegistration("PostToolBatch"),
    HookRegistration("SubagentStart", ".*"),
    HookRegistration("SubagentStop", ".*"),
    HookRegistration("TaskCreated"),
    HookRegistration("TaskCompleted"),
    HookRegistration("Stop"),
    HookRegistration("StopFailure", ".*"),
    HookRegistration("PreCompact", "manual|auto"),
    HookRegistration("PostCompact", "manual|auto"),
    HookRegistration(
        "SessionEnd",
        "clear|resume|logout|prompt_input_exit|bypass_permissions_disabled|other",
    ),
    HookRegistration("WorktreeRemove"),
)
_SAFE_REQUIRED_EVENTS = tuple(registration.event_name for registration in _SAFE_REGISTRATIONS)


def parse_hook_input(raw: bytes) -> dict[str, object]:
    """Decode one bounded Claude Code command-hook JSON object."""

    if len(raw) > MAX_CLAUDE_HOOK_INPUT_BYTES:
        return {"ok": False, "error": "Claude Code hook input exceeds protocol limit"}
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "error": "Claude Code hook input is not UTF-8"}
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        return {"ok": False, "error": "Claude Code hook input is not valid JSON"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "Claude Code hook input must be an object"}
    return payload


def dispatch_claude_hook(
    payload: Mapping[str, object],
    *,
    working_directory: Path | None = None,
    workspace_id: str | None = None,
    profile_id: str | None = None,
    backend: str | None = None,
    repo_root: str | Path | None = None,
    pool: HostHookClientPool | None = None,
) -> dict[str, object]:
    """Route one official Claude Code lifecycle event through a hook-only lease."""

    try:
        cwd = (working_directory or Path.cwd()).resolve()
    except (OSError, RuntimeError):
        return {"ok": False, "error": "Claude Code hook working directory is invalid"}
    event, scheduler_event_type, error = _event_from_payload(payload, cwd)
    if error is not None:
        return {"ok": False, "error": error}
    assert event is not None and scheduler_event_type is not None
    return capture_host_hook(
        event,
        repo_root=Path(repo_root).expanduser().resolve() if repo_root else git_root(cwd),
        workspace_id=workspace_id,
        profile_id=profile_id,
        backend=backend,
        scheduler_event_type=scheduler_event_type,
        timeout_seconds=CLAUDE_HOOK_TIMEOUT_SECONDS,
        pool=pool,
    )


def install_claude_hooks(
    project_directory: str | Path,
    *,
    python_executable: str | None = None,
    watch_files: Sequence[str] = (),
    module_arguments: Sequence[str] = (),
    settings_path: str | Path | None = None,
) -> Path:
    """Install passive Claude Code hooks while preserving unrelated settings."""

    registrations = list(_SAFE_REGISTRATIONS)
    normalized_watch_files = _normalized_watch_files(watch_files)
    if normalized_watch_files:
        registrations.append(HookRegistration("FileChanged", "|".join(normalized_watch_files)))
    target = (
        Path(settings_path).expanduser().resolve()
        if settings_path is not None
        else Path(project_directory).expanduser().resolve() / ".claude" / "settings.json"
    )
    return install_project_hooks(
        target,
        module_name=_CLAUDE_HOOK_MODULE,
        registrations=registrations,
        python_executable=python_executable,
        timeout_seconds=20,
        module_arguments=module_arguments,
    )


def claude_hooks_status(
    project_directory: str | Path,
    *,
    watch_files: Sequence[str] = (),
    settings_path: str | Path | None = None,
) -> dict[str, object]:
    """Report safe hook coverage without claiming Claude authorization or E2E."""

    normalized_watch_files = _normalized_watch_files(watch_files)
    required_events = list(_SAFE_REQUIRED_EVENTS)
    if normalized_watch_files:
        required_events.append("FileChanged")
    target = (
        Path(settings_path).expanduser().resolve()
        if settings_path is not None
        else Path(project_directory).expanduser().resolve() / ".claude" / "settings.json"
    )
    status = project_hooks_status(
        target,
        module_name=_CLAUDE_HOOK_MODULE,
        required_events=required_events,
    )
    status.update(
        {
            "implementation": "packaged-native-hooks",
            "real_host_conformance": "unconfirmed",
            "optional_unconfigured_events": [] if normalized_watch_files else ["FileChanged"],
            "unsafe_omissions": {"WorktreeCreate": "hook_replaces_default_worktree_creation"},
            "capability_manifest": claude_host_capability_manifest().to_dict(),
        }
    )
    return status


def claude_host_capability_manifest() -> HostCapabilityManifest:
    """Declare observable Claude surfaces while keeping deep support unqualified."""

    return HostCapabilityManifest(
        host_id="claude_code",
        adapter_id="claude_hooks",
        transport="native_hooks+local_service+mcp",
        conformance_passed=False,
        stable_workspace_id=True,
        stable_session_id=True,
        stable_task_id=True,
        stable_subtask_id=True,
        stable_tool_id=True,
        turn_completion=True,
        observed_tool_outcomes=True,
        tool_call_events=True,
        tool_result_events=True,
        tool_failure_events=True,
        live_state_transitions=True,
        compaction_callbacks=True,
        prospective_trigger_events=True,
        parent_child_lineage=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Run as a non-blocking Claude command hook with intentionally empty stdout."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--backend", default="")
    args = parser.parse_args(argv)
    payload = parse_hook_input(sys.stdin.buffer.read(MAX_CLAUDE_HOOK_INPUT_BYTES + 1))
    if payload.get("ok") is not False:
        dispatch_claude_hook(
            payload,
            repo_root=args.repo_root or None,
            workspace_id=args.workspace or None,
            profile_id=args.profile or None,
            backend=args.backend or None,
        )
    return 0


def _event_from_payload(
    payload: Mapping[str, object],
    working_directory: Path,
) -> tuple[HostHookEvent | None, str | None, str | None]:
    event_name, error = _required_identifier(payload, "hook_event_name")
    if error is not None:
        return None, None, error
    assert event_name is not None
    spec = _EVENT_SPECS.get(event_name)
    if spec is None:
        return None, None, "unsupported Claude Code hook event"
    session_id, error = _required_identifier(payload, "session_id")
    if error is not None:
        return None, None, error
    assert session_id is not None
    payload_cwd, error = _payload_cwd(payload)
    if error is not None:
        return None, None, error
    assert payload_cwd is not None
    if payload_cwd != working_directory:
        return None, None, "Claude Code hook cwd does not match process cwd"

    if event_name == "PostToolBatch":
        values, error = _batch_identity_fields(payload)
        if error is not None:
            return None, None, error
    else:
        values = []
        for field_name in spec.identity_fields:
            value, error = _required_identifier(payload, field_name)
            if error is not None:
                return None, None, error
            assert value is not None
            values.append((field_name, value))
    value_by_name = dict(values)
    if not spec.replay_stable:
        values.append(("capture_nonce", secrets.token_hex(16)))
    return (
        HostHookEvent(
            host_slug="claude",
            host_label="Claude Code",
            adapter_id="claude_hooks",
            capability_code="claude_code_hook",
            schema_version=CLAUDE_HOOK_SCHEMA_VERSION,
            event_name=event_name,
            session_id=session_id,
            identity_fields=tuple(values),
            stage_kind=spec.stage_kind,
            stage_id=value_by_name.get(spec.stage_field, ""),
            subtask_kind=spec.subtask_kind,
            subtask_id=value_by_name.get(spec.subtask_field, ""),
        ),
        spec.scheduler_event_type,
        None,
    )


def _batch_identity_fields(
    payload: Mapping[str, object],
) -> tuple[list[tuple[str, str]], str | None]:
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls or len(raw_calls) > _MAX_BATCH_TOOL_CALLS:
        return [], "Claude Code hook tool_calls must be a bounded non-empty list"
    values: list[tuple[str, str]] = []
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            return [], "Claude Code hook tool_calls entries must be objects"
        tool_name, error = _required_identifier(raw_call, "tool_name")
        if error is not None:
            return [], error
        tool_use_id, error = _required_identifier(raw_call, "tool_use_id")
        if error is not None:
            return [], error
        assert tool_name is not None and tool_use_id is not None
        values.extend(
            (
                (f"tool_{index}_name", tool_name),
                (f"tool_{index}_id", tool_use_id),
            )
        )
    return values, None


def _required_identifier(
    payload: Mapping[str, object],
    field_name: str,
) -> tuple[str | None, str | None]:
    value = payload.get(field_name)
    if not isinstance(value, str):
        return None, f"Claude Code hook {field_name} is required"
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_IDENTIFIER_LENGTH or "\x00" in normalized:
        return None, f"Claude Code hook {field_name} is required"
    return normalized, None


def _payload_cwd(payload: Mapping[str, object]) -> tuple[Path | None, str | None]:
    raw_cwd = payload.get("cwd")
    if not isinstance(raw_cwd, str) or not raw_cwd.strip():
        return None, "Claude Code hook cwd is required"
    try:
        return Path(raw_cwd).resolve(), None
    except (OSError, RuntimeError):
        return None, "Claude Code hook cwd is invalid"


def _normalized_watch_files(watch_files: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_name in watch_files:
        name = str(raw_name).strip()
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or "|" in name
            or "\x00" in name
        ):
            raise ValueError("Claude FileChanged watch entries must be literal basenames")
        normalized.append(name)
    return tuple(dict.fromkeys(normalized))


if __name__ == "__main__":
    raise SystemExit(main())
