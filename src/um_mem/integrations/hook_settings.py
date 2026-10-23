"""Atomic, non-destructive project hook settings installation."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HookRegistration:
    """One host hook event registration owned by a packaged adapter."""

    event_name: str
    matcher: str = ""
    timeout_seconds: int | None = None


def python_module_command(
    module_name: str,
    python_executable: str | None = None,
    module_arguments: Sequence[str] = (),
) -> str:
    """Build a command for the interpreter that installed UM-MEM."""

    executable = str(python_executable or sys.executable)
    arguments = [executable, "-m", module_name, *module_arguments]
    if os.name == "nt":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


def install_project_hooks(
    settings_path: Path,
    *,
    module_name: str,
    registrations: Iterable[HookRegistration],
    python_executable: str | None = None,
    timeout_seconds: int,
    codex_command_fields: bool = False,
    module_arguments: Sequence[str] = (),
) -> Path:
    """Merge adapter-owned command hooks while preserving unrelated settings."""

    settings = _read_settings(settings_path)
    hooks = _hooks_mapping(settings)
    _remove_module_handlers(hooks, module_name)
    command = python_module_command(module_name, python_executable, module_arguments)
    for registration in registrations:
        handler: dict[str, object] = {
            "type": "command",
            "command": command,
            "timeout": registration.timeout_seconds or timeout_seconds,
        }
        if codex_command_fields:
            handler.update(
                {
                    "commandWindows": f"& {command}" if os.name == "nt" else command,
                    "statusMessage": "Capturing UM-MEM lifecycle",
                }
            )
        group: dict[str, object] = {"hooks": [handler]}
        if registration.matcher:
            group["matcher"] = registration.matcher
        hooks.setdefault(registration.event_name, []).append(group)
    settings["hooks"] = hooks
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(settings_path, settings)
    return settings_path


def project_hooks_status(
    settings_path: Path,
    *,
    module_name: str,
    required_events: Iterable[str],
) -> dict[str, object]:
    """Report exact adapter registration coverage without inferring host qualification."""

    required = tuple(dict.fromkeys(str(event) for event in required_events))
    if not settings_path.exists():
        return {
            "installed": False,
            "path": str(settings_path),
            "installed_events": [],
            "missing_events": sorted(required),
        }
    try:
        settings = _read_settings(settings_path)
        hooks = _hooks_mapping(settings)
    except (OSError, UnicodeError, ValueError) as exc:
        return {
            "installed": False,
            "path": str(settings_path),
            "installed_events": [],
            "missing_events": sorted(required),
            "configuration_error": str(exc),
        }
    installed_events = sorted(
        event_name
        for event_name, groups in hooks.items()
        if _groups_contain_module(groups, module_name)
    )
    installed_set = set(installed_events)
    missing_events = sorted(event for event in required if event not in installed_set)
    return {
        "installed": not missing_events,
        "path": str(settings_path),
        "installed_events": installed_events,
        "missing_events": missing_events,
    }


def _read_settings(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"hook settings are not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"hook settings must contain a JSON object: {path}")
    return payload


def _hooks_mapping(settings: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    raw_hooks = settings.get("hooks", {})
    if not isinstance(raw_hooks, dict):
        raise ValueError("hook settings 'hooks' must be an object")
    hooks: dict[str, list[dict[str, object]]] = {}
    for event_name, raw_groups in raw_hooks.items():
        if not isinstance(event_name, str) or not isinstance(raw_groups, list):
            raise ValueError("each hook event must map to a list")
        groups: list[dict[str, object]] = []
        for raw_group in raw_groups:
            if not isinstance(raw_group, dict):
                raise ValueError("each hook group must be an object")
            raw_handlers = raw_group.get("hooks")
            if not isinstance(raw_handlers, list) or not all(
                isinstance(handler, dict) for handler in raw_handlers
            ):
                raise ValueError("each hook group must contain a handler list")
            groups.append(dict(raw_group))
        hooks[event_name] = groups
    return hooks


def _remove_module_handlers(
    hooks: dict[str, list[dict[str, object]]],
    module_name: str,
) -> None:
    for event_name in list(hooks):
        retained_groups: list[dict[str, object]] = []
        for group in hooks[event_name]:
            raw_handlers = group["hooks"]
            assert isinstance(raw_handlers, list)
            retained_handlers = [
                handler
                for handler in raw_handlers
                if isinstance(handler, dict) and not _handler_uses_module(handler, module_name)
            ]
            if retained_handlers:
                retained = dict(group)
                retained["hooks"] = retained_handlers
                retained_groups.append(retained)
        if retained_groups:
            hooks[event_name] = retained_groups
        else:
            hooks.pop(event_name)


def _groups_contain_module(groups: list[dict[str, object]], module_name: str) -> bool:
    for group in groups:
        raw_handlers = group.get("hooks")
        if isinstance(raw_handlers, list) and any(
            isinstance(handler, dict) and _handler_uses_module(handler, module_name)
            for handler in raw_handlers
        ):
            return True
    return False


def _handler_uses_module(handler: dict[str, object], module_name: str) -> bool:
    return any(
        isinstance(handler.get(field), str) and module_name in str(handler[field])
        for field in ("command", "commandWindows")
    )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
