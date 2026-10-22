"""User settings and onboarding helpers for the Lethyx CLI.

The command line kept every knob as an explicit flag for a long time.  That is
useful for automation, but a new user should not have to remember a store path,
a workspace name and a backend on every invocation.  This module owns the small
user-level config file that supplies those defaults, plus the harness wiring
used by ``lethyx init`` and ``lethyx connect``.

Resolution order is always: explicit CLI flag, environment variable, config
file, built-in default.  The config file is deliberately plain JSON so a user
can inspect or edit it with any editor.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

DEFAULT_BACKEND = "mnemosyne"
DEFAULT_WORKSPACE = "default"
DEFAULT_PROFILE = "default"
DEFAULT_MODEL = "intfloat/multilingual-e5-large"

HARNESSES: dict[str, str] = {
    "opencode": "OpenCode",
    "claude": "Claude Code",
    "codex": "Codex",
    "hermes": "Hermes",
}


def home() -> Path:
    raw = os.environ.get("LETHYX_HOME", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".lethyx"


def config_path() -> Path:
    raw = os.environ.get("LETHYX_CONFIG", "").strip()
    return Path(raw).expanduser() if raw else home() / "config.json"


def default_store() -> Path:
    return home() / "store"


def default_cache_dir() -> Path:
    return home() / "cache" / "fastembed"


def default_map_path() -> Path:
    return home() / "memory-map.html"


def load_config() -> dict[str, Any]:
    path = config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(config: dict[str, Any]) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(config)
    payload.setdefault("version", 1)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def apply_config_to_env(config: dict[str, Any] | None = None) -> None:
    """Fill unset semantic environment variables from the config.

    Explicit environment variables keep priority.  The config is a convenience
    layer for people who run the CLI directly, not a replacement for the
    environment a harness sets on the MCP process.
    """

    data = config if config is not None else load_config()
    if "semantic" in data:
        os.environ.setdefault("LETHYX_SEMANTIC", "1" if data.get("semantic") else "0")
    model = str(data.get("semantic_model") or "").strip()
    if model:
        os.environ.setdefault("LETHYX_SEMANTIC_MODEL", model)
    cache = str(data.get("semantic_cache_dir") or "").strip()
    if cache:
        os.environ.setdefault("LETHYX_SEMANTIC_CACHE_DIR", cache)


def resolve_cli_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply config defaults to the shared CLI namespace."""

    config = load_config()
    apply_config_to_env(config)
    args.repo_root = str(
        getattr(args, "store", "") or args.repo_root or _env("LETHYX_REPO_ROOT") or
        config.get("repo_root") or ""
    ).strip()
    args.workspace = str(
        args.workspace or _env("LETHYX_WORKSPACE") or config.get("workspace") or
        DEFAULT_WORKSPACE
    ).strip()
    args.profile = str(
        args.profile or _env("LETHYX_PROFILE") or config.get("profile") or
        DEFAULT_PROFILE
    ).strip()
    args.backend = str(
        args.backend or _env("LETHYX_BACKEND") or config.get("backend") or ""
    ).strip()
    return args


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def semantic_available() -> bool:
    return has_module("fastembed")


def backend_available(backend: str) -> bool:
    if backend == "hindsight":
        return has_module("hindsight_client") or has_module("hindsight")
    if backend == "zvec":
        return has_module("zvec")
    if backend == "mnemosyne":
        return has_module("mnemosyne")
    if backend == "mempalace":
        return has_module("mempalace")
    return True


def console_script(name: str) -> str:
    """Return the console script next to the running interpreter, if present."""

    suffix = ".exe" if os.name == "nt" else ""
    candidate = Path(sys.executable).resolve().parent / f"{name}{suffix}"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(name)
    return found or ""


def mcp_command() -> list[str]:
    """Command list that starts the Lethyx MCP server.

    Prefer the console script because harnesses display it nicely; fall back to
    ``python -m`` so a source checkout also works.
    """

    script = console_script("lethyx-mcp")
    if script:
        return [script]
    return [str(Path(sys.executable).resolve()), "-m", "lethyx.memory.mcp_server"]


def mcp_environment(config: dict[str, Any]) -> dict[str, str]:
    env = {
        "LETHYX_REPO_ROOT": str(config.get("repo_root") or ""),
        "LETHYX_WORKSPACE": str(config.get("workspace") or DEFAULT_WORKSPACE),
        "LETHYX_PROFILE": str(config.get("profile") or DEFAULT_PROFILE),
        "LETHYX_BACKEND": str(config.get("backend") or DEFAULT_BACKEND),
        "LETHYX_SESSION_ID": "host-mcp",
        "LETHYX_MCP_RESPONSE_MODE": "compact",
        "LETHYX_MCP_INSTRUCTIONS": "compact",
        "LETHYX_MCP_AWAIT_PROJECTION_SECONDS": "8",
    }
    if config.get("semantic"):
        env["LETHYX_SEMANTIC"] = "1"
        env["LETHYX_SEMANTIC_MODEL"] = str(
            config.get("semantic_model") or DEFAULT_MODEL
        )
        env["LETHYX_SEMANTIC_CACHE_DIR"] = str(
            config.get("semantic_cache_dir") or default_cache_dir()
        )
    else:
        env["LETHYX_SEMANTIC"] = "0"
    return env


def _config_root(*parts: str) -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", "") or Path.home() / ".config")
    return base.joinpath(*parts)


def harness_config_path(harness: str) -> Path:
    if harness == "opencode":
        override = os.environ.get("OPENCODE_CONFIG", "").strip()
        return Path(override).expanduser() if override else _config_root("opencode", "opencode.json")
    if harness == "claude":
        return Path.home() / ".claude.json"
    if harness == "codex":
        root = Path(os.environ.get("CODEX_HOME", "") or Path.home() / ".codex")
        return root.expanduser() / "config.toml"
    if harness == "hermes":
        root = Path(os.environ.get("HERMES_HOME", "") or Path.home() / ".hermes")
        return root.expanduser() / "config.yaml"
    raise ValueError(f"unknown harness: {harness}")


def detect_harnesses() -> list[str]:
    found: list[str] = []
    for harness in HARNESSES:
        path = harness_config_path(harness)
        if harness == "claude":
            # ``~/.claude.json`` lives directly in the home folder, so its
            # parent is always present; check the dedicated directory too.
            present = path.exists() or (Path.home() / ".claude").is_dir()
        else:
            present = path.exists() or path.parent.is_dir()
        if present:
            found.append(harness)
    return found


def _write_json_with_backup(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            backup.write_text(path.read_text(encoding="utf-8-sig"), encoding="utf-8")
        except OSError:
            pass
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def connect_opencode(config: dict[str, Any]) -> dict[str, Any]:
    path = harness_config_path("opencode")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    except ValueError as error:
        raise ValueError(
            f"refusing to overwrite {path}: it is not valid JSON ({error})"
        ) from error
    except OSError as error:
        raise OSError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"refusing to overwrite {path}: the top level is not an object")
    data.setdefault("$schema", "https://opencode.ai/config.json")
    mcp = data.setdefault("mcp", {})
    if not isinstance(mcp, dict):
        raise ValueError(f"refusing to overwrite {path}: 'mcp' is not an object")
    servers = mcp.setdefault("servers", {})
    if not isinstance(servers, dict):
        raise ValueError(f"refusing to overwrite {path}: 'mcp.servers' is not an object")
    previous = servers.get("lethyx") if isinstance(servers.get("lethyx"), dict) else {}
    entry = {
        "type": "local",
        "command": mcp_command(),
        "environment": mcp_environment(config),
        "enabled": True,
    }
    # Preserve extra keys a user may have configured (for example a timeout)
    # and custom environment variables; the generated keys win.
    for key, value in previous.items():
        entry.setdefault(key, value)
    previous_env = previous.get("environment")
    merged_env = dict(previous_env) if isinstance(previous_env, dict) else {}
    merged_env.update(mcp_environment(config))
    entry["command"] = mcp_command()
    entry["environment"] = merged_env
    servers["lethyx"] = entry
    written = _write_json_with_backup(path, data)
    return {"ok": True, "host": "opencode", "mcp_settings": str(written)}


def connect_harness(harness: str, config: dict[str, Any]) -> dict[str, Any]:
    """Install or refresh one harness integration."""

    harness = harness.strip().lower()
    if harness == "opencode":
        return connect_opencode(config)
    from lethyx.integrations.mcp_settings import (
        install_claude_mcp,
        install_codex_mcp,
        install_hermes_mcp,
        validate_identity,
    )

    identity = validate_identity(
        config.get("repo_root") or "",
        config.get("workspace") or DEFAULT_WORKSPACE,
        config.get("profile") or DEFAULT_PROFILE,
        config.get("backend") or DEFAULT_BACKEND,
    )
    if harness == "claude":
        from lethyx.integrations.claude_hooks import install_claude_hooks

        arguments = [
            "--repo-root", identity["repo_root"],
            "--workspace", identity["workspace"],
            "--profile", identity["profile"],
            "--backend", identity["backend"],
        ]
        hooks = install_claude_hooks(
            Path.home(),
            module_arguments=arguments,
            settings_path=Path.home() / ".claude" / "settings.json",
        )
        mcp = install_claude_mcp(
            Path.home() / ".claude.json",
            repo_root=identity["repo_root"],
            workspace_id=identity["workspace"],
            profile_id=identity["profile"],
            backend=identity["backend"],
        )
        return {"ok": True, "host": "claude", "mcp_settings": str(mcp), "hooks": str(hooks)}
    if harness == "codex":
        from lethyx.integrations.codex_hooks import install_codex_hooks

        codex_home = Path(os.environ.get("CODEX_HOME", "") or Path.home() / ".codex").expanduser()
        arguments = [
            "--repo-root", identity["repo_root"],
            "--workspace", identity["workspace"],
            "--profile", identity["profile"],
            "--backend", identity["backend"],
        ]
        hooks = install_codex_hooks(
            Path.home(),
            module_arguments=arguments,
            settings_path=codex_home / "hooks.json",
        )
        mcp = install_codex_mcp(
            codex_home / "config.toml",
            repo_root=identity["repo_root"],
            workspace_id=identity["workspace"],
            profile_id=identity["profile"],
            backend=identity["backend"],
        )
        return {"ok": True, "host": "codex", "mcp_settings": str(mcp), "hooks": str(hooks)}
    if harness == "hermes":
        path = harness_config_path("hermes")
        mcp = install_hermes_mcp(
            path,
            repo_root=identity["repo_root"],
            workspace_id=identity["workspace"],
            profile_id=identity["profile"],
            backend=identity["backend"],
        )
        return {"ok": True, "host": "hermes", "mcp_settings": str(mcp)}
    raise ValueError(f"unknown harness: {harness}")


def harness_connected(harness: str, config: dict[str, Any] | None = None) -> bool:
    """Best-effort check that the Lethyx entry exists for a harness."""

    try:
        path = harness_config_path(harness)
        if not path.exists():
            return False
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return False
    if harness == "opencode":
        try:
            data = json.loads(text)
        except ValueError:
            return False
        servers = ((data.get("mcp") or {}).get("servers") or {})
        return isinstance(servers, dict) and "lethyx" in servers
    return "lethyx" in text
