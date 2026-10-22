"""Install identity-bound UM-MEM MCP servers in user-owned host settings."""

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import yaml

SERVER_NAME = "um_mem"
_MANAGED_MARKER = "um-memory"


def default_repo_root() -> Path:
    """Return a user-data store path that can never default to the source tree."""

    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "UM-MEM" / "store"
    if os.environ.get("XDG_DATA_HOME"):
        return Path(os.environ["XDG_DATA_HOME"]) / "um-mem" / "store"
    return Path.home() / ".local" / "share" / "um-mem" / "store"


def validate_identity(
    repo_root: str | Path | None,
    workspace_id: str,
    profile_id: str,
    backend: str,
) -> dict[str, str]:
    """Validate and normalize the explicit host installation identity."""

    from um_mem.memory.local_service import build_local_service_identity

    identity = build_local_service_identity(
        repo_root=repo_root or default_repo_root(),
        workspace_id=workspace_id,
        profile_id=profile_id,
        backend=backend,
        caller_role="operator",
    )
    return {
        "repo_root": str(identity.repo_root),
        "workspace": identity.workspace_id,
        "profile": identity.profile_id,
        "backend": identity.backend,
    }


def _server_config(identity: dict[str, str]) -> dict[str, Any]:
    _require_mcp_runtime()
    _require_backend_runtime(identity["backend"])
    return {
        "command": str(Path(sys.executable).resolve()),
        "args": ["-m", "um_mem.memory.mcp_server"],
        "env": {
            "UM_MEM_REPO_ROOT": identity["repo_root"],
            "UM_MEM_WORKSPACE": identity["workspace"],
            "UM_MEM_PROFILE": identity["profile"],
            "UM_MEM_BACKEND": identity["backend"],
            "UM_MEM_MANAGED_CONFIG": _MANAGED_MARKER,
        },
    }


def _require_backend_runtime(backend: str) -> None:
    optional_modules = {"hindsight": "hindsight_client", "zvec": "zvec"}
    module_name = optional_modules.get(backend)
    if module_name is None:
        return
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util; assert importlib.util.find_spec({module_name!r})",
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"The selected Python lacks the {backend} backend dependency; "
            f"install with `python -m pip install 'um-mem[{backend}]'` first"
        )


def _require_mcp_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mcp.server.fastmcp import FastMCP; assert FastMCP",
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "The selected Python cannot import the UM-MEM MCP dependency; "
            "install with `python -m pip install 'um-mem[mcp]'` first"
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"host settings are not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"host settings must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    content = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write(path, content)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
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


def _assert_managed_or_missing(entry: object, host: str) -> None:
    if entry is None:
        return
    if not isinstance(entry, Mapping):
        raise ValueError(f"{host} already has a non-object MCP server named {SERVER_NAME!r}")
    environment = entry.get("env")
    if (
        not isinstance(environment, Mapping)
        or environment.get("UM_MEM_MANAGED_CONFIG") != _MANAGED_MARKER
    ):
        raise ValueError(
            f"{host} already has an unmanaged MCP server named {SERVER_NAME!r}; "
            "rename it before installing UM-MEM"
        )


def install_claude_mcp(
    settings_path: str | Path,
    *,
    repo_root: str | Path | None,
    workspace_id: str,
    profile_id: str,
    backend: str,
) -> Path:
    """Install a global Claude Code stdio MCP server without replacing other entries."""

    identity = validate_identity(repo_root, workspace_id, profile_id, backend)
    config = _server_config(identity)
    path = Path(settings_path).expanduser().resolve()
    settings = _read_json(path)
    servers = settings.setdefault("mcpServers", {})
    if not isinstance(servers, MutableMapping):
        raise ValueError("Claude Code mcpServers setting must be an object")
    _assert_managed_or_missing(servers.get(SERVER_NAME), "Claude Code")
    servers[SERVER_NAME] = config
    _write_json(path, settings)
    return path


def install_codex_mcp(
    settings_path: str | Path,
    *,
    repo_root: str | Path | None,
    workspace_id: str,
    profile_id: str,
    backend: str,
) -> Path:
    """Install a global Codex stdio MCP server while preserving TOML comments/settings."""

    import tomlkit

    identity = validate_identity(repo_root, workspace_id, profile_id, backend)
    config = _server_config(identity)
    path = Path(settings_path).expanduser().resolve()
    document = (
        tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
    )
    servers = document.get("mcp_servers")
    if servers is None:
        servers = tomlkit.table()
        document["mcp_servers"] = servers
    if not isinstance(servers, MutableMapping):
        raise ValueError("Codex mcp_servers setting must be a TOML table")
    _assert_managed_or_missing(servers.get(SERVER_NAME), "Codex")
    server = tomlkit.table()
    server.add("command", config["command"])
    server.add("args", config["args"])
    env = tomlkit.table()
    for key, value in config["env"].items():
        env.add(key, value)
    server.add("env", env)
    server.add("enabled", True)
    server.add("startup_timeout_sec", 20)
    server.add("tool_timeout_sec", 120)
    server.add("default_tools_approval_mode", "prompt")
    servers[SERVER_NAME] = server
    _atomic_write(path, tomlkit.dumps(document).encode("utf-8"))
    return path


def mcp_server_status(settings_path: str | Path, *, host: str) -> dict[str, Any]:
    """Inspect registration without claiming that the host loaded or called the server."""

    path = Path(settings_path).expanduser().resolve()
    try:
        if host == "claude":
            servers = _read_json(path).get("mcpServers", {})
            if not isinstance(servers, Mapping):
                raise ValueError("Claude Code mcpServers setting must be an object")
            entry = servers.get(SERVER_NAME)
        elif host == "codex":
            import tomllib

            data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            servers = data.get("mcp_servers", {})
            if not isinstance(servers, Mapping):
                raise ValueError("Codex mcp_servers setting must be a TOML table")
            entry = servers.get(SERVER_NAME)
        else:
            raise ValueError("host must be 'claude' or 'codex'")
    except (OSError, UnicodeError, ValueError) as exc:
        return {
            "configured": False,
            "path": str(path),
            "host_connection_tested": False,
            "configuration_error": str(exc),
        }
    managed = isinstance(entry, Mapping) and isinstance(entry.get("env"), Mapping)
    environment = entry.get("env", {}) if isinstance(entry, dict) else {}
    command = str(entry.get("command") or "") if isinstance(entry, Mapping) else ""
    configured = bool(
        managed
        and environment.get("UM_MEM_MANAGED_CONFIG") == _MANAGED_MARKER
        and isinstance(entry, Mapping)
        and command
        and Path(command).expanduser().is_file()
        and entry.get("args") == ["-m", "um_mem.memory.mcp_server"]
    )
    return {
        "configured": configured,
        "path": str(path),
        "server_name": SERVER_NAME if configured else "",
        "identity": {
            "repo_root": environment.get("UM_MEM_REPO_ROOT", ""),
            "workspace": environment.get("UM_MEM_WORKSPACE", ""),
            "profile": environment.get("UM_MEM_PROFILE", ""),
            "backend": environment.get("UM_MEM_BACKEND", ""),
        }
        if configured
        else {},
        "integration_mode": "mcp_tools_only",
        "active_mask_delivery": False,
        "host_connection_tested": False,
    }


def install_hermes_mcp(
    settings_path: str | Path,
    *,
    repo_root: str | Path | None,
    workspace_id: str,
    profile_id: str,
    backend: str,
) -> Path:
    """Install the MCP-only Hermes path in its profile YAML without losing comments."""

    from io import StringIO

    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.error import YAMLError

    identity = validate_identity(repo_root, workspace_id, profile_id, backend)
    config = _server_config(identity)
    path = Path(settings_path).expanduser().resolve()
    yaml = YAML()
    yaml.preserve_quotes = True
    if path.exists():
        try:
            document = yaml.load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, YAMLError) as exc:
            raise ValueError(f"Hermes settings are not valid YAML: {path}") from exc
    else:
        document = CommentedMap()
    if document is None:
        document = CommentedMap()
    if not isinstance(document, MutableMapping):
        raise ValueError("Hermes config.yaml must contain a YAML mapping")
    servers = document.get("mcp_servers")
    if servers is None:
        servers = CommentedMap()
        document["mcp_servers"] = servers
    if not isinstance(servers, MutableMapping):
        raise ValueError("Hermes mcp_servers setting must be a YAML mapping")
    existing = servers.get(SERVER_NAME)
    _assert_managed_or_missing(existing, "Hermes")
    if existing is None:
        server = CommentedMap()
        servers[SERVER_NAME] = server
    else:
        server = existing
    server["command"] = config["command"]
    server["args"] = config["args"]
    environment = server.get("env")
    if environment is None:
        environment = CommentedMap()
        server["env"] = environment
    if not isinstance(environment, MutableMapping):
        raise ValueError("Hermes UM-MEM env setting must be a YAML mapping")
    environment.update(config["env"])
    server["enabled"] = True
    server["connect_timeout"] = 20
    server["timeout"] = 120
    rendered = StringIO()
    yaml.dump(document, rendered)
    _atomic_write(path, rendered.getvalue().encode("utf-8"))
    return path


def hermes_mcp_server_status(settings_path: str | Path) -> dict[str, Any]:
    """Inspect Hermes MCP registration without launching its process."""

    path = Path(settings_path).expanduser().resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ValueError("Hermes config.yaml must contain a YAML mapping")
        servers = data.get("mcp_servers", {})
        if not isinstance(servers, Mapping):
            raise ValueError("Hermes mcp_servers setting must be a YAML mapping")
        entry = servers.get(SERVER_NAME)
        if not isinstance(entry, Mapping):
            entry = {}
        environment = entry.get("env", {})
        if not isinstance(environment, Mapping):
            environment = {}
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        return {
            "configured": False,
            "path": str(path),
            "host_connection_tested": False,
            "configuration_error": str(exc),
        }
    command = str(entry.get("command") or "")
    configured = bool(
        environment.get("UM_MEM_MANAGED_CONFIG") == _MANAGED_MARKER
        and command
        and Path(command).expanduser().is_file()
        and entry.get("args") == ["-m", "um_mem.memory.mcp_server"]
    )
    return {
        "configured": configured,
        "path": str(path),
        "server_name": SERVER_NAME if configured else "",
        "identity": {
            "repo_root": environment.get("UM_MEM_REPO_ROOT", ""),
            "workspace": environment.get("UM_MEM_WORKSPACE", ""),
            "profile": environment.get("UM_MEM_PROFILE", ""),
            "backend": environment.get("UM_MEM_BACKEND", ""),
        }
        if configured
        else {},
        "integration_mode": "mcp_tools_only",
        "active_mask_delivery": False,
        "host_connection_tested": False,
    }
