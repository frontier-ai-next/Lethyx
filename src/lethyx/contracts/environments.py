"""Execution-environment records for capability-carrying proof runs."""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from lethyx.contracts.hashing import hash_value

log = logging.getLogger(__name__)

DEFAULT_EXECUTION_ENVIRONMENT_ID = "workspace_default"
ENVIRONMENT_RECORDS_FILENAME = "execution_environments.json"
CAPABILITY_BINDINGS_FILENAME = "capability_bindings.json"

# Known aliases that LLMs / agent probes sometimes emit instead of the canonical id.
_ENV_ID_ALIASES: frozenset[str] = frozenset({"", "default", "workspace_default", "workspace"})


def normalize_env_id(env_id: str | None) -> str:
    """Coerce any known env-id alias to the canonical ``workspace_default``."""
    v = str(env_id or "").strip()
    if v in _ENV_ID_ALIASES:
        return DEFAULT_EXECUTION_ENVIRONMENT_ID
    return v


@dataclass(frozen=True)
class ExecutionEnvironmentRecord:
    env_id: str = DEFAULT_EXECUTION_ENVIRONMENT_ID
    python_executable: str = ""
    cwd: str = "."
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    python_version: str = ""
    installed_packages_hash: str = ""
    toolchains: Mapping[str, str] = field(default_factory=dict)
    primary_runner: str = ""
    tcl_tk_status: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    created_at: float = 0.0

    @property
    def env_hash(self) -> str:
        return environment_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "python_executable": self.python_executable,
            "cwd": self.cwd,
            "env_overrides": dict(self.env_overrides),
            "python_version": self.python_version,
            "installed_packages_hash": self.installed_packages_hash,
            "toolchains": dict(self.toolchains),
            "primary_runner": self.primary_runner,
            "tcl_tk_status": dict(self.tcl_tk_status),
            "evidence_refs": list(self.evidence_refs),
            "created_at": self.created_at,
            "env_hash": self.env_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionEnvironmentRecord":
        return cls(
            env_id=normalize_env_id(value.get("env_id")),
            python_executable=str(value.get("python_executable") or ""),
            cwd=str(value.get("cwd") or "."),
            env_overrides={
                str(key): str(raw)
                for key, raw in (value.get("env_overrides") or {}).items()
                if str(key).strip()
            }
            if isinstance(value.get("env_overrides"), dict)
            else {},
            python_version=str(value.get("python_version") or ""),
            installed_packages_hash=str(value.get("installed_packages_hash") or ""),
            toolchains={
                str(key): str(raw)
                for key, raw in (value.get("toolchains") or {}).items()
                if str(key).strip()
            }
            if isinstance(value.get("toolchains"), dict)
            else {},
            primary_runner=str(value.get("primary_runner") or ""),
            tcl_tk_status=dict(value.get("tcl_tk_status") or {})
            if isinstance(value.get("tcl_tk_status"), dict)
            else {},
            evidence_refs=tuple(
                str(item).strip()
                for item in (value.get("evidence_refs") or ())
                if str(item).strip()
            )
            if isinstance(value.get("evidence_refs"), (list, tuple))
            else (),
            created_at=float(value.get("created_at") or 0.0),
        )


@dataclass(frozen=True)
class CapabilityBinding:
    capability_id: str
    available: bool
    env_id: str
    python_executable: str = ""
    cwd: str = "."
    env_hash: str = ""
    probe_command: tuple[str, ...] = ()
    probe_exit_code: int | None = None
    evidence_ref: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "capability_id": self.capability_id,
            "available": self.available,
            "env_id": self.env_id,
            "python_executable": self.python_executable,
            "cwd": self.cwd,
            "env_hash": self.env_hash,
            "probe_command": list(self.probe_command),
            "reason": self.reason,
        }
        if self.probe_exit_code is not None:
            payload["probe_exit_code"] = self.probe_exit_code
        if self.evidence_ref:
            payload["evidence_ref"] = self.evidence_ref
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CapabilityBinding":
        return cls(
            capability_id=str(value.get("capability_id") or ""),
            available=bool(value.get("available")),
            env_id=normalize_env_id(value.get("env_id")),
            python_executable=str(value.get("python_executable") or ""),
            cwd=str(value.get("cwd") or "."),
            env_hash=str(value.get("env_hash") or ""),
            probe_command=tuple(str(item) for item in (value.get("probe_command") or ()))
            if isinstance(value.get("probe_command"), (list, tuple))
            else (),
            probe_exit_code=(
                int(value.get("probe_exit_code"))
                if value.get("probe_exit_code") is not None
                else None
            ),
            evidence_ref=str(value.get("evidence_ref") or ""),
            reason=str(value.get("reason") or ""),
        )


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def resolve_python_executable(repo_root: Path, workspace_root: Path) -> str:
    for candidate in (
        _venv_python(workspace_root / ".venv"),
        _venv_python(repo_root / ".venv"),
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def environment_hash(record: ExecutionEnvironmentRecord | Mapping[str, Any]) -> str:
    if isinstance(record, ExecutionEnvironmentRecord):
        payload = {
            "env_id": record.env_id,
            "python_executable": record.python_executable,
            "cwd": record.cwd,
            "env_overrides": dict(record.env_overrides),
            "python_version": record.python_version,
            "installed_packages_hash": record.installed_packages_hash,
        }
    else:
        payload = dict(record)
    payload.pop("env_hash", None)
    payload.pop("created_at", None)
    payload.pop("evidence_refs", None)
    payload.pop("toolchains", None)
    payload.pop("primary_runner", None)
    payload.pop("tcl_tk_status", None)
    return hash_value(payload)[:24]


def resolve_execution_environment(
    *,
    repo_root: Path,
    workspace_root: Path,
    env_id: str = "",
    cwd: Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
    python_executable: str = "",
) -> ExecutionEnvironmentRecord:
    resolved_id = str(env_id or DEFAULT_EXECUTION_ENVIRONMENT_ID).strip()
    resolved_cwd = cwd or workspace_root
    overrides = {
        str(key): str(value) for key, value in (env_overrides or {}).items() if str(key).strip()
    }
    return ExecutionEnvironmentRecord(
        env_id=resolved_id,
        python_executable=python_executable or resolve_python_executable(repo_root, workspace_root),
        cwd=str(resolved_cwd),
        env_overrides=overrides,
        python_version=sys.version.split()[0],
        toolchains=_detect_toolchains(workspace_root),
        primary_runner=_primary_runner_for_workspace(workspace_root),
        created_at=time.time(),
    )


def _detect_toolchains(workspace_root: Path) -> dict[str, str]:
    toolchains: dict[str, str] = {}
    if (workspace_root / "pyproject.toml").exists() or (
        workspace_root / "requirements.txt"
    ).exists():
        toolchains["python"] = "python"
    if (workspace_root / "package.json").exists():
        if (workspace_root / "pnpm-lock.yaml").exists():
            toolchains["node"] = "pnpm"
        elif (workspace_root / "yarn.lock").exists():
            toolchains["node"] = "yarn"
        else:
            toolchains["node"] = "npm"
    if (workspace_root / "go.mod").exists():
        toolchains["go"] = "go"
    if (workspace_root / "Cargo.toml").exists():
        toolchains["rust"] = "cargo"
    return toolchains


def _primary_runner_for_workspace(workspace_root: Path) -> str:
    toolchains = _detect_toolchains(workspace_root)
    for key in ("python", "node", "go", "rust"):
        if key in toolchains:
            return toolchains[key]
    return "shell"


def _state_path(drive_root: Path, filename: str) -> Path:
    state = drive_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    return state / filename


def load_environment_records(drive_root: Path | None) -> dict[str, ExecutionEnvironmentRecord]:
    if drive_root is None:
        return {}
    path = _state_path(drive_root, ENVIRONMENT_RECORDS_FILENAME)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    records = payload.get("environments") if isinstance(payload, dict) else None
    if not isinstance(records, dict):
        return {}
    out: dict[str, ExecutionEnvironmentRecord] = {}
    for key, raw in records.items():
        if isinstance(raw, dict):
            record = ExecutionEnvironmentRecord.from_mapping(raw)
            out[str(key)] = record
    return out


def persist_environment_record(drive_root: Path, record: ExecutionEnvironmentRecord) -> Path:
    path = _state_path(drive_root, ENVIRONMENT_RECORDS_FILENAME)
    records = {key: value.to_dict() for key, value in load_environment_records(drive_root).items()}
    records[record.env_id] = record.to_dict()
    path.write_text(
        json.dumps({"environments": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_capability_bindings(drive_root: Path | None) -> dict[str, CapabilityBinding]:
    if drive_root is None:
        return {}
    path = _state_path(drive_root, CAPABILITY_BINDINGS_FILENAME)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw_bindings = payload.get("bindings") if isinstance(payload, dict) else None
    if not isinstance(raw_bindings, dict):
        return {}
    out: dict[str, CapabilityBinding] = {}
    for key, raw in raw_bindings.items():
        if isinstance(raw, dict):
            out[str(key)] = CapabilityBinding.from_mapping(raw)
    return out


def persist_capability_binding(drive_root: Path, binding: CapabilityBinding) -> Path:
    path = _state_path(drive_root, CAPABILITY_BINDINGS_FILENAME)
    bindings = {key: value.to_dict() for key, value in load_capability_bindings(drive_root).items()}
    key = f"{binding.capability_id}:{binding.env_id}:{binding.env_hash}"
    bindings[key] = binding.to_dict()
    path.write_text(
        json.dumps({"bindings": bindings}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def find_capability_binding(
    drive_root: Path | None,
    *,
    capability_id: str,
    env_id: str,
    env_hash: str = "",
) -> CapabilityBinding | None:
    wanted_cap = str(capability_id or "").strip()
    wanted_env = normalize_env_id(env_id)
    env_fallback: CapabilityBinding | None = None
    for binding in load_capability_bindings(drive_root).values():
        if binding.capability_id != wanted_cap or binding.env_id != wanted_env:
            continue
        if env_hash and binding.env_hash and binding.env_hash != env_hash:
            if binding.available and env_fallback is None:
                env_fallback = binding
            continue
        return binding
    if env_fallback is not None:
        log.info(
            "degraded capability binding match by env_id: capability=%s env_id=%s "
            "wanted_env_hash=%s binding_env_hash=%s",
            wanted_cap,
            wanted_env,
            env_hash,
            env_fallback.env_hash,
        )
        return env_fallback
    return None


def classify_tcl_tk_status(output: str) -> dict[str, Any]:
    text = str(output or "")
    lowered = text.lower()
    missing_tokens = [
        token
        for token in ("tk.tcl", "wintheme.tcl", "xptheme.tcl", "panedwindow.tcl")
        if token in lowered
    ]
    if "_tkinter.tclerror" in lowered or missing_tokens:
        return {
            "available": False,
            "reason": "tk_runtime_setup_failed",
            "missing": missing_tokens,
        }
    if "no display name" in lowered or "couldn't connect to display" in lowered:
        return {"available": False, "reason": "display_unavailable"}
    return {}
