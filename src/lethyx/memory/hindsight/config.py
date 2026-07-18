"""Environment-backed Hindsight configuration."""

import os
from dataclasses import dataclass


def _env(*names: str, default: str = "") -> str:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None:
            return raw
    return default


def _env_bool(name: str, default: bool = False, *aliases: str) -> bool:
    raw = _env(name, *aliases)
    if raw is None:
        return default
    if not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, *aliases: str) -> float:
    raw = _env(name, *aliases)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int, *aliases: str) -> int:
    raw = _env(name, *aliases)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class HindsightConfig:
    enabled: bool = True
    base_url: str = "http://localhost:8888"
    api_key: str = ""
    timeout_seconds: float = 30.0
    embedded: bool = False
    profile: str = "lethyx-dev"
    llm_provider: str = "openai"
    llm_model: str = "gemma-4-31B-it"
    retain_async: bool = True
    fail_closed: bool = True
    reflect_enabled: bool = False
    max_candidates: int = 3
    backend_mode: str = "hindsight"

    @classmethod
    def from_env(cls) -> "HindsightConfig":
        return cls(
            enabled=_env_bool("LETHYX_HINDSIGHT_ENABLED", True, "LETHYX_HINDSIGHT_ENABLED"),
            base_url=_env(
                "LETHYX_HINDSIGHT_BASE_URL",
                "LETHYX_HINDSIGHT_BASE_URL",
                default="http://localhost:8888",
            ).strip()
            or "http://localhost:8888",
            api_key=_env("LETHYX_HINDSIGHT_API_KEY", "LETHYX_HINDSIGHT_API_KEY").strip(),
            timeout_seconds=_env_float(
                "LETHYX_HINDSIGHT_TIMEOUT_SECONDS", 30.0, "LETHYX_HINDSIGHT_TIMEOUT_SECONDS"
            ),
            embedded=_env_bool("LETHYX_HINDSIGHT_EMBEDDED", False, "LETHYX_HINDSIGHT_EMBEDDED"),
            profile=_env(
                "LETHYX_HINDSIGHT_PROFILE",
                "LETHYX_HINDSIGHT_PROFILE",
                default="lethyx-dev",
            ).strip()
            or "lethyx-dev",
            llm_provider=_env(
                "LETHYX_HINDSIGHT_LLM_PROVIDER",
                "LETHYX_HINDSIGHT_LLM_PROVIDER",
                default="openai",
            ).strip()
            or "openai",
            llm_model=_env(
                "LETHYX_HINDSIGHT_LLM_MODEL",
                "LETHYX_HINDSIGHT_LLM_MODEL",
                default=os.environ.get("LLM_MODEL", "gemma-4-31B-it"),
            ).strip()
            or os.environ.get("LLM_MODEL", "gemma-4-31B-it"),
            retain_async=_env_bool(
                "LETHYX_HINDSIGHT_RETAIN_ASYNC", True, "LETHYX_HINDSIGHT_RETAIN_ASYNC"
            ),
            fail_closed=_env_bool(
                "LETHYX_HINDSIGHT_FAIL_CLOSED", True, "LETHYX_HINDSIGHT_FAIL_CLOSED"
            ),
            reflect_enabled=_env_bool(
                "LETHYX_HINDSIGHT_REFLECT_ENABLED", False, "LETHYX_HINDSIGHT_REFLECT_ENABLED"
            ),
            max_candidates=max(
                1,
                min(
                    20,
                    _env_int(
                        "LETHYX_HINDSIGHT_MAX_CANDIDATES", 3, "LETHYX_HINDSIGHT_MAX_CANDIDATES"
                    ),
                ),
            ),
            backend_mode=_env("LETHYX_BACKEND", "LETHYX_MEMORY_BACKEND", default="hindsight")
            .strip()
            .lower()
            or "hindsight",
        )
