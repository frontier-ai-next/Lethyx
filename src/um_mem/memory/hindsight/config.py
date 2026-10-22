"""Environment-backed Hindsight configuration."""

import os
from dataclasses import dataclass


def _env(name: str, *, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class HindsightConfig:
    enabled: bool = True
    base_url: str = "http://localhost:8888"
    api_key: str = ""
    timeout_seconds: float = 30.0
    embedded: bool = False
    profile: str = "um-mem-dev"
    llm_provider: str = "openai"
    llm_model: str = "gemma-4-31B-it"
    llm_base_url: str = ""
    retain_async: bool = False
    fail_closed: bool = True
    backend_mode: str = "hindsight"

    @classmethod
    def from_env(cls) -> "HindsightConfig":
        if os.environ.get("UM_MEM_MEMORY_BACKEND") is not None:
            raise ValueError("UM_MEM_MEMORY_BACKEND is not supported; use UM_MEM_BACKEND")
        return cls(
            enabled=_env_bool("UM_MEM_HINDSIGHT_ENABLED", True),
            base_url=_env(
                "UM_MEM_HINDSIGHT_BASE_URL",
                default="http://localhost:8888",
            ).strip()
            or "http://localhost:8888",
            api_key=_env("UM_MEM_HINDSIGHT_API_KEY").strip(),
            timeout_seconds=_env_float("UM_MEM_HINDSIGHT_TIMEOUT_SECONDS", 30.0),
            embedded=_env_bool("UM_MEM_HINDSIGHT_EMBEDDED", False),
            profile=_env(
                "UM_MEM_HINDSIGHT_PROFILE",
                default="um-mem-dev",
            ).strip()
            or "um-mem-dev",
            llm_provider=_env(
                "UM_MEM_HINDSIGHT_LLM_PROVIDER",
                default="openai",
            ).strip()
            or "openai",
            llm_model=_env(
                "UM_MEM_HINDSIGHT_LLM_MODEL",
                default="gemma-4-31B-it",
            ).strip()
            or "gemma-4-31B-it",
            llm_base_url=_env("UM_MEM_HINDSIGHT_LLM_BASE_URL").strip(),
            retain_async=_env_bool("UM_MEM_HINDSIGHT_RETAIN_ASYNC", False),
            fail_closed=_env_bool("UM_MEM_HINDSIGHT_FAIL_CLOSED", True),
            backend_mode=_env("UM_MEM_BACKEND", default="hindsight").strip().lower() or "hindsight",
        )
