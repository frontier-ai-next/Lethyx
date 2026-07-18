import logging
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Mapping

log = logging.getLogger(__name__)

DEFAULT_WORKSPACE_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_CODE_ANALYZER_MODEL = "claude-opus-4"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"

WATCHER_BUDGET_ENABLED_ENV = "LETHYX_WATCHER_BUDGET_ENABLED"

_LLM_API_KEY_ALIASES = (
    "LLM_API_KEY",
    "LETHYX_LLM_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
)
_LLM_BASE_URL_ALIASES = (
    "LLM_BASE_URL",
    "LETHYX_LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
)
_LLM_MODEL_ALIASES = ("LLM_MODEL", "LETHYX_MODEL")


@dataclass(frozen=True)
class LLMRuntimeConfig:
    """Resolved Lethyx LLM runtime aliases for launchers and probes."""

    model: str = ""
    api_key: str = ""
    base_url: str = ""
    model_source: str = ""
    api_key_source: str = ""
    base_url_source: str = ""

    @property
    def available(self) -> bool:
        return bool(self.model and self.api_key and self.base_url)

    def missing_alias_groups(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.api_key:
            missing.append("api_key")
        if not self.base_url:
            missing.append("base_url")
        if not self.model:
            missing.append("model")
        return tuple(missing)

    def public_env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.api_key:
            out["LLM_API_KEY"] = self.api_key
        if self.base_url:
            out["LLM_BASE_URL"] = self.base_url
        if self.model:
            out["LLM_MODEL"] = self.model
        return out


def _first_env_value(env: Mapping[str, str | None], aliases: tuple[str, ...]) -> tuple[str, str]:
    for alias in aliases:
        value = str(env.get(alias) or "").strip()
        if value:
            return value, alias
    return "", ""


def resolve_llm_runtime_config(
    env: Mapping[str, str | None] | None = None,
) -> LLMRuntimeConfig:
    """Resolve the same LLM runtime aliases used by Lethyx launch paths.

    The resolver is intentionally provider-neutral. Generated workspace code should
    consume public ``LLM_*`` aliases, while Lethyx may be launched with
    ``LETHYX_*`` or provider-specific key aliases. Capability probes and
    launchers must call this resolver instead of reimplementing a smaller alias
    list.
    """

    source = env if env is not None else os.environ
    api_key, api_key_source = _first_env_value(source, _LLM_API_KEY_ALIASES)
    base_url, base_url_source = _first_env_value(source, _LLM_BASE_URL_ALIASES)
    model, model_source = _first_env_value(source, _LLM_MODEL_ALIASES)
    return LLMRuntimeConfig(
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model_source=model_source,
        api_key_source=api_key_source,
        base_url_source=base_url_source,
    )


def llm_runtime_available(env: Mapping[str, str | None] | None = None) -> bool:
    return resolve_llm_runtime_config(env).available


def llm_runtime_probe_diagnostics(
    env: Mapping[str, str | None] | None = None,
) -> dict[str, object]:
    config = resolve_llm_runtime_config(env)
    return {
        "available": config.available,
        "missing": list(config.missing_alias_groups()),
        "sources": {
            "model": config.model_source,
            "api_key": config.api_key_source,
            "base_url": config.base_url_source,
        },
    }


def apply_public_llm_runtime_env(
    env: dict[str, str],
    config: LLMRuntimeConfig | None = None,
) -> None:
    """Populate public LLM_* aliases in ``env`` without clobbering existing values."""

    resolved = config or resolve_llm_runtime_config(env)
    for key, value in resolved.public_env().items():
        env.setdefault(key, value)


def env_truthy(name: str) -> bool:
    """Return whether ``os.environ[name]`` is a truthy flag (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def watcher_budget_enforcement_enabled() -> bool:
    """Whether watcher enforces phase ``{id}.budget.json`` time/tool-call limits.

    Configure via ``LETHYX_WATCHER_BUDGET_ENABLED`` in repo ``.env`` (see ``load_env``).
    """
    return env_truthy(WATCHER_BUDGET_ENABLED_ENV)


def _iter_env_candidates(
    *,
    repo_root: Path,
    extra_search_roots: Iterable[Path] = (),
) -> list[Path]:
    candidates = [
        repo_root / ".env",
        *(Path(root) / ".env" for root in extra_search_roots),
        Path.cwd() / ".env",
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _load_simple_env_file(env_path: Path, *, override: bool = False) -> None:
    """Load a minimal .env file without requiring python-dotenv."""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not override and key in os.environ:
            continue
        os.environ[key] = value


def read_default_llm_model_from_repo_dotenv(repo_root: Path) -> str | None:
    """Read ``LETHYX_MODEL`` / ``LLM_MODEL`` from ``<repo_root>/.env`` only (no ``os.environ``).

    Used by web bridge so the Settings UI matches the file after edits even when
    ``load_dotenv(override=False)`` left stale model ids from the parent shell.
    """
    path = Path(repo_root).resolve() / ".env"
    if not path.is_file():
        return None
    lethyx, llm = "", ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if key not in ("LETHYX_MODEL", "LLM_MODEL"):
            continue
        value = rest.split("#", 1)[0].strip() if rest else ""
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key == "LETHYX_MODEL":
            lethyx = value
        else:
            llm = value
    pick = (lethyx or llm).strip()
    return pick or None


def load_env(
    *,
    repo_root: Path | None = None,
    extra_search_roots: Iterable[Path] = (),
    override: bool = False,
) -> Path | None:
    """Load the first available .env file into os.environ."""
    search_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    except Exception as exc:
        log.debug("Failed to import python-dotenv: %s", exc)
        load_dotenv = None

    for env_path in _iter_env_candidates(
        repo_root=search_root,
        extra_search_roots=(Path(root).resolve() for root in extra_search_roots),
    ):
        if not env_path.exists():
            continue
        if load_dotenv is not None:
            load_dotenv(env_path, override=override)
        else:
            _load_simple_env_file(env_path, override=override)
        log.info("Loaded environment variables from %s", env_path)
        # Debug: log key model env vars
        log.debug("LLM_API_KEY: %s", "SET" if os.environ.get("LLM_API_KEY") else "NOT SET")
        log.debug("LLM_BASE_URL: %s", os.environ.get("LLM_BASE_URL"))
        log.debug("LETHYX_MODEL_LIGHT: %s", os.environ.get("LETHYX_MODEL_LIGHT"))
        log.debug("LETHYX_LLM_BASE_URL: %s", os.environ.get("LETHYX_LLM_BASE_URL"))
        return env_path

    return None


def get_llm_env_config() -> tuple[str | None, str | None, str | None]:
    """Return model, api key, and base URL from the shared runtime resolver."""
    config = resolve_llm_runtime_config()
    return (config.model or None, config.api_key or None, config.base_url or None)


def get_default_workspace_model() -> str:
    """Return the default model for workspace execution paths."""
    return (
        os.environ.get("LLM_MODEL") or DEFAULT_WORKSPACE_LLM_MODEL
    ).strip() or DEFAULT_WORKSPACE_LLM_MODEL


def get_code_analyzer_model() -> str:
    """Return the model used by code-analysis helpers."""
    return (
        os.environ.get("LETHYX_CODE_ANALYZER_MODEL")
        or os.environ.get("LLM_MODEL")
        or DEFAULT_CODE_ANALYZER_MODEL
    ).strip() or DEFAULT_CODE_ANALYZER_MODEL


def get_openai_base_url(base_url: str | None = None) -> str:
    """Return an OpenAI-compatible base URL with a stable default."""
    return (
        base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_OPENAI_BASE_URL
    ).strip() or DEFAULT_OPENAI_BASE_URL


def get_anthropic_base_url(base_url: str | None = None) -> str:
    """Return an Anthropic-compatible base URL with a stable default."""
    return (
        base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL
    ).strip() or DEFAULT_ANTHROPIC_BASE_URL
