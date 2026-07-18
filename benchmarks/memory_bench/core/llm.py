"""LLM call wrapper for the memory_bench framework.

Provides:
- Unified chat-completion call (OpenAI-compatible API via .env config).
- Exponential retry with jitter.
- Token and cost accounting via Usage.
- Disk cache keyed by sha256(model, messages, temperature) for
  reproducibility and cost control on repeated judge calls.

No imports from lethyx except the env config helper (read-only usage).
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_bench.core.types import Usage

log = logging.getLogger(__name__)

# Cost table USD per 1000 tokens (input, output).
# Approximate figures; update as pricing changes.
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "o1": (0.015, 0.060),
    "o1-mini": (0.003, 0.012),
}


def _cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a call given model and token counts."""
    for prefix, (in_rate, out_rate) in _COST_TABLE.items():
        if model.startswith(prefix):
            return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1000
    return 0.0


def _get_env_config() -> dict[str, str]:
    """Read LLM config from environment, mirroring lethyx.env convention."""
    try:
        from lethyx.env import get_llm_env_config, load_env

        load_env()
        raw = get_llm_env_config()
        if isinstance(raw, dict):
            return {str(k): str(v or "") for k, v in raw.items()}
        if isinstance(raw, (tuple, list)) and len(raw) >= 3:
            model, api_key, base_url = raw[:3]
            return {
                "model": str(model or ""),
                "api_key": str(api_key or ""),
                "base_url": str(base_url or ""),
            }
    except Exception:
        pass
    return {
        "api_key": os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        "base_url": os.environ.get("LLM_BASE_URL") or "",
        "model": os.environ.get("LLM_MODEL") or "",
    }


@dataclass
class LLMConfig:
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    cache_dir: str = ".memory_bench_cache/llm"
    max_retries: int = 5
    retry_base_s: float = 1.0
    timeout_s: float = 120.0

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        env = _get_env_config()
        model = str(env.get("model") or "").strip()
        if not model:
            raise RuntimeError(
                "LLM model not configured. Set LLM_MODEL in .env or pass an explicit model "
                "in the benchmark suite; memory_bench does not silently fall back to a "
                "different model."
            )
        return model


def _cache_key(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / f"{key}.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_cache(cache_dir: Path, key: str, data: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        (cache_dir / f"{key}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        log.debug("cache write failed: %s", exc)


def call_llm(
    messages: list[dict[str, str]],
    *,
    config: LLMConfig | None = None,
) -> tuple[str, Usage]:
    """Make one chat-completion call, returning (response_text, usage).

    Uses disk cache when available.  Retries with exponential backoff on
    rate-limit and server errors.
    """
    cfg = config or LLMConfig()
    model = cfg.resolved_model()
    env = _get_env_config()
    api_key = env.get("api_key") or ""
    base_url = env.get("base_url") or ""

    reasoning_effort = os.environ.get("MEMORY_BENCH_REASONING_EFFORT", "").strip()
    cache_dir = Path(cfg.cache_dir)
    key = _cache_key(model, messages, cfg.temperature, cfg.max_tokens, reasoning_effort)
    cached = _read_cache(cache_dir, key)
    if cached:
        log.debug("llm cache hit: %s", key[:16])
        usage = Usage(
            prompt_tokens=cached.get("prompt_tokens", 0),
            completion_tokens=cached.get("completion_tokens", 0),
            cost_usd=cached.get("cost_usd", 0.0),
            calls=0,  # cache hit doesn't count as an API call
        )
        return cached["text"], usage

    if not api_key:
        raise RuntimeError("LLM API key not configured. Set LLM_API_KEY or OPENAI_API_KEY in .env")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package required: pip install openai") from exc

    timeout_s = float(os.environ.get("MEMORY_BENCH_LLM_TIMEOUT_S") or cfg.timeout_s)
    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    last_exc: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            request: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
            }
            if reasoning_effort:
                request["reasoning_effort"] = reasoning_effort
            resp = client.chat.completions.create(**request)  # type: ignore[arg-type]
            text = str(resp.choices[0].message.content or "")
            pt = resp.usage.prompt_tokens if resp.usage else 0
            ct = resp.usage.completion_tokens if resp.usage else 0
            cost = _cost_usd(model, pt, ct)
            _write_cache(
                cache_dir,
                key,
                {
                    "text": text,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "cost_usd": cost,
                },
            )
            return text, Usage(prompt_tokens=pt, completion_tokens=ct, cost_usd=cost, calls=1)
        except Exception as exc:
            last_exc = exc
            wait = cfg.retry_base_s * (2**attempt)
            log.warning("llm call attempt %d failed (%s), retry in %.1fs", attempt + 1, exc, wait)
            time.sleep(wait)

    raise RuntimeError(
        f"LLM call failed after {cfg.max_retries} attempts: {last_exc}"
    ) from last_exc
