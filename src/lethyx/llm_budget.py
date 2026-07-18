"""Token budget helpers from Lethyx environment settings."""

import os


def estimate_tokens(text: str) -> int:
    """Heuristic ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def get_model_context_tokens() -> int:
    """Soft cap for full prompt vs model window (default 200k)."""
    try:
        return max(4096, int(os.environ.get("LETHYX_MODEL_CONTEXT_TOKENS", "200000")))
    except (ValueError, TypeError):
        return 200000


def get_gmas_context_tokens() -> int:
    """Budget for ``build_gmas_context`` aggregated retrieval (default 60k)."""
    try:
        return max(2000, int(os.environ.get("LETHYX_GMAS_CONTEXT_TOKENS", "60000")))
    except (ValueError, TypeError):
        return 60000


def get_prior_knowledge_tokens() -> int:
    """Budget for Prior knowledge section in workspace prompt (default 20k)."""
    try:
        return max(1000, int(os.environ.get("LETHYX_PRIOR_KNOWLEDGE_TOKENS", "20000")))
    except (ValueError, TypeError):
        return 20000
