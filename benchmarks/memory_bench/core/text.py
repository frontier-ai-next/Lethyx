"""Text normalisation utilities for deterministic scoring metrics.

Conventions match LoCoMo / SQuAD-style normalisation used in the original
benchmark papers: lowercase, strip punctuation, remove English articles,
collapse whitespace.
"""

import re
import string
import unicodedata

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalise a string for F1 / exact-match scoring.

    Steps (matching SQuAD / LoCoMo upstream):
    1. Unicode NFKC normalisation.
    2. Lowercase.
    3. Remove punctuation.
    4. Remove English articles (a / an / the).
    5. Collapse and strip whitespace.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split normalised text into tokens (whitespace split)."""
    return normalize(text).split()


def is_abstain_response(text: str) -> bool:
    """Return True if the model response indicates it cannot answer.

    Covers common phrasings used by LLMs when abstaining.
    """
    lower = text.lower().strip()
    patterns = [
        "i don't know",
        "i do not know",
        "cannot answer",
        "can't answer",
        "i'm not sure",
        "i am not sure",
        "no information",
        "not mentioned",
        "not provided",
        "unanswerable",
        "i have no",
        "there is no",
        "there's no",
        "unable to find",
        "no relevant",
    ]
    return any(p in lower for p in patterns)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens (1 token ≈ 4 chars)."""
    if max_tokens <= 0:
        return text
    limit = max_tokens * 4
    if len(text) <= limit:
        return text
    return text[:limit]


def _metadata_line(item: object) -> str:
    metadata = dict(getattr(item, "metadata", {}) or {})
    parts: list[str] = []
    source_id = str(getattr(item, "source_id", "") or metadata.get("evidence_id") or "")
    if source_id:
        parts.append(f"source_id={source_id}")
    role = str(metadata.get("role") or metadata.get("producer") or "")
    if role:
        parts.append(f"role={role}")
    session = str(metadata.get("session_id") or metadata.get("run_id") or "")
    if session:
        parts.append(f"session={session}")
    ts = str(metadata.get("ts") or metadata.get("timestamp") or metadata.get("created_at") or "")
    if ts:
        parts.append(f"ts={ts}")
    backend = str(metadata.get("memory_block_backend") or metadata.get("native_backend") or "")
    if backend:
        parts.append(f"backend={backend}")
    tags = metadata.get("tags") or []
    if isinstance(tags, str):
        tags_text = tags
    else:
        tags_text = ",".join(str(tag) for tag in tags if str(tag))
    if tags_text:
        parts.append(f"tags={tags_text}")
    return f"[{'; '.join(parts)}]" if parts else ""


def format_context(
    items: list, *, max_tokens: int = 0, sep: str = "\n\n", include_metadata: bool = False
) -> str:
    """Format a list of RetrievedItem objects into a context string.

    Args:
        items: list of RetrievedItem (must have .text attribute).
        max_tokens: if > 0, truncate total context to this many tokens.
        sep: separator between items.
    """
    parts: list[str] = []
    budget = max_tokens * 4 if max_tokens > 0 else 0
    used = 0
    for item in items:
        text = str(getattr(item, "text", item))
        prefix = _metadata_line(item) if include_metadata else ""
        chunk = f"{prefix}\n{text}" if prefix else text
        if budget and used + len(chunk) > budget:
            remaining = budget - used
            if remaining > 0:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        used += len(chunk) + len(sep)
    return sep.join(parts)
