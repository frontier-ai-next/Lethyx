"""Prompt-oriented recall helpers on top of MemPalace search."""

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import os
import re
from typing import Any

from lethyx.llm_budget import estimate_tokens

_LOW_SIGNAL_MEMORY_RE = re.compile(
    r"(seed_backup_|^updated\s+\S+|^backup:|^args:\s*workspace_id=|^result:\s*$|^success:\s*true$)",
    re.IGNORECASE,
)
_NON_GMAS_IMPL_RE = re.compile(
    r"\b(fastapi|flask|httpx|aiohttp|uvicorn|django|"
    r"requests\.get|requests\.post|requests\.session)\b",
    re.IGNORECASE,
)
_GMAS_IMPL_RE = re.compile(
    r"\b(gmas|rolegraph|macprunner|agentprofile|build_property_graph|"
    r"graphbuilder|autographbuilder)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class PromptRecallBundle:
    """Flat prompt-injection recall lines.

    Distinct from the canonical ``lethyx.memory.palace.recall.RecallBundle``
    (tiered always_on/hot/warm/graph bundle built by ``MemPalace.recall``).
    """

    entries: list[str]
    flagged_non_gmas: list[str]


def summarized_palace_for_prompt(
    *,
    palace: Any,
    query: str,
    workspace_id: str,
    token_budget: int,
    decay_half_life_days: float = 30.0,
    require_gmas: bool = False,
) -> PromptRecallBundle:
    """Recall best MemPalace items for prompt injection."""
    token_budget = max(120, int(token_budget))
    try:
        if hasattr(palace, "search"):
            hits = palace.search(
                query=query or workspace_id or "workspace context",
                workspace_id=workspace_id,
                n_results=20,
            )
        elif hasattr(palace, "recent"):
            recent_hits = palace.recent(workspace_id=workspace_id, limit=8)
            hits = [
                {
                    "content": item.get("content", ""),
                    "room": item.get("room", ""),
                    "distance": 0.35,
                    "metadata": item.get("metadata", {}),
                }
                for item in recent_hits
            ]
        else:
            hits = []
    except Exception:
        try:
            recent_hits = palace.recent(workspace_id=workspace_id, limit=8)
        except Exception:
            recent_hits = []
        hits = [
            {
                "content": item.get("content", ""),
                "room": item.get("room", ""),
                "distance": 0.35,
                "metadata": item.get("metadata", {}),
            }
            for item in recent_hits
        ]

    scored: list[tuple[float, dict[str, Any]]] = []
    for hit in hits:
        content = str(hit.get("content") or "").strip()
        if _looks_low_signal_memory(content):
            continue
        distance = float(hit.get("distance") or 1.0)
        semantic_score = max(0.0, 1.0 - distance)
        age_days = _age_days(hit)
        decay = math.exp(-(math.log(2) * age_days / max(0.1, decay_half_life_days)))
        scored.append((semantic_score * decay, hit))
    scored.sort(key=lambda item: item[0], reverse=True)

    entries: list[str] = []
    flagged: list[str] = []
    for _, hit in scored:
        content = str(hit.get("content") or "").strip()
        if not content:
            continue
        room = str(hit.get("room") or (hit.get("metadata") or {}).get("room") or "")
        line = f"- [{room}] {content[:650]}"
        if require_gmas and _looks_like_non_gmas_attempt(content):
            flagged.append(line)
        else:
            entries.append(line)

    entries = _fit_lines(entries, token_budget=token_budget)
    flagged = _fit_lines(flagged, token_budget=max(80, token_budget // 2))

    if entries and _recall_llm_enabled():
        rendered = "\n".join(entries)
        if estimate_tokens(rendered) > token_budget:
            summarized = _llm_summarize_lines(entries, token_budget=token_budget)
            if summarized:
                entries = [f"- {line}" for line in summarized.splitlines() if line.strip()]
                entries = _fit_lines(entries, token_budget=token_budget)

    return PromptRecallBundle(entries=entries, flagged_non_gmas=flagged)


def _fit_lines(lines: list[str], *, token_budget: int) -> list[str]:
    out = list(lines)
    while out and estimate_tokens("\n".join(out)) > token_budget:
        out.pop()
    return out


def _age_days(hit: dict[str, Any]) -> float:
    metadata = hit.get("metadata") or {}
    ts = metadata.get("timestamp")
    try:
        event_time = float(ts)
    except (TypeError, ValueError):
        return 0.0
    now = datetime.now(UTC).timestamp()
    return max(0.0, (now - event_time) / 86400.0)


def _looks_like_non_gmas_attempt(content: str) -> bool:
    has_non_gmas = bool(_NON_GMAS_IMPL_RE.search(content or ""))
    has_gmas = bool(_GMAS_IMPL_RE.search(content or ""))
    return has_non_gmas and not has_gmas


def _looks_low_signal_memory(content: str) -> bool:
    normalized = " ".join(str(content or "").split())
    if not normalized:
        return True
    return bool(_LOW_SIGNAL_MEMORY_RE.search(normalized))


def _recall_llm_enabled() -> bool:
    return os.environ.get("LETHYX_RECALL_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _llm_summarize_lines(lines: list[str], *, token_budget: int) -> str:
    try:
        from lethyx.control_plane.code_analyzer import get_llm_client
    except Exception:
        return ""
    client = get_llm_client()
    if client is None:
        return ""
    prompt = (
        "Summarize memory lines into concise actionable bullets for a coding agent. "
        "Keep only concrete technical lessons. Return plain text bullets only."
    )
    user_text = "\n".join(lines[:40])
    try:
        response, _meta = client.chat(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Target tokens <= {token_budget}\n\n{user_text}",
                },
            ]
        )
    except Exception:
        return ""
    content = response.get("content") if isinstance(response, dict) else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks).strip()
    return ""
