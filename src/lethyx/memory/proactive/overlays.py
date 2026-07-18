"""Phase-specific always-loaded memory policy."""

from typing import Any

from lethyx.memory.proactive.models import OverlaySection

_PHASE_ALWAYS: dict[str, list[str]] = {
    "research": [
        "workspace charter",
        "research anti-patterns",
        "BKB",
        "open threads",
    ],
    "plan": [
        "workspace charter",
        "current strategy",
        "BKB",
        "workspace lessons",
    ],
    "execute": [
        "accepted plan",
        "BKB",
    ],
    "verify": [
        "verification contract",
        "BKB",
        "active risks",
    ],
    "reflexion": [
        "run timeline",
        "BKB",
        "failure patterns",
        "promotion rules",
    ],
    "preflight": [
        "workspace charter",
        "BKB",
        "active risks",
    ],
    "research_review": [
        "workspace charter",
        "BKB",
        "research anti-patterns",
        "phase commitments",
    ],
    "plan_review": [
        "workspace charter",
        "current strategy",
        "BKB",
        "implementation anti-patterns",
        "phase commitments",
    ],
    "subtask_review": [
        "accepted plan",
        "verification contract",
        "implementation anti-patterns",
        "BKB",
        "active risks",
    ],
    "final_review": [
        "accepted plan",
        "verification contract",
        "current strategy",
        "BKB",
        "active risks",
        "failure patterns",
    ],
}

_MANDATORY_CORE_KEYWORDS = ("identity", "constitution", "bkb")

_KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
    "workspace charter": ("charter", "workspace charter"),
    "research anti-patterns": ("antipattern", "anti-pattern", "anti pattern"),
    "implementation anti-patterns": ("antipattern", "anti-pattern", "anti pattern"),
    "BKB": ("bkb", "behavior rules", "verified behavior"),
    "open threads": ("open thread",),
    "current strategy": ("strategy",),
    "workspace lessons": ("lesson",),
    "accepted plan": ("plan", "phase plan", "run state"),
    "active risks": ("risk",),
    "verification contract": ("verification", "verify"),
    "run timeline": ("run state", "timeline"),
    "failure patterns": ("failure", "pattern"),
    "promotion rules": ("promotion", "bkb"),
}


def phase_policy(phase_id: str) -> dict[str, Any]:
    key = str(phase_id or "").strip().lower()
    return {
        "always_sections": list(_PHASE_ALWAYS.get(key, ["BKB", "workspace charter"])),
        "forbidden_in_core": ["raw_recall_hits", "unverified_candidates"],
    }


def _section_matches_policy(section: OverlaySection, policy_keywords: list[str]) -> bool:
    name = section.name.lower()
    content_head = section.content[:400].lower()
    for keyword in policy_keywords:
        key = keyword.lower()
        aliases = _KEYWORD_ALIASES.get(keyword, (key,))
        if any(alias in name or alias in content_head for alias in aliases):
            return True
    return False


def _is_mandatory_core(section: OverlaySection) -> bool:
    name = section.name.lower()
    return any(keyword in name for keyword in _MANDATORY_CORE_KEYWORDS)


def _dedupe_sections(sections: list[OverlaySection]) -> list[OverlaySection]:
    seen: set[str] = set()
    out: list[OverlaySection] = []
    for section in sections:
        key = section.name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(section)
    return out


def select_sections_by_policy(
    sections: list[OverlaySection],
    policy: dict[str, Any],
) -> list[OverlaySection]:
    """Always keep L0 identity/BKB; add phase-specific sections by policy."""
    mandatory = [s for s in sections if _is_mandatory_core(s)]
    always = [str(item) for item in (policy.get("always_sections") or [])]
    phase_extra = [s for s in sections if _section_matches_policy(s, always) and s not in mandatory]
    merged = _dedupe_sections(mandatory + phase_extra)
    return merged or list(sections)


def build_phase_state_section(
    *,
    phase_id: str,
    manifest_description: str,
    task_brief: str,
    active_risks: list[str],
    forbidden_repeats: list[str],
    open_threads: list[str],
    max_tokens: int = 600,
) -> str:
    lines = [f"Phase: {phase_id}", manifest_description.strip()]
    if task_brief.strip():
        lines.append(f"Task brief: {task_brief.strip()[:800]}")
    if active_risks:
        lines.append("Active risks:")
        lines.extend(f"- {r}" for r in active_risks[:8])
    if forbidden_repeats:
        lines.append("Forbidden repeats:")
        lines.extend(f"- {r}" for r in forbidden_repeats[:8])
    if open_threads:
        lines.append("Open threads:")
        lines.extend(f"- {t}" for t in open_threads[:6])
    text = "\n".join(lines)
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n...[phase state truncated]"
    return text
