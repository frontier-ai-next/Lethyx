"""Token budget resolution for proactive memory overlay."""

import os

from lethyx.memory.proactive.models import OverlaySection

_DEFAULT_TOKENS = 4500
_MIN_TOKENS = 1800
_MAX_NORMAL = 7500
_MAX_HIGH_RISK = 10000
_PHASE_CAP_FRACTION = 0.35
_BUDGET_MARGIN = 50

_HIGH_RISK_PHASES = frozenset({"reflexion", "verify", "final_review"})

_SECTION_PRIORITY: list[tuple[int, tuple[str, ...]]] = [
    (1, ("identity", "constitution")),
    (2, ("bkb", "invariant", "anti_pattern", "antipattern", "failure pattern")),
    (3, ("charter", "strategy", "operating principle")),
    (4, ("phase commitment", "phase commitments")),
    (5, ("risk", "forbidden")),
    (6, ("run state", "current run")),
    (7, ("lesson", "open thread")),
    (8, ("archive",)),
]


def _parse_env_int(name: str) -> int | None:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


def resolve_proactive_budget(
    *,
    phase: str,
    manifest_budget: int,
    env_override: str | None = None,
) -> int:
    explicit = _parse_env_int("LETHYX_PROACTIVE_MEMORY_BUDGET")
    if env_override and env_override.strip():
        try:
            explicit = max(1, int(env_override.strip()))
        except ValueError:
            pass

    phase_key = str(phase or "").strip().lower()
    base = _DEFAULT_TOKENS
    ceiling = _MAX_HIGH_RISK if phase_key in _HIGH_RISK_PHASES else _MAX_NORMAL

    if explicit is not None:
        return max(1, min(explicit, ceiling))

    budget = max(_MIN_TOKENS, min(base, ceiling))

    if manifest_budget > 0 and explicit is None:
        phase_cap = int(manifest_budget * _PHASE_CAP_FRACTION)
        if phase_cap >= _MIN_TOKENS:
            budget = min(budget, phase_cap)

    return max(_MIN_TOKENS, budget)


def _section_priority(section: OverlaySection) -> int:
    name = section.name.lower()
    for rank, keywords in _SECTION_PRIORITY:
        if any(keyword in name for keyword in keywords):
            return rank
    return 99


def trim_sections_to_budget(
    sections: list[OverlaySection],
    budget: int,
    *,
    phase_id: str = "",
) -> list[OverlaySection]:
    """Drop lowest-priority sections until total tokens fit budget."""
    del phase_id
    if not sections:
        return []
    ordered = sorted(sections, key=_section_priority)
    kept: list[OverlaySection] = []
    total = 0
    for section in ordered:
        cost = max(1, section.token_count)
        if total + cost <= budget + _BUDGET_MARGIN:
            kept.append(section)
            total += cost
            continue
        if _section_priority(section) <= 2:
            remaining = budget + _BUDGET_MARGIN - total
            if remaining >= 80:
                max_chars = max(80, remaining * 4)
                trimmed = section.content[: max_chars - 24].rstrip() + "\n...[budget trim]"
                trimmed_section = OverlaySection(
                    name=section.name,
                    content=trimmed,
                    source_refs=list(section.source_refs),
                    source_hashes=list(section.source_hashes),
                    trust=section.trust,
                    token_count=estimate_tokens(trimmed),
                )
                kept.append(trimmed_section)
                total += trimmed_section.token_count
    if not kept and ordered:
        first = ordered[0]
        max_chars = max(80, budget * 4)
        trimmed = first.content[: max_chars - 24].rstrip() + "\n...[budget trim]"
        kept.append(
            OverlaySection(
                name=first.name,
                content=trimmed,
                source_refs=list(first.source_refs),
                source_hashes=list(first.source_hashes),
                trust=first.trust,
                token_count=estimate_tokens(trimmed),
            )
        )
    return sorted(kept, key=lambda s: (_section_priority(s), s.name))
