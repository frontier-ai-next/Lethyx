"""Contrastive memory retrieval.

Instead of only returning top-k similar lessons, this module returns
both confirmers (successes) and challengers (failures) so the agent
can see what worked and what did not in similar contexts.
"""

import logging
from typing import Any

from lethyx.memory.models import LessonRecord, MemoryQuery
from lethyx.memory.store import MemoryStore

log = logging.getLogger(__name__)


def _classify_outcome(lesson: LessonRecord) -> str:
    """Classify a lesson as success, failure, or neutral."""
    tags = lesson.tags or set()
    avoid = lesson.avoid_tags or []
    repeat = lesson.repeat_tags or []
    observed = (lesson.observed_effect or "").lower()
    conclusion = (lesson.conclusion or "").lower()

    failure_signals = {
        "failure",
        "eval_failure",
        "partial",
        "HIGH_COST_NO_GAIN",
        "error",
        "regression",
    }
    success_signals = {"success", "improved", "promoted", "complete"}

    if tags.intersection(failure_signals) or avoid:
        return "failure"
    if tags.intersection(success_signals) or repeat:
        return "success"

    negative_words = ("failed", "error", "broken", "regression", "timeout", "crash")
    positive_words = (
        "success",
        "improved",
        "working",
        "passed",
        "complete",
        "promoted",
    )

    for word in negative_words:
        if word in observed or word in conclusion:
            return "failure"
    for word in positive_words:
        if word in observed or word in conclusion:
            return "success"

    return "neutral"


def _lesson_to_entry(lesson: LessonRecord) -> dict[str, Any]:
    return {
        "lesson_id": lesson.id,
        "workspace_id": lesson.workspace_id or "",
        "conclusion": lesson.conclusion[:300],
        "change_summary": lesson.change_summary[:200],
        "observed_effect": lesson.observed_effect[:200],
        "tags": sorted(lesson.tags)[:10],
        "raw_evidence_paths": list(getattr(lesson, "raw_evidence_paths", []) or [])[:5],
    }


def retrieve_contrastive_lessons(
    store: MemoryStore,
    *,
    query: str = "",
    workspace_id: str | None = None,
    limit_successes: int = 3,
    limit_failures: int = 3,
) -> dict[str, Any]:
    """Retrieve contrastive lessons: successes vs failures."""
    mq = MemoryQuery(
        workspace_id=workspace_id,
        limit=max((limit_successes + limit_failures) * 4, 40),
        include_stale=False,
    )
    all_lessons = store.query_lessons(mq)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    repeated_failures: list[dict[str, Any]] = []
    challengers: list[dict[str, Any]] = []

    avoid_tags: set[str] = set()
    repeat_tags: set[str] = set()

    for lesson in all_lessons:
        outcome = _classify_outcome(lesson)
        entry = _lesson_to_entry(lesson)

        if outcome == "success" and len(successes) < limit_successes:
            successes.append(entry)
            repeat_tags.update(lesson.repeat_tags or [])
        elif outcome == "failure" and len(failures) < limit_failures:
            failures.append(entry)
            avoid_tags.update(lesson.avoid_tags or [])

    # Same-workspace repeated failures
    if workspace_id:
        ws_query = MemoryQuery(
            workspace_id=workspace_id,
            limit=20,
            include_stale=True,
        )
        ws_lessons = store.query_lessons(ws_query)
        seen_summaries: set[str] = set()
        for lesson in ws_lessons:
            if _classify_outcome(lesson) == "failure":
                summary_key = lesson.change_summary[:100].lower()
                if summary_key in seen_summaries:
                    repeated_failures.append(_lesson_to_entry(lesson))
                seen_summaries.add(summary_key)

    # Challengers: lessons with opposite outcome for similar changes
    if successes and failures:
        success_summaries = {s["change_summary"][:50].lower() for s in successes}
        for f in failures:
            f_summary = f["change_summary"][:50].lower()
            for s_summary in success_summaries:
                overlap = len(set(f_summary.split()) & set(s_summary.split()))
                if overlap >= 2:
                    challengers.append(f)
                    break

    # Tier 2.3 — [DISPUTED] clusters. When >=2 lessons share substantial
    # ``change_summary`` overlap and none of them is verified (priority
    # >= 5 / `unverified_lesson` not in tags), flag the cluster so the
    # recall block surfaces it as DISPUTED rather than as authoritative
    # knowledge. This is what kept happening with ``core_files_exist``:
    # three contradictory "lessons" with identical summary topic, none
    # verified, all surfaced equally.
    disputed_clusters = _detect_disputed_clusters(all_lessons)

    return {
        "query": query,
        "workspace_id": workspace_id or "",
        "successes": successes[:limit_successes],
        "failures": failures[:limit_failures],
        "repeated_failures": repeated_failures[:3],
        "challengers": challengers[:3],
        "avoid_tags": sorted(avoid_tags)[:10],
        "repeat_tags": sorted(repeat_tags)[:10],
        "disputed_clusters": disputed_clusters,
    }


def _detect_disputed_clusters(lessons: list[LessonRecord]) -> list[dict[str, Any]]:
    """Group lessons by overlapping change_summary tokens and flag any
    cluster that has >=2 unverified members with no verified anchor.

    "Unverified" here means either:
    - lesson carries the ``unverified_lesson`` tag (Tier 2.2 writes this
      explicitly when verify_run_id is missing or failed_step_count > 0),
    - OR priority is below the verified threshold (``< 5``).

    A cluster with a verified anchor is *not* disputed — the verified
    lesson resolves the disagreement.
    """

    if not lessons:
        return []

    def _is_verified(lesson: LessonRecord) -> bool:
        tags = lesson.tags or set()
        if "unverified_lesson" in tags:
            return False
        return int(getattr(lesson, "priority", 0) or 0) >= 5

    # Hash buckets by the top-3 longest tokens in the change_summary —
    # cheap proximity without an embedding pass. Good enough to catch
    # the "core_files_exist" / "core_files_exist_path_resolution" /
    # "core_files_exist_line_endings" style of clustering.
    buckets: dict[str, list[LessonRecord]] = {}
    for lesson in lessons:
        summary = (lesson.change_summary or "").lower()
        if not summary:
            continue
        tokens = sorted(
            (t for t in summary.split() if len(t) >= 4),
            key=len,
            reverse=True,
        )[:3]
        if not tokens:
            continue
        key = ":".join(sorted(tokens))
        buckets.setdefault(key, []).append(lesson)

    clusters: list[dict[str, Any]] = []
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        if any(_is_verified(m) for m in members):
            continue
        clusters.append(
            {
                "topic_key": key,
                "lesson_count": len(members),
                "lessons": [_lesson_to_entry(m) for m in members[:5]],
                "label": "[DISPUTED — verify before trusting]",
            }
        )
    clusters.sort(key=lambda c: c["lesson_count"], reverse=True)
    return clusters[:5]


def render_contrastive_memory_section(bundle: dict[str, Any]) -> str:
    """Render contrastive memory as a markdown section."""
    lines: list[str] = []

    successes = bundle.get("successes", [])
    failures = bundle.get("failures", [])
    repeated = bundle.get("repeated_failures", [])
    disputed = bundle.get("disputed_clusters", [])
    avoid = bundle.get("avoid_tags", [])
    repeat = bundle.get("repeat_tags", [])

    if successes:
        lines.append("### What Worked")
        for s in successes:
            lines.append(f"- {s['change_summary']}: {s['conclusion']}")

    if failures:
        lines.append("### What Failed")
        for f in failures:
            lines.append(f"- {f['change_summary']}: {f['conclusion']}")

    if repeated:
        lines.append("### Repeated Failures (same workspace)")
        for r in repeated:
            lines.append(f"- {r['change_summary']}: {r['conclusion']}")

    if disputed:
        lines.append("### [DISPUTED — verify before trusting]")
        for cluster in disputed:
            lines.append(
                f"- topic `{cluster['topic_key']}`: "
                f"{cluster['lesson_count']} unverified lessons disagree. "
                "Re-run verification to settle which (if any) is correct."
            )

    if avoid:
        lines.append(f"### Avoid: {', '.join(avoid)}")
    if repeat:
        lines.append(f"### Repeat: {', '.join(repeat)}")

    return "\n".join(lines) if lines else ""
