"""LLM-based reflection phase for verified complex iterations."""

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Any


from lethyx.memory import (
    get_workspace_store,
    record_competency_signal,
    record_workspace_lesson,
)
from lethyx.memory.models import SignalCategory

log = logging.getLogger(__name__)

_REFLECTION_SYSTEM = (
    "You are an autonomous reflection module for a coding workspace manager. "
    "Given run evidence, output strict JSON with schema: "
    '{"lesson":{"change_summary":"...","expected_effect":"...","observed_effect":"...",'
    '"conclusion":"...","repeat_tags":["..."],"avoid_tags":["..."]},'
    '"candidate_skill":{"name":"...","domains":["..."],"when_to_use":"...",'
    '"params":[{"name":"...","description":"..."}],"steps":["..."]}|null,'
    '"gap_signal":{"capability_area":"...","evidence_summary":"...","strength":-0.4}|null}. '
    "Always include lesson. Do not emit prose."
)


@dataclass(slots=True)
class ReflectionResult:
    status: str
    reason: str = ""
    lesson_id: str = ""
    skill_slug: str = ""
    signal_id: str = ""

    @classmethod
    def skipped(cls, reason: str) -> "ReflectionResult":
        return cls(status="skipped", reason=reason)


def run_reflection_phase(
    *,
    repo_root: Path,
    workspace_id: str,
    task_id: str,
    verification_report: dict[str, Any] | None,
    tool_call_count: int,
    final_message: str,
    changes_made: list[str] | None = None,
    critic_review: dict[str, Any] | None = None,
) -> ReflectionResult:
    """Reflect on a verified run; AVOID-lesson on anything not verified."""
    _ingest_harness_selections(repo_root=repo_root, workspace_id=workspace_id)
    _derive_user_insights(
        repo_root=repo_root,
        workspace_id=workspace_id,
        task_id=task_id,
        critic_review=critic_review,
    )
    if not (verification_report and verification_report.get("passed")):
        _record_avoid_lesson(
            repo_root=repo_root,
            workspace_id=workspace_id,
            task_id=task_id,
            verification_report=verification_report,
            critic_review=critic_review,
            changes_made=changes_made or [],
        )
        return ReflectionResult.skipped("verification_not_passed")
    if int(tool_call_count) < 5:
        return ReflectionResult.skipped("not_complex_enough")

    payload = _reflect_with_llm(
        workspace_id=workspace_id,
        task_id=task_id,
        verification_report=verification_report,
        tool_call_count=tool_call_count,
        final_message=final_message,
        changes_made=changes_made or [],
    )

    store = get_workspace_store(repo_root, workspace_id)
    lesson_data = payload.get("lesson") if isinstance(payload, dict) else {}
    lesson = record_workspace_lesson(
        store=store,
        task_id=task_id,
        workspace_id=workspace_id,
        change_summary=str(lesson_data.get("change_summary") or "Iteration changes"),
        expected_effect=str(lesson_data.get("expected_effect") or "Improve task quality"),
        observed_effect=str(lesson_data.get("observed_effect") or "Verified run completed"),
        conclusion=str(lesson_data.get("conclusion") or "No concise conclusion provided"),
        evidence_summary=f"tool_calls={tool_call_count}; changed={len(changes_made or [])}",
        repeat_tags=[str(tag) for tag in lesson_data.get("repeat_tags", [])][:8]
        if isinstance(lesson_data.get("repeat_tags"), list)
        else [],
        avoid_tags=[str(tag) for tag in lesson_data.get("avoid_tags", [])][:8]
        if isinstance(lesson_data.get("avoid_tags"), list)
        else [],
        priority=7,
        metadata={
            "source": "reflection_phase",
            "verified_at": time.time(),
            "critic_verdict": (critic_review or {}).get("verdict") if critic_review else None,
            "evidence_sha": _evidence_sha(verification_report, changes_made or []),
        },
    )
    _retain_reflection_lesson_block(
        repo_root=repo_root,
        workspace_id=workspace_id,
        task_id=task_id,
        title=f"verified reflection {task_id}",
        content=(
            f"{lesson.change_summary} | expected:{lesson.expected_effect}"
            f" | observed:{lesson.observed_effect}"
        ),
        verified=True,
        metadata={"lesson_id": lesson.id, "phase": "verify"},
        verification_report=verification_report,
        changes_made=changes_made or [],
    )

    skill_slug = ""
    candidate_skill = payload.get("candidate_skill") if isinstance(payload, dict) else None
    if isinstance(candidate_skill, dict):
        skill_slug = _write_candidate_skill(
            repo_root=repo_root, skill_payload=candidate_skill, run_task_id=task_id
        )

    signal_id = ""
    gap_signal = payload.get("gap_signal") if isinstance(payload, dict) else None
    if isinstance(gap_signal, dict):
        signal = record_competency_signal(
            store=store,
            category=SignalCategory.MISSING_CAPABILITY,
            capability_area=str(gap_signal.get("capability_area") or "unknown_capability"),
            strength=float(gap_signal.get("strength") or -0.4),
            evidence_summary=str(
                gap_signal.get("evidence_summary") or "Reflection phase detected a capability gap"
            ),
            task_id=task_id,
            workspace_id=workspace_id,
            metadata={"source": "reflection_phase"},
        )
        signal_id = signal.id

    try:
        from lethyx.memory.palace.write import canonical_palace_add

        canonical_palace_add(
            repo_root,
            workspace_id=workspace_id,
            content=lesson.conclusion,
            title=f"reflection {task_id}",
            kind="insight",
            store="palace.idea",
            tags=["reflection", "lesson"],
            phase="reflection",
            source_path=f"reflection/{task_id}",
            extra={"room": "reflection", "skill_slug": skill_slug},
        )
    except Exception:
        log.debug("Reflection phase palace write failed", exc_info=True)

    return ReflectionResult(
        status="recorded",
        lesson_id=lesson.id,
        skill_slug=skill_slug,
        signal_id=signal_id,
    )


def _ingest_harness_selections(*, repo_root: Path, workspace_id: str) -> None:
    """Fold harness selection rows into stats/lessons (idempotent sidecar)."""
    try:
        from lethyx.memory.harness_stats import ingest_run_harness_selections
        from lethyx.memory.paths import workspace_memory_root

        drive_root = workspace_memory_root(repo_root, workspace_id) / "drive"
        ingest_run_harness_selections(repo_root, workspace_id, drive_root)
    except Exception:
        log.debug("Harness selection ingest failed in reflection", exc_info=True)


def _derive_user_insights(
    *,
    repo_root: Path,
    workspace_id: str,
    task_id: str,
    critic_review: dict[str, Any] | None,
) -> None:
    """Distill at most 1-2 user-level insights per run into the user profile.

    Typed heuristics only: an explicit ``human_feedback`` field on the critic
    review, plus repeat_tags recurring across this workspace's lessons.
    """
    try:
        from lethyx.memory.user_profile import record_user_insight

        recorded = 0
        run_id = task_id.split(":", 1)[0]
        feedback = ""
        if isinstance(critic_review, dict):
            feedback = str(critic_review.get("human_feedback") or "").strip()
        if feedback:
            record_user_insight(
                repo_root,
                content=f"Human feedback pattern: {feedback[:500]}",
                source="reflection.human_feedback",
                workspace_id=workspace_id,
                run_id=run_id,
                tags=("human_feedback",),
            )
            recorded += 1
        repeated_tag = _most_repeated_lesson_tag(repo_root, workspace_id)
        if repeated_tag and recorded < 2:
            record_user_insight(
                repo_root,
                content=(
                    f"Recurring success pattern in workspace {workspace_id}: '{repeated_tag}'"
                ),
                source="reflection.repeat_tags",
                workspace_id=workspace_id,
                run_id=run_id,
                tags=("repeat_pattern",),
            )
    except Exception:
        log.debug("User insight derivation failed in reflection", exc_info=True)


def _most_repeated_lesson_tag(repo_root: Path, workspace_id: str, *, min_count: int = 3) -> str:
    from lethyx.memory.models import MemoryQuery

    store = get_workspace_store(repo_root, workspace_id)
    lessons = store.query_lessons(MemoryQuery(workspace_id=workspace_id, limit=200))
    counts: dict[str, int] = {}
    for lesson in lessons:
        for tag in getattr(lesson, "repeat_tags", None) or []:
            text = str(tag).strip()
            if text:
                counts[text] = counts.get(text, 0) + 1
    if not counts:
        return ""
    tag, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return tag if count >= min_count else ""


def _record_avoid_lesson(
    *,
    repo_root: Path,
    workspace_id: str,
    task_id: str,
    verification_report: dict[str, Any] | None,
    critic_review: dict[str, Any] | None,
    changes_made: list[str],
) -> None:
    try:
        store = get_workspace_store(repo_root, workspace_id)
        summary = (
            (verification_report or {}).get("summary")
            or (critic_review or {}).get("rationale")
            or "Run did not pass verification+critic gates"
        )
        record_workspace_lesson(
            store=store,
            task_id=task_id,
            workspace_id=workspace_id,
            change_summary="Avoid treating unverified agent run as success",
            expected_effect="Prevent false positive memory from contaminating future runs",
            observed_effect=str(summary)[:1000],
            conclusion="AVOID reusing this run as a positive lesson; require runtime verification and critic pass.",
            evidence_summary=f"changed={len(changes_made)}; critic={(critic_review or {}).get('verdict')}",
            repeat_tags=[],
            avoid_tags=["unverified_run", "false_positive_success"],
            priority=3,
            tags={"avoid", "unverified_lesson"},
            metadata={
                "source": "reflection_phase",
                "stale": True,
                "critic_verdict": (critic_review or {}).get("verdict"),
                "verification_passed": bool(
                    verification_report and verification_report.get("passed")
                ),
                "evidence_sha": _evidence_sha(verification_report, changes_made),
            },
        )
        _retain_reflection_lesson_block(
            repo_root=repo_root,
            workspace_id=workspace_id,
            task_id=task_id,
            title=f"avoid reflection {task_id}",
            content=f"AVOID unverified run | {str(summary)[:400]}",
            verified=False,
            metadata={"phase": "verify", "avoid": True},
            verification_report=verification_report,
            changes_made=changes_made,
        )
    except Exception:
        log.debug("Failed to record AVOID lesson for unverified run", exc_info=True)


def _retain_reflection_lesson_block(
    *,
    repo_root: Path,
    workspace_id: str,
    task_id: str,
    title: str,
    content: str,
    verified: bool,
    metadata: dict[str, Any],
    verification_report: dict[str, Any] | None,
    changes_made: list[str],
) -> None:
    try:
        from lethyx.contracts import EvidenceRef
        from lethyx.memory.backends.contracts import MemoryBlockRecord
        from lethyx.memory.backends.factory import create_durable_backend

        evidence_hash = _evidence_sha(verification_report, changes_made)
        evidence_refs = ()
        if verified:
            evidence_refs = (
                EvidenceRef(
                    ref_type="artifact",
                    ref_id=evidence_hash,
                    produced_by="verifier",
                    phase="verify",
                ),
            )
        backend = create_durable_backend(repo_root, workspace_id=workspace_id)
        backend.retain_block(
            MemoryBlockRecord(
                block_id=f"reflection-{task_id}-{evidence_hash[:12]}",
                workspace_id=workspace_id,
                run_id=task_id.split(":", 1)[0],
                phase_id="verify",
                title=title,
                content=content,
                kind="lesson" if verified else "lesson_candidate",
                lifecycle="active",
                trust_level="workspace_verified" if verified else "agent_claim",
                scope="cross_run_durable" if verified else "run_scoped",
                tier="warm",
                surface="supplemental_evidence",
                tags=(
                    ("lesson", "verified", "reflection")
                    if verified
                    else ("lesson", "avoid", "unverified_lesson", "reflection")
                ),
                evidence_refs=evidence_refs,
                verified=verified,
                metadata={
                    **metadata,
                    "room": "lesson",
                    "task_id": task_id,
                    "evidence_sha": evidence_hash,
                    "palace_store": "palace.lesson",
                },
            )
        )
    except Exception:
        log.debug("Reflection selected-backend lesson retain failed", exc_info=True)


def _evidence_sha(verification_report: dict[str, Any] | None, changes_made: list[str]) -> str:
    blob = json.dumps(
        {"verification": verification_report or {}, "changes": changes_made},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()


def _reflect_with_llm(
    *,
    workspace_id: str,
    task_id: str,
    verification_report: dict[str, Any] | None,
    tool_call_count: int,
    final_message: str,
    changes_made: list[str],
) -> dict[str, Any]:
    fallback = {
        "lesson": {
            "change_summary": "Workspace iteration update",
            "expected_effect": "Higher reliability and quality",
            "observed_effect": "Verified status achieved",
            "conclusion": "Verified changes should be reused for similar tasks.",
            "repeat_tags": ["verified_workflow"],
            "avoid_tags": [],
        },
        "candidate_skill": None,
        "gap_signal": None,
    }
    try:
        from lethyx.control_plane.code_analyzer import get_llm_client
    except Exception:
        return fallback
    client = get_llm_client()
    if client is None:
        return fallback
    user_payload = {
        "workspace_id": workspace_id,
        "task_id": task_id,
        "tool_call_count": tool_call_count,
        "changes_made": changes_made[:30],
        "verification_summary": (verification_report or {}).get("summary", ""),
        "final_message": (final_message or "")[:2200],
    }
    try:
        response, _meta = client.chat(
            [
                {"role": "system", "content": _REFLECTION_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ]
        )
    except Exception:
        return fallback
    text = _extract_text(response if isinstance(response, dict) else {})
    parsed = _parse_first_json(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("lesson"), dict):
        return parsed
    return fallback


def _extract_text(response: dict[str, Any]) -> str:
    content = response.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks)
    return ""


def _parse_first_json(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    for candidate in re.findall(r"\{.*\}", text, flags=re.DOTALL):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _write_candidate_skill(repo_root: Path, skill_payload: dict[str, Any], run_task_id: str) -> str:
    name = str(skill_payload.get("name") or "reflected-skill").strip()
    slug = _slugify(name)
    domains = skill_payload.get("domains") if isinstance(skill_payload.get("domains"), list) else []
    when_to_use = str(skill_payload.get("when_to_use") or "Use when this pattern appears.").strip()
    params = skill_payload.get("params") if isinstance(skill_payload.get("params"), list) else []
    steps = skill_payload.get("steps") if isinstance(skill_payload.get("steps"), list) else []
    root = repo_root / "Lethyx" / "skills" / "library" / slug
    root.mkdir(parents=True, exist_ok=True)
    skill_md = root / "SKILL.md"
    step_lines = [
        f"{idx + 1}. {str(step).strip()}"
        for idx, step in enumerate(steps[:12])
        if str(step).strip()
    ]
    if not step_lines:
        step_lines = ["1. Re-run the verified sequence from the reflection outcome."]
    text = (
        "---\n"
        f"name: {name}\n"
        "status: candidate\n"
        f"domains: {json.dumps([str(d).strip().lower() for d in domains if str(d).strip()])}\n"
        f"when_to_use: {json.dumps(when_to_use)}\n"
        f"params: {json.dumps(params, ensure_ascii=False)}\n"
        "created_by: reflection\n"
        f"source_run_id: {run_task_id}\n"
        "---\n\n"
        "## Steps\n" + "\n".join(step_lines) + "\n"
    )
    skill_md.write_text(text, encoding="utf-8")
    return slug


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return cleaned or "reflected-skill"
