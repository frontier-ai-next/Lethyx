"""Text payload renderers for Hindsight retain calls."""

import json
from typing import Any

from lethyx.memory.backends.base import DurableEvent, DurableLesson
from lethyx.memory.hindsight.mapping import evidence_to_lines, redact_sensitive, stable_hash


def _section(title: str, body: str) -> str:
    text = str(body or "").strip()
    if not text:
        return ""
    return f"{title}:\n{text}"


def build_lesson_payload(lesson: DurableLesson) -> str:
    evidence = "\n".join(evidence_to_lines(lesson.evidence_refs))
    parts = [
        f"Title: {lesson.title}",
        f"Kind: {lesson.kind}",
        f"Trust: {lesson.trust_level}",
        _section("Lesson", lesson.content),
        _section("Evidence", evidence),
        _section(
            "Lethyx metadata",
            "\n".join(
                f"- {key}: {value}"
                for key, value in {
                    "workspace_id": lesson.workspace_id,
                    "run_id": lesson.run_id,
                    "phase_id": lesson.phase_id,
                    "source_hash": lesson.metadata.get("source_hash")
                    or stable_hash(lesson.content),
                }.items()
                if value
            ),
        ),
    ]
    return redact_sensitive("\n\n".join(part for part in parts if part))


def build_event_payload(event: DurableEvent) -> str:
    evidence = "\n".join(evidence_to_lines(event.evidence_refs))
    parts = [
        f"Title: Lethyx durable event: {event.kind}",
        f"Kind: {event.kind}",
        f"Trust: {event.trust_level}",
        _section("Summary", event.content),
        _section("Evidence", evidence),
        _section(
            "Lethyx metadata",
            "\n".join(
                f"- {key}: {value}"
                for key, value in {
                    "workspace_id": event.workspace_id,
                    "run_id": event.run_id,
                    "phase_id": event.phase_id,
                    "subtask_id": event.subtask_id,
                    "event_id": event.event_id,
                    "source_hash": event.metadata.get("source_hash") or stable_hash(event.content),
                }.items()
                if value
            ),
        ),
    ]
    return redact_sensitive("\n\n".join(part for part in parts if part))


def render_bkb_rule_for_hindsight(rule: dict[str, Any], patch: Any) -> str:
    rule_body = rule.get("rule") if isinstance(rule.get("rule"), dict) else {}
    applies_to = rule.get("applies_to") if isinstance(rule.get("applies_to"), dict) else {}
    evidence = "\n".join(evidence_to_lines(list(getattr(patch, "source_evidence", []) or [])))
    payload = {
        "trigger": rule_body.get("trigger"),
        "behavior": rule_body.get("behavior"),
        "forbidden": rule_body.get("forbidden"),
        "applies_to": applies_to,
    }
    return redact_sensitive(
        "\n\n".join(
            [
                f"Accepted BKB Rule: {rule.get('title') or rule.get('id')}",
                f"Kind: {rule.get('type') or rule.get('rule_type') or 'behavior'}",
                f"Scope: {rule.get('scope') or getattr(patch, 'workspace_id', '') or 'manager'}",
                "Rule:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
                "Why accepted:\nLethyx accepted this rule through the BKB evidence gate.",
                "Evidence:\n" + evidence,
            ]
        )
    )


def document_id_for_lesson(lesson: DurableLesson) -> str:
    return f"Lethyx:lesson:{lesson.lesson_id}"


def document_id_for_event(event: DurableEvent) -> str:
    return f"Lethyx:event:{event.event_id}"
