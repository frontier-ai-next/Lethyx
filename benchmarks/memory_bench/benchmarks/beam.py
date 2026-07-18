"""BEAM benchmark loader.

Beyond a Million Tokens (ICLR 2026) — tests memory at 100K–10M token scales.
This is the primary benchmark where context-stuffing fails and a proper memory
architecture is required (at 1M+ tokens no context window fits the history).

Dataset structure (upstream: mohammadtavakoli78/BEAM):
  data/memory_bench/beam/conversations/<conversation_id>/
    conversation.json    — list of {turn_id, timestamp, speaker, text}
    questions.json       — list of {
        question_id, question, answer, category,
        evidence_turn_ids, tier (100k|500k|1m|10m)
      }

10 memory categories:
  abstention, contradiction_resolution, event_ordering, instruction_following,
  preference_tracking, information_extraction, knowledge_update,
  multi_hop_reasoning, summarization, temporal_reasoning

Scoring: nugget-based (BEAM paper §4).

Reference: arXiv 2510.27246, ICLR 2026. Repo: mohammadtavakoli78/BEAM.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Iterator

from memory_bench.core.datasets import resolve_path, DatasetNotFoundError
from memory_bench.core.registry import register_benchmark
from memory_bench.core.types import (
    Episode,
    MemoryRecord,
    Probe,
    RunSpec,
    Session,
)

log = logging.getLogger(__name__)

_ALL_TIERS = ("100k", "500k", "1m", "10m")

_ABSTAIN_CATEGORIES = frozenset({"abstention"})


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_as_str_list(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_as_str_list(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def _gold_and_metadata(q: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Map BEAM's heterogeneous answer/rubric fields to benchmark gold text.

    Upstream BEAM uses different keys per category.  Several categories have no
    plain ``answer`` field; treating those as empty gold silently awards perfect
    nugget scores to trivial no-memory answers.  Prefer the explicit reference
    answer/summary when present, then compliance text plus rubric bullets.
    """
    rubric = _as_str_list(q.get("rubric"))
    direct = _as_str_list(
        q.get("answer")
        or q.get("ideal_answer")
        or q.get("ideal_response")
        or q.get("ideal_summary")
    )
    if direct:
        gold = tuple(direct)
    else:
        compliance = _as_str_list(q.get("expected_compliance"))
        gold_parts: list[str] = []
        seen: set[str] = set()
        for part in compliance + rubric:
            cleaned = part.strip().rstrip(".")
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                gold_parts.append(cleaned)
        gold = (". ".join(gold_parts),) if gold_parts else ()
    metadata: dict[str, Any] = {}
    if rubric:
        metadata["rubric"] = rubric
    if q.get("expected_compliance"):
        metadata["expected_compliance"] = str(q.get("expected_compliance"))
    return gold, metadata


def _load_conversation(conv_dir: Path) -> Episode:
    conv_file = conv_dir / "conversation.json"
    data: list[dict[str, Any]] = json.loads(conv_file.read_text(encoding="utf-8"))

    conv_id = conv_dir.name
    # Group turns into pseudo-sessions of 50 turns each for manageable isolation
    chunk_size = 50
    sessions: list[Any] = []
    for chunk_idx in range(0, len(data), chunk_size):
        chunk = data[chunk_idx : chunk_idx + chunk_size]
        sess_id = f"s{chunk_idx // chunk_size:04d}"
        ts = str(chunk[0].get("timestamp") or "") if chunk else ""
        records: list[MemoryRecord] = []
        for turn in chunk:
            turn_id = str(turn.get("turn_id") or turn.get("id") or uuid.uuid4().hex[:6])
            text = str(turn.get("text") or turn.get("content") or "")
            speaker = str(turn.get("speaker") or "")
            ts_turn = str(turn.get("timestamp") or ts)
            if not text:
                continue
            records.append(
                MemoryRecord(
                    id=f"{conv_id}_{turn_id}",
                    content=text,
                    ts=ts_turn,
                    role=speaker.lower(),
                    session_id=sess_id,
                    kind="message",
                    tags=(sess_id, speaker.lower()) if speaker else (sess_id,),
                    evidence_id=turn_id,
                )
            )
        from memory_bench.core.types import Session as Sess

        sessions.append(Sess(id=sess_id, ts=ts, records=tuple(records)))

    return Episode(id=conv_id, sessions=tuple(sessions))


def _load_questions(
    conv_dir: Path,
    conv_id: str,
    allowed_tiers: set[str],
    max_probes: int,
) -> list[Probe]:
    q_file = conv_dir / "questions.json"
    data: list[dict[str, Any]] = json.loads(q_file.read_text(encoding="utf-8"))
    probes: list[Probe] = []
    for q in data:
        if max_probes > 0 and len(probes) >= max_probes:
            break
        tier = _normalise_tier(str(q.get("tier") or "100k"))
        if allowed_tiers and tier not in allowed_tiers:
            continue
        question = str(q.get("question") or "")
        gold, metadata = _gold_and_metadata(q)
        category = str(q.get("category") or "general").lower().replace(" ", "_")
        evidence_ids = [str(e) for e in (q.get("evidence_turn_ids") or [])]
        abstain = category in _ABSTAIN_CATEGORIES
        qid = str(q.get("question_id") or f"{conv_id}_{uuid.uuid4().hex[:6]}")
        probes.append(
            Probe(
                id=qid,
                question=question,
                gold=gold,
                category=category,
                evidence_ids=tuple(evidence_ids),
                abstain=abstain,
                tier=tier,
                kind="qa",
                metadata=metadata,
            )
        )
    return probes


def _normalise_tier(raw: str) -> str:
    return str(raw or "").lower()


def _iter_upstream_turn_batches(value: Any) -> Iterator[dict[str, Any]]:
    """Yield BEAM upstream batch dicts from flat or nested plan-* chat JSON."""
    if isinstance(value, dict):
        turns = value.get("turns")
        if isinstance(turns, list):
            yield value
            return
        for child in value.values():
            yield from _iter_upstream_turn_batches(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_upstream_turn_batches(child)


def _flatten_source_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_source_ids(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_source_ids(item))
        return out
    return [str(value)]


def _load_upstream_chat_dir(chat_dir: Path, chats_root: Path) -> tuple[Episode, list[Probe]]:
    """Parse the current upstream BEAM layout under chats/<tier>/<id>/..."""
    rel_parts = chat_dir.relative_to(chats_root).parts
    tier_raw = rel_parts[0] if rel_parts else ""
    tier = _normalise_tier(tier_raw)
    conv_id = "_".join(rel_parts) if rel_parts else chat_dir.name

    raw_chat = json.loads((chat_dir / "chat.json").read_text(encoding="utf-8"))
    records: list[MemoryRecord] = []
    for batch_i, batch in enumerate(_iter_upstream_turn_batches(raw_chat)):
        turns_groups = batch.get("turns")
        if not isinstance(turns_groups, list):
            continue
        for group_i, group in enumerate(turns_groups):
            messages = group if isinstance(group, list) else [group]
            for msg_i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue
                content = str(msg.get("content") or msg.get("text") or "")
                if not content:
                    continue
                turn_id = str(msg.get("id") or msg.get("turn_id") or f"{batch_i}_{group_i}_{msg_i}")
                role = str(msg.get("role") or msg.get("speaker") or "").lower()
                ts = str(msg.get("time_anchor") or msg.get("timestamp") or "")
                records.append(
                    MemoryRecord(
                        id=f"{conv_id}_{turn_id}",
                        content=content,
                        ts=ts,
                        role=role,
                        session_id=f"batch_{batch_i}",
                        kind="message",
                        tags=(tier, f"batch_{batch_i}", role)
                        if role
                        else (tier, f"batch_{batch_i}"),
                        evidence_id=turn_id,
                    )
                )

    episode = Episode(
        id=conv_id,
        sessions=(Session(id="s0", ts="", records=tuple(records)),),
        metadata={"tier": tier},
    )

    q_file = chat_dir / "probing_questions" / "probing_questions.json"
    questions_raw = json.loads(q_file.read_text(encoding="utf-8"))
    probes: list[Probe] = []
    if isinstance(questions_raw, dict):
        iterable = [
            (str(category).lower().replace(" ", "_"), item)
            for category, items in questions_raw.items()
            for item in (items if isinstance(items, list) else [items])
        ]
    elif isinstance(questions_raw, list):
        iterable = [
            (str(item.get("category") or "general").lower().replace(" ", "_"), item)
            for item in questions_raw
            if isinstance(item, dict)
        ]
    else:
        iterable = []

    for i, (category, q) in enumerate(iterable):
        if not isinstance(q, dict):
            continue
        question = str(q.get("question") or "")
        gold, metadata = _gold_and_metadata(q)
        evidence_ids = _flatten_source_ids(
            q.get("evidence_turn_ids")
            or q.get("source_chat_ids")
            or q.get("conversation_references")
        )
        qid = str(q.get("question_id") or q.get("id") or f"{conv_id}_{category}_{i}")
        probes.append(
            Probe(
                id=qid,
                question=question,
                gold=gold,
                category=category,
                evidence_ids=tuple(evidence_ids),
                abstain=category in _ABSTAIN_CATEGORIES,
                tier=tier,
                kind="qa",
                metadata=metadata,
            )
        )
    return episode, probes


@register_benchmark("beam")
class BEAMBenchmark:
    """BEAM Beyond-a-Million-Tokens benchmark.

    Default judge: nugget (atomic nugget coverage, as in the BEAM paper).
    10 memory categories evaluated across 4 context-length tiers.

    Note on 10M tier: evaluating at 10M tokens requires very large model context
    windows or a real memory architecture. This tier is included for completeness
    but will be slow and expensive. Start with 100k for smoke tests.
    """

    name = "beam"
    kind = "recall"
    default_judge = "nugget"

    def load(self, spec: RunSpec) -> Iterator[tuple[Episode, list[Probe]]]:
        try:
            base = resolve_path("beam", strict=True)
        except DatasetNotFoundError as exc:
            log.error("BEAM dataset not found: %s", exc)
            return

        assert base is not None
        chats_dir = base / "chats"
        if chats_dir.is_dir():
            allowed_tiers: set[str] = (
                set(spec.dataset.tiers) if spec.dataset.tiers else set(_ALL_TIERS)
            )
            max_ep = spec.dataset.max_episodes
            max_pr = spec.dataset.max_probes_per_episode
            n = 0
            for q_file in sorted(chats_dir.rglob("probing_questions.json")):
                chat_dir = q_file.parent.parent
                rel_parts = chat_dir.relative_to(chats_dir).parts
                tier = _normalise_tier(rel_parts[0] if rel_parts else "")
                if allowed_tiers and tier not in allowed_tiers:
                    continue
                if max_ep > 0 and n >= max_ep:
                    break
                if not (chat_dir / "chat.json").is_file():
                    continue
                try:
                    episode, probes = _load_upstream_chat_dir(chat_dir, chats_dir)
                    if max_pr > 0:
                        probes = probes[:max_pr]
                    if probes:
                        yield episode, probes
                        n += 1
                except Exception as exc:
                    log.warning("BEAM: failed to load upstream chat %s: %s", chat_dir, exc)
            return

        conversations_dir = base / "conversations"
        if not conversations_dir.is_dir():
            # Try root directly
            conversations_dir = base

        # Filter tiers from spec
        allowed_tiers: set[str] = set(spec.dataset.tiers) if spec.dataset.tiers else set(_ALL_TIERS)
        max_ep = spec.dataset.max_episodes
        max_pr = spec.dataset.max_probes_per_episode

        conv_dirs = sorted(conversations_dir.iterdir()) if conversations_dir.is_dir() else []
        n = 0
        for conv_dir in conv_dirs:
            if not conv_dir.is_dir():
                continue
            if max_ep > 0 and n >= max_ep:
                break
            conv_file = conv_dir / "conversation.json"
            q_file = conv_dir / "questions.json"
            if not conv_file.is_file() or not q_file.is_file():
                log.debug("Skipping %s: missing conversation.json or questions.json", conv_dir)
                continue
            try:
                episode = _load_conversation(conv_dir)
                probes = _load_questions(conv_dir, episode.id, allowed_tiers, max_pr)
                if probes:
                    yield episode, probes
                    n += 1
            except Exception as exc:
                log.warning("BEAM: failed to load conversation %s: %s", conv_dir.name, exc)
