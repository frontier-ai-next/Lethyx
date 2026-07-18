"""LongMemEval benchmark loader.

Format: longmemeval_s.json (or longmemeval_m.json for the large variant).

JSON structure (one list of question objects):
  [
    {
      "question_id": "...",
      "question": "...",
      "answer": "...",
      "question_type": "single-session-user|single-session-assistant|
                        single-session-preference|multi-session|
                        knowledge-update|temporal-reasoning|false-premise",
      "evidence_session_ids": [...],   # sessions containing the answer
      "sessions": [
        {
          "session_id": "...",
          "timestamp": "...",
          "messages": [{"role": "user|assistant", "content": "..."}, ...]
        }
      ]
    }
  ]

5 memory abilities tested via 7 question types:
  - Information extraction: single-session-user, single-session-assistant
  - Preference: single-session-preference
  - Multi-session reasoning: multi-session
  - Knowledge updates: knowledge-update
  - Temporal reasoning: temporal-reasoning
  - Abstention: false-premise

Default judge: llm_judge (GPT-4o, as in the original paper).

Reference: Wu et al., ICLR 2025. Upstream: xiaowu0162/long-mem-eval.
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

_ABSTAIN_TYPES = frozenset({"false-premise", "false_premise", "abstention", "abs"})


def _parse_question(item: dict[str, Any], max_probes: int) -> tuple[Episode, list[Probe]]:
    qid = str(item.get("question_id") or uuid.uuid4().hex[:8])
    question = str(item.get("question") or "")
    answer = str(item.get("answer") or "")
    qtype = str(item.get("question_type") or item.get("type") or "general")
    evidence_session_ids = [
        str(s) for s in (item.get("evidence_session_ids") or item.get("answer_session_ids") or [])
    ]
    abstain = qtype in _ABSTAIN_TYPES

    # Build episode from sessions.  The cleaned LongMemEval release stores the
    # history as parallel haystack_session_ids / haystack_dates /
    # haystack_sessions arrays instead of a list of session dicts.
    sessions_raw: list[dict[str, Any]] = item.get("sessions") or []
    if not sessions_raw and item.get("haystack_sessions"):
        ids = list(item.get("haystack_session_ids") or [])
        dates = list(item.get("haystack_dates") or [])
        sessions_raw = [
            {
                "session_id": str(ids[i]) if i < len(ids) else uuid.uuid4().hex[:6],
                "timestamp": str(dates[i]) if i < len(dates) else "",
                "messages": messages,
            }
            for i, messages in enumerate(item.get("haystack_sessions") or [])
        ]
    sessions: list[Session] = []
    for sess_index, sess in enumerate(sessions_raw):
        sess_id = str(sess.get("session_id") or sess.get("id") or uuid.uuid4().hex[:6])
        ts = str(sess.get("timestamp") or sess.get("date") or "")
        messages = sess.get("messages") or []
        records: list[MemoryRecord] = []
        for j, msg in enumerate(messages):
            role = str(msg.get("role") or "user")
            content = str(msg.get("content") or "")
            if not content:
                continue
            records.append(
                MemoryRecord(
                    id=f"{qid}_s{sess_index}_{sess_id}_{j}",
                    content=content,
                    ts=ts,
                    role=role,
                    session_id=sess_id,
                    kind="message",
                    tags=(sess_id, role),
                    # For retrieval accuracy: use session_id as evidence_id
                    evidence_id=sess_id,
                )
            )
        sessions.append(Session(id=sess_id, ts=ts, records=tuple(records)))

    episode = Episode(id=qid, sessions=tuple(sessions))

    probe = Probe(
        id=f"{qid}_p0",
        question=question,
        gold=(answer,) if answer else (),
        category=qtype,
        evidence_ids=tuple(evidence_session_ids),
        abstain=abstain,
        kind="qa",
    )
    return episode, [probe]


@register_benchmark("longmemeval")
class LongMemEvalBenchmark:
    """LongMemEval benchmark (S and M variants).

    Default judge: llm_judge (GPT-4o as judge, matches original paper).
    7 question types mapped to 5 memory abilities.
    Abstention (false-premise) questions tested separately.
    """

    name = "longmemeval"
    kind = "recall"
    default_judge = "llm_judge"

    def load(self, spec: RunSpec) -> Iterator[tuple[Episode, list[Probe]]]:
        try:
            base = resolve_path("longmemeval", strict=True)
        except DatasetNotFoundError as exc:
            log.error("LongMemEval dataset not found: %s", exc)
            return

        assert base is not None
        # Try S first, then M
        candidates = ["longmemeval_s.json", "longmemeval_m.json", "data.json"]
        data_path: Path | None = None
        for name in candidates:
            p = base / name
            if p.is_file():
                data_path = p
                break

        if data_path is None:
            log.error(
                "No LongMemEval JSON file found in %s (tried: %s)",
                base,
                ", ".join(candidates),
            )
            return

        raw = json.loads(data_path.read_text(encoding="utf-8"))
        items: list[dict[str, Any]] = raw if isinstance(raw, list) else list(raw.values())

        max_ep = spec.dataset.max_episodes
        max_pr = spec.dataset.max_probes_per_episode

        for i, item in enumerate(items):
            if max_ep > 0 and i >= max_ep:
                break
            try:
                episode, probes = _parse_question(item, max_pr)
                if probes:
                    yield episode, probes
            except Exception as exc:
                log.warning("Failed to parse LongMemEval item %d: %s", i, exc)
