"""LoCoMo benchmark loader.

Format: locomo10.json — 10 long-term conversations, each with sessions and
annotated QA pairs.

JSON structure (per conversation):
  {
    "conversation_id": "...",
    "speaker_a": "...", "speaker_b": "...",
    "session_<N>": {
      "session_date_time": "...",
      "dialog": [{"speaker": "...", "dialog_id": "...", "text": "..."}, ...]
    },
    "qa": [
      {
        "question": "...", "answer": "...",
        "category": "single_hop|multi_hop|temporal|open_domain|adversarial",
        "evidence": ["dialog_id", ...]
      }
    ]
  }

Categories:
  1 = single_hop, 2 = multi_hop, 3 = temporal, 4 = open_domain, 5 = adversarial

Reference: Maharana et al., ACL 2024.  Upstream repo: snap-research/locomo.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Iterator

from memory_bench.core.datasets import resolve_file, DatasetNotFoundError
from memory_bench.core.registry import register_benchmark
from memory_bench.core.types import (
    Episode,
    MemoryRecord,
    Probe,
    RunSpec,
    Session,
)

log = logging.getLogger(__name__)

_CATEGORY_MAP: dict[str | int, str] = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
    "1": "single_hop",
    "2": "multi_hop",
    "3": "temporal",
    "4": "open_domain",
    "5": "adversarial",
    "single_hop": "single_hop",
    "multi_hop": "multi_hop",
    "temporal": "temporal",
    "open_domain": "open_domain",
    "adversarial": "adversarial",
}


def _load_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Some versions wrap in a dict keyed by conversation id
        return list(data.values())
    raise ValueError(f"Unexpected locomo10.json top-level type: {type(data)}")


def _parse_conversation(conv: dict[str, Any], max_probes: int) -> tuple[Episode, list[Probe]]:
    conv_payload = conv.get("conversation") if isinstance(conv.get("conversation"), dict) else conv
    conv_id = str(
        conv.get("conversation_id")
        or conv.get("sample_id")
        or conv.get("id")
        or uuid.uuid4().hex[:8]
    )

    # Collect sessions in chronological order.  Upstream LoCoMo has appeared in
    # two shapes: older fixtures store session_N as {session_date_time, dialog},
    # while current locomo10.json nests session_N lists under "conversation" and
    # stores timestamps as sibling session_N_date_time keys.
    sessions: list[Session] = []
    session_keys = sorted(
        (
            k
            for k, value in conv_payload.items()
            if k.startswith("session_")
            and not k.endswith("_date_time")
            and k != "session_date"
            and (isinstance(value, dict) or isinstance(value, list))
        ),
        key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0,
    )

    for sk in session_keys:
        sess_data = conv_payload[sk]
        if isinstance(sess_data, dict):
            sess_ts = str(
                sess_data.get("session_date_time") or conv_payload.get(f"{sk}_date_time") or ""
            )
            dialog = sess_data.get("dialog") or []
        elif isinstance(sess_data, list):
            sess_ts = str(conv_payload.get(f"{sk}_date_time") or "")
            dialog = sess_data
        else:
            continue
        records: list[MemoryRecord] = []
        for turn in dialog:
            dialog_id = str(turn.get("dialog_id") or turn.get("dia_id") or turn.get("id") or "")
            text = str(turn.get("text") or turn.get("content") or "")
            speaker = str(turn.get("speaker") or "")
            if not text:
                continue
            records.append(
                MemoryRecord(
                    id=f"{conv_id}_{dialog_id}"
                    if dialog_id
                    else f"{conv_id}_{uuid.uuid4().hex[:6]}",
                    content=text,
                    ts=sess_ts,
                    role=speaker.lower(),
                    session_id=sk,
                    kind="message",
                    tags=(sk, speaker.lower()) if speaker else (sk,),
                    evidence_id=dialog_id,
                )
            )
        sessions.append(Session(id=sk, ts=sess_ts, records=tuple(records)))

    episode = Episode(
        id=conv_id,
        sessions=tuple(sessions),
        metadata={
            "speaker_a": conv_payload.get("speaker_a", ""),
            "speaker_b": conv_payload.get("speaker_b", ""),
        },
    )

    # Parse QA probes
    qa_raw: list[dict[str, Any]] = conv.get("qa") or []
    probes: list[Probe] = []
    for i, qa in enumerate(qa_raw):
        if max_probes > 0 and len(probes) >= max_probes:
            break
        question = str(qa.get("question") or "")
        answer = str(qa.get("answer") or "")
        category_raw = qa.get("category")
        category = _CATEGORY_MAP.get(
            str(category_raw) if category_raw is not None else "", "general"
        )
        evidence = [str(e) for e in (qa.get("evidence") or []) if e]
        # adversarial = unanswerable
        abstain = category == "adversarial"

        probe_id = f"{conv_id}_q{i:04d}"
        probes.append(
            Probe(
                id=probe_id,
                question=question,
                gold=(answer,) if answer else (),
                category=category,
                evidence_ids=tuple(evidence),
                abstain=abstain,
                kind="qa",
            )
        )

    return episode, probes


@register_benchmark("locomo")
class LoCoMoBenchmark:
    """LoCoMo long-term conversational memory benchmark.

    Default judge: f1 (token-level F1 + optional LLM judge via suite.toml).
    5 question categories: single_hop, multi_hop, temporal, open_domain, adversarial.
    """

    name = "locomo"
    kind = "recall"
    default_judge = "f1"

    def load(self, spec: RunSpec) -> Iterator[tuple[Episode, list[Probe]]]:
        try:
            data_path = resolve_file("locomo", "locomo10.json", strict=True)
        except DatasetNotFoundError as exc:
            log.error("LoCoMo dataset not found: %s", exc)
            return

        assert data_path is not None
        conversations = _load_conversations(data_path)
        max_ep = spec.dataset.max_episodes
        max_pr = spec.dataset.max_probes_per_episode

        for i, conv in enumerate(conversations):
            if max_ep > 0 and i >= max_ep:
                break
            try:
                episode, probes = _parse_conversation(conv, max_pr)
                if probes:
                    yield episode, probes
            except Exception as exc:
                log.warning("Failed to parse locomo conversation %d: %s", i, exc)
