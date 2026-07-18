"""EvoMemBench benchmark loader.

EvoMemBench tests self-evolving memory from a self-evolving perspective:
  - In-episode evolution: memory is updated within a single episode.
  - Cross-episode evolution: memory is carried across multiple episodes.
  - Knowledge-oriented: tests factual knowledge retention and update.
  - Execution-oriented: tests procedure/skill memory (how to do things).

This creates a 2×2 matrix of evaluation modes:
  in_episode × knowledge
  in_episode × execution
  cross_episode × knowledge
  cross_episode × execution

Dataset structure (upstream: arXiv 2605.18421):
  data/memory_bench/evomembench/<mode>/
    episodes.json  — list of episode dicts
    probes.json    — list of probe dicts

Probes specifically test:
  - update: model should reflect updated information (requires supports_update=True)
  - reuse: model should recall past experience for a new task

NOTE: The public repository currently advertises the dataset as "coming soon".
When absent, this benchmark skips rows rather than emitting fake scored stubs.

Reference: arXiv 2605.18421 (2025).
"""

import hashlib
import json
import logging
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

_MODES = (
    "in_episode_knowledge",
    "in_episode_execution",
    "cross_episode_knowledge",
    "cross_episode_execution",
)


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _metadata_without_none(**items: Any) -> dict[str, Any]:
    return {key: value for key, value in items.items() if value is not None}


def _parse_episode(ep_raw: dict[str, Any], mode: str) -> Episode:
    ep_id = str(ep_raw.get("episode_id") or ep_raw.get("id") or _stable_id("episode", mode, ep_raw))
    sessions_raw = ep_raw.get("sessions") or ep_raw.get("turns") or []
    sessions: list[Session] = []

    if isinstance(sessions_raw, list) and sessions_raw and isinstance(sessions_raw[0], str):
        # Flat list of turn strings
        records = [
            MemoryRecord(
                id=f"{ep_id}_t{i}",
                content=str(t),
                session_id="s0",
                kind="message",
            )
            for i, t in enumerate(sessions_raw)
        ]
        sessions.append(Session(id="s0", ts="", records=tuple(records)))
    else:
        for j, sess in enumerate(sessions_raw):
            if isinstance(sess, str):
                sess_data: dict[str, Any] = {"session_id": f"s{j}", "messages": [{"content": sess}]}
            else:
                sess_data = sess
            sess_id = str(sess_data.get("session_id") or f"s{j}")
            ts = str(sess_data.get("timestamp") or "")
            messages = sess_data.get("messages") or []
            records = []
            for k, msg in enumerate(messages):
                content = str(msg.get("content") or msg.get("text") or "")
                role = str(msg.get("role") or "")
                if not content:
                    continue
                records.append(
                    MemoryRecord(
                        id=f"{ep_id}_{sess_id}_{k}",
                        content=content,
                        ts=ts,
                        role=role,
                        session_id=sess_id,
                        kind="message",
                    )
                )
            sessions.append(Session(id=sess_id, ts=ts, records=tuple(records)))

    return Episode(id=ep_id, sessions=tuple(sessions), metadata={"mode": mode})


def _parse_probe(p_raw: dict[str, Any], ep_id: str, mode: str) -> Probe:
    pid = str(p_raw.get("probe_id") or p_raw.get("id") or _stable_id("probe", mode, ep_id, p_raw))
    question = str(p_raw.get("question") or p_raw.get("instruction") or "")
    answer = str(p_raw.get("answer") or "")
    ptype = str(p_raw.get("type") or p_raw.get("probe_type") or "qa")
    is_action = "execution" in mode or ptype == "action"

    return Probe(
        id=pid,
        question=question,
        gold=(answer,) if answer else (),
        category=mode,
        kind="action" if is_action else "qa",
        metadata=_metadata_without_none(
            success_threshold=0.5 if is_action else None,
            probe_type=ptype,
        ),
    )


@register_benchmark("evomembench")
class EvoMemBenchBenchmark:
    """EvoMemBench self-evolving memory benchmark.

    Default judge: f1 for knowledge probes. Execution probes are dispatched
    through the agentic driver only when a real upstream environment is present;
    otherwise they are skipped rather than assigned fake task-success scores.
    4 evaluation modes: in-/cross-episode × knowledge/execution.

    NOTE: Dataset not yet publicly distributed.  Missing data yields no rows.
    """

    name = "evomembench"
    kind = "mixed"
    default_judge = "f1"

    def load(self, spec: RunSpec) -> Iterator[tuple[Episode, list[Probe]]]:
        try:
            base = resolve_path("evomembench", strict=True)
        except DatasetNotFoundError as exc:
            log.warning(
                "EvoMemBench dataset not found: %s\n"
                "Skipping EvoMemBench instead of emitting fake scored stub rows.",
                exc,
            )
            return

        assert base is not None
        max_ep = spec.dataset.max_episodes
        max_pr = spec.dataset.max_probes_per_episode
        n = 0

        for mode in _MODES:
            mode_dir = base / mode
            ep_file = mode_dir / "episodes.json"
            pr_file = mode_dir / "probes.json"

            if not mode_dir.is_dir() or not ep_file.is_file():
                log.warning("EvoMemBench: mode directory or episodes.json missing: %s", mode_dir)
                continue

            episodes_raw: list[dict[str, Any]] = json.loads(ep_file.read_text(encoding="utf-8"))
            probes_raw: list[dict[str, Any]] = (
                json.loads(pr_file.read_text(encoding="utf-8")) if pr_file.is_file() else []
            )
            # Index probes by episode_id
            probes_by_ep: dict[str, list[dict[str, Any]]] = {}
            for p in probes_raw:
                ep_ref = str(p.get("episode_id") or "")
                probes_by_ep.setdefault(ep_ref, []).append(p)

            for ep_raw in episodes_raw:
                if max_ep > 0 and n >= max_ep:
                    return
                try:
                    episode = _parse_episode(ep_raw, mode)
                    ep_probes_raw = probes_by_ep.get(episode.id, [])
                    probes = [_parse_probe(p, episode.id, mode) for p in ep_probes_raw]
                    if max_pr > 0:
                        probes = probes[:max_pr]
                    if not probes:
                        log.warning(
                            "EvoMemBench: no probes defined for episode %s in %s", episode.id, mode
                        )
                        continue
                    yield episode, probes
                    n += 1
                except Exception as exc:
                    log.warning("EvoMemBench: failed to parse episode: %s", exc)
