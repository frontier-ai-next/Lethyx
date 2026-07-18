"""MemoryArena benchmark loader.

MemoryArena tests agentic memory across 4 domains with multi-session,
interdependent tasks:
  - Web Shopping
  - Travel Planning
  - Progressive Search
  - Formal Reasoning

Tasks are structured so that information from earlier sessions is required
to complete later tasks — measuring if the agent built and can retrieve
the right memory.

Dataset structure (upstream: memoryarena.github.io):
  data/memory_bench/memoryarena/tasks/<domain>/<task_id>.json
  Each task JSON:
  {
    "task_id": "...", "domain": "...", "sessions": [...],
    "subtasks": [
      {"subtask_id": "...", "instruction": "...", "success_criteria": "..."}
    ]
  }

Current status: dataset may require access request (see plan notes).
This loader returns a graceful stub when the data directory is absent.

Reference: He et al., arXiv 2602.16313 (2026). Site: memoryarena.github.io.
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

_DOMAINS = ("web_shopping", "travel_planning", "progressive_search", "formal_reasoning")


def _stable_id(prefix: str, value: Any, *, length: int = 12) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


class _StubEnv:
    """Stub environment used when the real MemoryArena dataset is unavailable."""

    def reset(self) -> dict[str, Any]:
        return {"observation": "stub: dataset not available", "done": True}

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return {"observation": "done"}, 0.0, True, {}

    def score(self) -> float:
        return 0.0


def _parse_task(
    task_data: dict[str, Any],
    *,
    stub: bool = False,
    domain_hint: str = "",
    fallback_task_id: str = "",
) -> tuple[Episode, list[Probe]]:
    raw_task_id = str(
        task_data.get("task_id")
        or task_data.get("id")
        or fallback_task_id
        or _stable_id("task", task_data, length=8)
    )
    domain = str(task_data.get("domain") or domain_hint or "unknown")
    task_id = raw_task_id
    if domain and domain != "unknown" and not raw_task_id.startswith(f"{domain}_"):
        task_id = f"{domain}_{raw_task_id}"
    sessions_raw: list[dict[str, Any]] = task_data.get("sessions") or []

    sessions: list[Session] = []
    for i, sess in enumerate(sessions_raw):
        sess_id = str(
            sess.get("session_id")
            or _stable_id("session", {"task_id": task_id, "index": i, "session": sess}, length=8)
        )
        ts = str(sess.get("timestamp") or "")
        messages = sess.get("messages") or []
        records: list[MemoryRecord] = []
        for j, msg in enumerate(messages):
            content = str(msg.get("content") or msg.get("text") or "")
            role = str(msg.get("role") or "user")
            if not content:
                continue
            records.append(
                MemoryRecord(
                    id=f"{task_id}_{sess_id}_{j}",
                    content=content,
                    ts=ts,
                    role=role,
                    session_id=sess_id,
                    kind="message",
                )
            )
        sessions.append(Session(id=sess_id, ts=ts, records=tuple(records)))

    default_evidence_ids: list[str] = []
    evidence_ids_by_probe: dict[int, list[str]] = {}
    if not sessions:
        records: list[MemoryRecord] = []
        for key in ("background", "context", "misc", "necessary_background_context"):
            value = task_data.get(key)
            if value:
                content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                evidence_id = f"{task_id}_{key}"
                records.append(
                    MemoryRecord(
                        id=evidence_id,
                        content=content,
                        session_id="background",
                        kind="background",
                        evidence_id=evidence_id,
                        tags=(domain, key),
                    )
                )
                default_evidence_ids.append(evidence_id)
        backgrounds = task_data.get("backgrounds") or []
        if isinstance(backgrounds, list):
            for i, value in enumerate(backgrounds):
                content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                if content:
                    evidence_id = f"{task_id}_backgrounds_{i}"
                    records.append(
                        MemoryRecord(
                            id=evidence_id,
                            content=content,
                            session_id="background",
                            kind="background",
                            evidence_id=evidence_id,
                            tags=(domain, "backgrounds"),
                        )
                    )
                    evidence_ids_by_probe.setdefault(i, []).append(evidence_id)
        elif backgrounds:
            content = (
                backgrounds
                if isinstance(backgrounds, str)
                else json.dumps(backgrounds, ensure_ascii=False)
            )
            evidence_id = f"{task_id}_backgrounds"
            records.append(
                MemoryRecord(
                    id=evidence_id,
                    content=content,
                    session_id="background",
                    kind="background",
                    evidence_id=evidence_id,
                    tags=(domain, "backgrounds"),
                )
            )
            default_evidence_ids.append(evidence_id)
        base_person = task_data.get("base_person")
        if base_person:
            content = (
                base_person
                if isinstance(base_person, str)
                else json.dumps(base_person, ensure_ascii=False)
            )
            evidence_id = f"{task_id}_base_person"
            records.append(
                MemoryRecord(
                    id=evidence_id,
                    content=content,
                    session_id="background",
                    kind="background",
                    evidence_id=evidence_id,
                    tags=(domain, "base_person"),
                )
            )
            default_evidence_ids.append(evidence_id)
        if records:
            sessions.append(Session(id="background", ts="", records=tuple(records)))

    episode = Episode(id=task_id, sessions=tuple(sessions), metadata={"domain": domain})

    subtasks = list(task_data.get("subtasks") or [])
    if not subtasks and task_data.get("questions"):
        questions = list(task_data.get("questions") or [])
        answers = list(task_data.get("answers") or [])
        subtasks = [
            {
                "subtask_id": f"{task_id}_q{i}",
                "instruction": str(question),
                "success_criteria": str(answers[i]) if i < len(answers) else "",
            }
            for i, question in enumerate(questions)
        ]

    probes: list[Probe] = []

    for i, sub in enumerate(subtasks):
        raw_sub_id = str(sub.get("subtask_id") or f"{task_id}_sub{i}")
        sub_id = raw_sub_id
        if not raw_sub_id.startswith(f"{task_id}_"):
            sub_id = f"{task_id}_{raw_sub_id}"
        instruction = str(sub.get("instruction") or sub.get("question") or "")
        criteria = str(sub.get("success_criteria") or sub.get("answer") or "")
        probes.append(
            Probe(
                id=sub_id,
                question=instruction,
                gold=(criteria,) if criteria else (),
                category=domain,
                evidence_ids=tuple(evidence_ids_by_probe.get(i, default_evidence_ids)),
                kind="qa",
                metadata={
                    "domain": domain,
                    "dataset_mode": "public_jsonl_recall_qa" if not stub else "stub_unscored",
                },
            )
        )

    return episode, probes


@register_benchmark("memoryarena")
class MemoryArenaBenchmark:
    """MemoryArena agentic memory benchmark.

    Default judge: task_success (score from environment).
    4 domains, interdependent multi-session tasks.

    NOTE: Full evaluation requires the MemoryArena dataset and environment
    SDK.  When the dataset is absent, the benchmark returns stub probes
    with score=0 and a clear log message.
    """

    name = "memoryarena"
    kind = "recall"
    default_judge = "f1"

    def load(self, spec: RunSpec) -> Iterator[tuple[Episode, list[Probe]]]:
        try:
            base = resolve_path("memoryarena", strict=True)
        except DatasetNotFoundError as exc:
            log.warning(
                "MemoryArena dataset not found: %s\n"
                "Skipping MemoryArena instead of emitting fake scored stub rows.",
                exc,
            )
            return
        assert base is not None

        tasks_dir = base / "tasks"
        if not tasks_dir.is_dir():
            tasks_dir = base

        max_ep = spec.dataset.max_episodes
        max_pr = spec.dataset.max_probes_per_episode
        n = 0

        for task_file in sorted(tasks_dir.rglob("*.json")) + sorted(tasks_dir.rglob("*.jsonl")):
            if max_ep > 0 and n >= max_ep:
                break
            try:
                if task_file.suffix == ".jsonl":
                    rows = [
                        json.loads(line)
                        for line in task_file.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                else:
                    loaded = json.loads(task_file.read_text(encoding="utf-8"))
                    rows = loaded if isinstance(loaded, list) else [loaded]
                for row_index, task_data in enumerate(rows):
                    if max_ep > 0 and n >= max_ep:
                        break
                    if not isinstance(task_data, dict):
                        continue
                    fallback_task_id = _stable_id(
                        "task",
                        {
                            "file": str(task_file.relative_to(tasks_dir)),
                            "row_index": row_index,
                            "task": task_data,
                        },
                        length=8,
                    )
                    episode, probes = _parse_task(
                        task_data,
                        domain_hint=task_file.parent.name,
                        fallback_task_id=fallback_task_id,
                    )
                    if max_pr > 0:
                        probes = probes[:max_pr]
                    if probes:
                        yield episode, probes
                        n += 1
            except Exception as exc:
                log.warning("MemoryArena: failed to load %s: %s", task_file, exc)
