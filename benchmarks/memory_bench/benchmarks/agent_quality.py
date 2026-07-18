"""Self-contained agent-memory quality benchmark.

This benchmark is intentionally small and dataset-free.  It gives a fast
agent-style quality signal before running external LoCoMo/LongMemEval/BEAM data:
write a compact multi-session agent history, retrieve evidence through each
memory provider, and ask the fixed reader LLM to answer from retrieved memory
only.
"""

from typing import Iterator

from memory_bench.core.registry import register_benchmark
from memory_bench.core.types import Episode, MemoryRecord, Probe, RunSpec, Session


def _record(
    record_id: str,
    content: str,
    *,
    session_id: str,
    evidence_id: str,
    role: str = "agent",
    kind: str = "observation",
    tags: tuple[str, ...] = (),
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=content,
        role=role,
        session_id=session_id,
        kind=kind,
        tags=tags,
        evidence_id=evidence_id,
    )


def _episodes() -> list[tuple[Episode, list[Probe]]]:
    handoff_records = (
        _record(
            "handoff-r0",
            "Project Atlas rollback owner is Nina. The deployment codename is ORCHID.",
            session_id="s1",
            evidence_id="atlas-owner",
            tags=("handoff", "owner"),
        ),
        _record(
            "handoff-r1",
            "For risky LangGraph memory releases, run pytest, then the Lethyx scenario harness, then a real LLM smoke test.",
            session_id="s2",
            evidence_id="release-gate",
            kind="procedure",
            tags=("verification", "langgraph"),
        ),
        _record(
            "handoff-r2-old",
            "Deprecated note: old agents sometimes skipped tests before deploy.",
            session_id="s2",
            evidence_id="skip-tests-old",
            tags=("deprecated", "conflict"),
        ),
        _record(
            "handoff-r3-current",
            "Current rule: never skip tests before deployment; verify first and only then deploy.",
            session_id="s3",
            evidence_id="skip-tests-current",
            tags=("current", "conflict"),
        ),
    )
    handoff = Episode(
        id="agent_handoff",
        sessions=(
            Session(id="s1", ts="2026-01-01T09:00:00Z", records=handoff_records[:1]),
            Session(id="s2", ts="2026-01-02T09:00:00Z", records=handoff_records[1:3]),
            Session(id="s3", ts="2026-01-03T09:00:00Z", records=handoff_records[3:]),
        ),
        metadata={"theme": "agent_handoff"},
    )
    handoff_probes = [
        Probe(
            id="handoff-p-owner",
            question="Who owns rollback for Project Atlas?",
            gold=("Project Atlas rollback owner is Nina",),
            category="cross_session_fact",
            evidence_ids=("atlas-owner",),
        ),
        Probe(
            id="handoff-p-gate",
            question="What is the verification sequence for risky LangGraph memory releases?",
            gold=("run pytest, then the Lethyx scenario harness, then a real LLM smoke test",),
            category="procedural_memory",
            evidence_ids=("release-gate",),
        ),
        Probe(
            id="handoff-p-conflict",
            question="Should the agent skip tests before deployment?",
            gold=("No, never skip tests before deployment; verify first and only then deploy",),
            category="conflict_resolution",
            evidence_ids=("skip-tests-current",),
        ),
        Probe(
            id="handoff-p-abstain",
            question="What is Project Atlas's production database password?",
            gold=(),
            category="abstention",
            evidence_ids=(),
            abstain=True,
        ),
    ]

    preference_records = (
        _record(
            "pref-r0",
            "User preference for benchmark reports: be concise, lead with metrics, and write summaries in Russian.",
            session_id="s1",
            evidence_id="report-style",
            kind="preference",
            tags=("preference", "report"),
        ),
        _record(
            "pref-r1",
            "Experiment B-72 uses the Nemotron model through the FrontierAI OpenAI-compatible gateway.",
            session_id="s2",
            evidence_id="experiment-model",
            tags=("experiment", "model"),
        ),
    )
    preferences = Episode(
        id="agent_preferences",
        sessions=(
            Session(id="s1", ts="2026-02-01T10:00:00Z", records=preference_records[:1]),
            Session(id="s2", ts="2026-02-02T10:00:00Z", records=preference_records[1:]),
        ),
        metadata={"theme": "agent_preferences"},
    )
    preference_probes = [
        Probe(
            id="pref-p-style",
            question="How should benchmark report summaries be written?",
            gold=("concise, lead with metrics, and write summaries in Russian",),
            category="preference_update",
            evidence_ids=("report-style",),
        ),
        Probe(
            id="pref-p-model",
            question="Which model does Experiment B-72 use?",
            gold=("Experiment B-72 uses the Nemotron model",),
            category="cross_session_fact",
            evidence_ids=("experiment-model",),
        ),
    ]

    return [(handoff, handoff_probes), (preferences, preference_probes)]


@register_benchmark("agent_quality")
class AgentQualityBenchmark:
    """Tiny built-in benchmark for agent memory quality.

    It measures the same output metrics as external recall benchmarks
    (answer score plus retrieval recall/MRR/nDCG), but does not require dataset
    downloads.  Use it for smoke and CI; use LoCoMo/LongMemEval/BEAM for
    publication-grade evaluation.
    """

    name = "agent_quality"
    kind = "recall"
    default_judge = "f1"

    def load(self, spec: RunSpec) -> Iterator[tuple[Episode, list[Probe]]]:
        max_ep = spec.dataset.max_episodes
        max_pr = spec.dataset.max_probes_per_episode
        for index, (episode, probes) in enumerate(_episodes()):
            if max_ep > 0 and index >= max_ep:
                break
            if max_pr > 0:
                probes = probes[:max_pr]
            if probes:
                yield episode, probes
