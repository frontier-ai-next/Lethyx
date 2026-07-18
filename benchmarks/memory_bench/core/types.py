"""Provider-agnostic domain types for the memory_bench framework.

These dataclasses represent the universal benchmark data model.
No imports from Lethyx internals or any specific memory backend here.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Memory input side
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryRecord:
    """Universal atomic unit of memory to be written to a store.

    Maps to one dialog turn, fact, observation, action result, or feedback.
    The ``evidence_id`` field links to the Probe.evidence_ids list, allowing
    retrieval metrics (recall@k, MRR) to be computed without knowing internals
    of the storage backend.
    """

    id: str
    content: str
    ts: str = ""  # ISO-8601 timestamp of session/turn (for temporal benchmarks)
    role: str = ""  # user | assistant | system | agent | env
    session_id: str = ""  # groups records into sessions within an episode
    kind: str = "message"  # message | fact | observation | action | feedback
    tags: tuple[str, ...] = ()
    evidence_id: str = ""  # logical ID for retrieval-metric matching
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", f"rec_{uuid.uuid4().hex[:12]}")
        if not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(str(t) for t in self.tags))


@dataclass(frozen=True)
class Session:
    """One temporal session (conversation window) inside an episode."""

    id: str
    ts: str  # session start timestamp
    records: tuple[MemoryRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            object.__setattr__(self, "records", tuple(self.records))


@dataclass(frozen=True)
class Episode:
    """The full history for one benchmark conversation/scenario.

    A single Episode is isolated into its own MemoryStore namespace so that
    runs across episodes cannot cross-contaminate.
    """

    id: str
    sessions: tuple[Session, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sessions, tuple):
            object.__setattr__(self, "sessions", tuple(self.sessions))

    @property
    def all_records(self) -> list[MemoryRecord]:
        return [r for s in self.sessions for r in s.records]

    @property
    def n_tokens_estimate(self) -> int:
        """Rough character-based token estimate (1 token ≈ 4 chars)."""
        total = sum(len(r.content) for r in self.all_records)
        return total // 4


# ---------------------------------------------------------------------------
# Query / probe side
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """One question or agentic task directed at the memory store.

    ``evidence_ids`` lists the MemoryRecord.evidence_id values that contain
    the answer, enabling retrieval accuracy computation alongside answer quality.
    ``abstain=True`` marks false-premise / unanswerable questions (LongMemEval ABS,
    BEAM abstention category).
    """

    id: str
    question: str
    gold: tuple[str, ...] = ()  # acceptable reference answers
    category: str = ""  # benchmark-specific category label
    evidence_ids: tuple[str, ...] = ()  # which record evidence_ids cover the answer
    abstain: bool = False
    kind: str = "qa"  # qa | action
    tier: str = ""  # BEAM context-length tier: 100k|500k|1m|10m
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.gold, tuple):
            object.__setattr__(self, "gold", tuple(str(g) for g in self.gold))
        if not isinstance(self.evidence_ids, tuple):
            object.__setattr__(self, "evidence_ids", tuple(str(e) for e in self.evidence_ids))


# ---------------------------------------------------------------------------
# Retrieval output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedItem:
    """One item returned by a MemoryStore.retrieve() call."""

    text: str
    score: float
    source_id: str = ""  # maps to MemoryRecord.evidence_id when available
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Answer and judgment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    """Token and cost accounting for one LLM call or aggregated across a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            calls=self.calls + other.calls,
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
        }


@dataclass(frozen=True)
class Answer:
    """The LLM response to a Probe after memory retrieval."""

    text: str
    abstained: bool = False  # model said "I don't know" / refused to answer
    usage: Usage = field(default_factory=Usage)
    raw: str = ""  # unprocessed model output


@dataclass(frozen=True)
class Judgment:
    """Correctness verdict for one (Probe, Answer) pair."""

    correct: bool
    score: float  # continuous [0, 1] for partial credit (F1, nugget)
    metric: str  # f1 | exact_match | llm_judge | nugget | task_success
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """One benchmark result row: provider × benchmark × probe.

    Written as JSON Lines to runs/memory_bench/<ts>/<provider>/<bench>.jsonl.
    The ``retrieval`` dict holds retrieval-quality metrics (recall@k, mrr, ndcg,
    n_retrieved) computed independently of answer quality.
    """

    provider: str
    benchmark: str
    episode_id: str
    probe_id: str
    category: str
    status: str  # ok | error | skipped
    judgment: Judgment | None = None
    retrieval: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    error: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        j = None
        if self.judgment is not None:
            j = {
                "correct": self.judgment.correct,
                "score": self.judgment.score,
                "metric": self.judgment.metric,
                "detail": self.judgment.detail,
            }
        return {
            "provider": self.provider,
            "benchmark": self.benchmark,
            "episode_id": self.episode_id,
            "probe_id": self.probe_id,
            "category": self.category,
            "status": self.status,
            "judgment": j,
            "retrieval": self.retrieval,
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "error": self.error,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseResult":
        j = None
        if data.get("judgment"):
            jd = data["judgment"]
            j = Judgment(
                correct=bool(jd["correct"]),
                score=float(jd["score"]),
                metric=str(jd["metric"]),
                detail=dict(jd.get("detail") or {}),
            )
        raw_usage = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            cost_usd=float(raw_usage.get("cost_usd", 0.0)),
            calls=int(raw_usage.get("calls", 0)),
        )
        return cls(
            provider=str(data["provider"]),
            benchmark=str(data["benchmark"]),
            episode_id=str(data["episode_id"]),
            probe_id=str(data["probe_id"]),
            category=str(data.get("category", "")),
            status=str(data["status"]),
            judgment=j,
            retrieval=dict(data.get("retrieval") or {}),
            usage=usage,
            latency_ms=float(data.get("latency_ms", 0.0)),
            error=str(data.get("error") or ""),
            diagnostics=dict(data.get("diagnostics") or {}),
        )


# ---------------------------------------------------------------------------
# Run specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Reference to a benchmark dataset."""

    name: str
    path: str = ""  # local path override; empty = use datasets registry default
    tiers: tuple[str, ...] = ()  # BEAM context-length tiers; empty = all
    max_episodes: int = 0  # 0 = unlimited
    max_probes_per_episode: int = 0  # 0 = unlimited

    def __post_init__(self) -> None:
        if not isinstance(self.tiers, tuple):
            object.__setattr__(self, "tiers", tuple(str(t) for t in self.tiers))


@dataclass(frozen=True)
class ProviderSpec:
    """Configuration for one memory provider instance."""

    id: str  # unique label used in results (e.g. "lethyx-hindsight")
    adapter: str  # registered adapter name (e.g. "lethyx", "no_memory", "bm25")
    params: dict[str, Any] = field(default_factory=dict)  # adapter-specific config


@dataclass(frozen=True)
class ResponderSpec:
    model: str = ""  # "" -> read from LLM_MODEL env
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass(frozen=True)
class JudgeConfig:
    llm_judge_model: str = ""  # "" -> read from LLM_MODEL env
    use_llm_judge: bool = True  # set False to use only deterministic metrics


@dataclass(frozen=True)
class LimitsSpec:
    recall_k: int = 10
    max_context_tokens: int = 4000
    seed: int = 0
    max_parallel: int = 1
    redo_existing: bool = False


@dataclass(frozen=True)
class RunSpec:
    """Full resolved specification for one benchmark run."""

    providers: tuple[ProviderSpec, ...]
    benchmarks: tuple[str, ...]
    dataset: DatasetSpec
    responder: ResponderSpec = field(default_factory=ResponderSpec)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    limits: LimitsSpec = field(default_factory=LimitsSpec)
    out_dir: str = "runs/memory_bench"
    cache_dir: str = ".memory_bench_cache"

    def __post_init__(self) -> None:
        if not isinstance(self.providers, tuple):
            object.__setattr__(self, "providers", tuple(self.providers))
        if not isinstance(self.benchmarks, tuple):
            object.__setattr__(self, "benchmarks", tuple(self.benchmarks))
