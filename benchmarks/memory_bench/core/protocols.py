"""Protocol interfaces for the memory_bench framework.

All major extension points are expressed as Protocols so that any conforming
class works — no inheritance required.  ProviderCapabilities acts as a
capability-advertisement contract: benchmarks interrogate it to decide which
test categories are runnable against a given provider.
"""

from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memory_bench.core.types import (
        Answer,
        Episode,
        Judgment,
        MemoryRecord,
        Probe,
        RetrievedItem,
        RunSpec,
    )

# ---------------------------------------------------------------------------
# Capability declaration
# ---------------------------------------------------------------------------


class ProviderCapabilities:
    """Declares what a memory provider supports.

    Benchmarks use these flags to decide whether to run, skip, or degrade
    specific test categories.  For example, ``knowledge-update`` probes in
    LongMemEval and EvoMemBench are skipped when ``supports_update=False``.
    """

    __slots__ = (
        "persistent",
        "supports_update",
        "supports_delete",
        "supports_temporal_filter",
        "supports_graph",
        "supports_namespacing",
        "memory_as_tool",
        "token_accounting",
        "extra",
    )

    def __init__(
        self,
        *,
        persistent: bool = True,
        supports_update: bool = False,
        supports_delete: bool = False,
        supports_temporal_filter: bool = False,
        supports_graph: bool = False,
        supports_namespacing: bool = True,
        memory_as_tool: bool = False,
        token_accounting: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.persistent = persistent
        self.supports_update = supports_update
        self.supports_delete = supports_delete
        self.supports_temporal_filter = supports_temporal_filter
        self.supports_graph = supports_graph
        self.supports_namespacing = supports_namespacing
        self.memory_as_tool = memory_as_tool
        self.token_accounting = token_accounting
        self.extra: dict[str, Any] = dict(extra or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent": self.persistent,
            "supports_update": self.supports_update,
            "supports_delete": self.supports_delete,
            "supports_temporal_filter": self.supports_temporal_filter,
            "supports_graph": self.supports_graph,
            "supports_namespacing": self.supports_namespacing,
            "memory_as_tool": self.memory_as_tool,
            "token_accounting": self.token_accounting,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Core memory protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """One isolated namespace of memory (one episode × one provider).

    A MemoryStore is created fresh for each (provider, benchmark, episode)
    triple.  After the episode is complete the store must be closed so the
    provider can free resources or flush writes.

    The ``update`` method is optional; callers MUST check
    ``provider.capabilities().supports_update`` before calling it.
    """

    def write(self, record: "MemoryRecord") -> None:
        """Write one record to the store."""
        ...

    def write_many(self, records: "list[MemoryRecord]") -> None:
        """Bulk-write records; default implementation loops over write()."""
        for r in records:
            self.write(r)

    def retrieve(
        self,
        query: str,
        *,
        k: int,
        max_tokens: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> "list[RetrievedItem]":
        """Retrieve at most k items most relevant to query."""
        ...

    def update(self, record_id: str, **changes: Any) -> bool:
        """Update a stored record (knowledge-update tests).

        Returns True if the record was found and updated.
        Raises NotImplementedError when supports_update=False.
        """
        raise NotImplementedError("this provider does not support updates")

    def stats(self) -> dict[str, Any]:
        """Return diagnostic info: n_records, n_tokens_stored, etc."""
        ...

    def close(self) -> None:
        """Release resources and finalize writes."""
        ...


@runtime_checkable
class MemoryProvider(Protocol):
    """Factory for isolated per-episode memory stores.

    Each call to ``namespace(ns)`` returns a fresh, empty MemoryStore.
    The namespace string is deterministic per (benchmark, episode) so runs
    are resumable and reproducible.
    """

    name: str

    def capabilities(self) -> ProviderCapabilities:
        """Return the capability declaration for this provider."""
        ...

    def namespace(self, ns: str) -> MemoryStore:
        """Return a fresh, isolated MemoryStore for the given namespace."""
        ...


# ---------------------------------------------------------------------------
# Responder (reader / policy LLM)
# ---------------------------------------------------------------------------


@runtime_checkable
class Responder(Protocol):
    """Generates a natural-language answer given retrieved memory context.

    In isolation mode the same Responder instance is shared across all
    providers for a benchmark run so that the LLM is the fixed constant and
    only the retrieved context (i.e. the memory quality) varies.
    """

    def answer(
        self,
        question: str,
        context: "list[RetrievedItem]",
        *,
        instructions: str = "",
    ) -> "Answer":
        """Generate an answer using the provided context items."""
        ...


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


@runtime_checkable
class Judge(Protocol):
    """Evaluates one (Probe, Answer) pair and returns a Judgment.

    Multiple judge implementations are registered:
    - ``f1``: token-level F1 partial match (LoCoMo style).
    - ``llm_judge``: LLM-as-judge (LongMemEval GPT-4o style).
    - ``nugget``: nugget-coverage (BEAM style).
    - ``exact_match``: strict string equality after normalisation.
    - ``abstention``: checks that the model said "I don't know".
    """

    name: str

    def judge(self, probe: "Probe", answer: "Answer") -> "Judgment": ...


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@runtime_checkable
class Benchmark(Protocol):
    """Loads benchmark data and produces (Episode, Probe list) pairs.

    ``kind`` must be "recall" or "agentic"; the runner dispatches to the
    appropriate driver based on this value.
    """

    name: str
    kind: str  # "recall" | "agentic"
    default_judge: str  # registered Judge name used when none is specified

    def load(self, spec: "RunSpec") -> Iterator[tuple["Episode", "list[Probe]"]]:
        """Yield (episode, probes) pairs.

        Implementations should respect spec.dataset.max_episodes and
        spec.dataset.max_probes_per_episode.
        """
        ...


# ---------------------------------------------------------------------------
# Agentic protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Environment(Protocol):
    """An interactive environment for agentic benchmarks.

    Follows the classic gym-style API.  Implementations wrap the upstream
    MemoryArena / EvoMemBench task environments.
    """

    def reset(self) -> dict[str, Any]:
        """Reset the environment and return the initial observation."""
        ...

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute action; return (observation, reward, done, info)."""
        ...

    def score(self) -> float:
        """Return the final task success score [0, 1]."""
        ...


@runtime_checkable
class AgentPolicy(Protocol):
    """Policy that decides what action to take given an observation.

    In memory-isolation mode this is a fixed LLM policy that:
    1. Calls store.retrieve() to get relevant past experience.
    2. Calls the policy LLM with obs + context.
    3. Calls store.write() to persist the new observation + action.
    """

    def act(self, obs: dict[str, Any], store: MemoryStore) -> str:
        """Return the next action string."""
        ...
