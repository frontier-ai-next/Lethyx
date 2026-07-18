"""Baseline memory providers for memory_bench.

These providers serve as reference points:
- NoMemoryProvider  — retrieve() always returns [] (lower bound).
- FullContextProvider — stores everything; retrieve() returns all records
  up to token budget (upper bound on short episodes; degrades on long ones).
- BM25Provider — classic BM25 full-text retrieval baseline
  (requires rank-bm25 package; gracefully disabled if absent).

All providers are pure Python with no Lethyx dependency.
"""

import logging
from typing import Any

from memory_bench.core.protocols import ProviderCapabilities
from memory_bench.core.registry import register_provider
from memory_bench.core.text import tokenize
from memory_bench.core.types import MemoryRecord, RetrievedItem

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NoMemory
# ---------------------------------------------------------------------------


class _NoMemoryStore:
    def write(self, record: MemoryRecord) -> None:
        pass

    def write_many(self, records: list[MemoryRecord]) -> None:
        pass

    def retrieve(
        self, query: str, *, k: int, max_tokens: int = 0, filters: dict[str, Any] | None = None
    ) -> list[RetrievedItem]:
        return []

    def update(self, record_id: str, **changes: Any) -> bool:
        return False

    def stats(self) -> dict[str, Any]:
        return {"backend": "no_memory"}

    def close(self) -> None:
        pass


@register_provider("no_memory")
class NoMemoryProvider:
    """Lower bound: no memory retrieved, ever."""

    name = "no_memory"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        pass

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            persistent=False,
            supports_update=False,
            supports_namespacing=True,
        )

    def namespace(self, ns: str) -> _NoMemoryStore:
        return _NoMemoryStore()


# ---------------------------------------------------------------------------
# FullContext
# ---------------------------------------------------------------------------


class _FullContextStore:
    """In-memory store that returns all records up to the token budget."""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    def write(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def write_many(self, records: list[MemoryRecord]) -> None:
        self._records.extend(records)

    def retrieve(
        self, query: str, *, k: int, max_tokens: int = 0, filters: dict[str, Any] | None = None
    ) -> list[RetrievedItem]:
        # Return records in chronological order (most recent first for recency bias)
        top = list(reversed(self._records))
        if max_tokens > 0:
            # Truncate to budget
            budget = max_tokens * 4
            used = 0
            capped: list[MemoryRecord] = []
            for r in top:
                if used + len(r.content) > budget:
                    break
                capped.append(r)
                used += len(r.content)
            top = capped
        else:
            top = top[:k]
        return [
            RetrievedItem(
                text=r.content,
                score=1.0,
                source_id=r.evidence_id or r.id,
                rank=i,
            )
            for i, r in enumerate(top)
        ]

    def update(self, record_id: str, **changes: Any) -> bool:
        for i, r in enumerate(self._records):
            if r.id == record_id:
                # Replace content if provided
                new_content = changes.get("content")
                if new_content:
                    from dataclasses import replace

                    self._records[i] = replace(r, content=str(new_content))
                return True
        return False

    def stats(self) -> dict[str, Any]:
        return {"backend": "full_context", "n_records": len(self._records)}

    def close(self) -> None:
        self._records.clear()


@register_provider("full_context")
class FullContextProvider:
    """Upper-bound baseline: dump all episode records into context.

    Degrades naturally on long episodes (BEAM 1M/10M) because max_tokens
    truncates the context, making the gap vs. a proper memory system visible.
    """

    name = "full_context"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        pass

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            persistent=False,
            supports_update=True,
            supports_namespacing=True,
        )

    def namespace(self, ns: str) -> _FullContextStore:
        return _FullContextStore()


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def _try_import_bm25() -> Any:
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import]

        return BM25Okapi
    except ImportError:
        return None


class _BM25Store:
    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []
        self._index: Any = None  # BM25Okapi instance, built on first retrieve

    def write(self, record: MemoryRecord) -> None:
        self._records.append(record)
        self._index = None  # invalidate

    def write_many(self, records: list[MemoryRecord]) -> None:
        self._records.extend(records)
        self._index = None

    def _build_index(self) -> None:
        BM25Okapi = _try_import_bm25()
        if BM25Okapi is None:
            raise RuntimeError(
                "rank-bm25 is required for the BM25 baseline; install rank-bm25 "
                "instead of falling back to recency ranking."
            )
        corpus = [tokenize(r.content) for r in self._records]
        self._index = BM25Okapi(corpus)

    def retrieve(
        self, query: str, *, k: int, max_tokens: int = 0, filters: dict[str, Any] | None = None
    ) -> list[RetrievedItem]:
        if not self._records:
            return []

        BM25Okapi = _try_import_bm25()
        if BM25Okapi is None:
            raise RuntimeError(
                "rank-bm25 is required for the BM25 baseline; install rank-bm25 "
                "instead of falling back to recency ranking."
            )

        if self._index is None:
            self._build_index()

        query_tokens = tokenize(query)
        scores = self._index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k]

        results: list[RetrievedItem] = []
        used_tokens = 0
        budget = max_tokens * 4 if max_tokens > 0 else 0
        for rank, (idx, score) in enumerate(ranked):
            r = self._records[idx]
            if budget and used_tokens + len(r.content) > budget:
                break
            results.append(
                RetrievedItem(
                    text=r.content,
                    score=float(score),
                    source_id=r.evidence_id or r.id,
                    rank=rank,
                )
            )
            used_tokens += len(r.content)
        return results

    def update(self, record_id: str, **changes: Any) -> bool:
        for i, r in enumerate(self._records):
            if r.id == record_id:
                new_content = changes.get("content")
                if new_content:
                    from dataclasses import replace

                    self._records[i] = replace(r, content=str(new_content))
                    self._index = None
                return True
        return False

    def stats(self) -> dict[str, Any]:
        return {"backend": "bm25", "n_records": len(self._records)}

    def close(self) -> None:
        self._records.clear()
        self._index = None


@register_provider("bm25")
class BM25Provider:
    """BM25 retrieval baseline; requires rank-bm25 and fails closed if absent."""

    name = "bm25"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        pass

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            persistent=False,
            supports_update=True,
            supports_namespacing=True,
        )

    def namespace(self, ns: str) -> _BM25Store:
        return _BM25Store()
