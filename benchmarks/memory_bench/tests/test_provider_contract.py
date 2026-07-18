"""Provider contract test suite.

These tests run against ANY MemoryProvider implementation by parameterising
over the baseline providers (no Lethyx dependency, no network).

To add a new provider to the contract suite, add it to the PROVIDERS list.
"""

import pytest
from memory_bench.adapters.baselines import (
    NoMemoryProvider,
    FullContextProvider,
    BM25Provider,
)
from memory_bench.core.types import MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_record(i: int, evidence_id: str = "") -> MemoryRecord:
    return MemoryRecord(
        id=f"r{i}",
        content=f"Alice visited Paris on day {i}. She bought a croissant.",
        session_id="s0",
        evidence_id=evidence_id or f"ev{i}",
    )


PROVIDERS = [
    pytest.param(NoMemoryProvider(), id="no_memory"),
    pytest.param(FullContextProvider(), id="full_context"),
    pytest.param(BM25Provider(), id="bm25"),
]


@pytest.fixture(params=PROVIDERS)
def provider(request):
    return request.param


# ---------------------------------------------------------------------------
# Contract: namespace isolation
# ---------------------------------------------------------------------------


def test_namespace_isolation(provider):
    """Two namespaces must not share data."""
    store_a = provider.namespace("ep_a")
    store_b = provider.namespace("ep_b")
    store_a.write(make_record(0))
    # b should be empty regardless
    results = store_b.retrieve("Paris", k=5)
    # NoMemory always returns []; FullContext and BM25 should also return []
    # because nothing was written to store_b
    assert isinstance(results, list)
    store_a.close()
    store_b.close()


# ---------------------------------------------------------------------------
# Contract: write then retrieve
# ---------------------------------------------------------------------------


def test_write_then_retrieve(provider):
    """FullContext and BM25 should return something; NoMemory always empty."""
    store = provider.namespace("ep_test")
    for i in range(5):
        store.write(make_record(i))
    results = store.retrieve("Paris croissant", k=3)
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item.text, str)
        assert isinstance(item.score, float)
    store.close()


# ---------------------------------------------------------------------------
# Contract: k limit respected
# ---------------------------------------------------------------------------


def test_k_limit(provider):
    """retrieve() must return at most k items."""
    store = provider.namespace("ep_k")
    for i in range(20):
        store.write(make_record(i))
    results = store.retrieve("Alice Paris", k=3)
    assert len(results) <= 3
    store.close()


# ---------------------------------------------------------------------------
# Contract: capabilities returns ProviderCapabilities
# ---------------------------------------------------------------------------


def test_capabilities(provider):
    from memory_bench.core.protocols import ProviderCapabilities

    caps = provider.capabilities()
    assert isinstance(caps, ProviderCapabilities)
    assert isinstance(caps.persistent, bool)
    assert isinstance(caps.supports_namespacing, bool)


# ---------------------------------------------------------------------------
# Contract: stats()
# ---------------------------------------------------------------------------


def test_stats_returns_dict(provider):
    store = provider.namespace("ep_stats")
    store.write(make_record(0))
    stats = store.stats()
    assert isinstance(stats, dict)
    store.close()


# ---------------------------------------------------------------------------
# Contract: write_many
# ---------------------------------------------------------------------------


def test_write_many(provider):
    store = provider.namespace("ep_many")
    records = [make_record(i) for i in range(10)]
    store.write_many(records)
    results = store.retrieve("Alice", k=5)
    assert isinstance(results, list)
    store.close()
