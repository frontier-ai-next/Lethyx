"""Coverage for the multilingual semantic retrieval tier."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from lethyx.memory import local_tier
from lethyx.memory import mcp_server
from lethyx.memory import semantic

_FAMILIES = {
    "config": ("config", "конфигурац", "конфиг"),
    "python": ("python", "питон"),
    "capital": ("capital", "столиц", "france", "франц"),
}


class FakeEmbedder:
    """Deterministic stand-in for the multilingual model."""

    model = "fake/multilingual-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts, *, kind: str = "document"):
        self.calls.append(tuple(texts))
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            vector = [0.0, 0.0, 0.0, 0.35]
            for index, keywords in enumerate(_FAMILIES.values()):
                if any(keyword in lowered for keyword in keywords):
                    vector[index] = 1.0
                    break
            total = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / total for value in vector])
        return vectors


@pytest.fixture
def fake_embedder(monkeypatch) -> FakeEmbedder:
    embedder = FakeEmbedder()
    monkeypatch.setenv("LETHYX_SEMANTIC", "1")
    monkeypatch.setattr(semantic, "get_embedder", lambda *args, **kwargs: embedder)
    return embedder


def _records() -> list[dict[str, object]]:
    return [
        {
            "record_id": "mem_config",
            "title": "OpenCode config locations",
            "content": "Global config lives at C:/Users/example/.config/opencode/opencode.json",
            "block_membership": {"primary_block_id": "leaf_config"},
        },
        {
            "record_id": "mem_python",
            "title": "Python and Node toolchain",
            "content": "Python tooling uses uv-managed CPython 3.12.14.",
            "block_membership": {"primary_block_id": "leaf_python"},
        },
    ]


def test_query_uses_non_ascii_script_detects_other_scripts() -> None:
    assert semantic.query_uses_non_ascii_script("где конфигурация") is True
    assert semantic.query_uses_non_ascii_script("配置文件在哪里") is True
    assert semantic.query_uses_non_ascii_script("where is the config") is False
    assert semantic.query_uses_non_ascii_script("") is False


def test_effective_floor_keeps_the_base_floor_for_other_scripts() -> None:
    base = semantic.relevance_floor("fake/multilingual-v1")
    assert semantic.effective_floor("fake/multilingual-v1", cross_script=True) == base
    assert semantic.effective_floor(
        "fake/multilingual-v1", cross_script=False
    ) == pytest.approx(base + semantic.LATIN_SCRIPT_MARGIN)


def test_rank_semantic_caches_vectors_between_calls(
    fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    index = semantic.SemanticIndex(tmp_path / "embeddings.sqlite")
    records = _records()

    first = semantic.rank_semantic(
        "где конфигурация", records, embedder=fake_embedder, index=index
    )
    assert first
    assert first[0][0] == "mem_config"
    assert first[0][1] > 0.9
    # One embedding call for the missing record vectors plus one for the query.
    assert len(fake_embedder.calls) == 2

    second = semantic.rank_semantic(
        "где конфигурация", records, embedder=fake_embedder, index=index
    )
    assert [record_id for record_id, _ in second] == [
        record_id for record_id, _ in first
    ]
    assert second[0][1] == pytest.approx(first[0][1], abs=1e-6)
    # Only the query is embedded once every record vector is cached.
    assert len(fake_embedder.calls) == 3


def test_rank_semantic_reembeds_after_the_content_changed(
    fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    index = semantic.SemanticIndex(tmp_path / "embeddings.sqlite")
    records = _records()
    semantic.rank_semantic(
        "где конфигурация", records, embedder=fake_embedder, index=index
    )
    calls_before = len(fake_embedder.calls)
    changed = [
        records[0],
        dict(records[1], title="Config tooling", content="Config tooling uses uv."),
    ]
    ranked = semantic.rank_semantic(
        "питон", changed, embedder=fake_embedder, index=index
    )
    # The changed record no longer belongs to the python family, so the stale
    # vector must not be reused: the record is re-embedded from its new text.
    assert ranked == []
    assert len(fake_embedder.calls) == calls_before + 2


def test_rank_semantic_drops_records_below_the_floor(
    fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    index = semantic.SemanticIndex(tmp_path / "embeddings.sqlite")
    ranked = semantic.rank_semantic(
        "what is the capital of France",
        _records(),
        embedder=fake_embedder,
        index=index,
    )
    assert ranked == []


def test_suggest_blocks_groups_semantic_hits_by_block() -> None:
    blocks = semantic.suggest_blocks(
        _records(),
        {"mem_config": 0.81, "mem_python": 0.62},
    )
    assert blocks[0]["block_id"] == "leaf_config"
    assert blocks[0]["hits"] == 1
    assert blocks[0]["score"] == pytest.approx(0.81)


def test_recall_fallback_answers_a_cross_script_question(
    fake_embedder: FakeEmbedder, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *args, **kwargs: {"ok": True, "records": _records()},
    )
    identity = SimpleNamespace(
        repo_root=tmp_path, workspace_id="opencode", profile_id="default"
    )
    result = mcp_server._apply_recall_fallback(
        identity,
        object(),
        "recall",
        {"query": "где находится конфигурация", "limit": 4},
        {"ok": True, "hits": []},
    )
    assert result["ok"] is True
    assert result["fallback"] == "semantic"
    assert result["semantic"] is True
    assert result["semantic_model"] == "fake/multilingual-v1"
    assert result["hits"][0]["title"] == "OpenCode config locations"
    assert result["suggested_blocks"][0]["block_id"] == "leaf_config"


def test_recall_fallback_stays_empty_for_an_off_topic_question(
    fake_embedder: FakeEmbedder, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *args, **kwargs: {"ok": True, "records": _records()},
    )
    identity = SimpleNamespace(
        repo_root=tmp_path, workspace_id="opencode", profile_id="default"
    )
    result = mcp_server._apply_recall_fallback(
        identity,
        object(),
        "recall",
        {"query": "what is the capital of France", "limit": 4},
        {"ok": True, "hits": []},
    )
    assert result["ok"] is True
    assert not result.get("hits")
    assert "hint" in result


def test_recall_fallback_keeps_lexical_hits_for_same_script_questions(
    fake_embedder: FakeEmbedder, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda *args, **kwargs: {"ok": True, "records": _records()},
    )
    identity = SimpleNamespace(
        repo_root=tmp_path, workspace_id="opencode", profile_id="default"
    )
    result = mcp_server._apply_recall_fallback(
        identity,
        object(),
        "recall",
        {"query": "python tooling", "limit": 4},
        {"ok": True, "hits": []},
    )
    assert result["fallback"] == "unicode_ngram"
    assert result["hits"][0]["title"] == "Python and Node toolchain"


def test_local_tier_marks_the_result_and_answers_across_scripts(
    fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    result = local_tier.apply_local_tier(
        {"ok": True, "hits": []},
        name="recall",
        arguments={
            "query": "где конфигурация",
            "limit": 4,
            "include_candidates": True,
            "source_horizon": "durable",
        },
        records_provider=lambda: _records(),
        repo_root=tmp_path,
        workspace_id="opencode",
    )
    assert result["_local_tier_applied"] is True
    assert result["fallback"] == "semantic"
    assert result["semantic"] is True
    assert result["hits"][0]["title"] == "OpenCode config locations"


def test_local_tier_declines_scoped_queries(fake_embedder: FakeEmbedder, tmp_path: Path) -> None:
    result = local_tier.apply_local_tier(
        {"ok": True, "hits": []},
        name="recall",
        arguments={
            "query": "где конфигурация",
            "limit": 4,
            "include_candidates": True,
            "source_horizon": "durable",
            "logical_block_ids": ["leaf_config"],
        },
        records_provider=lambda: _records(),
        repo_root=tmp_path,
        workspace_id="opencode",
    )
    assert result["_local_tier_applied"] is True
    assert not result.get("hits")
    assert "fallback" not in result


def test_global_ranking_reaches_a_record_outside_the_window(
    fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    index = semantic.SemanticIndex(tmp_path / "embeddings.sqlite")
    semantic.embed_record(
        index,
        fake_embedder,
        record_id="mem_old",
        title="Old deployment note",
        content="Old deployment note about the restart procedure.",
        kind="project_fact",
    )
    old = {
        "record_id": "mem_old",
        "title": "Old deployment note",
        "content": "Old deployment note about the restart procedure.",
        "kind": "project_fact",
        "tags": [],
        "status": "candidate",
        "metadata": {},
        "block_membership": {"primary_block_id": "leaf_old"},
    }
    ranked = semantic.rank_global(
        "как перезапустить деплой",
        embedder=fake_embedder,
        index=index,
        limit=3,
        loader=lambda identifiers: [old] if identifiers else [],
    )
    assert ranked
    assert ranked[0][0] == "mem_old"


def test_global_ranking_can_be_narrowed_to_one_kind(
    fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    index = semantic.SemanticIndex(tmp_path / "embeddings.sqlite")
    semantic.embed_record(
        index,
        fake_embedder,
        record_id="mem_fact",
        title="Config location",
        content="The opencode config lives in the config folder.",
        kind="project_fact",
    )
    semantic.embed_record(
        index,
        fake_embedder,
        record_id="mem_procedure",
        title="Config procedure",
        content="The opencode config lives in the config folder.",
        kind="procedure",
    )
    fact = {
        "record_id": "mem_fact",
        "title": "Config location",
        "content": "The opencode config lives in the config folder.",
        "kind": "project_fact",
        "tags": [],
        "status": "candidate",
        "metadata": {},
        "block_membership": {"primary_block_id": "leaf"},
    }
    procedure = dict(fact, record_id="mem_procedure", kind="procedure")
    records = {item["record_id"]: item for item in (fact, procedure)}

    def loader(identifiers):
        return [records[identifier] for identifier in identifiers if identifier in records]

    ranked = semantic.rank_global(
        "opencode config",
        embedder=fake_embedder,
        index=index,
        limit=3,
        loader=loader,
        kinds=["procedure"],
    )
    assert [identifier for identifier, _score, _record in ranked] == ["mem_procedure"]


def test_procedure_entry_reports_its_freshness() -> None:
    from lethyx.memory import procedures as procedure_rules

    record = {
        "record_id": "mem_procedure",
        "title": "Restart the local service",
        "content": "Restart the MCP service after editing the source.",
        "kind": "procedure",
        "tags": ["lethyx"],
        "recorded_at": "2026-10-01T00:00:00Z",
        "metadata": {
            "steps": ["edit the source", "restart the service"],
            "verify": "the health call answers",
            "verified_at": "2026-09-01T00:00:00Z",
            "stale_after_days": 30,
        },
    }
    entry = procedure_rules.procedure_entry(record)
    assert entry["steps"] == ["edit the source", "restart the service"]
    assert entry["verify"] == "the health call answers"
    assert entry["freshness"] == "stale"
    assert entry["age_days"] > 30

    unverified = procedure_rules.procedure_entry(dict(record, metadata={}))
    assert unverified["freshness"] == "unverified"


def test_procedure_staleness_is_caused_by_the_environment() -> None:
    from lethyx.memory import procedures as procedure_rules

    descriptor = procedure_rules.current_descriptor(
        repo_root="/tmp/store", model="model-a", workspace_id="ws"
    )
    metadata = procedure_rules.stamp_metadata(
        {
            "verified_at": "2026-10-21T00:00:00Z",
            "stale_after_days": 365,
        },
        descriptor,
    )
    same = procedure_rules.procedure_freshness(metadata, descriptor)
    assert same["freshness"] == "fresh"

    moved = dict(descriptor, semantic_model="model-b")
    changed = procedure_rules.procedure_freshness(metadata, moved)
    assert changed["freshness"] == "stale"
    assert "semantic_model" in changed["reason"]


def test_calibration_measures_a_floor_between_relevant_and_irrelevant(
    tmp_path: Path,
) -> None:
    index = semantic.SemanticIndex(tmp_path / "embeddings.sqlite")
    embedder = FakeEmbedder()
    for position in range(4):
        semantic.embed_record(
            index,
            embedder,
            record_id=f"mem_{position}",
            title=f"Config note {position}",
            content="The opencode config lives in the config folder.",
            kind="project_fact",
        )
    records = [
        {
            "record_id": f"mem_{position}",
            "title": f"Config note {position}",
            "content": "The opencode config lives in the config folder.",
        }
        for position in range(4)
    ]
    measured = semantic.calibrate(records, index, embedder)
    assert measured["ok"] is True
    assert measured["relevant"]["min"] > measured["irrelevant"]["max"]
    assert 0.0 < measured["floor"] < 1.0
    assert measured["floor"] >= measured["default_floor"]

    path = semantic.store_calibration(tmp_path, "opencode", measured)
    assert path.exists()
    assert semantic.calibrated_floor(tmp_path, "opencode", embedder.model) == measured["floor"]
    # A different model has its own floor.
    assert semantic.calibrated_floor(tmp_path, "opencode", "other/model") is None


def test_vector_cache_reuses_the_parsed_index(
    fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    index = semantic.SemanticIndex(tmp_path / "embeddings.sqlite")
    semantic.embed_record(
        index,
        fake_embedder,
        record_id="mem_one",
        title="Config note",
        content="The opencode config lives in the config folder.",
    )
    first = semantic.cached_vectors(index, "fake/multilingual-v1", "default")
    second = semantic.cached_vectors(index, "fake/multilingual-v1", "default")
    assert first is second
    # A new record changes the version, so the cache is refreshed.
    semantic.embed_record(
        index,
        fake_embedder,
        record_id="mem_two",
        title="Second note",
        content="Another note about the config folder.",
    )
    third = semantic.cached_vectors(index, "fake/multilingual-v1", "default")
    assert third is not first
    assert len(third) == 2
