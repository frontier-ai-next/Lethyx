from dataclasses import replace
from pathlib import Path

import pytest

from lethyx.memory.backends.query_policy import merge_hybrid_block_results
from lethyx.memory.backends.mempalace import MemPalaceBackend, _record_partition_identity
from lethyx.memory.recall_plan import RecallPlan
from test_native_candidate_sidecar_recall import (
    _mempalace_backend,
    _record,
    _retain_projected_sidecar,
)


def test_retrieval_mode_round_trip_and_closed_boundary() -> None:
    plan = RecallPlan.compile("a date", logical_block_ids=("leaf",), retrieval_mode="hybrid")
    assert RecallPlan.from_mapping(plan.to_backend_payload()).retrieval_mode == "hybrid"
    assert RecallPlan.compile("a date").retrieval_mode == "semantic"
    for mode in ("lexical", True, None):
        with pytest.raises(ValueError, match="retrieval_mode"):
            RecallPlan.compile("a date", retrieval_mode=mode)
    for fields in (
        {"record_ids": ("record",)},
        {"temporal_mode": "historical"},
        {"as_of": "2026-01-01"},
    ):
        with pytest.raises(ValueError, match="hybrid"):
            RecallPlan.compile("a date", retrieval_mode="hybrid", **fields)


def test_rrf_qualifies_dense_scores_and_lexical_partition_ranks() -> None:
    a = _record("a", "same text", primary_block_id="a-leaf")
    b = _record("b", "same text", primary_block_id="b-leaf")
    c = _record("c", "date and number", primary_block_id="a-leaf")
    dense = [("a-leaf", [(a, 0.4), (c, 0.3)]), ("b-leaf", [(b, 0.9)])]
    lexical = [("b-leaf", [(b, None)]), ("a-leaf", [(c, None), (a, None)])]
    hits = merge_hybrid_block_results(dense, lexical, limit=3)
    assert [hit.block_id for hit in hits] == ["b", "c", "a"]
    assert hits[0].metadata["channel_ranks"] == {"semantic": 1, "lexical": 2}
    assert hits[1].metadata["channel_ranks"] == {"semantic": 3, "lexical": 1}
    assert hits[0].metadata["retrieval_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert hits[0].metadata["native_score_mode"] == "rrf"
    assert len(hits) == 3  # Equal text is not equal identity.
    reordered = merge_hybrid_block_results(list(reversed(dense)), list(reversed(lexical)), limit=3)
    assert [hit.block_id for hit in reordered] == [hit.block_id for hit in hits]


def test_rrf_deduplicates_identity_and_revision_not_content() -> None:
    record = _record("a", "fact")
    revised = replace(record, revision_id="revision:new")
    hits = merge_hybrid_block_results(
        [("leaf", [(record, 0.9), (record, 0.9), (revised, 0.8)])],
        [("leaf", [(record, None)])],
        limit=10,
    )
    assert len(hits) == 2
    assert hits[0].metadata["channel_ranks"] == {"semantic": 1, "lexical": 1}


def test_hybrid_validates_all_candidates_before_global_limit(tmp_path: Path) -> None:
    records = [_record("a", "irrelevant"), _record("b", "exact date 2023-04-12")]
    backend = _mempalace_backend(tmp_path, records)
    _retain_projected_sidecar(backend, records)
    native = backend._native
    native.lexical_rows = lambda **kwargs: [dict(native._rows[1])]
    hits = backend.recall(
        {
            "query": "exact date",
            "scope_chain": ["workspace:ws"],
            "logical_block_ids": ["workspace_episodes"],
            "retrieval_mode": "hybrid",
            "limit": 1,
        }
    )
    assert [hit.block_id for hit in hits] == ["b"]
    assert native.calls[-1]["limit"] == 64
    assert hits[0].metadata["channel_ranks"] == {"semantic": 2, "lexical": 1}


def test_hybrid_fts_failure_is_explicit(tmp_path: Path) -> None:
    record = _record("a", "fact")
    backend = _mempalace_backend(tmp_path, [record])
    _retain_projected_sidecar(backend, [record])

    def failed(**kwargs):
        raise RuntimeError("mempalace_fulltext_index_failed")

    backend._native.lexical_rows = failed
    with pytest.raises(RuntimeError, match="fulltext_index_failed"):
        backend.recall(
            {
                "query": "fact",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["workspace_episodes"],
                "retrieval_mode": "hybrid",
            }
        )


def test_real_native_fts_tail_scope_and_restart(tmp_path: Path) -> None:
    record = _record(
        "tail", "ordinary background " * 400 + " Invoice ZXR48212: 74 kilograms on 2023-04-12."
    )
    foreign = replace(_record("foreign", "ZXR48212 is 999 kilograms"), profile_id="other")
    elsewhere = _record("elsewhere", "ZXR48212 is 888 kilograms", primary_block_id="other-leaf")
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="ws")
    try:
        for row in (record, foreign, elsewhere):
            result = backend.retain_block(row)
            if row.block_id == "tail":
                assert result["embedding_diagnostics"]["window_exceeded"]
                assert result["embedding_diagnostics"]["token_count"] > 256
    finally:
        backend.close()
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="ws")
    try:
        hits = backend.recall(
            {
                "query": "ZXR48212 2023-04-12",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["workspace_episodes"],
                "retrieval_mode": "hybrid",
                "limit": 1,
            }
        )
        assert [hit.block_id for hit in hits] == ["tail"]
        assert hits[0].content == record.content
        assert hits[0].metadata["channel_ranks"]["lexical"] == 1
        assert backend.last_recall_trace["fusion"] == "rrf"
    finally:
        backend.close()


def test_long_source_semantic_windows_recover_tail_after_reopen(tmp_path: Path) -> None:
    source = "ordinary background " * 400 + " Invoice ZXR48212: 74 kilograms."
    record = _record("window-tail", source)
    backend = MemPalaceBackend(repo_root=tmp_path, workspace_id="ws")
    try:
        assert backend.retain_block(record)["saved"] is True
        partition = _record_partition_identity(record)
        native = backend._native.collection("palace.idea", partition=partition)
        rows = native.get(include=["documents", "metadatas"])
        windows = [
            (document, metadata)
            for document, metadata in zip(rows.documents, rows.metadatas, strict=True)
            if metadata and metadata.get("derived_window") == 1
        ]
        assert len(windows) > 1
        for document, metadata in windows:
            assert document == source.encode("utf-8")[
                metadata["window_start_byte"] : metadata["window_end_byte"]
            ].decode("utf-8")
    finally:
        backend.close()
    reopened = MemPalaceBackend(repo_root=tmp_path, workspace_id="ws")
    try:
        native_top = reopened._native.rows(
            query="Invoice ZXR48212 74 kilograms",
            limit=1,
            partition=_record_partition_identity(record),
        )
        assert ":window:" in native_top[0]["id"]
        assert "ZXR48212" in native_top[0]["content"]
        hits = reopened.recall(
            {
                "query": "Invoice ZXR48212 74 kilograms",
                "scope_chain": ["workspace:ws"],
                "logical_block_ids": ["workspace_episodes"],
                "retrieval_mode": "semantic",
                "limit": 1,
            }
        )
        assert [hit.block_id for hit in hits] == [record.block_id]
        assert hits[0].content == source
    finally:
        reopened.close()
