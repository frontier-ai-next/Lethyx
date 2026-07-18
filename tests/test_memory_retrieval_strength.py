from pathlib import Path

from lethyx.memory.backends.base import MemoryHit
from lethyx.memory.backends.composite import CompositeMemoryBackend, MemoryBackendBlock
from lethyx.memory.backends.contracts import MemoryBlockRecord, block_record_to_memory_hit
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex


def _block(block_id: str, content: str) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        title=block_id,
        content=content,
        kind="observation",
        lifecycle="active",
        trust_level="workspace_verified",
        workspace_id="ws",
        source_backend="test",
        verified=True,
    )


def test_sqlite_index_preserves_retrieval_score_on_returned_blocks(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    try:
        index.retain_block(_block("weak", "alpha filler"))
        index.retain_block(_block("strong", "alpha beta exact evidence"))

        rows = index.recall({"query": "alpha beta", "workspace_id": "ws", "limit": 2})

        assert [row.block_id for row in rows] == ["strong", "weak"]
        assert rows[0].metadata["retrieval_score"] > rows[1].metadata["retrieval_score"]
        assert rows[0].metadata["retrieval_rank"] == 1
        assert rows[0].metadata["retrieval_backend"] == "test"

        hit = block_record_to_memory_hit(rows[0])
        assert hit.score == rows[0].metadata["retrieval_score"]
    finally:
        index.close()


class _FakeEvidenceBackend:
    def __init__(self, hits: list[MemoryHit]) -> None:
        self._hits = hits

    def recall_evidence(self, query: object) -> list[MemoryHit]:
        return list(self._hits)

    def health(self) -> dict[str, object]:
        return {"ok": True, "strict_ok": True, "native_ok": True}


def test_composite_uses_rank_fusion_not_incompatible_raw_scores() -> None:
    active = _FakeEvidenceBackend(
        [MemoryHit(text="active exact evidence", source="active", score=0.05)]
    )
    passive = _FakeEvidenceBackend(
        [MemoryHit(text="passive lexical distractor", source="passive", score=1.0)]
    )
    backend = CompositeMemoryBackend(
        None,
        workspace_id="ws",
        blocks=[
            MemoryBackendBlock(
                block_id="active.block",
                backend_id="mempalace",
                backend=active,
                role="active",
                weight=1.0,
            ),
            MemoryBackendBlock(
                block_id="passive.block",
                backend_id="mnemosyne",
                backend=passive,
                role="passive",
                weight=0.9,
            ),
        ],
    )

    hits = backend.recall_evidence({"query": "evidence", "include_passive": True, "limit": 2})

    assert [hit.source for hit in hits] == ["active", "passive"]
    assert hits[0].metadata["memory_block_rank"] == 1
    assert hits[0].metadata["raw_backend_score"] == 0.05
    assert hits[0].metadata["memory_block_rrf_score"] > hits[1].metadata["memory_block_rrf_score"]
