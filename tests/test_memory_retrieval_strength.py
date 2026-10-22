from pathlib import Path

from um_mem.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from um_mem.memory.backends.base import MemoryHit
from um_mem.memory.backends.contracts import block_record_to_memory_hit
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex


def _block(block_id: str, content: str, **overrides) -> MemoryBlockRecord:
    data = {
        "block_id": block_id,
        "revision_id": f"rev_{block_id}",
        "title": block_id,
        "content": content,
        "block_membership": RecordBlockMembership(
            primary_block_id="workspace_episodes",
            catalog_revision_id="test-catalog",
        ),
        "kind": "observation",
        "status": "accepted",
        "retention": "durable",
        "delivery_policy": "on_demand",
        "temperature": "warm",
        "authority": "advisory",
        "trust_level": "workspace_verified",
        "scope": "workspace",
        "workspace_id": "ws",
        "source_backend": "test",
        "verification_status": "verified",
        "verification_methods": ("test_run",),
        "promotion_decision": "accepted",
        "evidence_refs": (
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"retrieval-strength:{block_id}",
                produced_by="verifier",
                producer_id="retrieval-strength-test",
            ),
        ),
    }
    data.update(overrides)
    return MemoryBlockRecord(**data)


def test_sqlite_index_preserves_retrieval_score_on_returned_blocks(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    try:
        index.retain_block(_block("weak", "alpha filler"))
        index.retain_block(_block("strong", "alpha beta exact evidence"))

        rows = index.recall(
            {
                "query": "alpha beta",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "limit": 2,
            }
        )

        assert [row.block_id for row in rows] == ["strong", "weak"]
        assert rows[0].metadata["retrieval_score"] > rows[1].metadata["retrieval_score"]
        assert rows[0].metadata["retrieval_rank"] == 1
        assert rows[0].metadata["retrieval_backend"] == "test"

        hit = block_record_to_memory_hit(rows[0])
        assert hit.score == rows[0].metadata["retrieval_score"]
    finally:
        index.close()


def test_sqlite_index_selected_block_ids_bypass_weak_query_match(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    try:
        index.retain_block(
            _block(
                "selected",
                "the exact retained detail is stored here",
                block_membership=RecordBlockMembership(
                    primary_block_id="verified_lessons",
                    catalog_revision_id="catalog:test",
                ),
            )
        )

        rows = index.recall(
            {
                "query": "unrelated weak query",
                "workspace_id": "ws",
                "record_ids": ["selected"],
                "logical_block_ids": ["verified_lessons"],
                "scope_chain": ["workspace:ws"],
                "limit": 5,
            }
        )

        assert [row.block_id for row in rows] == ["selected"]
        assert rows[0].metadata["retrieval_score"] > 0
    finally:
        index.close()


def test_sqlite_index_selected_block_ids_bypass_recent_scan_window(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    try:
        index.retain_block(
            _block(
                "old-selected",
                "the explicitly selected architectural memory",
                block_membership=RecordBlockMembership(
                    primary_block_id="stable_project_facts",
                    catalog_revision_id="catalog:test",
                ),
                created_at="2020-01-01T00:00:00Z",
                updated_at="2020-01-01T00:00:00Z",
            )
        )
        for idx in range(5):
            index.retain_block(
                _block(
                    f"new-{idx}",
                    f"new unrelated record {idx}",
                    created_at=f"2025-01-0{idx + 1}T00:00:00Z",
                    updated_at=f"2025-01-0{idx + 1}T00:00:00Z",
                )
            )
        index._conn.execute(
            "UPDATE blocks SET updated_at=? WHERE block_id=?",
            ("2020-01-01T00:00:00Z", "old-selected"),
        )
        index._conn.execute(
            "UPDATE blocks SET updated_at=? WHERE block_id LIKE 'new-%'",
            ("2025-01-01T00:00:00Z",),
        )
        index._conn.commit()

        rows = index.recall(
            {
                "query": "unrelated weak query",
                "workspace_id": "ws",
                "record_ids": ["old-selected"],
                "logical_block_ids": ["stable_project_facts"],
                "scope_chain": ["workspace:ws"],
                "limit": 5,
            }
        )

        assert [row.block_id for row in rows] == ["old-selected"]
    finally:
        index.close()


def test_sqlite_index_passive_and_candidate_flags_are_explicit(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    try:
        index.retain_block(_block("active", "alpha active memory"))
        index.retain_block(
            _block(
                "archive",
                "alpha archived memory",
                retention="archive",
                temperature="cold",
            )
        )
        index.retain_block(
            _block(
                "candidate",
                "alpha candidate memory",
                status="candidate",
                trust_level="agent_reported",
                verification_status="unverified",
                verification_methods=(),
                promotion_decision="not_requested",
            )
        )

        default = index.recall(
            {
                "query": "alpha",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "limit": 10,
            }
        )
        archive = index.recall(
            {
                "query": "alpha",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "limit": 10,
                "source_horizon": "archive",
            }
        )
        candidates = index.recall(
            {
                "query": "alpha",
                "workspace_id": "ws",
                "scope_chain": ["workspace:ws"],
                "limit": 10,
                "source_horizon": "archive",
                "include_candidates": True,
            }
        )

        assert [row.block_id for row in default] == ["active"]
        assert {row.block_id for row in archive} == {"archive", "active"}
        assert {row.block_id for row in candidates} == {"candidate", "archive", "active"}
    finally:
        index.close()


class _FakeEvidenceBackend:
    def __init__(self, hits: list[MemoryHit]) -> None:
        self._hits = hits

    def recall_evidence(self, query: object) -> list[MemoryHit]:
        return list(self._hits)

    def health(self) -> dict[str, object]:
        return {"ok": True, "strict_ok": True, "native_ok": True}
