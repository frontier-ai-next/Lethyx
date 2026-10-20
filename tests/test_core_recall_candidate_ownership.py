from pathlib import Path
from typing import Any

import pytest

from lethyx.api import MemoryClient
from lethyx.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from lethyx.memory.backends.contracts import block_record_to_memory_hit


def _record(
    block_id: str,
    content: str,
    *,
    logical_block_id: str = "stable_project_facts",
) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        content=content,
        title=block_id,
        status="accepted",
        retention="durable",
        delivery_policy="on_demand",
        temperature="warm",
        authority="advisory",
        trust_level="workspace_verified",
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
        scope="workspace",
        workspace_id="alpha",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"candidate-order:{block_id}",
                produced_by="verifier",
                producer_id="candidate-order-test",
            ),
        ),
        block_membership=RecordBlockMembership(
            primary_block_id=logical_block_id,
            catalog_revision_id="catalog:test",
        ),
    )


class _CandidateBackend:
    def __init__(self, records: list[MemoryBlockRecord]) -> None:
        self.records = records
        self.recall_calls: list[dict[str, Any]] = []

    def recall_evidence(self, payload: dict[str, Any]) -> list[Any]:
        self.recall_calls.append(dict(payload))
        return [block_record_to_memory_hit(record) for record in self.records]


@pytest.mark.parametrize("backend_name", ["mempalace", "hindsight", "mnemosyne"])
def test_scoped_recall_preserves_each_backend_order_without_hidden_filtering(
    tmp_path: Path,
    backend_name: str,
) -> None:
    correct = _record(f"{backend_name}-correct", "opaque correct evidence")
    second = _record(f"{backend_name}-second", "opaque secondary evidence")
    backend = _CandidateBackend([correct, second])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    with pytest.raises(ValueError, match="logical_block_ids or search_all"):
        client.recall("opaque lookup", include_candidates=True)

    hits = client.recall(
        "opaque lookup",
        logical_block_ids=("stable_project_facts",),
        include_candidates=True,
        limit=2,
    )

    assert [hit.metadata["block_id"] for hit in hits] == [correct.block_id, second.block_id]
    assert backend.recall_calls == [
        {
            **backend.recall_calls[0],
            "limit": 2,
            "logical_block_ids": ["stable_project_facts"],
        }
    ]


def test_search_all_is_explicit_and_still_preserves_backend_order(tmp_path: Path) -> None:
    first = _record("first", "first backend result")
    second = _record("second", "second backend result")
    backend = _CandidateBackend([first, second])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    hits = client.recall("unknown location", search_all=True, limit=2)

    assert [hit.metadata["block_id"] for hit in hits] == ["first", "second"]
    assert backend.recall_calls[0]["search_all"] is True
    assert backend.recall_calls[0]["logical_block_ids"] == []


def test_post_backend_deduplication_keeps_the_first_ranked_record(tmp_path: Path) -> None:
    first = _record("same", "backend first revision")
    duplicate = _record("same", "backend lower-ranked duplicate")
    backend = _CandidateBackend([first, duplicate])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    hits = client.recall(
        "opaque lookup",
        logical_block_ids=("stable_project_facts",),
        limit=2,
    )

    assert [hit.text for hit in hits] == ["backend first revision"]
