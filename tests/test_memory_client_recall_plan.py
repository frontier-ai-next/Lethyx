from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from um_mem.api import MemoryClient
from um_mem.contracts import RecordBlockMembership
from um_mem.memory.backends.base import MemoryHit
from um_mem.memory.recall_plan import RecallPlan


class RecordingEvidenceBackend:
    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []

    def recall_evidence(self, query: Any) -> list[MemoryHit]:
        if isinstance(query, dict):
            payload = dict(query)
        elif is_dataclass(query):
            payload = asdict(query)
        else:
            payload = dict(getattr(query, "__dict__", {}) or {})
        self.queries.append(payload)
        return [
            MemoryHit(
                text="mnemosyne episodic answer",
                source="record-1",
                score=0.9,
                kind="observation",
                tags=["mnemosyne", "episodic"],
                metadata={
                    "block_id": "record-1",
                    "status": "accepted",
                    "retention": "durable",
                    "delivery_policy": "on_demand",
                    "temperature": "warm",
                    "authority": "advisory",
                    "trust_level": "workspace_verified",
                    "verification_status": "verified",
                    "verification_methods": ["test_run"],
                    "promotion_decision": "accepted",
                    "sensitivity": "internal",
                    "content_mode": "full",
                    "operational_risk": "low",
                    "evidence_refs": [
                        {
                            "ref_type": "test_run",
                            "ref_id": "record-1:recall",
                            "produced_by": "verifier",
                            "producer_id": "verifier:recall",
                        }
                    ],
                    "workspace_id": "alpha",
                    "block_membership": asdict(
                        RecordBlockMembership(
                            primary_block_id="workspace_episodes",
                            catalog_revision_id="catalog:test",
                        )
                    ),
                },
            )
        ]


def test_memory_client_recall_forwards_explicit_block_scope(tmp_path: Path) -> None:
    backend = RecordingEvidenceBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    hits = client.recall(
        "episodic memory",
        source_horizon="archive",
        include_candidates=True,
        scope_chain=["workspace:alpha"],
        logical_block_ids=["workspace_episodes"],
        record_ids=["record-1"],
    )

    assert hits[0].text == "mnemosyne episodic answer"
    payload = backend.queries[-1]
    assert payload["source_horizon"] == "archive"
    assert payload["include_candidates"] is True
    assert payload["scope_chain"] == ["workspace:alpha"]
    assert "workspace_episodes" in payload["logical_block_ids"]
    assert payload["record_ids"] == ["record-1"]


def test_memory_client_keeps_historical_query_time_on_returned_evidence(tmp_path: Path) -> None:
    client = MemoryClient(
        tmp_path, workspace_id="alpha", backend_instance=RecordingEvidenceBackend()
    )
    hits = client.recall(
        "calibration history",
        temporal_mode="historical",
        as_of="2020-01-15T00:00:00Z",
        scope_chain=["workspace:alpha"],
        logical_block_ids=["workspace_episodes"],
        record_ids=["record-1"],
    )
    assert len(hits) == 1
    assert hits[0].metadata["historical"] is True
    assert hits[0].metadata["as_of"] == "2020-01-15T00:00:00Z"
    assert hits[0].metadata["authority"] == "advisory"


def test_memory_client_forwards_known_as_of_without_losing_scope(tmp_path: Path) -> None:
    backend = RecordingEvidenceBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    hits = client.recall(
        "framework",
        known_as_of="2020-01-15T00:00:00Z",
        as_of="2019-12-01T00:00:00Z",
        logical_block_ids=["workspace_episodes"],
    )
    assert backend.queries[-1]["known_as_of"] == "2020-01-15T00:00:00Z"
    assert backend.queries[-1]["workspace_id"] == "alpha"
    assert hits[0].metadata["known_as_of"] == "2020-01-15T00:00:00Z"


def test_known_as_of_rejects_ambiguous_local_time() -> None:
    with pytest.raises(ValueError, match="requires a timezone"):
        RecallPlan.compile("framework", known_as_of="2020-01-15T00:00:00")


def test_memory_client_build_context_keeps_explicit_logical_scope_advisory(
    tmp_path: Path,
) -> None:
    backend = RecordingEvidenceBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    items = client.build_context(
        "mnemosyne episodic workspace memory",
        source_horizon="durable",
        logical_block_ids=["workspace_episodes"],
        max_tokens=512,
    )

    assert items[0]["authority"] == "advisory"
    assert items[0]["metadata"]["block_id"] == "record-1"
    payload = backend.queries[-1]
    assert payload["logical_block_ids"] == ["workspace_episodes"]
    assert payload["source_horizon"] == "durable"


def test_memory_client_recall_plan_argument_wins_over_logical_block_kwargs(tmp_path: Path) -> None:
    backend = RecordingEvidenceBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    plan = RecallPlan(
        query="planned query",
        workspace_id="alpha",
        logical_block_ids=("workspace_episodes",),
        source_horizon="archive",
    )

    client.recall(
        "ignored query",
        recall_plan=plan,
        logical_block_ids=["ignored"],
    )

    payload = backend.queries[-1]
    assert payload["query"] == "planned query"
    assert payload["logical_block_ids"] == ["workspace_episodes"]
    assert payload["source_horizon"] == "archive"


def test_recall_plan_zero_limit_disables_delivery(tmp_path: Path) -> None:
    backend = RecordingEvidenceBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    plan = RecallPlan(
        query="episodic memory",
        workspace_id="alpha",
        logical_block_ids=("workspace_episodes",),
        limit=0,
    )

    hits = client.recall("episodic memory", recall_plan=plan)

    assert plan.limit == 0
    assert hits == []
