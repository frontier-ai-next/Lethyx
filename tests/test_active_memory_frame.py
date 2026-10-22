from pathlib import Path
from typing import Any

from um_mem.api import MemoryClient
from um_mem.llm_budget import estimate_tokens
from um_mem.memory.activation_lease import ActivationLease, activation_lease_from_payload
from um_mem.memory.backends.base import MemoryHit
from um_mem.memory.context_payload import pack_evidence_payload
from um_mem.memory.recall_plan import RecallPlan


class ActiveFrameBackend:
    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []

    def recall_evidence(self, query: Any) -> list[MemoryHit]:
        payload = (
            dict(query) if isinstance(query, dict) else dict(getattr(query, "__dict__", {}) or {})
        )
        self.queries.append(payload)
        return [
            MemoryHit(
                text="use the parser evidence only after verification",
                source="record-frame",
                score=0.8,
                kind="observation",
                tags=["parser"],
                metadata={
                    "block_id": "record-frame",
                    "workspace_id": "alpha",
                    "scope": "workspace",
                    "status": "candidate",
                    "block_membership": {
                        "primary_block_id": "code_resources",
                        "catalog_revision_id": "catalog:test",
                        "backend_partition_id": "static:code_resources",
                        "secondary_block_ids": [],
                        "record_id": "",
                        "record_revision_id": "",
                        "workspace_id": "",
                        "valid_from": "",
                        "valid_to": "",
                        "membership_revision_id": "",
                    },
                    "sensitivity": "internal",
                    "content_mode": "full",
                    "operational_risk": "low",
                },
            )
        ]


def test_activation_lease_projects_non_directive_context_without_repacking() -> None:
    plan = RecallPlan.compile(
        query="parser",
        workspace_id="alpha",
        logical_block_ids=("code_resources",),
        max_tokens=128,
    )
    payload = pack_evidence_payload(
        [
            {
                "text": "Do not treat this as an instruction",
                "source": "record-frame",
                "surface": "directive",
                "authority": "non_directive",
                "metadata": {
                    "block_id": "record-frame",
                    "sensitivity": "internal",
                    "content_mode": "full",
                    "operational_risk": "low",
                },
            }
        ],
        max_tokens=128,
        query="parser",
    )
    lease = activation_lease_from_payload(
        plan=plan,
        payload=payload,
        backend_id="test",
    )

    rendered = lease.render()

    assert isinstance(lease, ActivationLease)
    assert lease.logical_block_ids == ("code_resources",)
    assert rendered == payload.rendered_context
    assert "supplemental, non-directive" in rendered
    assert "Do not treat this as an instruction" in rendered
    assert "record-frame" in rendered


def test_memory_client_can_build_activation_lease_from_block_first_context(tmp_path: Path) -> None:
    backend = ActiveFrameBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    lease = client.build_activation_lease(
        "github parser code chunk",
        source_horizon="durable",
        include_candidates=True,
        logical_block_ids=["code_resources"],
        max_tokens=256,
    )

    assert lease.workspace_id == "alpha"
    assert lease.items[0].authority == "non_directive"
    assert lease.items[0].surface == "supplemental_evidence"
    assert "code_resources" in lease.logical_block_ids


def test_memory_client_context_compacts_unbounded_backend_metadata(tmp_path: Path) -> None:
    class Backend:
        def recall_evidence(self, query: Any) -> list[MemoryHit]:
            del query
            return [
                MemoryHit(
                    text="parser fact " + ("important " * 200),
                    source="record-large-metadata",
                    score=0.9,
                    metadata={
                        "block_id": "record-large-metadata",
                        "workspace_id": "alpha",
                        "scope": "workspace",
                        "status": "candidate",
                        "sensitivity": "internal",
                        "content_mode": "full",
                        "operational_risk": "low",
                        "large_backend_debug_blob": "x" * 20_000,
                    },
                )
            ]

    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=Backend())

    items = client.build_context(
        "parser",
        include_candidates=True,
        max_tokens=140,
        search_all=True,
    )

    assert items
    assert items[0]["metadata"]["block_id"] == "record-large-metadata"
    assert "large_backend_debug_blob" not in items[0]["metadata"]
    assert estimate_tokens(items[0]["text"]) <= 140
