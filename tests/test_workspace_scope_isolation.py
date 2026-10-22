from pathlib import Path

import pytest

from um_mem import MemoryClient
from um_mem.contracts import EvidenceRef, MemoryBlockRecord


def _evidence(name: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="verification_report",
            ref_id=name,
            produced_by="verifier",
            producer_id="verifier:workspace-isolation",
        ),
    )


def _record(
    block_id: str,
    workspace_id: str,
    *,
    scope: str = "workspace",
    project_id: str = "",
    user_id: str = "",
) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        content=f"isolated fact {block_id}",
        title=block_id,
        kind="observation",
        status="accepted",
        retention="durable",
        delivery_policy="on_demand",
        temperature="warm",
        authority="advisory",
        trust_level="workspace_verified",
        scope=scope,
        project_id=project_id,
        user_id=user_id,
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
        workspace_id=workspace_id,
        evidence_refs=_evidence(block_id),
    )


class _LeakyBackend:
    def __init__(self) -> None:
        self.records = [
            _record("fact-a", "workspace-a"),
            _record("fact-b", "workspace-b"),
            _record("project-b", "workspace-b", scope="project", project_id="project-a"),
            _record("global-b", "workspace-b", scope="user_global", user_id="user-1"),
        ]

    def recall_evidence(self, _query: dict[str, object]) -> list[MemoryBlockRecord]:
        return list(self.records)


def _ids(client: MemoryClient) -> tuple[str, ...]:
    return tuple(
        hit.metadata["block_id"]
        for hit in client.recall("isolated fact", search_all=True, limit=10)
    )


def test_a_to_b_to_a_recall_never_contaminates_foreign_workspace(tmp_path: Path) -> None:
    backend = _LeakyBackend()
    client_a = MemoryClient(tmp_path, workspace_id="workspace-a", backend_instance=backend)
    client_b = MemoryClient(tmp_path, workspace_id="workspace-b", backend_instance=backend)

    assert _ids(client_a) == ("fact-a",)
    assert _ids(client_b) == ("fact-b",)
    assert _ids(client_a) == ("fact-a",)


def test_only_a_user_global_scope_may_cross_the_workspace_boundary(tmp_path: Path) -> None:
    backend = _LeakyBackend()
    client = MemoryClient(
        tmp_path,
        workspace_id="workspace-a",
        project_id="project-a",
        user_id="user-1",
        backend_instance=backend,
    )

    ids = {
        hit.metadata["block_id"]
        for hit in client.recall(
            "isolated fact",
            search_all=True,
            scope_chain=(
                "workspace:workspace-a",
                "project:project-a",
                "user_global:user-1",
            ),
            limit=10,
        )
    }

    assert ids == {"fact-a", "global-b"}


def test_project_scope_keeps_the_physical_workspace_filter() -> None:
    from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex

    project_chain = {"scope_chain": ["workspace:alpha", "project:demo"]}
    user_global_chain = {"scope_chain": ["workspace:alpha", "user_global:u1"]}

    assert BackendSQLiteIndex._workspace_scope_filter(project_chain) == ("alpha",)
    assert BackendSQLiteIndex._workspace_scope_filter(user_global_chain) == ()


def test_cross_workspace_isolation_is_declared_only_for_user_global_blocks() -> None:
    from um_mem.memory.blocks.models import MemoryBlock

    with pytest.raises(ValueError, match="user_global"):
        MemoryBlock(
            block_id="leaky",
            block_type="semantic_leaf",
            temperature="warm",
            scope="workspace",
            authority="advisory",
            delivery_policy="on_demand",
            retention="durable",
            name="Leaky block",
            parent_id="root",
            agent_writable=True,
            index_only=False,
            owner="test",
            producer="test",
            consumers=("memory_agent",),
            recall_path="hybrid_recall",
            lifecycle_policy={
                "initial_status": "active",
                "mutable": True,
                "terminal_statuses": ("archived",),
            },
            isolation_policy={"dimension": "workspace", "cross_workspace": True},
            token_budget=128,
        )
