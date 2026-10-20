from pathlib import Path

import pytest

from lethyx.api import MemoryClient
from lethyx.memory import mcp_facade
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex


def _client(tmp_path: Path) -> tuple[MemoryClient, BackendSQLiteIndex]:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    return (
        MemoryClient(
            tmp_path,
            workspace_id="alpha",
            session_id="session:test",
            task_id="task:test",
            subtask_id="subtask:test",
            backend_instance=backend,
        ),
        backend,
    )


def test_normal_remember_materializes_the_explicit_logical_block(tmp_path: Path) -> None:
    client, backend = _client(tmp_path)
    try:
        saved = client.remember(
            "The parser repair requires a verified regression test.",
            title="Parser regression lesson",
            kind="lesson",
            requested_block_id="artifact_claims",
            evidence_refs=[
                {
                    "ref_type": "test_run",
                    "ref_id": "test-parser-regression",
                    "produced_by": "verifier",
                    "producer_id": "verifier:parser-regression",
                }
            ],
        )
        record = backend.get_block(saved.memory_id, workspace_id="alpha")

        assert saved.saved is True
        assert record is not None
        assert record.block_membership is not None
        assert record.block_membership.primary_block_id == "artifact_claims"
        assert record.metadata["route_decision_reason"] == "requested_block_policy"
    finally:
        backend.close()


def test_selected_block_drives_both_write_and_recall(tmp_path: Path) -> None:
    client, backend = _client(tmp_path)
    try:
        saved = client.remember(
            "Repository test command: uv run --group test executes the full suite.",
            title="Repository testing command",
            kind="project_fact",
            requested_block_id="stable_project_facts",
            evidence_refs=[
                {
                    "ref_type": "artifact",
                    "ref_id": "pyproject.toml",
                    "produced_by": "harness",
                    "producer_id": "harness:pyproject",
                }
            ],
        )
        items = client.build_context(
            "Which test command runs the repository suite?",
            logical_block_ids=["stable_project_facts"],
            include_candidates=True,
            max_tokens=256,
        )

        assert saved.saved is True
        assert "uv run --group test" in items[0]["text"]
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("kind", "selected_block"),
    [
        ("project_state", "mutable_project_state"),
        ("decision", "decisions_and_contracts"),
        ("artifact_claim", "artifact_claims"),
        ("research_finding", "execution_working_state"),
    ],
)
def test_agent_selected_generic_semantic_and_working_blocks(
    tmp_path: Path,
    kind: str,
    selected_block: str,
) -> None:
    client, backend = _client(tmp_path)
    try:
        saved = client.remember(
            f"Evidence-backed {kind} record.",
            kind=kind,
            requested_block_id=selected_block,
            evidence_refs=[
                {
                    "ref_type": "artifact",
                    "ref_id": f"artifact:{kind}",
                    "produced_by": "harness",
                    "producer_id": "harness:explicit-block-write",
                }
            ],
        )
        record = backend.get_block(saved.memory_id, workspace_id="alpha")

        assert record is not None
        assert record.block_membership is not None
        assert record.block_membership.primary_block_id == selected_block
    finally:
        backend.close()


def test_mcp_remember_requires_and_preserves_selected_block(monkeypatch, tmp_path: Path) -> None:
    client, backend = _client(tmp_path)
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)
    try:
        saved = mcp_facade.lethyx_remember(
            "The MCP path preserves a verified regression finding.",
            workspace_id="alpha",
            backend="test",
            repo_root=str(tmp_path),
            kind="lesson",
            requested_block_id="artifact_claims",
            evidence_refs=[
                {
                    "ref_type": "test_run",
                    "ref_id": "test-mcp-regression",
                    "produced_by": "verifier",
                    "producer_id": "verifier:mcp-regression",
                }
            ],
        )
        record = backend.get_block(saved["memory_id"], workspace_id="alpha")

        assert saved["saved"] is True
        assert record is not None
        assert record.block_membership is not None
        assert record.block_membership.primary_block_id == "artifact_claims"
    finally:
        backend.close()
