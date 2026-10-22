from pathlib import Path
from typing import Any

import pytest

from um_mem.api import MemoryClient
from um_mem.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from um_mem.memory import mcp_facade
from um_mem.memory.backends.query_policy import filter_records
from um_mem.memory.recall_plan import RecallPlan


class _CapturingBackend:
    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []

    def recall_evidence(self, query: dict[str, Any]) -> list[Any]:
        self.queries.append(dict(query))
        return []


def _record(block_id: str, **overrides: Any) -> MemoryBlockRecord:
    values: dict[str, Any] = {
        "block_id": block_id,
        "content": f"evidence from {block_id}",
        "title": block_id,
        "status": "accepted",
        "retention": "durable",
        "delivery_policy": "on_demand",
        "temperature": "warm",
        "authority": "advisory",
        "trust_level": "workspace_verified",
        "scope": "workspace",
        "verification_status": "verified",
        "verification_methods": ("test_run",),
        "promotion_decision": "accepted",
        "workspace_id": "alpha",
        "block_membership": RecordBlockMembership(
            primary_block_id="code_resources",
            catalog_revision_id="catalog:test",
        ),
        "evidence_refs": (
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"query-typing:{block_id}",
                produced_by="verifier",
                producer_id="query-typing-test",
            ),
        ),
        "valid_from": "2020-01-01T00:00:00Z",
        "valid_to": "2099-01-01T00:00:00Z",
    }
    values.update(overrides)
    return MemoryBlockRecord(**values)


@pytest.mark.parametrize(
    ("query", "kwargs", "expected"),
    [
        ("record", {"record_ids": ("record-1",)}, "exact_lookup"),
        ("inspect src/parser.py", {"paths": ("src/parser.py",)}, "exact_lookup"),
        ("RecallPlan.compile", {"symbols": ("RecallPlan.compile",)}, "exact_lookup"),
        ("inspect src/parser.py", {}, "broad_recall"),
        ("RecallPlan.compile", {}, "broad_recall"),
        ("what changed yesterday?", {}, "broad_recall"),
        ("what changed yesterday?", {"temporal_mode": "historical"}, "temporal"),
        ("summarize and compare decisions", {}, "broad_recall"),
        ("what should I implement next?", {}, "broad_recall"),
    ],
)
def test_recall_plan_uses_only_explicit_structural_query_types(
    query: str,
    kwargs: dict[str, Any],
    expected: str,
) -> None:
    plan = RecallPlan.compile(query, **kwargs)

    assert plan.query_type == expected
    assert plan.evidence_roles == ()
    assert plan.requirements == ()


def test_post_backend_filtering_is_safety_only_and_preserves_backend_order() -> None:
    first = _record("backend-first")
    second = _record("backend-second")
    records = [
        first,
        _record("foreign", workspace_id="beta"),
        _record("protected", sensitivity="secret"),
        _record(
            "retracted",
            status="retracted",
            retention="archive",
            delivery_policy="hidden",
        ),
        second,
    ]

    result = filter_records(
        records,
        {
            "query": "second should not rerank first",
            "workspace_id": "alpha",
            "logical_block_ids": ["code_resources"],
            "source_horizon": "durable",
            "scope_chain": ["workspace:alpha"],
            "temporal_mode": "current",
        },
    )

    assert result == [first, second]


def test_memoryclient_and_mcp_forward_the_same_agent_selected_blocks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = _CapturingBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    query = "inspect parser"
    selected = ["code_resources"]

    direct = client.compile_recall_plan(query, limit=4, logical_block_ids=selected)
    client.build_context(query, limit=4, logical_block_ids=selected)
    assert backend.queries[-1]["logical_block_ids"] == selected

    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)
    payload = mcp_facade.um_mem_build_context(
        query=query,
        workspace_id="alpha",
        repo_root=str(tmp_path),
        limit=4,
        logical_block_ids=selected,
    )

    assert payload["ok"] is True
    assert payload["diagnostics"]["logical_block_ids"] == selected
    assert direct.logical_block_ids == tuple(selected)


def test_product_surfaces_do_not_construct_a_parallel_semantic_router() -> None:
    root = Path(__file__).parents[1] / "src" / "um_mem"

    assert not (root / "memory" / "blocks" / "selection.py").exists()
    assert not (root / "memory" / "coverage_selection.py").exists()
    assert "compile_recall_plan_for_query" not in (root / "api.py").read_text(encoding="utf-8")
