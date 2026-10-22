from pathlib import Path
from typing import Any

from um_mem.contracts import EvidenceRef, MemoryBlockRecord
from um_mem.memory.selection import MemorySelectionPolicy, SelectionContext

ROOT = Path(__file__).resolve().parents[1]


def _block(**overrides: Any) -> MemoryBlockRecord:
    base = {
        "block_id": "mem-1",
        "content": "alpha evidence",
        "title": "Alpha",
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
        "evidence_refs": (
            EvidenceRef(
                ref_type="test_run",
                ref_id="inv",
                produced_by="verifier",
                producer_id="verifier:invariant",
            ),
        ),
        "metadata": {},
    }
    base.update(overrides)
    return MemoryBlockRecord(**base)


class _RecordingBackend:
    def __init__(self, rows: list[MemoryBlockRecord] | None = None) -> None:
        self.rows = rows or []
        self.writes: list[MemoryBlockRecord] = []

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        self.writes.append(record)
        return {"ok": True, "block_id": record.block_id}

    def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
        del query
        return list(self.rows)


def test_selection_policy_is_authority_gated_not_lane_gated() -> None:
    policy = MemorySelectionPolicy()
    ctx = SelectionContext(query="alpha", scope_chain=("workspace:alpha",))

    assert policy.eligible_for_recall(_block(), ctx)
    assert not policy.eligible_for_recall(_block(status="deprecated"), ctx)
    assert policy.eligible_for_recall(_block(temperature="cold"), ctx)
    assert not policy.eligible_for_recall(_block(retention="archive"), ctx)
    assert policy.eligible_for_recall(
        _block(retention="archive"),
        SelectionContext(query="alpha", scope_chain=("workspace:alpha",), source_horizon="archive"),
    )
    assert not policy.eligible_for_directive(_block(), ctx)
    assert policy.eligible_for_directive(_block(authority="directive"), ctx)
