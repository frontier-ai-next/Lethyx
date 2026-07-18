from pathlib import Path
from typing import Any

from lethyx.memory.backends.composite import CompositeMemoryBackend, MemoryBackendBlock
from lethyx.memory.backends.contracts import MemoryBlockRecord


def _record(block_id: str, content: str = "kept record") -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        content=content,
        title="record",
        kind="observation",
        lifecycle="active",
        trust_level="workspace_verified",
        scope="workspace",
        tier="warm",
        surface="supplemental_evidence",
        workspace_id="ws",
        run_id="",
        phase_id="",
        producer="test",
        source_backend="mempalace",
        tags=("test",),
        metadata={},
        verified=True,
    )


class _GoodBackend:
    def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
        return [_record("good")]

    def health(self) -> dict[str, Any]:
        return {"ok": True, "strict_ok": True, "native_ok": True}


class _FailingBackend:
    def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
        raise RuntimeError("child recall down")

    def health(self) -> dict[str, Any]:
        return {"ok": True, "strict_ok": True, "native_ok": True}


def test_composite_recall_reports_degraded_child_backend(tmp_path: Path) -> None:
    backend = CompositeMemoryBackend(
        tmp_path,
        workspace_id="ws",
        blocks=[
            MemoryBackendBlock("good.active", "mempalace", _GoodBackend()),
            MemoryBackendBlock("bad.passive", "hindsight", _FailingBackend()),
        ],
    )

    rows = backend.recall({"query": "anything", "limit": 10, "include_passive": True})

    assert [row.block_id for row in rows] == ["good"]
    diagnostics = backend.last_recall_diagnostics()
    assert diagnostics["degraded"] is True
    assert diagnostics["errors"] == [
        {
            "memory_block_id": "bad.passive",
            "backend": "hindsight",
            "operation": "recall",
            "error": "child recall down",
        }
    ]
    assert backend.health()["last_recall"]["degraded"] is True
