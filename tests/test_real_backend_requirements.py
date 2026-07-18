from pathlib import Path
from typing import Any

import pytest

from lethyx.memory.backends.hindsight import HindsightBackend
from lethyx.memory.backends.mempalace import MemPalaceBackend
from lethyx.memory.backends.mnemosyne import MnemosyneBackend
from lethyx.memory.backends.contracts import MemoryBlockRecord
from lethyx.memory.hindsight.config import HindsightConfig
from lethyx.memory.hindsight.errors import HindsightUnavailableError


def _block(block_id: str = "b1") -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        content="Native backend must store this record.",
        title="native-required",
        kind="observation",
        lifecycle="active",
        trust_level="workspace_verified",
        scope="run_scoped",
        tier="warm",
        surface="supplemental_evidence",
        workspace_id="ws",
        run_id="run",
        phase_id="phase",
        producer="test",
        tags=("native",),
        metadata={"evidence_id": block_id},
        verified=True,
    )


class _RecordingIndex:
    def __init__(self) -> None:
        self.retained: list[MemoryBlockRecord] = []

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        self.retained.append(record)
        return {"ok": True, "block_id": record.block_id, "backend": "index"}


class _RaisingPalace:
    def add(self, **kwargs: Any) -> str:
        raise RuntimeError("native palace down")


class _WorkingPalace:
    def add(self, **kwargs: Any) -> str:
        return "native-id"


def test_mempalace_retain_block_fails_closed_when_native_write_fails() -> None:
    backend = MemPalaceBackend.__new__(MemPalaceBackend)
    backend._workspace_id = "ws"
    backend._index = _RecordingIndex()
    backend._palace = _RaisingPalace()

    with pytest.raises(RuntimeError, match="native palace down"):
        backend.retain_block(_block())

    assert backend._index.retained == []


def test_mempalace_ignores_benchmark_skip_native_flag() -> None:
    backend = MemPalaceBackend.__new__(MemPalaceBackend)
    backend._workspace_id = "ws"
    backend._index = _RecordingIndex()
    backend._palace = _WorkingPalace()
    record = _block("skip-flag")
    record.metadata["memory_bench_skip_native_mempalace"] = True

    result = backend.retain_block(record)

    assert result["ok"] is True
    assert result.get("native_skipped") is not True
    assert backend._index.retained[0].block_id == "skip-flag"


def test_hindsight_config_default_mode_names_hindsight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LETHYX_BACKEND", raising=False)
    monkeypatch.delenv("LETHYX_MEMORY_BACKEND", raising=False)

    assert HindsightConfig.from_env().backend_mode == "hindsight"


def test_hindsight_selected_backend_requires_native_enabled(tmp_path: Path) -> None:
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=False),
    )

    with pytest.raises(HindsightUnavailableError, match="disabled"):
        backend.retain_block(_block())

    health = backend.health()
    assert health["ok"] is False
    assert health["native_ok"] is False
    assert health["effective_mode"] == "unavailable"


def test_mnemosyne_selected_backend_requires_native_enabled(tmp_path: Path) -> None:
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=False)

    with pytest.raises(RuntimeError, match="native Mnemosyne"):
        backend.retain_block(_block())

    health = backend.health()
    assert health["ok"] is False
    assert health["native_ok"] is False
    assert health["effective_mode"] == "unavailable"
