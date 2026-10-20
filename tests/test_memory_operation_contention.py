from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import monotonic

import pytest
from filelock import FileLock

from lethyx import MemoryBusyError
from lethyx.memory.backends.mempalace import MemPalaceBackend


@pytest.mark.parametrize("gate", ["thread", "file"])
def test_busy_memory_returns_within_wait_budget_and_remains_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    monkeypatch.setenv("LETHYX_MEMPALACE_LOCK_TIMEOUT_SECONDS", "0.05")
    backend = MemPalaceBackend(tmp_path, workspace_id="contention")
    entered = Event()
    release = Event()

    def hold_gate() -> None:
        lock = (
            backend.operation_gate()
            if gate == "thread"
            else FileLock(backend._native_operation_gate.file_lock.lock_file, timeout=1)
        )
        with lock:
            entered.set()
            release.wait(timeout=2)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            holder = executor.submit(hold_gate)
            assert entered.wait(timeout=2)
            started = monotonic()
            try:
                with pytest.raises(MemoryBusyError, match="memory_operation_lock_timeout"):
                    backend.expire_scope("session", "finished-session")
                assert monotonic() - started < 1.5
            finally:
                release.set()
                holder.result(timeout=2)
        with backend.operation_gate():
            assert backend.expire_scope("session", "finished-session") == 0
    finally:
        backend.close()


@pytest.mark.parametrize("value", ["", "0", "-1", "nan", "inf", "3601"])
def test_memory_wait_budget_rejects_unbounded_or_invalid_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LETHYX_MEMPALACE_LOCK_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="LETHYX_MEMPALACE_LOCK_TIMEOUT_SECONDS"):
        MemPalaceBackend(tmp_path, workspace_id="invalid-budget")
    assert not list(tmp_path.iterdir())
