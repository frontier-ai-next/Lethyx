from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from test_native_candidate_sidecar_recall import _record

from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex


def test_failed_append_does_not_abort_another_thread_append(tmp_path, monkeypatch):
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    entered = Event()
    release = Event()
    original = index._sync_memory_indexes

    def sync(record, **kwargs):
        if record.block_id == "failed":
            entered.set()
            assert release.wait(5)
            raise ValueError("injected index failure")
        return original(record, **kwargs)

    monkeypatch.setattr(index, "_sync_memory_indexes", sync)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            failed = pool.submit(index.retain_block, _record("failed", "rolled back"))
            assert entered.wait(5)
            saved = pool.submit(index.retain_block, _record("saved", "survives"))
            # An independent writer must wait for the first transaction, not enter it.
            with pytest.raises(TimeoutError):
                saved.result(timeout=0.1)
            release.set()
            with pytest.raises(ValueError, match="injected"):
                failed.result(timeout=5)
            assert saved.result(timeout=5)["saved"]
        assert index.get_block("failed", workspace_id="ws") is None
        assert index.get_block("saved", workspace_id="ws").content == "survives"
    finally:
        release.set()
        index.close()
