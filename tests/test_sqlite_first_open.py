from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.catalog_store import WorkspaceCatalogStore


def test_backend_index_first_open_is_safe_across_concurrent_sessions(tmp_path: Path) -> None:
    path = tmp_path / "shared" / "memory.sqlite"
    barrier = Barrier(8)

    def open_index(_: int) -> None:
        barrier.wait()
        index = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
        index.close()

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(open_index, range(8)))

    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="alpha")
    reopened.close()


def test_workspace_catalog_first_open_is_safe_across_concurrent_sessions(
    tmp_path: Path,
) -> None:
    barrier = Barrier(8)

    def open_catalog(_: int) -> None:
        barrier.wait()
        store = WorkspaceCatalogStore(tmp_path, "alpha", "default")
        store.close()

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(open_catalog, range(8)))

    reopened = WorkspaceCatalogStore(tmp_path, "alpha", "default")
    reopened.close()
