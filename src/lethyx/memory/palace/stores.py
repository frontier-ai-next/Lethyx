import logging
import os
import pathlib
import threading
import importlib.util
from typing import Any

from lethyx.memory.palace.graph import GraphStore
from lethyx.memory.palace.transient import TransientStore

log = logging.getLogger(__name__)

_CHROMA_STORES = [
    "palace.charter",
    "palace.lesson",
    "palace.idea",
    "palace.codeptr",
    "palace.skill_index",
    "palace.run",
    "palace.phase",
    "palace.subtask",
    "palace.durable",
]


def _chroma_collection_name(store: str) -> str:
    return store.replace(".", "_")


# One PersistentClient per resolved path, shared across PalaceStores
# instances: parallel clients on the same directory race on the embedded
# sqlite even with WAL enabled.
_PERSISTENT_CLIENT_LOCK = threading.Lock()
_PERSISTENT_CLIENTS: dict[str, Any] = {}


def _shared_persistent_client(path: pathlib.Path) -> Any:
    import chromadb

    key = str(path.resolve())
    with _PERSISTENT_CLIENT_LOCK:
        client = _PERSISTENT_CLIENTS.get(key)
        if client is None:
            client = chromadb.PersistentClient(path=key)
            _PERSISTENT_CLIENTS[key] = client
        return client


def _release_persistent_client(key: str) -> None:
    with _PERSISTENT_CLIENT_LOCK:
        client = _PERSISTENT_CLIENTS.pop(key, None)
    if client is None:
        return
    try:
        if hasattr(client, "close"):
            client.close()
    except Exception:
        log.debug("chroma client close failed for %s", key, exc_info=True)


class PalaceStores:
    def __init__(self, palace_root: pathlib.Path) -> None:
        self._root = palace_root
        self._root.mkdir(parents=True, exist_ok=True)
        self._graph = GraphStore(palace_root / "graph.sqlite")
        self._transient = TransientStore(
            palace_root / "transient.sqlite",
            default_ttl_seconds=int(os.environ.get("LETHYX_PALACE_TRANSIENT_TTL_SEC", "86400")),
            max_rows=int(os.environ.get("LETHYX_PALACE_TRANSIENT_MAX_ROWS", "5000")),
        )
        self._chroma_clients: dict[str, Any] = {}
        self._client_paths: set[str] = set()

    def chroma(self, store: str) -> Any:
        if store not in self._chroma_clients:
            self._chroma_clients[store] = self._open_chroma(store)
        return self._chroma_clients[store]

    def _open_chroma(self, store: str) -> Any:
        if importlib.util.find_spec("chromadb") is None:
            raise RuntimeError(
                "chromadb is not installed; real MemPalace memory is unavailable. "
                "Install chromadb; Lethyx no longer substitutes a volatile in-memory stub."
            )
        collection_dir = self._root / _chroma_collection_name(store)
        collection_dir.mkdir(parents=True, exist_ok=True)
        client = _shared_persistent_client(collection_dir)
        self._client_paths.add(str(collection_dir.resolve()))
        return client.get_or_create_collection(
            name=_chroma_collection_name(store),
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def graph(self) -> GraphStore:
        return self._graph

    @property
    def transient(self) -> TransientStore:
        return self._transient

    def health(self) -> dict[str, Any]:
        ok_stores = []
        bad_stores = []
        for s in _CHROMA_STORES:
            try:
                col = self.chroma(s)
                col.count()
                ok_stores.append(s)
            except Exception as exc:
                bad_stores.append(f"{s}: {exc}")
        return {
            "ok": not bad_stores,
            "stores_ok": ok_stores,
            "stores_fail": bad_stores,
            "volatile_stub": False,
        }

    def close(self) -> None:
        self._chroma_clients.clear()
        for key in list(self._client_paths):
            _release_persistent_client(key)
        self._client_paths.clear()
        try:
            from chromadb.api.shared_system_client import SharedSystemClient

            SharedSystemClient._identifier_to_system.clear()
        except Exception:
            log.debug("chroma shared-system cleanup failed", exc_info=True)
        import gc

        gc.collect()
        self._graph.close()
        self._transient.close()
