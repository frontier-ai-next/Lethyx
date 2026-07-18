"""MemPalace-backed memory for Lethyx.

Maps Lethyx's workspace-centric memory model onto the MemPalace
wing/room/drawer hierarchy with ChromaDB semantic search.

Wing mapping:
    workspace_id  ->  wing_{workspace_id}
    "system"      ->  wing_LETHYX_system

Hall mapping (MemPalace standard halls):
    command/test  ->  hall_events
    change/code   ->  hall_facts
    error/bug     ->  hall_events
    lesson/idea   ->  hall_discoveries
    decision      ->  hall_facts
    preference    ->  hall_preferences

Room = free-form topic within the wing (e.g. "scoring-system", "api-client").
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)

_HALL_MAP: dict[str, str] = {
    "command": "hall_events",
    "test": "hall_events",
    "error": "hall_events",
    "bug": "hall_events",
    "warning": "hall_events",
    "change": "hall_facts",
    "code": "hall_facts",
    "decision": "hall_facts",
    "seed": "hall_facts",
    "commit": "hall_facts",
    "lesson": "hall_discoveries",
    "idea": "hall_discoveries",
    "observation": "hall_discoveries",
    "insight": "hall_discoveries",
    "completion": "hall_discoveries",
    "preference": "hall_preferences",
    "config": "hall_preferences",
    "advice": "hall_advice",
    "recommendation": "hall_advice",
}

_DEFAULT_HALL = "hall_events"


def _event_to_hall(event_type: str) -> str:
    key = (event_type or "").strip().lower().rstrip("s")
    return _HALL_MAP.get(key, _DEFAULT_HALL)


def _workspace_to_wing(workspace_id: str) -> str:
    clean = (workspace_id or "").strip().replace(" ", "_").lower()
    if not clean or clean == "system":
        return "wing_LETHYX_system"
    return f"wing_{clean}"


def _make_drawer_id(wing: str, room: str, content: str, timestamp: float) -> str:
    raw = f"{wing}:{room}:{content[:200]}:{timestamp}"
    return f"drawer_{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


class PalaceBackend:
    """Thin adapter between Lethyx memory operations and MemPalace storage."""

    def __init__(self, palace_path: Path | str):
        self._palace_path = str(Path(palace_path).resolve())
        os.environ["MEMPALACE_PALACE_PATH"] = self._palace_path
        self._collection = None

    @property
    def palace_path(self) -> str:
        return self._palace_path

    def _get_collection(self):
        if self._collection is None:
            from mempalace.palace import get_collection

            try:
                self._collection = get_collection(
                    self._palace_path,
                    create=True,
                )
            except Exception as exc:
                # ChromaDB SharedSystemClient keeps a singleton cache that
                # can become corrupted (stale Rust bindings, stopped System,
                # KeyError on identifier, etc.).  Flush it and retry once.
                log.debug("ChromaDB init failed (%s), clearing cache and retrying", exc)
                self._flush_chromadb_cache()
                self._collection = get_collection(
                    self._palace_path,
                    create=True,
                )
        return self._collection

    @staticmethod
    def _flush_chromadb_cache():
        """Remove stale entries from ChromaDB's internal singleton caches."""
        try:
            from chromadb.api.shared_system_client import SharedSystemClient

            SharedSystemClient._identifier_to_system.clear()
        except Exception:
            pass

    def close(self) -> None:
        """Best-effort release of underlying Chroma resources.

        On Windows the ChromaDB SQLite WAL file and the HNSW
        ``data_level0.bin`` are kept mmap-locked for the lifetime of the
        ``PersistentClient``. Without an explicit ``client.close()`` the
        files remain locked even after every Python reference is dropped,
        which makes ``shutil.rmtree`` fail with WinError 32 when the user
        tries to delete the run.

        ``self._collection`` is mempalace's ``ChromaCollection`` wrapper.
        The actual chromadb collection lives under ``._collection`` and
        the persistent client under ``._collection._client``.
        """
        collection = self._collection
        if collection is None:
            self._flush_chromadb_cache()
            return
        self._collection = None

        inner = getattr(collection, "_collection", None) or collection
        client = getattr(inner, "_client", None) or getattr(collection, "_client", None)

        try:
            if client is not None and hasattr(client, "close"):
                client.close()
        except Exception:
            log.debug("Failed to close palace client", exc_info=True)

        try:
            from mempalace import palace as mempalace_palace

            backend = getattr(mempalace_palace, "_DEFAULT_BACKEND", None)
            close_palace = getattr(backend, "close_palace", None)
            if callable(close_palace):
                close_palace(self._palace_path)
        except Exception:
            log.debug("Failed to close mempalace backend cache", exc_info=True)

        try:
            del inner
        except UnboundLocalError:
            pass
        try:
            del collection
        except UnboundLocalError:
            pass
        try:
            del client
        except UnboundLocalError:
            pass

        try:
            import gc

            gc.collect()
        except Exception:
            pass

        # Flush ChromaDB singleton cache so next _get_collection creates
        # fresh state instead of reusing a stopped system with dead Rust
        # bindings.
        self._flush_chromadb_cache()

    def add(
        self,
        *,
        workspace_id: str = "",
        event_type: str = "observation",
        room: str = "",
        title: str,
        content: str,
        kind: str = "info",
        tags: list[str] | None = None,
        task_id: str = "",
        source_path: str = "",
        metadata_extra: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Store a memory in MemPalace with proper wing/hall/room classification."""
        wing = _workspace_to_wing(workspace_id)
        hall = _event_to_hall(event_type)
        room_name = (room or event_type or "general").strip().lower().replace(" ", "-")

        ts = time.time()
        drawer_id = _make_drawer_id(wing, room_name, content, ts)

        full_content = f"[{title}]\n{content}" if title else content

        meta: dict[str, Any] = {
            "wing": wing,
            "room": room_name,
            "hall": hall,
            "kind": kind,
            "event_type": event_type,
            "workspace_id": workspace_id,
            "task_id": task_id,
            "source_file": source_path or f"Lethyx/{event_type}",
            "added_by": "Lethyx",
            "timestamp": ts,
        }
        if tags:
            meta["tags"] = ",".join(sorted(tags))
        if metadata_extra:
            for k, v in metadata_extra.items():
                meta[str(k)] = str(v)[:500]

        col = self._get_collection()
        col.upsert(
            ids=[drawer_id],
            documents=[full_content],
            metadatas=[meta],
        )

        log.debug("Palace add: %s/%s/%s [%s]", wing, hall, room_name, drawer_id[:16])

        return {"id": drawer_id, "wing": wing, "hall": hall, "room": room_name}

    # Tier 2.3 — default list of rooms that are *write-time* mirrors of
    # tool events (ideas, changes, scratchpad, terminal log). These tend
    # to crowd out actual ``lessons`` / ``verify_runs`` content when the
    # palace is queried for "what do we know". Callers can override.
    DEFAULT_EXCLUDE_ROOMS: frozenset[str] = frozenset(
        {
            "ideas-hypothesis",
            "ideas-observation_from_log",
            "scratchpad",
            "terminal_scrollback",
            "changes",
        }
    )

    def search(
        self,
        query: str,
        *,
        workspace_id: str = "",
        room: str = "",
        n_results: int = 10,
        exclude_rooms: frozenset[str] | set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search across the palace, optionally scoped to wing/room.

        ``exclude_rooms`` filters out drawers whose ``room`` metadata is
        in the given set. Defaults to :attr:`DEFAULT_EXCLUDE_ROOMS` when
        ``None``. Pass an empty set to disable filtering entirely (legacy
        behaviour). The over-fetch of ``n_results * 3`` keeps the final
        result count close to what the caller asked for after filtering
        away noisy rooms.
        """

        wing = _workspace_to_wing(workspace_id) if workspace_id else ""
        if exclude_rooms is None:
            exclude_rooms = self.DEFAULT_EXCLUDE_ROOMS
        exclude_rooms_set: set[str] = {str(r).strip() for r in exclude_rooms if r}

        where: dict[str, Any] | None = None
        if wing and room:
            where = {"$and": [{"wing": wing}, {"room": room}]}
        elif wing:
            where = {"wing": wing}
        elif room:
            where = {"room": room}

        col = self._get_collection()
        # Over-fetch a bit so post-filter still hits ``n_results``.
        raw_limit = min(n_results * (3 if exclude_rooms_set else 1), 150)
        try:
            results = col.query(
                query_texts=[query],
                n_results=raw_limit,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.warning("Palace search failed: %s", e)
            return []

        hits: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for i, drawer_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            entry_room = str((meta or {}).get("room") or "").strip()
            if entry_room and entry_room in exclude_rooms_set:
                continue
            hits.append(
                {
                    "id": drawer_id,
                    "content": docs[i] if i < len(docs) else "",
                    "metadata": meta,
                    "distance": dists[i] if i < len(dists) else 1.0,
                    "wing": (meta or {}).get("wing", ""),
                    "room": entry_room,
                    "hall": (meta or {}).get("hall", ""),
                }
            )
            if len(hits) >= n_results:
                break

        return hits

    def list_wings(self) -> dict[str, int]:
        """Return {wing_name: drawer_count}."""
        col = self._get_collection()
        total = col.count()
        if total == 0:
            return {}
        wings: dict[str, int] = {}
        batch_size = 5000
        offset = 0
        while offset < total:
            batch = col.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            for meta in batch.get("metadatas") or []:
                w = (meta or {}).get("wing", "unknown")
                wings[w] = wings.get(w, 0) + 1
            offset += batch_size
        return dict(sorted(wings.items()))

    def list_rooms(self, workspace_id: str = "") -> dict[str, int]:
        """Return {room_name: count} for a given wing or globally."""
        col = self._get_collection()
        total = col.count()
        if total == 0:
            return {}

        wing = _workspace_to_wing(workspace_id) if workspace_id else ""
        rooms: dict[str, int] = {}
        batch_size = 5000
        offset = 0
        while offset < total:
            batch = col.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
                where={"wing": wing} if wing else None,
            )
            for meta in batch.get("metadatas") or []:
                r = (meta or {}).get("room", "unknown")
                rooms[r] = rooms.get(r, 0) + 1
            offset += batch_size
        return dict(sorted(rooms.items()))

    def get_taxonomy(self) -> dict[str, dict[str, int]]:
        """Return {wing: {room: count}} tree."""
        col = self._get_collection()
        total = col.count()
        if total == 0:
            return {}
        tree: dict[str, dict[str, int]] = {}
        batch_size = 5000
        offset = 0
        while offset < total:
            batch = col.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            for meta in batch.get("metadatas") or []:
                w = (meta or {}).get("wing", "unknown")
                r = (meta or {}).get("room", "unknown")
                if w not in tree:
                    tree[w] = {}
                tree[w][r] = tree[w].get(r, 0) + 1
            offset += batch_size
        return tree

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        col = self._get_collection()
        total = col.count()
        wings = self.list_wings()
        return {
            "total_drawers": total,
            "wings_count": len(wings),
            "wings": wings,
            "palace_path": self._palace_path,
            "backend": "mempalace_chromadb",
        }

    def recent(
        self,
        *,
        workspace_id: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return most recent entries for a workspace (by timestamp metadata)."""
        wing = _workspace_to_wing(workspace_id) if workspace_id else ""
        col = self._get_collection()
        total = col.count()
        if total == 0:
            return []

        batch = col.get(
            include=["documents", "metadatas"],
            limit=min(limit * 5, 500),
            where={"wing": wing} if wing else None,
        )

        entries = []
        ids = batch.get("ids") or []
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        for i, drawer_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            entries.append(
                {
                    "id": drawer_id,
                    "content": docs[i] if i < len(docs) else "",
                    "metadata": meta,
                    "timestamp": float(meta.get("timestamp", 0)),
                    "wing": meta.get("wing", ""),
                    "room": meta.get("room", ""),
                }
            )

        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        return entries[:limit]

    def fetch_document_by_metadata(
        self,
        *,
        workspace_id: str,
        room: str,
        field: str,
        value: str,
    ) -> str | None:
        """Return first document whose metadata matches ``field`` == ``value`` (exact).

        Used for deterministic cache keys (e.g. GMAS chunk summaries) without
        relying on semantic search.
        """
        wing = _workspace_to_wing(workspace_id)
        col = self._get_collection()
        try:
            res = col.get(
                where={
                    "$and": [
                        {"wing": {"$eq": wing}},
                        {"room": {"$eq": room}},
                        {field: {"$eq": value}},
                    ]
                },
                limit=1,
                include=["documents", "metadatas"],
            )
        except Exception:
            try:
                res = col.get(
                    where={"wing": wing, "room": room, field: value},
                    limit=1,
                    include=["documents", "metadatas"],
                )
            except Exception as exc:
                log.debug("fetch_document_by_metadata failed: %s", exc)
                return None

        docs = res.get("documents") or []
        if not docs:
            return None
        raw = docs[0] if docs[0] is not None else ""
        text = str(raw).strip()
        if not text:
            return None
        # ``add()`` stores ``[title]\\ncontent`` — return full body for cache hits.
        return text


_backend_cache: dict[str, PalaceBackend] = {}


def get_palace_backend(palace_path: Path | str) -> PalaceBackend:
    """Return a cached PalaceBackend instance for the given path."""
    key = str(Path(palace_path).resolve())
    if key not in _backend_cache:
        _backend_cache[key] = PalaceBackend(key)
    return _backend_cache[key]


def clear_palace_backend_cache(palace_path: Path | str | None = None) -> None:
    """Release cached palace backends for one path or for all paths."""
    if palace_path is None:
        targets = list(_backend_cache.items())
        _backend_cache.clear()
    else:
        key = str(Path(palace_path).resolve())
        backend = _backend_cache.pop(key, None)
        targets = [(key, backend)] if backend is not None else []

    for _key, backend in targets:
        if backend is None:
            continue
        backend.close()
