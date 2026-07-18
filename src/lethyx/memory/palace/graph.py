import sqlite3
import pathlib
import time
from dataclasses import dataclass
from typing import Iterable, Literal

from lethyx.memory.backends.contracts import ALLOWED_EDGE_TYPES


@dataclass(frozen=True)
class Edge:
    src_id: str
    dst_id: str
    edge_type: str
    weight: float
    phase: str | None
    created_at: float


_CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    src_id     TEXT NOT NULL,
    dst_id     TEXT NOT NULL,
    edge_type  TEXT NOT NULL,
    weight     REAL DEFAULT 1.0,
    phase      TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY (src_id, dst_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, edge_type);
"""


class GraphStore:
    def __init__(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.executescript(_CREATE_EDGES)
        self._conn.commit()

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        phase: str | None = None,
    ) -> None:
        src = str(src_id or "").strip()
        dst = str(dst_id or "").strip()
        if not src or not dst:
            raise ValueError(
                f"add_edge requires non-empty src/dst ids (got {src_id!r} -> {dst_id!r})"
            )
        if edge_type not in ALLOWED_EDGE_TYPES:
            raise ValueError(
                f"unknown edge type {edge_type!r}; allowed: {sorted(ALLOWED_EDGE_TYPES)}"
            )
        self._conn.execute(
            "INSERT OR REPLACE INTO edges(src_id,dst_id,edge_type,weight,phase,created_at) VALUES(?,?,?,?,?,?)",
            (src, dst, edge_type, weight, phase, time.time()),
        )
        self._conn.commit()

    def delete_edges_for(self, node_ids: Iterable[str]) -> int:
        """Cascade-delete every edge touching any of *node_ids*."""
        ids = [str(node_id) for node_id in node_ids if str(node_id or "").strip()]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        cur = self._conn.execute(
            f"DELETE FROM edges WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})",
            ids + ids,
        )
        self._conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def walk(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None = None,
        hops: int = 1,
        direction: Literal["in", "out", "both"] = "both",
        limit: int = 50,
    ) -> list[Edge]:
        visited: set[str] = {node_id}
        frontier: set[str] = {node_id}
        result: list[Edge] = []

        for _ in range(hops):
            new_frontier: set[str] = set()
            for nid in frontier:
                rows = self._fetch_adjacent(nid, edge_types=edge_types, direction=direction)
                for row in rows:
                    edge = Edge(*row)
                    result.append(edge)
                    other = edge.dst_id if edge.src_id == nid else edge.src_id
                    if other not in visited:
                        new_frontier.add(other)
                        visited.add(other)
            frontier = new_frontier
            if not frontier:
                break

        return result[:limit]

    def _fetch_adjacent(
        self, node_id: str, *, edge_types: list[str] | None, direction: str
    ) -> list[tuple]:
        placeholders = ",".join("?" * len(edge_types)) if edge_types else None
        results: list[tuple] = []
        if direction in ("out", "both"):
            if edge_types:
                q = f"SELECT src_id,dst_id,edge_type,weight,phase,created_at FROM edges WHERE src_id=? AND edge_type IN ({placeholders})"
                results += self._conn.execute(q, [node_id] + edge_types).fetchall()
            else:
                results += self._conn.execute(
                    "SELECT src_id,dst_id,edge_type,weight,phase,created_at FROM edges WHERE src_id=?",
                    (node_id,),
                ).fetchall()
        if direction in ("in", "both"):
            if edge_types:
                q = f"SELECT src_id,dst_id,edge_type,weight,phase,created_at FROM edges WHERE dst_id=? AND edge_type IN ({placeholders})"
                results += self._conn.execute(q, [node_id] + edge_types).fetchall()
            else:
                results += self._conn.execute(
                    "SELECT src_id,dst_id,edge_type,weight,phase,created_at FROM edges WHERE dst_id=?",
                    (node_id,),
                ).fetchall()
        return results

    def close(self) -> None:
        self._conn.close()
