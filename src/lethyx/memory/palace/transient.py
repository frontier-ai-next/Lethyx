import logging
import sqlite3
import pathlib
import time
import uuid
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TransientNode:
    id: str
    workspace_id: str
    run_id: str | None
    phase: str | None
    subtask_id: str | None
    tags: list[str]
    source_path: str | None
    summary: str
    body: str | None
    created_at: float
    ttl_seconds: int | None
    extra: dict[str, Any]


_DDL = """
CREATE TABLE IF NOT EXISTS transient_nodes (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    run_id TEXT,
    phase TEXT,
    subtask_id TEXT,
    tags TEXT,
    source_path TEXT,
    summary TEXT,
    body TEXT,
    created_at REAL NOT NULL,
    ttl_seconds INTEGER,
    extra_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_transient_run     ON transient_nodes(run_id, phase, subtask_id);
CREATE INDEX IF NOT EXISTS idx_transient_created ON transient_nodes(created_at);
CREATE VIRTUAL TABLE IF NOT EXISTS transient_fts USING fts5(
    summary, body, tags,
    content='transient_nodes', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS transient_nodes_fts_delete
AFTER DELETE ON transient_nodes BEGIN
    INSERT INTO transient_fts(transient_fts, rowid, summary, body, tags)
    VALUES('delete', old.rowid, old.summary, old.body, old.tags);
END;
"""


class TransientStore:
    DEFAULT_TTL = int(86400)
    DEFAULT_MAX_ROWS = 5000

    def __init__(
        self,
        path: pathlib.Path,
        default_ttl_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.executescript(_DDL)
        self._conn.commit()
        self._default_ttl = default_ttl_seconds or self.DEFAULT_TTL
        self._max_rows = max_rows if max_rows is not None else self.DEFAULT_MAX_ROWS

    def add(
        self,
        *,
        workspace_id: str,
        summary: str,
        run_id: str | None = None,
        phase: str | None = None,
        subtask_id: str | None = None,
        tags: list[str] | None = None,
        source_path: str | None = None,
        body: str | None = None,
        ttl_seconds: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        import json

        node_id = str(uuid.uuid4())
        tags_str = ",".join(tags or [])
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        self._conn.execute(
            """INSERT INTO transient_nodes
               (id,workspace_id,run_id,phase,subtask_id,tags,source_path,summary,body,created_at,ttl_seconds,extra_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                node_id,
                workspace_id,
                run_id,
                phase,
                subtask_id,
                tags_str,
                source_path,
                summary,
                body,
                time.time(),
                ttl,
                json.dumps(extra or {}),
            ),
        )
        try:
            self._conn.execute(
                "INSERT INTO transient_fts(rowid,summary,body,tags) SELECT rowid,summary,body,tags FROM transient_nodes WHERE id=?",
                (node_id,),
            )
        except sqlite3.Error:
            log.debug("transient FTS insert failed for %s", node_id, exc_info=True)
        # Hard cap: evict oldest rows in the same transaction (delete trigger
        # keeps the FTS index in sync).
        self._evict_over_cap()
        self._conn.commit()
        return node_id

    def _evict_over_cap(self) -> int:
        """Delete oldest rows beyond ``max_rows``; caller commits."""
        if self._max_rows <= 0:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM transient_nodes").fetchone()
        overflow = int(row[0]) - self._max_rows
        if overflow <= 0:
            return 0
        cur = self._conn.execute(
            """DELETE FROM transient_nodes WHERE rowid IN (
                   SELECT rowid FROM transient_nodes
                   ORDER BY created_at ASC, rowid ASC LIMIT ?
               )""",
            (overflow,),
        )
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def search_fts(
        self, query: str, *, workspace_id: str | None = None, n: int = 20
    ) -> list[TransientNode]:
        try:
            if workspace_id:
                rows = self._conn.execute(
                    """SELECT t.id,t.workspace_id,t.run_id,t.phase,t.subtask_id,t.tags,t.source_path,
                              t.summary,t.body,t.created_at,t.ttl_seconds,t.extra_json
                       FROM transient_fts f JOIN transient_nodes t ON f.rowid=t.rowid
                       WHERE f.transient_fts MATCH ? AND t.workspace_id=? LIMIT ?""",
                    (query, workspace_id, n),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT t.id,t.workspace_id,t.run_id,t.phase,t.subtask_id,t.tags,t.source_path,
                              t.summary,t.body,t.created_at,t.ttl_seconds,t.extra_json
                       FROM transient_fts f JOIN transient_nodes t ON f.rowid=t.rowid
                       WHERE f.transient_fts MATCH ? LIMIT ?""",
                    (query, n),
                ).fetchall()
        except Exception:
            return []
        return [self._row_to_node(r) for r in rows]

    def recent(
        self, *, workspace_id: str, n: int = 50, phase: str | None = None
    ) -> list[TransientNode]:
        if phase:
            rows = self._conn.execute(
                "SELECT id,workspace_id,run_id,phase,subtask_id,tags,source_path,summary,body,created_at,ttl_seconds,extra_json FROM transient_nodes WHERE workspace_id=? AND phase=? ORDER BY created_at DESC LIMIT ?",
                (workspace_id, phase, n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id,workspace_id,run_id,phase,subtask_id,tags,source_path,summary,body,created_at,ttl_seconds,extra_json FROM transient_nodes WHERE workspace_id=? ORDER BY created_at DESC LIMIT ?",
                (workspace_id, n),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def expire_ttl(self) -> int:
        cutoff = time.time()
        cur = self._conn.execute(
            "DELETE FROM transient_nodes WHERE ttl_seconds IS NOT NULL AND (created_at + ttl_seconds) < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cur.rowcount

    def _row_to_node(self, row: tuple) -> TransientNode:
        import json

        return TransientNode(
            id=row[0],
            workspace_id=row[1],
            run_id=row[2],
            phase=row[3],
            subtask_id=row[4],
            tags=(row[5] or "").split(",") if row[5] else [],
            source_path=row[6],
            summary=row[7],
            body=row[8],
            created_at=row[9],
            ttl_seconds=row[10],
            extra=json.loads(row[11] or "{}"),
        )

    def close(self) -> None:
        self._conn.close()
