"""Small persistent backend-owned index for memory backend parity.

The index stores Lethyx-neutral records for adapters that need parity metadata
outside their native content store. It is intentionally not a MemPalace mirror;
it is owned by the selected backend implementation directory.
"""

import json
import os
import re
import sqlite3
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lethyx.llm_budget import estimate_tokens
from lethyx.memory.backends.contracts import (
    ALLOWED_EDGE_TYPES,
    MemoryBlockRecord,
    MemoryEdgeRecord,
    MemoryUsageRecord,
    block_from_json,
    block_to_json,
    validate_block_record,
    validate_lifecycle_transition,
    with_backend,
)

_SUPPRESSED_LIFECYCLES = {
    "quarantined",
    "superseded",
    "retracted",
    "deprecated",
    "archived",
}
_SCOPE_ALIASES = {
    "phase": "phase_scoped",
    "phase_scoped": "phase_scoped",
    "subtask": "subtask_scoped",
    "subtask_scoped": "subtask_scoped",
    "run": "run_scoped",
    "run_scoped": "run_scoped",
}

_CREATE = """
CREATE TABLE IF NOT EXISTS blocks (
    block_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    phase_id TEXT NOT NULL,
    subtask_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    surface TEXT NOT NULL,
    content TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_blocks_workspace ON blocks(workspace_id, lifecycle, surface);
CREATE INDEX IF NOT EXISTS idx_blocks_scope ON blocks(workspace_id, run_id, phase_id, subtask_id);
CREATE TABLE IF NOT EXISTS edges (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(src_id, dst_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    block_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _parse_time(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class BackendSQLiteIndex:
    def __init__(self, path: Path, *, backend_id: str, workspace_id: str = "") -> None:
        self.path = Path(path)
        self.backend_id = backend_id
        self.workspace_id = workspace_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.executescript(_CREATE)
        self._migrate_schema()
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _migrate_schema(self) -> None:
        columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(blocks)").fetchall()
        }
        if "scope" not in columns:
            self._conn.execute("ALTER TABLE blocks ADD COLUMN scope TEXT NOT NULL DEFAULT ''")
        rows = self._conn.execute(
            "SELECT block_id, payload_json FROM blocks WHERE scope='' OR scope IS NULL"
        ).fetchall()
        for block_id, payload_json in rows:
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                payload = {}
            scope = str(payload.get("scope") or "run_scoped")
            self._conn.execute("UPDATE blocks SET scope=? WHERE block_id=?", (scope, block_id))
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blocks_contract_scope "
            "ON blocks(workspace_id, scope, run_id, phase_id, subtask_id)"
        )

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        record = with_backend(record, self.backend_id)
        if not record.workspace_id and self.workspace_id:
            record = replace(record, workspace_id=self.workspace_id)
        validate_block_record(record)
        payload = block_to_json(record)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO blocks(
                block_id, workspace_id, run_id, phase_id, subtask_id,
                scope, lifecycle, surface, content, payload_json, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.block_id,
                record.workspace_id,
                record.run_id,
                record.phase_id,
                record.subtask_id,
                record.scope,
                record.lifecycle,
                record.surface,
                record.content,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                record.updated_at,
            ),
        )
        evicted = self._evict_over_cap()
        self._conn.commit()
        result: dict[str, Any] = {
            "ok": True,
            "saved": True,
            "block_id": record.block_id,
            "backend": self.backend_id,
        }
        if evicted:
            result["evicted"] = evicted
        return result

    def get_block(self, block_id: str, *, workspace_id: str = "") -> MemoryBlockRecord | None:
        row = self._conn.execute(
            "SELECT workspace_id, payload_json FROM blocks WHERE block_id=?",
            (str(block_id),),
        ).fetchone()
        if row is None:
            return None
        if workspace_id and row[0] and row[0] != workspace_id:
            return None
        data = json.loads(row[1])
        return block_from_json(data)

    def recall(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
        text = str(query.get("query") or "").strip().lower()
        workspace_id = str(query.get("workspace_id") or self.workspace_id or "")
        current_run_id = str(query.get("current_run_id") or "")
        run_id = str(query.get("run_id") or "")
        phase_id = str(query.get("phase_id") or "")
        subtask_id = str(query.get("subtask_id") or "")
        include_candidates = bool(query.get("include_candidates", False))
        limit = max(1, int(query.get("limit") or query.get("n") or 10))
        scan_limit = max(limit, int(query.get("scan_limit") or 2000))
        token_budget = max(0, int(query.get("token_budget") or query.get("max_tokens") or 0))
        clauses = []
        params: list[Any] = []
        if workspace_id:
            clauses.append("workspace_id=?")
            params.append(workspace_id)
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        if phase_id:
            clauses.append("phase_id=?")
            params.append(phase_id)
        if subtask_id:
            clauses.append("subtask_id=?")
            params.append(subtask_id)
        sql = "SELECT payload_json FROM blocks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(scan_limit)
        rows = self._conn.execute(sql, params).fetchall()
        tokens = [tok for tok in re.split(r"\W+", text) if tok]
        scored: list[tuple[int, int, MemoryBlockRecord, list[str]]] = []
        now = datetime.now(timezone.utc)
        for order, (payload_json,) in enumerate(rows):
            block = block_from_json(json.loads(payload_json))
            if self._is_expired(block, now=now):
                continue
            if block.lifecycle in _SUPPRESSED_LIFECYCLES:
                continue
            if not include_candidates and block.lifecycle == "candidate":
                continue
            if (
                current_run_id
                and block.surface == "current_run_state"
                and block.run_id
                and block.run_id != current_run_id
            ):
                continue
            score = 0
            matched_terms: list[str] = []
            if tokens:
                haystack = (block.content + " " + block.title + " " + " ".join(block.tags)).lower()
                matched_terms = [tok for tok in tokens if tok in haystack]
                score = len(matched_terms)
                if score == 0:
                    continue
            scored.append((score, order, block, matched_terms))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected: list[MemoryBlockRecord] = []
        used_tokens = 0
        for rank, (score, _, block, matched_terms) in enumerate(scored, start=1):
            retrieval_score = (float(score) / float(len(tokens))) if tokens else 1.0 / float(rank)
            block = replace(
                block,
                metadata={
                    **dict(block.metadata or {}),
                    "retrieval_backend": self.backend_id,
                    "retrieval_rank": rank,
                    "retrieval_score": retrieval_score,
                    "retrieval_score_raw": score,
                    "retrieval_matched_terms": matched_terms[:20],
                },
            )
            block_tokens = estimate_tokens(
                "\n".join(part for part in (block.title, block.content) if part)
            )
            if token_budget and selected and used_tokens + block_tokens > token_budget:
                break
            if token_budget and not selected and block_tokens > token_budget:
                break
            selected.append(block)
            used_tokens += block_tokens
            if len(selected) >= limit:
                break
        return selected

    def update_lifecycle(
        self,
        block_id: str,
        lifecycle: str,
        *,
        reason: str = "",
        evidence_refs: list[Any] | None = None,
        surface: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_block(block_id)
        if current is None:
            return {
                "ok": False,
                "error": "not_found",
                "block_id": block_id,
                "backend": self.backend_id,
            }
        validate_lifecycle_transition(
            current,
            lifecycle,
            reason=reason,
            evidence_refs=evidence_refs,
            surface=surface,
        )
        metadata = {**current.metadata, "lifecycle_reason": reason}
        updated = replace(
            current,
            lifecycle=lifecycle,
            surface=surface or current.surface,
            evidence_refs=tuple(evidence_refs or current.evidence_refs),
            metadata=metadata,
        )
        self.retain_block(updated)
        return {
            "ok": True,
            "block_id": block_id,
            "lifecycle": lifecycle,
            "backend": self.backend_id,
        }

    def expire_scope(self, scope: str, scope_id: str = "", *, run_id: str = "") -> int:
        target_scope = _SCOPE_ALIASES.get(str(scope or ""), str(scope or ""))
        block_ids: set[str] = set(self._expired_block_ids())
        clauses = []
        params: list[Any] = []
        if target_scope:
            clauses.append("scope=?")
            params.append(target_scope)
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        if target_scope == "phase_scoped" and scope_id:
            clauses.append("phase_id=?")
            params.append(scope_id)
        elif target_scope == "subtask_scoped" and scope_id:
            clauses.append("subtask_id=?")
            params.append(scope_id)
        elif target_scope == "run_scoped" and scope_id:
            clauses.append("run_id=?")
            params.append(scope_id)
        if clauses:
            rows = self._conn.execute(
                "SELECT block_id FROM blocks WHERE " + " AND ".join(clauses),
                params,
            ).fetchall()
            block_ids.update(str(row[0]) for row in rows)
        return self._delete_blocks(block_ids)

    def add_edge(self, edge: MemoryEdgeRecord) -> dict[str, Any]:
        src = str(edge.src_id or "").strip()
        dst = str(edge.dst_id or "").strip()
        if not src or not dst:
            raise ValueError(
                f"add_edge requires non-empty src/dst ids (got {edge.src_id!r} -> {edge.dst_id!r})"
            )
        if edge.edge_type not in ALLOWED_EDGE_TYPES:
            raise ValueError(
                f"unknown edge type {edge.edge_type!r}; allowed: {sorted(ALLOWED_EDGE_TYPES)}"
            )
        payload = asdict(edge)
        payload["src_id"] = src
        payload["dst_id"] = dst
        payload["evidence_refs"] = [asdict(ref) for ref in edge.evidence_refs]
        self._conn.execute(
            "INSERT OR REPLACE INTO edges(src_id,dst_id,edge_type,workspace_id,payload_json) VALUES(?,?,?,?,?)",
            (
                src,
                dst,
                edge.edge_type,
                edge.workspace_id or self.workspace_id,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        if edge.edge_type == "supersedes":
            old = self.get_block(dst)
            if old is not None:
                self.retain_block(replace(old, lifecycle="superseded", superseded_by=src))
        self._conn.commit()
        return {
            "ok": True,
            "src_id": src,
            "dst_id": dst,
            "edge_type": edge.edge_type,
            "backend": self.backend_id,
        }

    def edges_for(self, block_id: str) -> list[MemoryEdgeRecord]:
        rows = self._conn.execute(
            "SELECT payload_json FROM edges WHERE src_id=? OR dst_id=? ORDER BY rowid",
            (block_id, block_id),
        ).fetchall()
        out: list[MemoryEdgeRecord] = []
        for (payload_json,) in rows:
            data = json.loads(payload_json)
            data["evidence_refs"] = tuple(data.get("evidence_refs") or ())
            out.append(
                MemoryEdgeRecord(
                    **{k: v for k, v in data.items() if k in MemoryEdgeRecord.__dataclass_fields__}
                )
            )
        return out

    def record_usage(self, usage: MemoryUsageRecord) -> dict[str, Any]:
        payload = asdict(usage)
        payload["evidence_refs"] = [asdict(ref) for ref in usage.evidence_refs]
        self._conn.execute(
            "INSERT INTO usage(block_id,workspace_id,outcome,payload_json) VALUES(?,?,?,?)",
            (
                usage.block_id,
                usage.workspace_id or self.workspace_id,
                usage.outcome,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._conn.commit()
        return {"ok": True, "block_id": usage.block_id, "backend": self.backend_id}

    def value_stats(self, block_id: str) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT outcome FROM usage WHERE block_id=?",
            (block_id,),
        ).fetchall()
        helped = sum(1 for (outcome,) in rows if outcome == "helped")
        hurt = sum(1 for (outcome,) in rows if outcome == "hurt")
        return {
            "block_id": block_id,
            "uses_count": len(rows),
            "helped_count": helped,
            "hurt_count": hurt,
        }

    def export_blocks(self) -> list[MemoryBlockRecord]:
        rows = self._conn.execute("SELECT payload_json FROM blocks ORDER BY block_id").fetchall()
        return [block_from_json(json.loads(payload_json)) for (payload_json,) in rows]

    def import_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        count = 0
        for record in records:
            target = (
                replace(record, workspace_id=self.workspace_id) if self.workspace_id else record
            )
            self.retain_block(target)
            count += 1
        return {"ok": True, "imported": count, "backend": self.backend_id}

    def compile_active_overlay(
        self, *, workspace_id: str = "", run_id: str = "", phase_id: str = ""
    ) -> dict[str, Any]:
        query: dict[str, Any] = {
            "workspace_id": workspace_id or self.workspace_id,
            "limit": 500,
            "include_candidates": True,
        }
        if run_id:
            query["run_id"] = run_id
        if phase_id:
            query["phase_id"] = phase_id
        rows = self.recall(query)
        directives = [
            row.block_id for row in rows if row.surface == "directive" and row.lifecycle == "active"
        ]
        candidates = [
            row.block_id
            for row in rows
            if row.lifecycle == "candidate" or row.surface == "reflection_candidate"
        ]
        return {
            "backend": self.backend_id,
            "directive_blocks": directives,
            "candidate_blocks": candidates,
        }

    def render_prompt_items(
        self, *, workspace_id: str = "", run_id: str = "", phase_id: str = ""
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"workspace_id": workspace_id or self.workspace_id, "limit": 500}
        if run_id:
            query["run_id"] = run_id
        if phase_id:
            query["phase_id"] = phase_id
        rows = self.recall(query)
        rendered: list[dict[str, Any]] = []
        for row in rows:
            if row.lifecycle != "active":
                continue
            text = row.content
            if row.privacy_class == "secret_pointer_only" or row.risk_level == "secret":
                text = f"[redacted secret_pointer_only memory: {row.block_id}]"
            rendered.append(
                {
                    "id": row.block_id,
                    "text": text,
                    "surface": row.surface,
                    "source_backend": self.backend_id,
                    "privacy_class": row.privacy_class,
                    "risk_level": row.risk_level,
                }
            )
        return rendered

    def _evict_over_cap(self) -> int:
        max_rows = _int_env("LETHYX_BACKEND_INDEX_MAX_ROWS", 20000)
        if max_rows <= 0:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM blocks").fetchone()
        overflow = int(row[0]) - max_rows
        if overflow <= 0:
            return 0
        rows = self._conn.execute(
            """
            SELECT block_id FROM blocks
            WHERE scope!='cross_run_durable' AND surface!='directive'
            ORDER BY updated_at ASC, rowid ASC
            LIMIT ?
            """,
            (overflow,),
        ).fetchall()
        return self._delete_blocks([str(row[0]) for row in rows], commit=False)

    def _delete_blocks(self, block_ids: set[str] | list[str], *, commit: bool = True) -> int:
        ids = [block_id for block_id in block_ids if str(block_id or "").strip()]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"DELETE FROM edges WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})",
            ids + ids,
        )
        cur = self._conn.execute(
            f"DELETE FROM blocks WHERE block_id IN ({placeholders})",
            ids,
        )
        if commit:
            self._conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def _expired_block_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT block_id, payload_json FROM blocks").fetchall()
        now = datetime.now(timezone.utc)
        expired: list[str] = []
        for block_id, payload_json in rows:
            try:
                block = block_from_json(json.loads(payload_json))
            except (TypeError, json.JSONDecodeError, ValueError):
                continue
            if self._is_expired(block, now=now):
                expired.append(str(block_id))
        return expired

    @staticmethod
    def _is_expired(block: MemoryBlockRecord, *, now: datetime) -> bool:
        expires_at = _parse_time(block.expires_at)
        return expires_at is not None and expires_at <= now
