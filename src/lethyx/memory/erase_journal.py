"""Durable, hash-only state for governed erase requests."""

import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


_CREATE = """
CREATE TABLE IF NOT EXISTS erase_requests (
    workspace_id_sha256 TEXT NOT NULL,
    request_id_sha256 TEXT NOT NULL,
    fingerprint_sha256 TEXT NOT NULL,
    target_id_sha256 TEXT NOT NULL,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(workspace_id_sha256, request_id_sha256)
);
"""


class EraseRequestConflict(ValueError):
    """Raised when a request ID is replayed with a different immutable fingerprint."""


@dataclass(frozen=True)
class EraseRequestState:
    workspace_id_sha256: str
    request_id_sha256: str
    fingerprint_sha256: str
    target_id_sha256: str
    backend: str
    status: str
    error: str
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def erase_request_journal_path(repo_root: str | Path) -> Path:
    return Path(repo_root) / ".Lethyx" / "memory" / "erase_requests.sqlite"


def open_erase_request(
    repo_root: str | Path,
    *,
    workspace_id_sha256: str,
    request_id_sha256: str,
    fingerprint_sha256: str,
    target_id_sha256: str,
    backend: str,
) -> tuple[EraseRequestState, bool]:
    path = erase_request_journal_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(_CREATE)
        row = connection.execute(
            """
            SELECT workspace_id_sha256, request_id_sha256, fingerprint_sha256,
                   target_id_sha256, backend, status, error, created_at, updated_at
            FROM erase_requests
            WHERE workspace_id_sha256=? AND request_id_sha256=?
            """,
            (workspace_id_sha256, request_id_sha256),
        ).fetchone()
        if row is not None:
            state = EraseRequestState(*[str(value) for value in row])
            if state.fingerprint_sha256 != fingerprint_sha256:
                raise EraseRequestConflict("erase request ID has a different fingerprint")
            return state, False
        state = EraseRequestState(
            workspace_id_sha256=workspace_id_sha256,
            request_id_sha256=request_id_sha256,
            fingerprint_sha256=fingerprint_sha256,
            target_id_sha256=target_id_sha256,
            backend=backend,
            status="requested",
            error="",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            """
            INSERT INTO erase_requests(
                workspace_id_sha256, request_id_sha256, fingerprint_sha256,
                target_id_sha256, backend, status, error, created_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                state.workspace_id_sha256,
                state.request_id_sha256,
                state.fingerprint_sha256,
                state.target_id_sha256,
                state.backend,
                state.status,
                state.error,
                state.created_at,
                state.updated_at,
            ),
        )
        return state, True


def record_erase_request_outcome(
    repo_root: str | Path,
    state: EraseRequestState,
    *,
    status: str,
    error: str,
) -> EraseRequestState:
    path = erase_request_journal_path(repo_root)
    now = datetime.now(UTC).isoformat()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(_CREATE)
        cursor = connection.execute(
            """
            UPDATE erase_requests
            SET status=?, error=?, updated_at=?
            WHERE workspace_id_sha256=? AND request_id_sha256=? AND fingerprint_sha256=?
            """,
            (
                status,
                error,
                now,
                state.workspace_id_sha256,
                state.request_id_sha256,
                state.fingerprint_sha256,
            ),
        )
        if cursor.rowcount != 1:
            raise LookupError("erase request state is unavailable")
    return EraseRequestState(
        workspace_id_sha256=state.workspace_id_sha256,
        request_id_sha256=state.request_id_sha256,
        fingerprint_sha256=state.fingerprint_sha256,
        target_id_sha256=state.target_id_sha256,
        backend=state.backend,
        status=status,
        error=error,
        created_at=state.created_at,
        updated_at=now,
    )
