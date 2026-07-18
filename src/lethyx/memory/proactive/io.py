"""Atomic writes and audit logging for core/BKB files."""

import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _file_hash(path: Path) -> str:
    if not path.is_file():
        return "sha256:empty"
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    if sys.platform == "win32":
        import msvcrt

        with open(lock_path, "a+b") as lock_fh:
            lock_fh.seek(0)
            lock_fh.write(b"\0")
            lock_fh.flush()
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        with open(lock_path, "a") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _atomic_write_unlocked(path: Path, content: str, *, encoding: str = "utf-8") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as tmp_fh:
            tmp_fh.write(content)
            tmp_fh.flush()
            os.fsync(tmp_fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return _file_hash(path) if path.is_file() else "sha256:empty"


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> str:
    """Write text atomically; return new content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path):
        return _atomic_write_unlocked(path, content, encoding=encoding)


def append_audit(
    audit_path: Path,
    entry: dict[str, Any],
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(entry)
    row.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _file_lock(audit_path):
        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


@contextmanager
def bkb_transaction(core_root: Path) -> Iterator[Path]:
    """Exclusive lock for BKB read/merge/write under core_root."""
    core_root.mkdir(parents=True, exist_ok=True)
    bkb_path = core_root / "bkb.yaml"
    with _file_lock(bkb_path):
        yield bkb_path


_RULE_TYPE_MD_FILES: dict[str, str] = {
    "lesson": "20_manager_lessons.md",
    "anti_pattern": "30_failure_patterns.md",
    "risk": "40_active_risks.md",
    "open_thread": "50_open_threads.md",
}

_WORKSPACE_RULE_TYPE_MD: dict[str, str] = {
    "lesson": "20_workspace_lessons.md",
    "anti_pattern": "30_workspace_antipatterns.md",
    "risk": "40_current_strategy.md",
    "open_thread": "50_open_threads.md",
}


def append_rule_md_slice(
    core_root: Path,
    *,
    rule_type: str,
    title: str,
    body: str,
    rule_id: str,
    target: str = "manager",
) -> None:
    """Append a short promoted rule line to the matching core MD file."""
    mapping = _WORKSPACE_RULE_TYPE_MD if target == "workspace" else _RULE_TYPE_MD_FILES
    filename = mapping.get(str(rule_type or "").strip().lower())
    if not filename:
        return
    path = core_root / filename
    if not path.is_file():
        path.write_text(f"# {filename}\n\n", encoding="utf-8")
    line = f"- [{rule_id}] {title}: {body}\n"
    with _file_lock(path):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
