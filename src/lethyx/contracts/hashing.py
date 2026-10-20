"""Stable hashing helpers for contract/evidence validation."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".memory",
    ".Lethyx",
    ".LETHYX_scratch",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
}


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_hashable_files(root: Path) -> Iterable[Path]:
    paths: list[Path] = []
    for directory, directories, files in os.walk(root, topdown=True):
        directories[:] = [name for name in directories if name not in SKIP_DIRS]
        paths.extend(Path(directory) / name for name in files if name not in SKIP_DIRS)
    for path in sorted(paths):
        if path.is_file():
            yield path


def _is_hash_skipped_rel(rel: str) -> bool:
    parts = [part for part in str(rel or "").replace("\\", "/").split("/") if part]
    return any(part in SKIP_DIRS for part in parts)


def workspace_hash(root: str | Path) -> str:
    base = Path(root).resolve()
    rows: list[tuple[str, str]] = []
    if not base.exists():
        return hash_value(rows)
    for path in _iter_hashable_files(base):
        try:
            rel = path.relative_to(base).as_posix()
            rows.append((rel, file_sha256(path)))
        except OSError:
            continue
    return hash_value(rows)


def diff_hash(root: str | Path, changed_files: Iterable[str] = ()) -> str:
    base = Path(root).resolve()
    rows: list[tuple[str, str]] = []
    for raw in changed_files:
        rel = str(raw or "").replace("\\", "/").strip().lstrip("./")
        if not rel:
            continue
        if _is_hash_skipped_rel(rel):
            continue
        path = (base / rel).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            continue
        if path.is_file():
            rows.append((rel, file_sha256(path)))
        else:
            rows.append((rel, "<missing>"))
    return hash_value(sorted(rows))
