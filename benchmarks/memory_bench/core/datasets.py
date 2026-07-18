"""Dataset registry and path resolution for memory_bench.

Datasets are NOT stored in the repository.  This module maintains a registry
of where each dataset lives (or should be downloaded from) and validates
presence and integrity (sha256) before a run.

Directory convention:
  data/memory_bench/<name>/       (relative to repo root)

Environment override:
  MEMORY_BENCH_DATA_DIR=<path>   overrides the default data root.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: name -> {description, subpath, url, notes}
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict[str, Any]] = {
    "locomo": {
        "description": "LoCoMo: Long-term Conversational Memory benchmark (Maharana et al. ACL 2024)",
        "subpath": "locomo10.json",
        "url": "https://github.com/snap-research/locomo (download locomo10.json from ./data/)",
        "notes": (
            "Clone snap-research/locomo and copy ./data/locomo10.json "
            "to data/memory_bench/locomo/locomo10.json"
        ),
    },
    "longmemeval": {
        "description": "LongMemEval: Chat assistant long-term memory benchmark (Wu et al. 2024)",
        "subpath": "longmemeval_s.json",
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "notes": (
            "Download xiaowu0162/longmemeval-cleaned; copy longmemeval_s_cleaned.json "
            "to data/memory_bench/longmemeval/longmemeval_s.json (optional M/oracle variants supported)"
        ),
    },
    "beam": {
        "description": "BEAM: Beyond a Million Tokens benchmark (ICLR 2026)",
        "subpath": "chats",
        "url": "https://github.com/mohammadtavakoli78/BEAM",
        "notes": (
            "Clone mohammadtavakoli78/BEAM or download Mohammadta/BEAM; expose the "
            "upstream chats/ directory under data/memory_bench/beam/chats/"
        ),
    },
    "memoryarena": {
        "description": "MemoryArena: Multi-session agentic memory benchmark (He et al. 2026)",
        "subpath": "tasks",
        "url": "https://memoryarena.github.io / arxiv:2602.16313",
        "notes": (
            "Dataset may require access request; see https://memoryarena.github.io "
            "Place task files under data/memory_bench/memoryarena/tasks/"
        ),
    },
    "evomembench": {
        "description": "EvoMemBench: Self-evolving memory benchmark (arXiv 2605.18421)",
        "subpath": "",
        "url": "https://arxiv.org/abs/2605.18421",
        "notes": (
            "Dataset access instructions in the paper; place files under "
            "data/memory_bench/evomembench/"
        ),
    },
}

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _data_root() -> Path:
    """Return the root directory for all benchmark datasets."""
    env = os.environ.get("MEMORY_BENCH_DATA_DIR", "").strip()
    if env:
        return Path(env)
    # Walk up from this file and prefer the outer repository root.  In this
    # monorepo memory_bench/ also has its own pyproject.toml, but datasets live
    # at <repo>/data/memory_bench rather than <repo>/memory_bench/data/...
    candidate = Path(__file__).resolve()
    roots: list[Path] = []
    for _ in range(8):
        candidate = candidate.parent
        if (candidate / "pyproject.toml").is_file():
            roots.append(candidate)
    if roots:
        return roots[-1] / "data" / "memory_bench"
    return Path.cwd() / "data" / "memory_bench"


def resolve_path(name: str, *, strict: bool = True) -> Path | None:
    """Return the Path to the dataset directory (or None if not strict and missing).

    Raises DatasetNotFoundError when strict=True and the path doesn't exist.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'; registered: {sorted(_REGISTRY)}")
    base = _data_root() / name
    if not base.exists():
        if strict:
            reg = _REGISTRY[name]
            raise DatasetNotFoundError(name, base, reg)
        return None
    return base


def resolve_file(name: str, subpath: str | None = None, *, strict: bool = True) -> Path | None:
    """Return the Path to a specific file within a dataset directory."""
    base = resolve_path(name, strict=strict)
    if base is None:
        return None
    sub = subpath or _REGISTRY.get(name, {}).get("subpath", "")
    if sub:
        return base / sub
    return base


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------


def check_sha256(path: Path, expected: str) -> bool:
    """Return True if file matches expected sha256 hex digest."""
    if not expected:
        return True
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest() == expected
    except OSError:
        return False


def check_all(names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Check presence (and sha256 if registered) of all requested datasets.

    Returns a dict name -> {present: bool, path: str, notes: str, ...}.
    """
    targets = names or list(_REGISTRY)
    report: dict[str, dict[str, Any]] = {}
    for name in targets:
        reg = _REGISTRY.get(name, {})
        base = _data_root() / name
        present = base.exists()
        entry: dict[str, Any] = {
            "present": present,
            "path": str(base),
            "url": reg.get("url", ""),
            "notes": reg.get("notes", ""),
            "description": reg.get("description", ""),
        }
        if present:
            sub = reg.get("subpath", "")
            if sub:
                fp = base / sub
                entry["file_exists"] = fp.exists()
                expected_sha = reg.get("sha256", "")
                if expected_sha and fp.is_file():
                    entry["sha256_ok"] = check_sha256(fp, expected_sha)
        report[name] = entry
    return report


def print_check_report(report: dict[str, dict[str, Any]]) -> None:
    """Print a human-readable dataset availability table."""
    print(f"\n{'Dataset':<20} {'Status':<10} {'Notes / Download'}")
    print("-" * 80)
    for name, info in sorted(report.items()):
        status = "OK" if info["present"] else "MISSING"
        notes = info["notes"] if not info["present"] else info["path"]
        print(f"  {name:<18} {status:<10} {notes[:60]}")
    print()


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DatasetNotFoundError(FileNotFoundError):
    def __init__(self, name: str, path: Path, reg: dict[str, Any]) -> None:
        self.name = name
        self.path = path
        self.reg = reg
        msg = (
            f"\nDataset '{name}' not found at: {path}\n\n"
            f"  Description: {reg.get('description', '')}\n"
            f"  URL:         {reg.get('url', '')}\n"
            f"  Setup:       {reg.get('notes', '')}\n\n"
            f"  After downloading, create the directory:\n"
            f"    {path}\n"
        )
        super().__init__(msg)


def register_dataset(
    name: str,
    *,
    description: str = "",
    subpath: str = "",
    url: str = "",
    notes: str = "",
    sha256: str = "",
) -> None:
    """Register a custom dataset (for external plugins or tests)."""
    _REGISTRY[name] = {
        "description": description,
        "subpath": subpath,
        "url": url,
        "notes": notes,
        "sha256": sha256,
    }
