"""Reproducibility manifest for memory_bench runs.

Written to runs/memory_bench/<ts>/manifest.json alongside results.
Records everything needed to reproduce or cite a run:
- provider IDs and capability declarations.
- dataset names, paths, and sha256 checksums.
- reader model + endpoint.
- seed and concurrency settings.
- git SHA of the repository.
- run timestamp and Python version.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def file_sha256(path: Path) -> str:
    """Compute sha256 of a file for dataset integrity verification."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _provider_info(provider: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"name": getattr(provider, "name", str(provider))}
    try:
        caps = provider.capabilities()
        info["capabilities"] = caps.to_dict()
    except Exception:
        pass
    try:
        import importlib.metadata as im

        pkg = getattr(provider, "_package", None)
        if pkg:
            info["version"] = im.version(pkg)
    except Exception:
        pass
    return info


def _lethyx_version() -> str:
    try:
        import importlib.metadata as im

        return im.version("lethyx")
    except Exception:
        pass
    try:
        root = Path(__file__).resolve().parents[4]
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            for line in pyproject.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("version"):
                    return line.split("=", 1)[-1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "unknown"


def build_manifest(
    *,
    spec: Any,
    providers: list[Any],
    dataset_paths: dict[str, Path],
    repo_root: Path,
) -> dict[str, Any]:
    """Build the full manifest dict from a RunSpec and resolved objects."""
    from memory_bench.core.types import RunSpec

    datasets: dict[str, Any] = {}
    for name, path in dataset_paths.items():
        entry: dict[str, Any] = {"path": str(path)}
        if path.is_file():
            entry["sha256"] = file_sha256(path)
        elif path.is_dir():
            # checksum the directory's first-level file list for repeatability
            files = sorted(path.iterdir())
            h = hashlib.sha256()
            for f in files:
                h.update(f.name.encode())
            entry["dir_hash"] = h.hexdigest()
            entry["n_files"] = len(files)
        datasets[name] = entry

    run_spec: RunSpec | None = spec if isinstance(spec, RunSpec) else None

    provider_entries = [_provider_info(p) for p in providers]

    return {
        "schema": "memory_bench/manifest/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": _git_sha(repo_root),
        "lethyx_version": _lethyx_version(),
        "providers": provider_entries,
        "datasets": datasets,
        "responder": {
            "model": (run_spec.responder.model if run_spec else ""),
            "temperature": (run_spec.responder.temperature if run_spec else 0.0),
        },
        "judge": {
            "llm_judge_model": (run_spec.judge.llm_judge_model if run_spec else ""),
            "use_llm_judge": (run_spec.judge.use_llm_judge if run_spec else True),
        },
        "limits": {
            "recall_k": (run_spec.limits.recall_k if run_spec else 10),
            "seed": (run_spec.limits.seed if run_spec else 0),
            "max_parallel": (run_spec.limits.max_parallel if run_spec else 1),
        },
    }


def write_manifest(manifest: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
