from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_NAMESPACE = "umbr" + "ella"
LEGACY_HOST = "ouro" + "boros"
LEGACY_REFERENCE_DIR = "external" + "_references"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_removed_legacy_namespace_and_integration_paths() -> None:
    forbidden_paths = [
        ROOT / "src" / LEGACY_NAMESPACE,
        ROOT / "tests" / LEGACY_NAMESPACE,
        ROOT / "integrations" / LEGACY_NAMESPACE,
        ROOT / "benchmarks" / "memory_bench" / "adapters" / f"{LEGACY_NAMESPACE}_provider.py",
        ROOT / "EXTRACTION_MANIFEST.md",
        ROOT / LEGACY_REFERENCE_DIR,
    ]

    assert [str(path.relative_to(ROOT)) for path in forbidden_paths if path.exists()] == []


def test_release_package_has_no_legacy_runtime_imports() -> None:
    roots = [
        ROOT / "src" / "lethyx",
        ROOT / "benchmarks" / "memory_bench" / "adapters" / "lethyx_provider.py",
    ]
    forbidden = (
        f"from {LEGACY_NAMESPACE}",
        f"import {LEGACY_NAMESPACE}",
        f"{LEGACY_NAMESPACE}.",
    )
    offenders: list[str] = []

    for root in roots:
        files = [root] if root.is_file() else list(root.rglob("*.py"))
        for path in files:
            text = _read(path)
            for needle in forbidden:
                if needle in text:
                    offenders.append(f"{path.relative_to(ROOT)} contains {needle!r}")

    assert offenders == []


def test_benchmark_suite_uses_lethyx_provider_ids_only() -> None:
    files = [
        ROOT / "benchmarks" / "memory_bench" / "suite.toml",
        ROOT / "benchmarks" / "memory_bench" / "scripts" / "full_bench.py",
        ROOT / "benchmarks" / "memory_bench" / "core" / "registry.py",
    ]
    text = "\n".join(_read(path) for path in files)

    assert f'adapter = "{LEGACY_NAMESPACE}"' not in text
    assert f"{LEGACY_NAMESPACE}-" not in text
    assert 'adapter = "lethyx"' in text
    assert "lethyx-mempalace" in text


def test_packaging_exposes_only_lethyx_namespace() -> None:
    pyproject = _read(ROOT / "pyproject.toml")

    assert 'include = ["lethyx*"]' in pyproject
    assert f'include = ["{LEGACY_NAMESPACE}*"' not in pyproject
    assert 'lethyx = "lethyx.memory.cli:main"' in pyproject
    assert 'lethyx-scenarios = "lethyx.evals.memory_scenarios.cli:main"' in pyproject


def test_release_tree_has_no_placeholder_modules() -> None:
    placeholders = [
        ROOT / "src" / "lethyx" / "memory" / "external_projects" / "github_ingest.py",
    ]

    assert [str(path.relative_to(ROOT)) for path in placeholders if path.exists()] == []


def test_release_tree_has_no_legacy_project_terms() -> None:
    terms = (
        LEGACY_NAMESPACE,
        LEGACY_NAMESPACE.capitalize(),
        LEGACY_NAMESPACE.upper(),
        LEGACY_HOST,
        LEGACY_HOST.capitalize(),
        LEGACY_HOST.upper(),
        LEGACY_REFERENCE_DIR,
    )
    ignored_dirs = {
        ".external",
        ".git",
        ".hermes",
        ".memory_bench_cache",
        ".mrt",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "data",
        "dist",
        "runs",
        "venv",
    }
    ignored_prefixes = {
        Path("benchmarks") / "memory_bench" / ".memory_bench_raw",
        Path("benchmarks") / "memory_bench" / "workspaces",
    }
    offenders: list[str] = []

    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in ignored_dirs for part in path.parts):
            continue
        if any(rel == prefix or prefix in rel.parents for prefix in ignored_prefixes):
            continue
        if not path.is_file() or path.suffix.lower() in {".pyc", ".sqlite", ".db"}:
            continue
        try:
            text = _read(path)
        except OSError:
            continue
        for term in terms:
            if term in text or term in path.name:
                offenders.append(f"{path.relative_to(ROOT)} contains legacy term")
                break

    assert offenders == []
