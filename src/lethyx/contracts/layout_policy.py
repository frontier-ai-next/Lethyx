"""Shared greenfield Python layout policy for plan-time and write-time gates."""

from pathlib import Path
from typing import TYPE_CHECKING

from lethyx.contracts.models import ContractIssue, PlanIR

if TYPE_CHECKING:
    from lethyx.contracts.policy_input import WorkspaceContext

_GREENFIELD_PY_NON_IMPL_TOPS = frozenset(
    {
        ".git",
        ".memory",
        ".Lethyx",
        ".LETHYX_scratch",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "doc",
        "docs",
        "frontend",
        "node_modules",
        "public",
        "reports",
        "test",
        "tests",
        "tmp",
        "venv",
    }
)
_GREENFIELD_PY_ALLOWED_ROOT_FILES = frozenset(
    {
        "asgi.py",
        "conftest.py",
        "manage.py",
        "noxfile.py",
        "setup.py",
        "tasks.py",
        "wsgi.py",
    }
)
_PROJECT_MARKERS = frozenset(
    {
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "package.json",
    }
)
_NESTED_SRC_PREFIXES = frozenset(
    {"backend", "server", "api", "app", "agents", "agent", "services", "service"}
)
_ALLOWED_NESTED_SRC_TOPS = frozenset({"backend", "server"})
_PRODUCTION_EXTENSIONS = frozenset({".py", ".pyi"})


def normalise_plan_path(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("/")


def _path_parts(path: str) -> list[str]:
    return [p for p in normalise_plan_path(path).split("/") if p and p != "."]


def collect_plan_paths(plan: PlanIR) -> list[str]:
    paths: list[str] = []
    for subtask in plan.subtasks:
        paths.extend(subtask.files_to_create)
        paths.extend(subtask.files_to_change)
        affected = getattr(subtask, "files_affected", ())
        if affected:
            paths.extend(affected)
        if subtask.proof is not None:
            paths.extend(subtask.proof.scope.files_under_test)
            paths.extend(subtask.proof.scope.changed_files_expected)
    return [normalise_plan_path(p) for p in paths if normalise_plan_path(p)]


def is_non_implementation_path(path: str) -> bool:
    parts = _path_parts(path)
    if not parts:
        return True
    lowered = [p.lower() for p in parts]
    top = lowered[0]
    if top in {"tests", "test", "docs", "doc", "frontend"}:
        return True
    if top in _GREENFIELD_PY_NON_IMPL_TOPS:
        return True
    name = lowered[-1]
    if len(parts) == 1 and name in _GREENFIELD_PY_ALLOWED_ROOT_FILES:
        return True
    if name.startswith("test_") or "tests" in lowered:
        return True
    return False


def is_python_implementation_path(path: str) -> bool:
    norm = normalise_plan_path(path)
    if not norm:
        return False
    if is_non_implementation_path(norm):
        return False
    ext = "." + norm.rsplit(".", 1)[-1].lower() if "." in norm else ""
    if ext not in _PRODUCTION_EXTENSIONS:
        return False
    return True


def _workspace_has_project_marker(workspace_root: Path | None, planned_paths: set[str]) -> bool:
    planned_lower = {p.lower() for p in planned_paths}
    if planned_lower & _PROJECT_MARKERS:
        return True
    if workspace_root is None or not workspace_root.is_dir():
        return False
    for marker in _PROJECT_MARKERS:
        if (workspace_root / marker).is_file():
            return True
    try:
        workspace_toml = workspace_root / "workspace.toml"
        if workspace_toml.is_file() and "multi_agent_gmas" in workspace_toml.read_text(
            encoding="utf-8", errors="replace"
        ):
            return True
    except OSError:
        return False
    return False


def is_greenfield_project(
    *,
    planned_paths: set[str],
    workspace_root: Path | None,
) -> bool:
    return _workspace_has_project_marker(workspace_root, planned_paths)


def existing_workspace_python_roots(workspace_root: Path | None) -> set[str]:
    """Top-level Python implementation roots already present on disk."""
    if workspace_root is None or not workspace_root.is_dir():
        return set()
    roots: set[str] = set()
    skip_dirs = {
        ".git",
        ".memory",
        ".Lethyx",
        ".LETHYX_scratch",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
        "tests",
        "test",
        "docs",
        "doc",
        "frontend",
    }
    try:
        for path in workspace_root.rglob("*.py"):
            try:
                rel = path.relative_to(workspace_root)
            except ValueError:
                continue
            parts = [p for p in rel.parts if p and p != "."]
            if not parts:
                continue
            lowered = [p.lower() for p in parts]
            if any(part in skip_dirs for part in lowered):
                continue
            name = lowered[-1]
            if len(parts) == 1 and name in _GREENFIELD_PY_ALLOWED_ROOT_FILES:
                continue
            if name.startswith("test_"):
                continue
            if lowered[0] != "src":
                roots.add(parts[0])
    except OSError:
        return set()
    return roots


def existing_workspace_src_package_roots(workspace_root: Path | None) -> set[str]:
    """Existing package roots under canonical ``src/<package>/...`` layout."""
    if workspace_root is None or not workspace_root.is_dir():
        return set()
    roots: set[str] = set()
    try:
        for path in workspace_root.rglob("*.py"):
            try:
                rel = path.relative_to(workspace_root)
            except ValueError:
                continue
            parts = [p for p in rel.parts if p and p != "."]
            if len(parts) < 3:
                continue
            lowered = [p.lower() for p in parts]
            if lowered[0] != "src":
                continue
            if lowered[-1].startswith("test_") or "tests" in lowered or "test" in lowered:
                continue
            roots.add(parts[1])
    except OSError:
        return set()
    return roots


def src_python_package_roots_from_paths(paths: set[str]) -> set[str]:
    roots: set[str] = set()
    for raw in paths:
        parts = _path_parts(raw)
        if len(parts) < 3 or parts[0].lower() != "src":
            continue
        if not parts[-1].lower().endswith(".py"):
            continue
        lowered = [p.lower() for p in parts]
        name = lowered[-1]
        if name.startswith("test_") or "tests" in lowered or "test" in lowered:
            continue
        roots.add(parts[1])
    return roots


def nested_src_python_package_roots_from_paths(paths: set[str]) -> dict[str, set[str]]:
    roots_by_top: dict[str, set[str]] = {}
    for raw in paths:
        parts = _path_parts(raw)
        lowered = [p.lower() for p in parts]
        if len(parts) < 4 or lowered[0] not in _ALLOWED_NESTED_SRC_TOPS:
            continue
        if lowered[1] != "src" or not lowered[-1].endswith(".py"):
            continue
        if lowered[-1].startswith("test_") or "tests" in lowered or "test" in lowered:
            continue
        roots_by_top.setdefault(lowered[0], set()).add(parts[2])
    return roots_by_top


def _layout_issue(
    *,
    subtask_id: str,
    path: str,
    message: str,
) -> ContractIssue:
    return ContractIssue(
        code="greenfield_python_src_layout_policy",
        severity="blocking",
        phase="plan",
        subtask_id=subtask_id,
        path=path,
        message=message,
    )


def _issue_for_single_path(
    path: str,
    *,
    subtask_id: str,
    planned_paths: set[str],
    existing_top_roots: set[str] | None = None,
    existing_src_roots: set[str] | None = None,
) -> ContractIssue | None:
    norm = normalise_plan_path(path)
    if not norm or not is_python_implementation_path(norm):
        return None
    parts = _path_parts(norm)
    lowered = [p.lower() for p in parts]
    top = lowered[0]
    existing_top_lower = {str(root).lower() for root in (existing_top_roots or set())}
    existing_src_lower = {str(root).lower() for root in (existing_src_roots or set())}

    if top != "src" and existing_top_lower and top in existing_top_lower:
        return None
    if existing_top_lower and len(parts) >= 2 and lowered[1] == "src" and top in existing_top_lower:
        return None

    if top == "src":
        if len(parts) < 3:
            return _layout_issue(
                subtask_id=subtask_id,
                path=norm,
                message=(
                    f"Plan declares `{norm}`, but greenfield Python "
                    "application/library modules must use canonical "
                    "`src/<package>/...` layout, not bare `src/*.py` or "
                    "`src/__init__.py`."
                ),
            )
        src_root = lowered[1]
        if existing_src_lower and src_root not in existing_src_lower:
            return _layout_issue(
                subtask_id=subtask_id,
                path=norm,
                message=(
                    f"Plan declares `{norm}`, but this workspace already has "
                    "canonical Python package root(s) under `src/`: "
                    f"{sorted(existing_src_lower)!r}. Keep new modules under "
                    "the existing package root instead of creating a parallel "
                    "`src/<package>/...` root."
                ),
            )
        return None

    if len(parts) >= 2 and lowered[1] == "src":
        if top in _ALLOWED_NESTED_SRC_TOPS:
            if len(parts) >= 4:
                return None
            return _layout_issue(
                subtask_id=subtask_id,
                path=norm,
                message=(
                    f"Plan declares `{norm}`, but full-stack backend Python "
                    "modules must use `backend/src/<package>/...`, not bare "
                    "`backend/src/*.py` or `backend/src/__init__.py`."
                ),
            )
        return _layout_issue(
            subtask_id=subtask_id,
            path=norm,
            message=(
                f"Plan declares `{norm}`, but greenfield Python "
                "application/library modules must use canonical "
                "`src/<package>/...` layout. Move backend modules under "
                "one package root, e.g. `src/<package>/backend/...`, and "
                "keep tests under `tests/`."
            ),
        )

    if top in _NESTED_SRC_PREFIXES and len(parts) >= 2:
        return _layout_issue(
            subtask_id=subtask_id,
            path=norm,
            message=(
                f"Plan declares `{norm}`, but greenfield Python "
                "application/library modules must use canonical "
                "`src/<package>/...` layout instead of parallel top-level "
                f"packages such as `{top}/...`."
            ),
        )

    if len(parts) >= 2 and parts[-1].lower().endswith(".py"):
        return _layout_issue(
            subtask_id=subtask_id,
            path=norm,
            message=(
                f"Plan declares `{norm}`, but greenfield Python "
                "application/library modules must live under "
                "`src/<package>/...`, not as top-level module trees."
            ),
        )
    return None


def _greenfield_layout_policy_enabled(context: "WorkspaceContext | None") -> bool:
    if context is None or not getattr(context, "workspace_root", None):
        return False
    from lethyx.workspace_registry.charter import load_workspace_charter

    charter = load_workspace_charter(Path(context.workspace_root))
    policies = charter.get("policies") if isinstance(charter.get("policies"), dict) else {}
    return bool(policies.get("greenfield_python_src_layout"))


def validate_plan_layout_policy(
    plan: PlanIR,
    *,
    context: "WorkspaceContext | None" = None,
) -> list[ContractIssue]:
    if not _greenfield_layout_policy_enabled(context):
        return []
    issues: list[ContractIssue] = []
    all_paths = collect_plan_paths(plan)
    planned_set = set(all_paths)
    workspace_root = (
        Path(context.workspace_root).resolve()
        if context is not None and getattr(context, "workspace_root", None)
        else None
    )
    if not is_greenfield_project(planned_paths=planned_set, workspace_root=workspace_root):
        return issues

    existing_top_roots = existing_workspace_python_roots(workspace_root)
    existing_src_roots = existing_workspace_src_package_roots(workspace_root)

    src_roots = src_python_package_roots_from_paths(planned_set)
    nested_src_roots = nested_src_python_package_roots_from_paths(planned_set)
    if len(src_roots) > 1:
        issues.append(
            _layout_issue(
                subtask_id="",
                path="src/",
                message=(
                    "Greenfield Python code under `src/` must use one canonical "
                    f"package root (`src/<package>/...`), not parallel roots "
                    f"{sorted(src_roots)!r}. Place related modules under one "
                    "package, e.g. `src/<package>/api/...` and "
                    "`src/<package>/agents/...`."
                ),
            )
        )
    elif existing_src_roots and src_roots:
        new_src_roots = {root.lower() for root in src_roots} - {
            root.lower() for root in existing_src_roots
        }
        if new_src_roots:
            issues.append(
                _layout_issue(
                    subtask_id="",
                    path="src/",
                    message=(
                        "Greenfield Python code under `src/` must stay under "
                        "the existing canonical package root(s) "
                        f"{sorted(existing_src_roots)!r}; do not add parallel "
                        f"package roots {sorted(src_roots)!r}."
                    ),
                )
            )
    for top, roots in sorted(nested_src_roots.items()):
        if len(roots) > 1:
            issues.append(
                _layout_issue(
                    subtask_id="",
                    path=f"{top}/src/",
                    message=(
                        "Greenfield full-stack Python code under "
                        f"`{top}/src/` must use one canonical package root "
                        f"(`{top}/src/<package>/...`), not parallel roots "
                        f"{sorted(roots)!r}."
                    ),
                )
            )

    for subtask in plan.subtasks:
        sub_paths = set()
        sub_paths.update(subtask.files_to_create)
        sub_paths.update(subtask.files_to_change)
        affected = getattr(subtask, "files_affected", ())
        if affected:
            sub_paths.update(affected)
        if subtask.proof is not None:
            sub_paths.update(subtask.proof.scope.files_under_test)
            sub_paths.update(subtask.proof.scope.changed_files_expected)
        for path in sub_paths:
            issue = _issue_for_single_path(
                path,
                subtask_id=subtask.id,
                planned_paths=planned_set,
                existing_top_roots=existing_top_roots,
                existing_src_roots=existing_src_roots,
            )
            if issue is not None:
                issues.append(issue)
    return issues
