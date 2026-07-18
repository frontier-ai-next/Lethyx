"""Repo/workspace preparation for memory scenarios."""

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACES_TEST_SRC = REPO_ROOT / "workspaces" / "test"
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures" / "memory_scenarios"
SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
EXPECTED_ROOT = FIXTURES_ROOT / "expected"


def manifest_path(manifest_name: str) -> Path:
    return Path(__file__).resolve().parent / "manifests" / f"{manifest_name}.yaml"


def fixture_core_dir(name: str) -> Path:
    return FIXTURES_ROOT / "core" / name


def overlay_core_files(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def apply_default_workspace_memory(repo_root: Path, workspace_id: str = "test") -> None:
    """Overlay committed default workspace core fixture (for conftest / live tests)."""
    ws_core = repo_root / "workspaces" / workspace_id / ".memory" / "core"
    overlay_core_files(fixture_core_dir("default"), ws_core)
    apply_builtin_core_fixture("default", ws_core)


def ensure_manager_core(repo_root: Path, fixture_name: str = "manager_default") -> None:
    core = repo_root / ".Lethyx" / "memory" / "core"
    overlay_core_files(fixture_core_dir(fixture_name), core)
    apply_builtin_core_fixture(fixture_name, core)
    core.mkdir(parents=True, exist_ok=True)
    if not (core / "10_operating_principles.md").is_file():
        (core / "10_operating_principles.md").write_text(
            "# Principles\nUse typed evidence refs.\n",
            encoding="utf-8",
        )


def prepare_scenario_repo(
    tmp_root: Path,
    *,
    workspace_id: str = "test",
    workspace_fixture: str = "default",
    manager_fixture: str = "manager_default",
    extra_workspaces: dict[str, str] | None = None,
) -> Path:
    """Copy workspaces/test seed and overlay memory fixtures into tmp_root."""
    (tmp_root / "Lethyx").mkdir(parents=True, exist_ok=True)
    dst_ws = tmp_root / "workspaces" / workspace_id
    if dst_ws.exists():
        shutil.rmtree(dst_ws)
    if WORKSPACES_TEST_SRC.is_dir():
        shutil.copytree(WORKSPACES_TEST_SRC, dst_ws)
    else:
        _create_default_workspace(dst_ws, workspace_id)
    overlay_core_files(fixture_core_dir(workspace_fixture), dst_ws / ".memory" / "core")
    apply_builtin_core_fixture(workspace_fixture, dst_ws / ".memory" / "core")
    ensure_manager_core(tmp_root, manager_fixture)
    for ws_name, fix_name in (extra_workspaces or {}).items():
        ws_path = tmp_root / "workspaces" / ws_name
        if not ws_path.is_dir():
            if WORKSPACES_TEST_SRC.is_dir():
                shutil.copytree(WORKSPACES_TEST_SRC, ws_path)
            else:
                _create_default_workspace(ws_path, ws_name)
        overlay_core_files(fixture_core_dir(fix_name), ws_path / ".memory" / "core")
        apply_builtin_core_fixture(fix_name, ws_path / ".memory" / "core")
    return tmp_root


def drive_root(repo: Path, workspace_id: str) -> Path:
    drive = repo / "workspaces" / workspace_id / ".memory" / "drive"
    drive.mkdir(parents=True, exist_ok=True)
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    (drive / "state").mkdir(parents=True, exist_ok=True)
    return drive


def _create_default_workspace(path: Path, workspace_id: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text("# Memory scenario workspace\n", encoding="utf-8")
    (path / "TASK_MAIN.md").write_text(
        "# TASK: memory scenario\n\n## Goal\nExercise Lethyx memory prompts.\n",
        encoding="utf-8",
    )
    (path / "workspace.toml").write_text(
        f'[workspace]\nid = "{workspace_id}"\nname = "{workspace_id}"\n',
        encoding="utf-8",
    )


def apply_builtin_core_fixture(name: str, core_root: Path) -> None:
    """Apply tiny built-in fixtures when external fixture dirs are absent."""

    fixture = str(name or "").strip()
    if not fixture:
        return
    core_root.mkdir(parents=True, exist_ok=True)
    if fixture == "default":
        bkb_path = core_root / "bkb.yaml"
        if not bkb_path.is_file():
            bkb_path.write_text(
                """
rules:
  - id: test_active_provenance
    title: Require source_id for research provenance
    scope: workspace
    type: behavior
    status: active
    trust: verified
    rule:
      behavior: "Record source_id before using research findings."
    applies_to:
      phases: [research]
      workspaces: ["*"]
      agents: ["*"]
  - id: test_candidate_only
    title: Candidate rule must not inject
    scope: workspace
    type: behavior
    status: candidate
    trust: candidate
    rule:
      behavior: "inject me please"
    applies_to:
      phases: ["*"]
      workspaces: ["*"]
      agents: ["*"]
""".strip()
                + "\n",
                encoding="utf-8",
            )
    elif fixture == "ws_a":
        (core_root / "00_workspace_charter.md").write_text(
            "# Workspace Charter\nUse API A for this workspace.\n",
            encoding="utf-8",
        )
    elif fixture == "ws_b":
        (core_root / "00_workspace_charter.md").write_text(
            "# Workspace Charter\nUse API B for this workspace.\n",
            encoding="utf-8",
        )
