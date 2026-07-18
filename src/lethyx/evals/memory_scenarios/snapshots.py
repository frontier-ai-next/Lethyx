"""Before/after snapshots for memory scenario steps."""

import json
from pathlib import Path
from typing import Any

from lethyx.memory.paths import manager_core_root, workspace_core_root
from lethyx.memory.palace.facade import MemPalace


def _read_core_files(core_root: Path) -> dict[str, str]:
    if not core_root.is_dir():
        return {}
    return {
        p.name: p.read_text(encoding="utf-8") for p in sorted(core_root.iterdir()) if p.is_file()
    }


def palace_snapshot(repo: Path, workspace_id: str, *, n: int = 200) -> dict[str, Any]:
    palace = MemPalace(repo, workspace_id)
    try:
        try:
            nodes = palace.list_all(n=n)
        except Exception as exc:
            return {"count": 0, "nodes": [], "available": False, "error": str(exc)}
        return {"count": len(nodes), "nodes": nodes}
    finally:
        palace.close()


def drive_state_snapshot(drive: Path) -> dict[str, Any]:
    state = drive / "state"
    if not state.is_dir():
        return {"files": []}
    files = sorted(p.name for p in state.iterdir() if p.is_file())
    return {"files": files}


def capture_snapshot(
    repo: Path,
    workspace_id: str,
    drive: Path | None,
) -> dict[str, Any]:
    ws_core = workspace_core_root(repo, workspace_id)
    mgr_core = manager_core_root(repo)
    bkb_before = ""
    bkb_path = ws_core / "bkb.yaml"
    if bkb_path.is_file():
        bkb_before = bkb_path.read_text(encoding="utf-8")
    mgr_bkb = mgr_core / "bkb.yaml"
    return {
        "palace": palace_snapshot(repo, workspace_id),
        "bkb_yaml": bkb_before,
        "manager_bkb_yaml": mgr_bkb.read_text(encoding="utf-8") if mgr_bkb.is_file() else "",
        "core_files": _read_core_files(ws_core),
        "manager_core_files": _read_core_files(mgr_core),
        "drive_state": drive_state_snapshot(drive) if drive else {},
    }


def write_step_snapshots(
    report_dir: Path,
    step_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"palace_before_{step_id}.json").write_text(
        json.dumps(before.get("palace", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / f"palace_after_{step_id}.json").write_text(
        json.dumps(after.get("palace", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / f"bkb_before_{step_id}.yaml").write_text(
        before.get("bkb_yaml") or "",
        encoding="utf-8",
    )
    (report_dir / f"bkb_after_{step_id}.yaml").write_text(
        after.get("bkb_yaml") or "",
        encoding="utf-8",
    )
    (report_dir / f"core_files_before_{step_id}.json").write_text(
        json.dumps(before.get("core_files") or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / f"core_files_after_{step_id}.json").write_text(
        json.dumps(after.get("core_files") or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / f"drive_state_before_{step_id}.json").write_text(
        json.dumps(before.get("drive_state") or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / f"drive_state_after_{step_id}.json").write_text(
        json.dumps(after.get("drive_state") or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    diff_lines = ["# Snapshot diff", ""]
    if before.get("bkb_yaml") != after.get("bkb_yaml"):
        diff_lines.append("- BKB yaml changed")
    b_palace = (before.get("palace") or {}).get("count", 0)
    a_palace = (after.get("palace") or {}).get("count", 0)
    if b_palace != a_palace:
        diff_lines.append(f"- Palace node count: {b_palace} -> {a_palace}")
    (report_dir / f"diff_{step_id}.md").write_text("\n".join(diff_lines), encoding="utf-8")
