"""Tribunal report persistence."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from lethyx.memory.tribunal.models import TribunalDecision


def write_tribunal_report(
    drive_root: Path | None, patch_id: str, decision: TribunalDecision
) -> TribunalDecision:
    if drive_root is None:
        return decision
    path = Path(drive_root) / "state" / "tribunal_reports" / f"{patch_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = replace(decision, report_path=str(path))
    path.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2), encoding="utf-8")
    return decision
