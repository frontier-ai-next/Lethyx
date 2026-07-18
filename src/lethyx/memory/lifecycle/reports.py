"""Lifecycle report writers."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from lethyx.memory.lifecycle.models import LifecycleProposal


def write_lifecycle_report(
    drive_root: Path, proposals: list[LifecycleProposal]
) -> dict[str, object]:
    reports = Path(drive_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "proposal_count": len(proposals),
        "proposals": [asdict(p) for p in proposals],
        "self_promotion_allowed": False,
    }
    latest = reports / "memory_lifecycle_report_latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = reports / f"memory_lifecycle_report_{time.strftime('%Y%m%d%H%M%S', time.gmtime())}.md"
    md.write_text(
        "# Memory lifecycle report\n\n"
        + "\n".join(f"- {p.rule_id}: {p.action} ({p.reason})" for p in proposals),
        encoding="utf-8",
    )
    return payload
