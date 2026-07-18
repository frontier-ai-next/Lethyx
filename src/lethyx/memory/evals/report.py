"""Static eval report writers."""

import json
import time
from pathlib import Path


def write_ablation_report(drive_root: Path, payload: dict[str, object]) -> dict[str, Path]:
    reports = Path(drive_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    latest = reports / "memory_ablation_report_latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = reports / f"memory_ablation_report_{time.strftime('%Y%m%d%H%M%S', time.gmtime())}.md"
    rows = payload.get("conditions", []) if isinstance(payload, dict) else []
    md.write_text(
        "# Memory ablation report\n\n"
        + "\n".join(
            f"- {r['condition']}: leakage={r['retrieval_leakage_rate']} contamination={r['contamination_rate']}"
            for r in rows
            if isinstance(r, dict)
        ),
        encoding="utf-8",
    )
    return {"json": latest, "markdown": md}
