"""Generate deterministic research artifacts from static evals."""

import csv
import json
from pathlib import Path

from lethyx.memory.evals.ablation import run_static_ablation
from lethyx.memory.research.latex_tables import latex_table, markdown_table
from lethyx.memory.research.paper_metrics import flatten_ablation_results


def write_research_reports(drive_root: Path) -> dict[str, str]:
    reports = Path(drive_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    payload = run_static_ablation()
    rows = flatten_ablation_results(payload)
    json_path = reports / "memory_ablation_results.json"
    csv_path = reports / "memory_ablation_results.csv"
    md_path = reports / "memory_metrics_table.md"
    tex_path = reports / "memory_metrics_table.tex"
    failures_path = reports / "memory_failure_cases.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()) if rows else ["condition"])
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown_table(rows), encoding="utf-8")
    tex_path.write_text(latex_table(rows), encoding="utf-8")
    failures_path.write_text(
        "# Memory failure cases\n\nStatic eval checks candidate poisoning and retrieval-only contamination.\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "latex": str(tex_path),
        "failure_cases": str(failures_path),
    }
