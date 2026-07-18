"""Generate markdown/LaTeX metric tables."""

from typing import Any


def markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| condition | directive_precision | retrieval_leakage_rate | contamination_rate |\n|---|---:|---:|---:|"
    body = [
        f"| {r.get('condition')} | {r.get('directive_precision', 0):.3f} | "
        f"{r.get('retrieval_leakage_rate', 0):.3f} | {r.get('contamination_rate', 0):.3f} |"
        for r in rows
    ]
    return "\n".join([header, *body])


def latex_table(rows: list[dict[str, Any]]) -> str:
    newline = r"\\"
    lines = [
        r"\begin{tabular}{lrrr}",
        "Condition & Precision & Leakage & Contamination " + newline,
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r.get('condition')} & {r.get('directive_precision', 0):.3f} & "
            f"{r.get('retrieval_leakage_rate', 0):.3f} & {r.get('contamination_rate', 0):.3f} "
            + newline
        )
    lines.append(r"\end{tabular}")
    return "\n".join(lines)
