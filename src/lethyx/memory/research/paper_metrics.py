"""Flatten eval metrics for paper artifacts."""

from typing import Any


def flatten_ablation_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("conditions") if isinstance(payload, dict) else []
    return [dict(row) for row in rows if isinstance(row, dict)]
