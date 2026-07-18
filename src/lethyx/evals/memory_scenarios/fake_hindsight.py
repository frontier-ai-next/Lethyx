"""Fake Hindsight client for offline memory scenarios."""

import json
from pathlib import Path
from typing import Any

from lethyx.memory.backends.base import ReflectionCandidate


class FakeBanks:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.missions: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, str]:
        self.created.append(kwargs)
        return {"ok": "true"}

    def set_mission(self, **kwargs: Any) -> dict[str, str]:
        self.missions.append(kwargs)
        return {"ok": "true"}

    def list(self) -> list[dict[str, Any]]:
        return []


class FakeHindsightClient:
    """Records retain/recall/reflect calls for audit."""

    def __init__(self) -> None:
        self.banks = FakeBanks()
        self.retained: list[dict[str, Any]] = []
        self.recall_rows: list[dict[str, Any]] = []
        self.reflect_payload: dict[str, Any] = {"candidates": []}
        self.reflect_kwargs: dict[str, Any] = {}

    def retain(self, **kwargs: Any) -> dict[str, Any]:
        self.retained.append(kwargs)
        return {"id": kwargs.get("document_id"), "status": "ok"}

    def recall(self, **kwargs: Any) -> dict[str, Any]:
        return {"results": list(self.recall_rows)}

    def reflect(self, **kwargs: Any) -> dict[str, Any]:
        self.reflect_kwargs = kwargs
        return self.reflect_payload


class FakeHindsightCallLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, payload: dict[str, Any]) -> None:
        row = {"event": event, **payload}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_reflect_candidate(evidence: list[dict[str, Any]]) -> ReflectionCandidate:
    return ReflectionCandidate(
        candidate_id="fake-c1",
        kind="behavior",
        title="Fake hindsight candidate",
        content="Hindsight reflect candidate for harness gate test.",
        confidence=0.85,
        scope="workspace",
        evidence_refs=evidence,
    )
