import pytest

from memory_bench.adapters.lethyx_provider import (
    _CompositeLethyxStore,
    _fail_if_native_requested_but_not_strict,
)
from memory_bench.core.text import format_context
from memory_bench.core.types import RetrievedItem


class _Store:
    def __init__(self, hits: list[RetrievedItem]) -> None:
        self._hits = hits

    def retrieve(self, query: str, *, k: int = 10, max_tokens: int = 0, filters=None):
        return list(self._hits)[:k]


class _Backend:
    def __init__(self, health: dict[str, object]) -> None:
        self._health = health

    def health(self) -> dict[str, object]:
        return dict(self._health)


def test_format_context_can_render_source_and_temporal_metadata() -> None:
    context = format_context(
        [
            RetrievedItem(
                text="Caroline said she met the support group on 7 May.",
                score=0.7,
                source_id="D9:11",
                metadata={
                    "role": "user",
                    "session_id": "session-9",
                    "ts": "2023-05-07T10:00:00Z",
                    "evidence_id": "D9:11",
                    "tags": ["speaker:Caroline", "topic:support"],
                },
            )
        ],
        include_metadata=True,
    )

    assert "source_id=D9:11" in context
    assert "role=user" in context
    assert "session=session-9" in context
    assert "ts=2023-05-07T10:00:00Z" in context
    assert "speaker:Caroline" in context
    assert "Caroline said she met" in context


def test_benchmark_composite_uses_rank_fusion_instead_of_raw_score_sorting() -> None:
    store = _CompositeLethyxStore(
        {
            "first": _Store(
                [RetrievedItem(text="first child top hit", score=0.05, source_id="first")]
            ),
            "second": _Store(
                [RetrievedItem(text="second child high-scale hit", score=1.0, source_id="second")]
            ),
        }
    )

    hits = store.retrieve("query", k=2)

    assert [hit.source_id for hit in hits] == ["first", "second"]
    assert (
        hits[0].metadata["composite_rank_fusion_score"]
        == hits[1].metadata["composite_rank_fusion_score"]
    )
    assert hits[0].metadata["raw_backend_score"] == 0.05
    assert hits[1].metadata["raw_backend_score"] == 1.0


def test_native_requested_backends_fail_instead_of_silent_sidecar_fallback() -> None:
    with pytest.raises(RuntimeError, match="native Hindsight"):
        _fail_if_native_requested_but_not_strict(
            _Backend(
                {
                    "native_ok": False,
                    "strict_ok": False,
                    "effective_mode": "sqlite_sidecar",
                    "reason": "native_disabled",
                }
            ),
            "hindsight",
        )


def test_native_requested_mempalace_allows_native_effective_mode_without_sidecar() -> None:
    _fail_if_native_requested_but_not_strict(
        _Backend(
            {
                "native_ok": True,
                "strict_ok": True,
                "ok": True,
                "effective_mode": "native",
            }
        ),
        "mempalace",
    )
