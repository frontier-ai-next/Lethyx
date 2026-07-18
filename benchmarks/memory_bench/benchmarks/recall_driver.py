"""Common driver for all recall-QA benchmarks.

Handles the ingest -> retrieve -> answer -> judge pipeline.
Called by runner.run_pair() for benchmarks with kind="recall".
"""

import logging
import time
from typing import Any

from memory_bench.core.metrics import retrieval_metrics
from memory_bench.core.types import CaseResult, Episode, Probe, RunSpec

log = logging.getLogger(__name__)

_DIAGNOSTIC_METADATA_KEYS = {
    "backend",
    "bench_kind",
    "block_id",
    "effective_mode",
    "evidence_id",
    "kind",
    "lifecycle",
    "native_backend",
    "phase_id",
    "raw_backend",
    "scope",
    "source_path",
    "surface",
    "tags",
    "tier",
    "workspace_id",
}


def _snippet(text: str, *, limit: int = 800) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in list(value.items())[:20]}
    return str(value)


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe(value)
        for key, value in (metadata or {}).items()
        if str(key) in _DIAGNOSTIC_METADATA_KEYS
    }


def _retrieval_diagnostics(retrieved: list[Any], relevant_ids: set[str]) -> dict[str, Any]:
    retrieved_ids = [str(item.source_id) for item in retrieved if getattr(item, "source_id", "")]
    evidence_ranks: dict[str, int] = {}
    context: list[dict[str, Any]] = []
    for index, item in enumerate(retrieved):
        rank = index + 1
        source_id = str(getattr(item, "source_id", "") or "")
        if source_id and source_id in relevant_ids and source_id not in evidence_ranks:
            evidence_ranks[source_id] = rank
        try:
            score = float(getattr(item, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        context.append(
            {
                "rank": rank,
                "source_id": source_id,
                "score": round(score, 6),
                "text": _snippet(str(getattr(item, "text", "") or "")),
                "metadata": _safe_metadata(dict(getattr(item, "metadata", {}) or {})),
            }
        )
    return {
        "retrieved_ids": retrieved_ids,
        "relevant_ids": sorted(relevant_ids),
        "missing_evidence": sorted(relevant_ids - set(retrieved_ids)),
        "evidence_ranks": evidence_ranks,
        "retrieved_context": context,
    }


def _failure_mode(*, probe: Probe, answer: Any, judgment: Any, missing_evidence: list[str]) -> str:
    if judgment is not None and getattr(judgment, "correct", False):
        return "ok"
    if probe.abstain and not getattr(answer, "abstained", False):
        return "abstention_failure"
    if missing_evidence:
        return "retrieval_miss"
    if getattr(answer, "abstained", False) and not probe.abstain:
        return "reader_abstained"
    if not str(getattr(answer, "text", "") or "").strip():
        return "reader_empty_answer"
    return "answer_or_judge_mismatch"


def run_recall_episode(
    *,
    store: Any,
    episode: Episode,
    probe: Probe,
    responder: Any,
    judge: Any,
    spec: RunSpec,
    provider_id: str,
    benchmark_name: str,
) -> CaseResult:
    """Evaluate one probe against the ingested episode store.

    Flow: retrieve -> answer -> judge -> compute retrieval metrics -> CaseResult
    """
    t0 = time.monotonic()

    # 1. Retrieve
    retrieved = store.retrieve(
        probe.question,
        k=spec.limits.recall_k,
        max_tokens=spec.limits.max_context_tokens,
    )

    # 2. Answer
    answer = responder.answer(probe.question, retrieved)

    # 3. Judge
    judgment = judge.judge(probe, answer)

    # 4. Retrieval metrics (recall@k, MRR, nDCG) based on evidence_ids
    retrieved_ids = [item.source_id for item in retrieved if item.source_id]
    relevant_ids = set(probe.evidence_ids)
    ret_metrics = retrieval_metrics(retrieved_ids, relevant_ids, spec.limits.recall_k)
    retrieval_diag = _retrieval_diagnostics(retrieved, relevant_ids)
    missing_evidence = retrieval_diag["missing_evidence"]

    latency = (time.monotonic() - t0) * 1000

    return CaseResult(
        provider=provider_id,
        benchmark=benchmark_name,
        episode_id=episode.id,
        probe_id=probe.id,
        category=probe.category,
        status="ok",
        judgment=judgment,
        retrieval={**ret_metrics, "n_context_items": len(retrieved)},
        usage=answer.usage,
        latency_ms=latency,
        diagnostics={
            "question": probe.question,
            "gold": list(probe.gold),
            "expected_abstain": probe.abstain,
            "answer_text": _snippet(answer.text, limit=1200),
            "answer_abstained": answer.abstained,
            "failure_mode": _failure_mode(
                probe=probe,
                answer=answer,
                judgment=judgment,
                missing_evidence=missing_evidence,
            ),
            **retrieval_diag,
        },
    )
