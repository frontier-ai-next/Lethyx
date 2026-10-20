"""Out-of-prompt recall explanations and fixed delivery quality gates."""

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from lethyx.memory.backends.base import MemoryHit
from lethyx.memory.context_payload import PackedEvidencePayload
from lethyx.memory.recall_plan import RecallPlan


_CANDIDATE_RECALL_THRESHOLD = 1.0
_RENDERED_LITERAL_COVERAGE_THRESHOLD = 1.0


def _hit_id(hit: MemoryHit) -> str:
    return str(hit.metadata.get("block_id") or hit.source or "")


@dataclass(frozen=True)
class RecallQualityGate:
    candidate_recall_threshold: float
    rendered_literal_coverage_threshold: float
    candidate_recall: float
    rendered_literal_coverage: float
    passed: bool
    status: str


@dataclass(frozen=True)
class RecallExplanation:
    query: str
    query_type: str
    workspace_id: str
    run_id: str
    backend_id: str
    plan_reason: str
    candidate_record_ids: tuple[str, ...]
    rendered_record_ids: tuple[str, ...]
    budget: dict[str, int]
    visibility: tuple[dict[str, Any], ...]
    exact_literal_capsule: tuple[str, ...]
    exact_literal_losses: tuple[dict[str, Any], ...]
    canonical_source_refs: tuple[dict[str, Any], ...]
    corrective_recall: dict[str, Any]
    quality_gate: RecallQualityGate
    rendered_context_sha256: str
    prompt_includes_trace: bool = False

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_record_ids"] = list(self.candidate_record_ids)
        payload["rendered_record_ids"] = list(self.rendered_record_ids)
        payload["exact_literal_capsule"] = list(self.exact_literal_capsule)
        payload["visibility"] = list(self.visibility)
        payload["exact_literal_losses"] = list(self.exact_literal_losses)
        payload["canonical_source_refs"] = list(self.canonical_source_refs)
        return payload


def _quality_gate(
    *,
    candidate_ids: tuple[str, ...],
    rendered_context: str,
    gold_block_ids: tuple[str, ...],
    gold_literals: tuple[str, ...],
) -> RecallQualityGate:
    if not gold_block_ids and not gold_literals:
        return RecallQualityGate(
            candidate_recall_threshold=_CANDIDATE_RECALL_THRESHOLD,
            rendered_literal_coverage_threshold=_RENDERED_LITERAL_COVERAGE_THRESHOLD,
            candidate_recall=0.0,
            rendered_literal_coverage=0.0,
            passed=False,
            status="not_evaluated",
        )
    candidate_recall = (
        sum(block_id in candidate_ids for block_id in gold_block_ids) / len(gold_block_ids)
        if gold_block_ids
        else 1.0
    )
    folded = rendered_context.casefold()
    rendered_coverage = (
        sum(literal.casefold() in folded for literal in gold_literals) / len(gold_literals)
        if gold_literals
        else 1.0
    )
    passed = (
        candidate_recall >= _CANDIDATE_RECALL_THRESHOLD
        and rendered_coverage >= _RENDERED_LITERAL_COVERAGE_THRESHOLD
    )
    return RecallQualityGate(
        candidate_recall_threshold=_CANDIDATE_RECALL_THRESHOLD,
        rendered_literal_coverage_threshold=_RENDERED_LITERAL_COVERAGE_THRESHOLD,
        candidate_recall=candidate_recall,
        rendered_literal_coverage=rendered_coverage,
        passed=passed,
        status="passed" if passed else "failed",
    )


def build_recall_explanation(
    *,
    plan: RecallPlan,
    hits: list[MemoryHit],
    payload: PackedEvidencePayload,
    backend_id: str,
    gold_block_ids: tuple[str, ...] = (),
    gold_literals: tuple[str, ...] = (),
) -> RecallExplanation:
    candidate_ids = tuple(dict.fromkeys(_hit_id(hit) for hit in hits if _hit_id(hit)))
    rendered_ids = tuple(
        dict.fromkeys(
            str((item.get("metadata") or {}).get("block_id") or item.get("source") or "")
            for item in payload.items
            if str((item.get("metadata") or {}).get("block_id") or item.get("source") or "")
        )
    )
    source_refs: list[dict[str, Any]] = []
    corrective_recall: dict[str, Any] = {}
    for hit in hits:
        metadata = dict(hit.metadata or {})
        if isinstance(metadata.get("corrective_recall"), dict):
            corrective_recall = dict(metadata["corrective_recall"])
        if metadata.get("reverse_provenance_depth") != 1:
            continue
        source_refs.append(
            {
                "record_id": _hit_id(hit),
                "derived_id": str(metadata.get("reverse_provenance_derived_id") or ""),
                "root_id": str(metadata.get("reverse_provenance_root_id") or ""),
                "reference_only": metadata.get("reverse_provenance_reference_only") is True,
            }
        )
    return RecallExplanation(
        query=plan.query,
        query_type=str(plan.query_type),
        workspace_id=plan.workspace_id,
        run_id=plan.run_id,
        backend_id=backend_id,
        plan_reason=plan.reason,
        candidate_record_ids=candidate_ids,
        rendered_record_ids=rendered_ids,
        budget={
            "requested_tokens": plan.max_tokens,
            "allocated_tokens": payload.token_budget,
            "estimated_tokens": payload.estimated_tokens,
            "metadata_tokens": payload.metadata_tokens,
        },
        visibility=tuple(asdict(row) for row in payload.visibility),
        exact_literal_capsule=payload.exact_literal_capsule,
        exact_literal_losses=tuple(asdict(loss) for loss in payload.exact_literal_losses),
        canonical_source_refs=tuple(source_refs),
        corrective_recall=corrective_recall,
        quality_gate=_quality_gate(
            candidate_ids=candidate_ids,
            rendered_context=payload.rendered_context,
            gold_block_ids=gold_block_ids,
            gold_literals=gold_literals,
        ),
        rendered_context_sha256=hashlib.sha256(
            payload.rendered_context.encode("utf-8")
        ).hexdigest(),
    )
