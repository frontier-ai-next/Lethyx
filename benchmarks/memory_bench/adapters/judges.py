"""Built-in Judge implementations for memory_bench.

Registered judges:
- f1           : token-level F1 (LoCoMo / SQuAD style)
- exact_match  : strict normalised match
- llm_judge    : configured LLM judge (LongMemEval style)
- nugget       : nugget coverage (BEAM style)
- abstention   : checks model abstained when it should (or didn't when it shouldn't)
- task_success : pass-through for agentic benchmarks (score from environment)
"""

import json
import logging
from typing import Any

from memory_bench.core.llm import LLMConfig, call_llm
from memory_bench.core.metrics import f1_max, exact_match_max, nugget_coverage, abstention_accuracy
from memory_bench.core.registry import register_judge
from memory_bench.core.types import Answer, Judgment, Probe

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# F1 judge
# ---------------------------------------------------------------------------


@register_judge("f1")
class F1Judge:
    """Token-level F1 against all gold answers.

    score = max F1 across gold answers.
    correct = score >= threshold (default 0.3, matching LoCoMo convention).
    """

    name = "f1"

    def __init__(self, threshold: float = 0.3) -> None:
        self._threshold = threshold

    def judge(self, probe: Probe, answer: Answer) -> Judgment:
        if probe.abstain:
            # For abstention probes, defer to abstention judge logic
            correct = answer.abstained
            return Judgment(
                correct=correct,
                score=1.0 if correct else 0.0,
                metric="f1_abstention",
                detail={"abstained": answer.abstained, "expected_abstain": True},
            )
        score = f1_max(answer.text, probe.gold) if probe.gold else 0.0
        correct = score >= self._threshold
        return Judgment(
            correct=correct,
            score=score,
            metric="f1",
            detail={"threshold": self._threshold, "gold": list(probe.gold)},
        )


# ---------------------------------------------------------------------------
# Exact match judge
# ---------------------------------------------------------------------------


@register_judge("exact_match")
class ExactMatchJudge:
    name = "exact_match"

    def judge(self, probe: Probe, answer: Answer) -> Judgment:
        if probe.abstain:
            correct = answer.abstained
            return Judgment(correct=correct, score=float(correct), metric="exact_match_abstention")
        score = exact_match_max(answer.text, probe.gold) if probe.gold else 0.0
        return Judgment(correct=score >= 1.0, score=score, metric="exact_match")


# ---------------------------------------------------------------------------
# LLM judge (LongMemEval style)
# ---------------------------------------------------------------------------

_LLM_JUDGE_SYSTEM = """\
You are an expert evaluator for question-answering systems.
Given a question, a reference answer, and a model's response, decide if the response is correct.
Output JSON on the LAST line: {"correct": true/false, "score": 0.0-1.0, "reason": "..."}
- correct: true if the response captures the essential information of the reference.
- score: 0.0=wrong, 0.5=partial, 1.0=correct.
For abstention questions (marked with abstain=true), correct means the model says it doesn't know.
"""


@register_judge("llm_judge")
class LLMJudge:
    """Configured LLM-as-judge following LongMemEval evaluation prompt design.

    Uses disk-cached LLM calls so repeated evaluation runs are free.
    Fails closed if the LLM call fails; silently changing metrics would make a
    run incomparable.
    """

    name = "llm_judge"

    def __init__(self, model: str = "", cache_dir: str = ".memory_bench_cache") -> None:
        self._config = LLMConfig(
            model=model,
            temperature=0.0,
            max_tokens=256,
            cache_dir=f"{cache_dir}/judge_llm",
        )

    def judge(self, probe: Probe, answer: Answer) -> Judgment:
        gold_str = " | ".join(probe.gold) if probe.gold else "(none)"
        user = (
            f"Question: {probe.question}\n"
            f"Reference answer: {gold_str}\n"
            f"Category: {probe.category or 'general'}\n"
            f"Abstention expected: {probe.abstain}\n\n"
            f"Model response:\n{answer.text}"
        )
        try:
            text, usage = call_llm(
                [
                    {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                config=self._config,
            )
            # Extract JSON from last line
            parsed: dict[str, Any] = {}
            for line in reversed(text.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        parsed = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        pass
            correct = bool(parsed.get("correct", False))
            score = float(parsed.get("score", 0.0))
            return Judgment(
                correct=correct,
                score=score,
                metric="llm_judge",
                detail={
                    "reason": str(parsed.get("reason", "")),
                    "model": self._config.resolved_model(),
                    "usage": usage.to_dict(),
                },
            )
        except Exception as exc:
            raise RuntimeError(f"LLM judge failed; no fallback metric is used: {exc}") from exc


# ---------------------------------------------------------------------------
# Nugget judge (BEAM style)
# ---------------------------------------------------------------------------


@register_judge("nugget")
class NuggetJudge:
    """Nugget coverage judge for BEAM benchmark.

    The gold answer is split into atomic nuggets (comma-separated or one per
    sentence).  Score = fraction of nuggets covered in the prediction.
    """

    name = "nugget"

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def _extract_nuggets(self, gold: tuple[str, ...]) -> list[str]:
        """Split gold answers into atomic nuggets."""
        nuggets: list[str] = []
        for g in gold:
            # Split on sentence boundaries or semicolons
            for part in g.replace(";", ".").split("."):
                part = part.strip()
                if len(part) > 3:
                    nuggets.append(part)
        return nuggets or list(gold)

    def judge(self, probe: Probe, answer: Answer) -> Judgment:
        if probe.abstain:
            correct = answer.abstained
            return Judgment(correct=correct, score=float(correct), metric="nugget_abstention")
        nuggets = self._extract_nuggets(probe.gold)
        if not nuggets:
            return Judgment(
                correct=False,
                score=0.0,
                metric="nugget_missing_gold",
                detail={"nuggets": [], "reason": "missing_non_abstention_gold"},
            )
        score = nugget_coverage(answer.text, nuggets)
        correct = score >= self._threshold
        return Judgment(
            correct=correct,
            score=score,
            metric="nugget",
            detail={
                "n_nuggets": len(nuggets),
                "threshold": self._threshold,
                "nuggets": nuggets[:5],  # first 5 for debugging
            },
        )


# ---------------------------------------------------------------------------
# Abstention judge
# ---------------------------------------------------------------------------


@register_judge("abstention")
class AbstentionJudge:
    """Checks whether the model's abstention behaviour matches the probe."""

    name = "abstention"

    def judge(self, probe: Probe, answer: Answer) -> Judgment:
        correct = abstention_accuracy(answer.abstained, probe.abstain) >= 1.0
        return Judgment(
            correct=correct,
            score=1.0 if correct else 0.0,
            metric="abstention",
            detail={
                "expected_abstain": probe.abstain,
                "model_abstained": answer.abstained,
            },
        )


# ---------------------------------------------------------------------------
# Task success judge (agentic benchmarks)
# ---------------------------------------------------------------------------


@register_judge("task_success")
class TaskSuccessJudge:
    """Pass-through judge for agentic benchmarks.

    The score is taken from probe.metadata["env_score"] (set by the agentic
    driver after calling environment.score()).
    """

    name = "task_success"

    def judge(self, probe: Probe, answer: Answer) -> Judgment:
        score = float(probe.metadata.get("env_score", 0.0))
        correct = score >= float(probe.metadata.get("success_threshold", 0.5))
        return Judgment(
            correct=correct,
            score=score,
            metric="task_success",
            detail={"env_score": score},
        )
