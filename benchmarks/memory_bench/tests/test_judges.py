"""Unit tests for judge implementations — no network, no LLM required."""

from memory_bench.core.types import Answer, Probe, Usage
from memory_bench.adapters.judges import (
    F1Judge,
    ExactMatchJudge,
    NuggetJudge,
    AbstentionJudge,
    TaskSuccessJudge,
)


def _probe(question="Q?", gold=("Alice",), category="", abstain=False, metadata=None):
    return Probe(
        id="p0",
        question=question,
        gold=gold,
        category=category,
        abstain=abstain,
        kind="qa",
        metadata=metadata or {},
    )


def _answer(text="Alice", abstained=False):
    return Answer(text=text, abstained=abstained, usage=Usage())


class TestF1Judge:
    def test_exact_answer_passes(self):
        j = F1Judge()
        jmt = j.judge(_probe(gold=("Alice",)), _answer("Alice"))
        assert jmt.correct
        assert jmt.score >= 0.9

    def test_wrong_answer_fails(self):
        j = F1Judge()
        jmt = j.judge(_probe(gold=("Alice",)), _answer("Bob"))
        assert not jmt.correct
        assert jmt.score < 0.3

    def test_abstain_probe_abstain_answer(self):
        j = F1Judge()
        jmt = j.judge(_probe(abstain=True), _answer("I don't know", abstained=True))
        assert jmt.correct

    def test_abstain_probe_non_abstain_answer(self):
        j = F1Judge()
        jmt = j.judge(_probe(abstain=True), _answer("Alice", abstained=False))
        assert not jmt.correct

    def test_empty_gold_zero_score(self):
        j = F1Judge()
        jmt = j.judge(_probe(gold=()), _answer("anything"))
        assert jmt.score == 0.0


class TestExactMatchJudge:
    def test_exact_normalised(self):
        j = ExactMatchJudge()
        jmt = j.judge(_probe(gold=("Alice",)), _answer("alice"))
        assert jmt.correct

    def test_no_match(self):
        j = ExactMatchJudge()
        jmt = j.judge(_probe(gold=("Alice",)), _answer("Bob"))
        assert not jmt.correct


class TestNuggetJudge:
    def test_all_nuggets_covered(self):
        j = NuggetJudge()
        probe = _probe(gold=("cat. mat.",))
        jmt = j.judge(probe, _answer("the cat sat on the mat"))
        assert jmt.correct

    def test_no_nuggets_covered(self):
        j = NuggetJudge()
        probe = _probe(gold=("alice. bob.",))
        jmt = j.judge(probe, _answer("xyz"))
        assert not jmt.correct

    def test_abstain_probe(self):
        j = NuggetJudge()
        jmt = j.judge(_probe(abstain=True), _answer("I don't know", abstained=True))
        assert jmt.correct

    def test_empty_gold_is_invalid_not_perfect(self):
        j = NuggetJudge()
        jmt = j.judge(_probe(gold=()), _answer("anything"))
        assert not jmt.correct
        assert jmt.score == 0.0
        assert jmt.metric == "nugget_missing_gold"


class TestAbstentionJudge:
    def test_correct_abstain(self):
        j = AbstentionJudge()
        jmt = j.judge(_probe(abstain=True), _answer(abstained=True))
        assert jmt.correct

    def test_incorrect_abstain(self):
        j = AbstentionJudge()
        jmt = j.judge(_probe(abstain=True), _answer(abstained=False))
        assert not jmt.correct

    def test_correct_answer(self):
        j = AbstentionJudge()
        jmt = j.judge(_probe(abstain=False), _answer(abstained=False))
        assert jmt.correct


class TestTaskSuccessJudge:
    def test_high_score(self):
        j = TaskSuccessJudge()
        probe = _probe(metadata={"env_score": 0.9, "success_threshold": 0.5})
        jmt = j.judge(probe, _answer())
        assert jmt.correct
        assert jmt.score == 0.9

    def test_low_score(self):
        j = TaskSuccessJudge()
        probe = _probe(metadata={"env_score": 0.1, "success_threshold": 0.5})
        jmt = j.judge(probe, _answer())
        assert not jmt.correct
