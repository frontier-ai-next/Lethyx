"""Unit tests for core/metrics.py — no network, no LLM required."""

from memory_bench.core.metrics import (
    f1,
    f1_max,
    exact_match,
    recall_at_k,
    mrr,
    ndcg_at_k,
    retrieval_metrics,
    nugget_coverage,
    abstention_accuracy,
    mean,
    aggregate_scores,
)


class TestF1:
    def test_identical(self):
        assert f1("the cat sat", "the cat sat") == 1.0

    def test_empty(self):
        assert f1("", "") == 1.0

    def test_partial(self):
        score = f1("alice saw bob", "alice saw charlie")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert f1("hello world", "foo bar") == 0.0

    def test_f1_max_picks_best(self):
        golds = ("alice lives in paris", "alice stays in paris")
        pred = "alice in paris"
        score = f1_max(pred, golds)
        assert score > f1(pred, "alice lives in london")


class TestExactMatch:
    def test_exact(self):
        assert exact_match("Alice", "alice") == 1.0  # normalised

    def test_no_match(self):
        assert exact_match("Alice", "Bob") == 0.0

    def test_articles_stripped(self):
        assert exact_match("the cat", "cat") == 1.0


class TestRetrievalMetrics:
    def test_perfect_recall(self):
        ids = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(ids, relevant, k=3) == 1.0

    def test_zero_recall(self):
        ids = ["x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(ids, relevant, k=2) == 0.0

    def test_partial_recall(self):
        ids = ["a", "x", "b"]
        relevant = {"a", "b", "c"}
        r = recall_at_k(ids, relevant, k=3)
        assert abs(r - 2 / 3) < 0.01

    def test_duplicate_retrieved_ids_do_not_push_recall_above_one(self):
        ids = ["a", "a", "a"]
        relevant = {"a"}
        assert recall_at_k(ids, relevant, k=3) == 1.0

    def test_mrr_first(self):
        assert mrr(["a", "b"], {"a"}) == 1.0

    def test_mrr_second(self):
        assert mrr(["x", "a"], {"a"}) == 0.5

    def test_mrr_miss(self):
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_ndcg_perfect(self):
        ids = ["a", "b"]
        assert ndcg_at_k(ids, {"a", "b"}, k=2) == 1.0

    def test_retrieval_metrics_bundle(self):
        m = retrieval_metrics(["a", "x"], {"a"}, k=2)
        assert "recall_at_2" in m
        assert "mrr" in m
        assert m["mrr"] == 1.0

    def test_empty_relevant(self):
        # Empty relevant set -> perfect by convention
        assert recall_at_k(["a", "b"], set(), k=2) == 1.0


class TestNugget:
    def test_full_coverage(self):
        assert nugget_coverage("the cat sat on the mat", ["cat", "mat"]) == 1.0

    def test_no_coverage(self):
        assert nugget_coverage("hello world", ["alice", "bob"]) == 0.0

    def test_partial_coverage(self):
        score = nugget_coverage("the cat sat", ["cat", "flying pig"])
        assert 0.0 < score < 1.0

    def test_empty_nuggets(self):
        assert nugget_coverage("anything", []) == 1.0


class TestAbstention:
    def test_abstain_correct(self):
        assert abstention_accuracy(True, True) == 1.0

    def test_abstain_wrong(self):
        assert abstention_accuracy(True, False) == 0.0

    def test_answer_correct(self):
        assert abstention_accuracy(False, False) == 1.0


class TestAggregation:
    def test_mean(self):
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_mean_empty(self):
        assert mean([]) == 0.0

    def test_aggregate_scores(self):
        r = aggregate_scores([0.5, 1.0, 0.0])
        assert r["n"] == 3
        assert abs(r["mean"] - 0.5) < 0.01
