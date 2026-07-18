from memory_bench.core import registry
from memory_bench.core.runner import run_pair
from memory_bench.core.types import (
    Answer,
    DatasetSpec,
    Episode,
    Judgment,
    JudgeConfig,
    LimitsSpec,
    MemoryRecord,
    Probe,
    ProviderSpec,
    ResponderSpec,
    RetrievedItem,
    RunSpec,
    Session,
)


def _make_spec(tmp_path):
    return RunSpec(
        providers=(ProviderSpec(id="test-mixed-memory", adapter="test-mixed-memory"),),
        benchmarks=("test-mixed-dispatch",),
        dataset=DatasetSpec(name="test", max_episodes=1),
        responder=ResponderSpec(model="fake"),
        judge=JudgeConfig(use_llm_judge=False),
        limits=LimitsSpec(recall_k=3),
        out_dir=str(tmp_path),
    )


class _Store:
    def __init__(self):
        self.records = []

    def write_many(self, records):
        self.records.extend(records)

    def retrieve(self, query, *, k=10, max_tokens=4000):
        return [RetrievedItem(text="The codeword is MAUVE.", score=1.0, source_id="e1")]

    def close(self):
        pass


class _Provider:
    def namespace(self, namespace):
        return _Store()


class _Responder:
    def answer(self, question, retrieved):
        return Answer(text="MAUVE")


class _Judge:
    def judge(self, probe, answer):
        return Judgment(correct=answer.text == "MAUVE", score=1.0, metric="fake")


def test_mixed_benchmark_dispatches_qa_to_recall_and_action_to_agentic_skip(tmp_path, monkeypatch):
    @registry.register_provider("test-mixed-memory")
    class TestProvider(_Provider):
        pass

    @registry.register_benchmark("test-mixed-dispatch")
    class MixedBenchmark:
        name = "test-mixed-dispatch"
        kind = "mixed"
        default_judge = "f1"

        def load(self, spec):
            episode = Episode(
                id="ep1",
                sessions=(
                    Session(
                        id="s1",
                        ts="",
                        records=(
                            MemoryRecord(
                                id="r1",
                                content="The codeword is MAUVE.",
                                evidence_id="e1",
                            ),
                        ),
                    ),
                ),
            )
            probes = [
                Probe(
                    id="p_qa",
                    question="What is the codeword?",
                    gold=("MAUVE",),
                    evidence_ids=("e1",),
                    kind="qa",
                ),
                Probe(
                    id="p_action",
                    question="Execute the task.",
                    kind="action",
                ),
            ]
            yield episode, probes

    monkeypatch.setattr("memory_bench.core.runner._resolve_responder", lambda spec: _Responder())
    monkeypatch.setattr("memory_bench.core.runner._resolve_judge", lambda bench, spec: _Judge())

    spec = _make_spec(tmp_path)
    rows = run_pair(spec.providers[0], "test-mixed-dispatch", spec, tmp_path)

    assert [(row.probe_id, row.status) for row in rows] == [
        ("p_qa", "ok"),
        ("p_action", "skipped"),
    ]
    assert rows[0].retrieval["recall_at_3"] == 1.0
    assert rows[1].error == "no environment provided for agentic probe"
