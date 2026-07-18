"""Unit tests for benchmark loaders using tiny inline fixtures.

All tests use in-memory fixture data (no disk datasets required).
"""

import json

from memory_bench.core.types import DatasetSpec, JudgeConfig, LimitsSpec, ResponderSpec, RunSpec


def _make_spec(**overrides) -> RunSpec:
    base = dict(
        providers=(),
        benchmarks=("locomo",),
        dataset=DatasetSpec(name="", max_episodes=5),
        responder=ResponderSpec(),
        judge=JudgeConfig(use_llm_judge=False),
        limits=LimitsSpec(recall_k=3),
    )
    base.update(overrides)
    return RunSpec(**base)


# ---------------------------------------------------------------------------
# LoCoMo loader
# ---------------------------------------------------------------------------


_LOCOMO_FIXTURE = [
    {
        "conversation_id": "conv1",
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1": {
            "session_date_time": "2024-01-01",
            "dialog": [
                {"dialog_id": "d0", "speaker": "Alice", "text": "I love Paris."},
                {"dialog_id": "d1", "speaker": "Bob", "text": "Me too!"},
            ],
        },
        "qa": [
            {
                "question": "What does Alice love?",
                "answer": "Paris",
                "category": "single_hop",
                "evidence": ["d0"],
            }
        ],
    }
]


def test_locomo_loader_parses_fixture(tmp_path, monkeypatch):
    """LoCoMo loader correctly parses conversation + QA into Episode/Probe."""
    # Write fixture file
    data_dir = tmp_path / "locomo"
    data_dir.mkdir()
    fixture = data_dir / "locomo10.json"
    fixture.write_text(json.dumps(_LOCOMO_FIXTURE), encoding="utf-8")

    # Patch datasets.resolve_file to point at our fixture
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.locomo import LoCoMoBenchmark

    bench = LoCoMoBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="locomo", max_episodes=10))
    items = list(bench.load(spec))
    assert len(items) == 1
    episode, probes = items[0]
    assert episode.id == "conv1"
    assert len(probes) == 1
    assert probes[0].question == "What does Alice love?"
    assert "Paris" in probes[0].gold
    assert "d0" in probes[0].evidence_ids


_LOCOMO_V2_FIXTURE = [
    {
        "sample_id": "sample1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "2024-01-01",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Alice", "text": "I keep the spare key in Paris."},
                {"dia_id": "D1:2", "speaker": "Bob", "text": "Good to know."},
            ],
        },
        "qa": [
            {
                "question": "Where is the spare key?",
                "answer": "Paris",
                "category": 1,
                "evidence": ["D1:1"],
            }
        ],
    }
]


def test_locomo_loader_parses_current_locomo10_shape(tmp_path, monkeypatch):
    data_dir = tmp_path / "locomo"
    data_dir.mkdir()
    (data_dir / "locomo10.json").write_text(json.dumps(_LOCOMO_V2_FIXTURE), encoding="utf-8")
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.locomo import LoCoMoBenchmark

    bench = LoCoMoBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="locomo", max_episodes=10))
    items = list(bench.load(spec))
    assert len(items) == 1
    episode, probes = items[0]
    assert episode.id == "sample1"
    assert len(episode.all_records) == 2
    assert episode.all_records[0].evidence_id == "D1:1"
    assert probes[0].category == "single_hop"
    assert probes[0].evidence_ids == ("D1:1",)


# ---------------------------------------------------------------------------
# LongMemEval loader
# ---------------------------------------------------------------------------


_LONGMEMEVAL_FIXTURE = [
    {
        "question_id": "q1",
        "question": "What is Alice's favourite city?",
        "answer": "Paris",
        "question_type": "single-session-user",
        "evidence_session_ids": ["s1"],
        "sessions": [
            {
                "session_id": "s1",
                "timestamp": "2024-01-01",
                "messages": [
                    {"role": "user", "content": "I love Paris!"},
                    {"role": "assistant", "content": "That's wonderful!"},
                ],
            }
        ],
    }
]


def test_longmemeval_loader(tmp_path, monkeypatch):
    data_dir = tmp_path / "longmemeval"
    data_dir.mkdir()
    (data_dir / "longmemeval_s.json").write_text(json.dumps(_LONGMEMEVAL_FIXTURE), encoding="utf-8")
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.longmemeval import LongMemEvalBenchmark

    bench = LongMemEvalBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="longmemeval", max_episodes=10))
    items = list(bench.load(spec))
    assert len(items) == 1
    episode, probes = items[0]
    assert episode.id == "q1"
    assert probes[0].category == "single-session-user"
    assert "Paris" in probes[0].gold


_LONGMEMEVAL_CLEANED_FIXTURE = [
    {
        "question_id": "q2",
        "question": "What degree did I graduate with?",
        "answer": "Business Administration",
        "question_type": "single-session-user",
        "answer_session_ids": ["answer_s1"],
        "haystack_dates": ["2024-01-01"],
        "haystack_session_ids": ["answer_s1"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I graduated with Business Administration."},
                {"role": "assistant", "content": "Congratulations."},
            ]
        ],
    }
]


def test_longmemeval_loader_parses_cleaned_haystack_shape(tmp_path, monkeypatch):
    data_dir = tmp_path / "longmemeval"
    data_dir.mkdir()
    (data_dir / "longmemeval_s.json").write_text(
        json.dumps(_LONGMEMEVAL_CLEANED_FIXTURE), encoding="utf-8"
    )
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.longmemeval import LongMemEvalBenchmark

    bench = LongMemEvalBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="longmemeval", max_episodes=10))
    items = list(bench.load(spec))
    assert len(items) == 1
    episode, probes = items[0]
    assert len(episode.all_records) == 2
    assert episode.all_records[0].evidence_id == "answer_s1"
    assert probes[0].evidence_ids == ("answer_s1",)


def test_longmemeval_loader_record_ids_remain_unique_when_session_ids_repeat(tmp_path, monkeypatch):
    data_dir = tmp_path / "longmemeval"
    data_dir.mkdir()
    (data_dir / "longmemeval_s.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "q-repeat",
                    "question": "What code did I repeat?",
                    "answer": "ORCHID",
                    "question_type": "multi-session",
                    "answer_session_ids": ["same-session"],
                    "haystack_dates": ["2024-01-01", "2024-01-02"],
                    "haystack_session_ids": ["same-session", "same-session"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "The code is ORCHID."}],
                        [{"role": "assistant", "content": "I remember ORCHID."}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.longmemeval import LongMemEvalBenchmark

    bench = LongMemEvalBenchmark()
    [(episode, _)] = list(bench.load(_make_spec(dataset=DatasetSpec(name="longmemeval"))))
    record_ids = [record.id for record in episode.all_records]

    assert len(record_ids) == 2
    assert len(set(record_ids)) == 2
    assert {record.evidence_id for record in episode.all_records} == {"same-session"}


# ---------------------------------------------------------------------------
# BEAM loader
# ---------------------------------------------------------------------------


_BEAM_CONVERSATION = [
    {"turn_id": "t0", "speaker": "Alice", "text": "I visited Paris last year."},
    {"turn_id": "t1", "speaker": "Bob", "text": "I heard it's beautiful."},
]
_BEAM_QUESTIONS = [
    {
        "question_id": "bq0",
        "question": "What city did Alice visit?",
        "answer": "Paris",
        "category": "information_extraction",
        "evidence_turn_ids": ["t0"],
        "tier": "100k",
    }
]


def test_beam_loader(tmp_path, monkeypatch):
    data_dir = tmp_path / "beam" / "conversations" / "conv1"
    data_dir.mkdir(parents=True)
    (data_dir / "conversation.json").write_text(json.dumps(_BEAM_CONVERSATION), encoding="utf-8")
    (data_dir / "questions.json").write_text(json.dumps(_BEAM_QUESTIONS), encoding="utf-8")
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.beam import BEAMBenchmark

    bench = BEAMBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="beam", max_episodes=10, tiers=("100k",)))
    items = list(bench.load(spec))
    assert len(items) == 1
    episode, probes = items[0]
    assert episode.id == "conv1"
    assert probes[0].tier == "100k"
    assert probes[0].category == "information_extraction"


def test_beam_loader_parses_upstream_chats_layout(tmp_path, monkeypatch):
    """Current BEAM release stores data as chats/<tier>/<id>/chat.json + probing_questions."""
    chat_dir = tmp_path / "beam" / "chats" / "100K" / "1"
    q_dir = chat_dir / "probing_questions"
    q_dir.mkdir(parents=True)
    (chat_dir / "chat.json").write_text(
        json.dumps(
            [
                {
                    "batch_number": 1,
                    "turns": [
                        [
                            {"role": "user", "id": 1, "content": "I visited Paris."},
                            {"role": "assistant", "id": 2, "content": "Paris is lovely."},
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (q_dir / "probing_questions.json").write_text(
        json.dumps(
            {
                "information_extraction": [
                    {
                        "question": "Which city did I visit?",
                        "answer": "Paris",
                        "source_chat_ids": [1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.beam import BEAMBenchmark

    bench = BEAMBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="beam", max_episodes=10, tiers=("100k",)))
    items = list(bench.load(spec))
    assert len(items) == 1
    episode, probes = items[0]
    assert episode.id == "100K_1"
    assert len(episode.all_records) == 2
    assert probes[0].category == "information_extraction"
    assert probes[0].gold == ("Paris",)
    assert probes[0].evidence_ids == ("1",)


def test_beam_loader_parses_nested_plan_chat_batches(tmp_path, monkeypatch):
    """Large BEAM tiers store chat turns inside plan-* keys in chat.json."""
    chat_dir = tmp_path / "beam" / "chats" / "10M" / "1"
    q_dir = chat_dir / "probing_questions"
    q_dir.mkdir(parents=True)
    (chat_dir / "chat.json").write_text(
        json.dumps(
            [
                {
                    "plan-1": [
                        {
                            "batch_number": 1,
                            "turns": [
                                [
                                    {
                                        "role": "user",
                                        "id": 1,
                                        "content": "I stored the relic in Oslo.",
                                    },
                                    {
                                        "role": "assistant",
                                        "id": 2,
                                        "content": "The Oslo relic is noted.",
                                    },
                                ]
                            ],
                        }
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    (q_dir / "probing_questions.json").write_text(
        json.dumps(
            {
                "information_extraction": [
                    {
                        "question": "Where is the relic?",
                        "answer": "Oslo",
                        "source_chat_ids": [1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.beam import BEAMBenchmark

    bench = BEAMBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="beam", max_episodes=10, tiers=("10m",)))
    items = list(bench.load(spec))
    assert len(items) == 1
    episode, probes = items[0]
    assert episode.id == "10M_1"
    assert len(episode.all_records) == 2
    assert episode.all_records[0].content == "I stored the relic in Oslo."
    assert episode.all_records[0].evidence_id == "1"
    assert probes[0].tier == "10m"
    assert probes[0].evidence_ids == ("1",)


def test_beam_loader_maps_all_upstream_answer_and_rubric_fields(tmp_path, monkeypatch):
    """BEAM upstream categories use ideal_summary/expected_compliance/rubric, not only answer."""
    chat_dir = tmp_path / "beam" / "chats" / "100K" / "1"
    q_dir = chat_dir / "probing_questions"
    q_dir.mkdir(parents=True)
    (chat_dir / "chat.json").write_text(
        json.dumps(
            [
                {
                    "turns": [
                        [
                            {
                                "role": "user",
                                "id": 1,
                                "content": "Use typed Python and concise code.",
                            },
                            {
                                "role": "assistant",
                                "id": 2,
                                "content": "The summary milestone is beta.",
                            },
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (q_dir / "probing_questions.json").write_text(
        json.dumps(
            {
                "summarization": [
                    {
                        "question": "Summarize the milestone.",
                        "ideal_summary": "The milestone is beta.",
                        "source_chat_ids": [2],
                        "rubric": ["mentions beta milestone"],
                    }
                ],
                "instruction_following": [
                    {
                        "question": "Show code for this feature.",
                        "expected_compliance": "Response should use typed Python.",
                        "source_chat_ids": [1],
                        "rubric": ["uses typed Python", "keeps code concise"],
                    }
                ],
                "preference_following": [
                    {
                        "question": "Which approach should I use?",
                        "expected_compliance": "Response should prefer concise code.",
                        "source_chat_ids": [1],
                        "rubric": ["recommends concise code"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.beam import BEAMBenchmark

    bench = BEAMBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="beam", max_episodes=10, tiers=("100k",)))
    [(_, probes)] = list(bench.load(spec))
    by_category = {probe.category: probe for probe in probes}

    assert by_category["summarization"].gold == ("The milestone is beta.",)
    assert by_category["instruction_following"].gold == (
        "Response should use typed Python. uses typed Python. keeps code concise",
    )
    assert by_category["preference_following"].gold == (
        "Response should prefer concise code. recommends concise code",
    )
    assert by_category["instruction_following"].metadata["rubric"] == [
        "uses typed Python",
        "keeps code concise",
    ]


# ---------------------------------------------------------------------------
# MemoryArena loader
# ---------------------------------------------------------------------------


def test_memoryarena_loader_parses_huggingface_jsonl(tmp_path, monkeypatch):
    data_dir = tmp_path / "memoryarena" / "tasks" / "bundled_shopping"
    data_dir.mkdir(parents=True)
    (data_dir / "data.jsonl").write_text(
        json.dumps(
            {
                "id": 7,
                "background": "The compatible cake base is strawberry.",
                "questions": ["Buy the compatible cake base."],
                "answers": ["strawberry"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.memoryarena import MemoryArenaBenchmark

    bench = MemoryArenaBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="memoryarena", max_episodes=10))
    items = list(bench.load(spec))
    assert len(items) == 1
    episode, probes = items[0]
    assert episode.id == "bundled_shopping_7"
    assert episode.metadata["domain"] == "bundled_shopping"
    assert episode.all_records[0].content == "The compatible cake base is strawberry."
    assert probes[0].question == "Buy the compatible cake base."
    assert probes[0].gold == ("strawberry",)


def test_memoryarena_loader_uses_real_hf_context_fields(tmp_path, monkeypatch):
    data_dir = tmp_path / "memoryarena" / "tasks" / "formal_reasoning_math"
    data_dir.mkdir(parents=True)
    (data_dir / "data.jsonl").write_text(
        json.dumps(
            {
                "id": 11,
                "paper_name": "paper-1",
                "backgrounds": ["Lemma A is required.", "Theorem B follows."],
                "questions": ["Which lemma is required?"],
                "answers": ["Lemma A"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.memoryarena import MemoryArenaBenchmark

    bench = MemoryArenaBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="memoryarena", max_episodes=10))
    items = list(bench.load(spec))
    assert len(items) == 1
    episode, probes = items[0]
    assert episode.metadata["domain"] == "formal_reasoning_math"
    assert len(episode.all_records) == 2
    assert episode.all_records[0].content == "Lemma A is required."
    assert episode.all_records[0].evidence_id == "formal_reasoning_math_11_backgrounds_0"
    assert episode.all_records[1].content == "Theorem B follows."
    assert episode.all_records[1].evidence_id == "formal_reasoning_math_11_backgrounds_1"
    assert probes[0].gold == ("Lemma A",)
    assert probes[0].evidence_ids == ("formal_reasoning_math_11_backgrounds_0",)


def test_memoryarena_loader_makes_domain_scoped_probe_keys(tmp_path, monkeypatch):
    for domain in ("bundled_shopping", "formal_reasoning_math"):
        data_dir = tmp_path / "memoryarena" / "tasks" / domain
        data_dir.mkdir(parents=True)
        (data_dir / "data.jsonl").write_text(
            json.dumps(
                {
                    "id": 1,
                    "questions": ["Question A", "Question B"],
                    "answers": ["A", "B"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.memoryarena import MemoryArenaBenchmark

    bench = MemoryArenaBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="memoryarena", max_episodes=10))
    items = list(bench.load(spec))

    keys = [f"{episode.id}::{probe.id}" for episode, probes in items for probe in probes]
    assert len(keys) == 4
    assert len(set(keys)) == 4
    assert {episode.id for episode, _ in items} == {
        "bundled_shopping_1",
        "formal_reasoning_math_1",
    }


def test_memoryarena_loader_uses_stable_ids_when_task_id_missing(tmp_path, monkeypatch):
    data_dir = tmp_path / "memoryarena" / "tasks" / "bundled_shopping"
    data_dir.mkdir(parents=True)
    (data_dir / "data.jsonl").write_text(
        json.dumps(
            {
                "background": "The hidden coupon code is ORCHID.",
                "questions": ["What is the coupon code?", "Repeat the code."],
                "answers": ["ORCHID", "ORCHID"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.memoryarena import MemoryArenaBenchmark

    bench = MemoryArenaBenchmark()
    spec = _make_spec(dataset=DatasetSpec(name="memoryarena", max_episodes=10))

    first = list(bench.load(spec))
    second = list(bench.load(spec))
    first_keys = [f"{episode.id}::{probe.id}" for episode, probes in first for probe in probes]
    second_keys = [f"{episode.id}::{probe.id}" for episode, probes in second for probe in probes]

    assert first_keys == second_keys
    assert len(first_keys) == 2
    assert first[0][0].id.startswith("bundled_shopping_")


def test_memoryarena_public_jsonl_is_recall_qa_not_stub_agentic_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "memoryarena" / "tasks" / "bundled_shopping"
    data_dir.mkdir(parents=True)
    (data_dir / "data.jsonl").write_text(
        json.dumps(
            {
                "id": 8,
                "background": "The allergy-safe cake base is oat.",
                "questions": ["Which cake base is allergy-safe?"],
                "answers": ["oat"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.memoryarena import MemoryArenaBenchmark

    bench = MemoryArenaBenchmark()
    assert bench.kind == "recall"
    assert bench.default_judge == "f1"
    [(_, probes)] = list(bench.load(_make_spec(dataset=DatasetSpec(name="memoryarena"))))
    assert probes[0].kind == "qa"
    assert "env" not in probes[0].metadata


def test_evomembench_missing_dataset_yields_no_fake_score_rows(tmp_path, monkeypatch):
    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.evomembench import EvoMemBenchBenchmark

    bench = EvoMemBenchBenchmark()
    assert list(bench.load(_make_spec(dataset=DatasetSpec(name="evomembench")))) == []


def test_evomembench_knowledge_probes_are_recall_with_stable_ids(tmp_path, monkeypatch):
    mode_dir = tmp_path / "evomembench" / "in_episode_knowledge"
    mode_dir.mkdir(parents=True)
    (mode_dir / "episodes.json").write_text(
        json.dumps(
            [
                {
                    "episode_id": "ep1",
                    "turns": ["The migration codeword is MAUVE."],
                }
            ]
        ),
        encoding="utf-8",
    )
    (mode_dir / "probes.json").write_text(
        json.dumps(
            [
                {
                    "episode_id": "ep1",
                    "question": "What is the migration codeword?",
                    "answer": "MAUVE",
                }
            ]
        ),
        encoding="utf-8",
    )

    import memory_bench.core.datasets as ds

    monkeypatch.setattr(ds, "_data_root", lambda: tmp_path)

    from memory_bench.benchmarks.evomembench import EvoMemBenchBenchmark

    bench = EvoMemBenchBenchmark()
    assert bench.kind == "mixed"
    assert bench.default_judge == "f1"

    first = list(bench.load(_make_spec(dataset=DatasetSpec(name="evomembench"))))
    second = list(bench.load(_make_spec(dataset=DatasetSpec(name="evomembench"))))

    first_keys = [f"{episode.id}::{probe.id}" for episode, probes in first for probe in probes]
    second_keys = [f"{episode.id}::{probe.id}" for episode, probes in second for probe in probes]
    assert first_keys == second_keys

    [(_, probes)] = first
    assert probes[0].kind == "qa"
    assert probes[0].gold == ("MAUVE",)
    assert "env" not in probes[0].metadata
