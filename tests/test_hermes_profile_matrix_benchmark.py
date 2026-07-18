import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "hermes_memory_profiles" / "run_profile_matrix.py"
LARGE_BENCHMARK = ROOT / "benchmarks" / "hermes_memory_profiles" / "run_large_project_sequence.py"


def _load_matrix():
    spec = importlib.util.spec_from_file_location("hermes_profile_matrix", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_profile_matrix"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_large():
    bench_dir = LARGE_BENCHMARK.parent
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    spec = importlib.util.spec_from_file_location("large_langchain_benchmark", LARGE_BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    sys.modules["large_langchain_benchmark"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profile_matrix_score_negates_forbidden_terms_locally() -> None:
    matrix = _load_matrix()
    question = matrix.Question(
        "negative",
        "Does repo use reportkit or Zustand?",
        ("repo",),
        ("reportkit", "Zustand"),
    )

    safe = matrix._score("No, repo does not use reportkit or Zustand.", question)
    absent = matrix._score("No memory entries found for repo, reportkit, or Zustand.", question)
    absent_multiline = matrix._score(
        "No memory entries found for:\n- repo\n- reportkit\n- Zustand", question
    )
    no_fact = matrix._score("No memory-backed fact says repo uses reportkit or Zustand.", question)
    no_support = matrix._score(
        "No memory-backed fact supports repo using either reportkit or Zustand.", question
    )
    no_fact_lists = matrix._score(
        "No memory-backed fact for repo lists reportkit or Zustand as dependencies.", question
    )
    neither = matrix._score(
        "So the memory-backed answer is: neither reportkit nor Zustand is supported.",
        question,
    )
    not_supported = matrix._score(
        "reportkit and Zustand are not supported by connected memory evidence.",
        question,
    )
    leaky = matrix._score(
        "No memory-backed fact supports repo using reportkit. reportkit belongs elsewhere.",
        question,
    )
    unsafe = matrix._score("No, repo does not use reportkit. It uses Zustand.", question)

    assert safe["forbidden_penalized"] == []
    assert absent["forbidden_penalized"] == []
    assert absent_multiline["forbidden_penalized"] == []
    assert no_fact["forbidden_penalized"] == []
    assert no_support["forbidden_penalized"] == []
    assert no_fact_lists["forbidden_penalized"] == []
    assert neither["forbidden_penalized"] == []
    assert not_supported["forbidden_penalized"] == []
    assert leaky["forbidden_penalized"] == ["reportkit"]
    assert set(safe["forbidden_negated"]) == {"reportkit", "Zustand"}
    assert set(absent["forbidden_negated"]) == {"reportkit", "Zustand"}
    assert unsafe["forbidden_negated"] == ["reportkit"]
    assert unsafe["forbidden_penalized"] == ["Zustand"]
    assert unsafe["score"] < safe["score"]


def test_large_benchmark_builds_67_questions_per_memory_with_session_probes() -> None:
    large = _load_large()
    cards = [
        large.FactCard(
            f"source_synthetic_{idx}",
            f"LangChain source: libs/core/langchain_core/synthetic_{idx}.py",
            (
                f"LangChain source file card for libs/core/langchain_core/synthetic_{idx}.py.\n"
                f"Classes: SyntheticClass{idx}, SyntheticHelper{idx}.\n"
                f"Functions: synthetic_function_{idx}, synthetic_helper_{idx}.\n"
                f"Imports/modules: synthetic.module{idx}, langchain_core.synthetic.\n"
                "Excerpt:\npass\n"
            ),
        )
        for idx in range(70)
    ]

    questions = large.build_langchain_questions(cards, target_count=67)
    ids = [question.id for question in questions]

    assert len(questions) == 67
    assert len(set(ids)) == 67
    assert ids[:8] == [question.id for question in large.LANGCHAIN_EXAMPLE_QUESTIONS[:8]]
    assert ids[-1] == "lc_late_update_recall"
    assert "lc_session_user_codename_recall" in ids
    assert "lc_session_profile_ack_recall" in ids
    assert any(question.id.startswith("lc_source_synthetic_") for question in questions)


def test_large_benchmark_hermes_command_resumes_same_session() -> None:
    large = _load_large()

    first = large._hermes_chat_command("profile-a", "prompt text", session_id=None, source="bench")
    resumed = large._hermes_chat_command(
        "profile-a", "prompt text", session_id="20260629_abc", source="bench"
    )

    assert "--resume" not in first
    assert resumed[0:3] == ["hermes", "-p", "profile-a"]
    assert "--resume" in resumed
    assert resumed[resumed.index("--resume") + 1] == "20260629_abc"
    assert resumed[resumed.index("--source") + 1] == "bench"


def test_profile_matrix_installs_source_loading_lethyx_wrapper(tmp_path, monkeypatch) -> None:
    matrix = _load_matrix()
    monkeypatch.setattr(matrix, "HERMES_HOME", tmp_path)

    matrix._install_lethyx_plugin_wrapper("profile-a")

    wrapper = tmp_path / "profiles" / "profile-a" / "plugins" / "lethyx" / "__init__.py"
    text = wrapper.read_text(encoding="utf-8")
    assert "lethyx.integrations.hermes_provider" in text
    assert "LethyxMemoryProvider" in text
    assert "sys.path.insert" not in text


def test_large_benchmark_metrics_dedupes_copied_tool_calls_after_compression(
    tmp_path, monkeypatch
) -> None:
    large = _load_large()
    monkeypatch.setattr(large.base, "HERMES_HOME", tmp_path)
    profile_home = tmp_path / "profiles" / "profile-a"
    profile_home.mkdir(parents=True)
    con = sqlite3.connect(profile_home / "state.db")
    con.executescript(
        """
        create table sessions (
            id text primary key,
            parent_session_id text,
            end_reason text,
            message_count integer,
            tool_call_count integer,
            api_call_count integer
        );
        create table messages (
            id integer primary key,
            session_id text,
            role text,
            content text,
            tool_call_id text,
            tool_calls text,
            tool_name text,
            compacted integer default 0
        );
        """
    )
    con.execute("insert into sessions values ('parent', null, 'compression', 10, 2, 5)")
    con.execute("insert into sessions values ('child', 'parent', null, 10, 2, 3)")
    call_1 = json.dumps([{"id": "call-1", "function": {"name": "lethyx_provider_recall"}}])
    call_2 = json.dumps([{"id": "call-2", "function": {"name": "lethyx_provider_recall"}}])
    con.executemany(
        "insert into messages values (?,?,?,?,?,?,?,?)",
        [
            (1, "parent", "assistant", "", None, call_1, None, 0),
            (2, "parent", "tool", "{}", "call-1", None, "lethyx_provider_recall", 0),
            # Context compression can copy recent tail messages into the child
            # continuation. The copied call id must not count as a new call.
            (3, "child", "assistant", "", None, call_1, None, 0),
            (4, "child", "tool", "{}", "call-1", None, "lethyx_provider_recall", 0),
            (5, "child", "assistant", "", None, call_2, None, 0),
            (6, "child", "tool", "{}", "call-2", None, "lethyx_provider_recall", 0),
        ],
    )
    con.commit()
    con.close()

    metrics = large._session_lineage_metrics("profile-a", "child")

    assert metrics["compression_count"] == 1
    assert metrics["tool_call_count"] == 4
    assert metrics["tool_message_count"] == 3
    assert metrics["unique_tool_call_count"] == 2
