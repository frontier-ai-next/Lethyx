"""CLI compatibility and full-benchmark launcher tests."""


def test_datasets_check_alias_checks_all(monkeypatch) -> None:
    """`memory_bench datasets check` remains a compatibility alias for all datasets."""
    from memory_bench import cli
    import memory_bench.core.datasets as datasets

    seen: dict[str, object] = {}

    def fake_check_all(names=None):
        seen["names"] = names
        return {"locomo": {"present": True, "path": "x", "notes": ""}}

    def fake_print_check_report(report) -> None:
        seen["report"] = report

    monkeypatch.setattr(datasets, "check_all", fake_check_all)
    monkeypatch.setattr(datasets, "print_check_report", fake_print_check_report)

    assert cli.main(["datasets", "check"]) == 0
    assert seen["names"] is None
    assert "locomo" in seen["report"]


def test_full_bench_suite_toml_defaults_to_gemma20_and_memory_matrix() -> None:
    """The launcher should default to the local honest Gemma-only matrix."""
    from memory_bench.scripts.full_bench import build_suite_toml

    suite = build_suite_toml(max_episodes=0, max_probes_per_episode=0)

    assert "max_parallel = 20" in suite
    assert 'model = "gemma-4-31B-it"' in suite
    assert 'llm_judge_model = "gemma-4-31B-it"' in suite
    assert "max_tokens = 2048" in suite
    for benchmark in (
        "agent_quality",
        "locomo",
        "longmemeval",
        "beam",
        "memoryarena",
        "evomembench",
    ):
        assert f'"{benchmark}"' in suite
    for provider_id in (
        "lethyx-mempalace",
        "lethyx-hindsight",
        "lethyx-mnemosyne",
        "baseline-no-memory",
        "baseline-full-context",
        "baseline-bm25",
    ):
        assert f'id = "{provider_id}"' in suite
    assert 'id = "raw-hindsight"' in suite
    assert 'id = "raw-mnemosyne"' in suite
    assert 'profile = "memory-bench-raw-no-llm"' in suite
    assert 'llm_provider = "none"' in suite
    assert 'profile = "memory-bench-lethyx-native-chunks"' in suite


def test_dataset_root_prefers_outer_repo_when_memory_bench_has_pyproject(
    tmp_path, monkeypatch
) -> None:
    """Dataset lookup should use repo-root/data, not memory_bench/data."""
    import memory_bench.core.datasets as datasets

    repo = tmp_path / "repo"
    module_file = repo / "memory_bench" / "core" / "datasets.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='lethyx'\n", encoding="utf-8")
    (repo / "memory_bench" / "pyproject.toml").write_text(
        "[project]\nname='memory-bench'\n", encoding="utf-8"
    )

    monkeypatch.delenv("MEMORY_BENCH_DATA_DIR", raising=False)
    monkeypatch.setattr(datasets, "__file__", str(module_file))

    assert datasets._data_root() == repo / "data" / "memory_bench"
