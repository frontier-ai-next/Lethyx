"""Tests for the memory_bench LLM wrapper."""

from types import SimpleNamespace

import pytest


def test_get_env_config_supports_lethyx_tuple_runtime_config(monkeypatch) -> None:
    """lethyx env resolver returns (model, api_key, base_url), not a dict."""
    import lethyx.env as lethyx_env
    import memory_bench.core.llm as llm

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setattr(lethyx_env, "load_env", lambda: None)
    monkeypatch.setattr(
        lethyx_env,
        "get_llm_env_config",
        lambda: ("tuple-model", "tuple-key", "https://tuple.example/v1"),
    )

    assert llm._get_env_config() == {
        "model": "tuple-model",
        "api_key": "tuple-key",
        "base_url": "https://tuple.example/v1",
    }


def test_call_llm_passes_timeout_to_openai_client(monkeypatch, tmp_path) -> None:
    import memory_bench.core.llm as llm

    monkeypatch.delenv("MEMORY_BENCH_LLM_TIMEOUT_S", raising=False)
    captured: dict[str, object] = {}

    class FakeMessage:
        content = "OK"

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=FakeMessage(), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = FakeChat()

    fake_openai_module = SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai_module)
    monkeypatch.setattr(
        llm,
        "_get_env_config",
        lambda: {"api_key": "test-key", "base_url": "", "model": "gpt-oss"},
    )

    llm.call_llm(
        [{"role": "user", "content": "Return OK"}],
        config=llm.LLMConfig(
            model="gpt-oss",
            max_tokens=16,
            cache_dir=str(tmp_path),
            max_retries=1,
            timeout_s=12.5,
        ),
    )

    assert captured["client_kwargs"] == {"api_key": "test-key", "timeout": 12.5}


def test_resolved_model_requires_explicit_model(monkeypatch) -> None:
    """Benchmarks must fail closed instead of silently switching to another model."""
    import memory_bench.core.llm as llm

    monkeypatch.setattr(
        llm,
        "_get_env_config",
        lambda: {"api_key": "test-key", "base_url": "", "model": ""},
    )

    with pytest.raises(RuntimeError, match="LLM model not configured"):
        llm.LLMConfig(model="").resolved_model()


def test_call_llm_passes_reasoning_effort_from_env(monkeypatch, tmp_path) -> None:
    import memory_bench.core.llm as llm

    captured: dict[str, object] = {}

    class FakeMessage:
        content = "OK"

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=FakeMessage(), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = FakeChat()

    fake_openai_module = SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai_module)
    monkeypatch.setenv("MEMORY_BENCH_REASONING_EFFORT", "low")
    monkeypatch.setattr(
        llm,
        "_get_env_config",
        lambda: {"api_key": "test-key", "base_url": "", "model": "gpt-oss"},
    )

    text, usage = llm.call_llm(
        [{"role": "user", "content": "Return OK"}],
        config=llm.LLMConfig(
            model="gpt-oss",
            max_tokens=16,
            cache_dir=str(tmp_path),
            max_retries=1,
        ),
    )

    assert text == "OK"
    assert usage.calls == 1
    assert captured["reasoning_effort"] == "low"


def test_llm_cache_key_includes_generation_budget_and_reasoning_effort() -> None:
    import memory_bench.core.llm as llm

    messages = [{"role": "user", "content": "Return OK"}]
    low_budget = llm._cache_key("gpt-oss", messages, 0.0, 16, "low")
    high_budget = llm._cache_key("gpt-oss", messages, 0.0, 2048, "low")
    medium_reasoning = llm._cache_key("gpt-oss", messages, 0.0, 2048, "medium")

    assert low_budget != high_budget
    assert high_budget != medium_reasoning
