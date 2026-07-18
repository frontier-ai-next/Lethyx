import importlib

from memory_bench.core.runner import _prewarm_worker_imports
from memory_bench.core.types import (
    DatasetSpec,
    JudgeConfig,
    LimitsSpec,
    ProviderSpec,
    ResponderSpec,
    RunSpec,
)


def test_prewarm_worker_imports_loads_native_memory_modules(monkeypatch):
    imported = []

    def fake_import_module(name):
        imported.append(name)
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    spec = RunSpec(
        providers=(
            ProviderSpec(id="raw-mempalace", adapter="raw-mempalace"),
            ProviderSpec(
                id="lethyx-hindsight-native",
                adapter="lethyx",
                params={"backend": "hindsight", "native": True},
            ),
            ProviderSpec(
                id="lethyx-mnemosyne-native",
                adapter="lethyx",
                params={"backend": "mnemosyne", "native": True},
            ),
        ),
        benchmarks=("locomo",),
        dataset=DatasetSpec(name="locomo"),
        responder=ResponderSpec(model="fake"),
        judge=JudgeConfig(use_llm_judge=False),
        limits=LimitsSpec(max_parallel=28),
    )

    _prewarm_worker_imports(spec)

    assert "lethyx.memory.backends" in imported
    assert "lethyx.memory.backends.contracts" in imported
    assert "lethyx.memory.palace.facade" in imported
    assert "lethyx.memory.palace.tiers" in imported
    assert "lethyx.memory.backends.hindsight" in imported
    assert "lethyx.memory.hindsight.config" in imported
    assert "lethyx.memory.backends.mnemosyne" in imported
    assert "mnemosyne" in imported
