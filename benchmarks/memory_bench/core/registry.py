"""Central registry for providers, benchmarks, and judges.

Registration works two ways:
1. Decorator-based: @register_provider("lethyx"), @register_benchmark("locomo"),
   @register_judge("f1")  — used by all built-in adapters and benchmarks.
2. Entry-point discovery: on first access, the registry scans the
   ``memory_bench.providers``, ``memory_bench.benchmarks``, and
   ``memory_bench.judges`` entry-point groups so that external pip packages
   can contribute providers/benchmarks/judges without editing this file.

All registries are module-level singletons (thread-safe for read-heavy usage;
writes happen only at import time or first access).
"""

import importlib
import importlib.metadata
import logging
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------

_providers: dict[str, Any] = {}  # name -> MemoryProvider class/factory
_benchmarks: dict[str, Any] = {}  # name -> Benchmark class/factory
_judges: dict[str, Any] = {}  # name -> Judge class/factory

_ep_loaded: dict[str, bool] = {
    "providers": False,
    "benchmarks": False,
    "judges": False,
}


def _load_entry_points(group: str, registry: dict[str, Any]) -> None:
    """Load all entry-points for the given group into registry (idempotent)."""
    key = group.split(".")[-1]  # "providers" | "benchmarks" | "judges"
    if _ep_loaded.get(key):
        return
    _ep_loaded[key] = True
    try:
        eps = importlib.metadata.entry_points(group=group)
    except Exception as exc:
        log.debug("entry_points(%s) failed: %s", group, exc)
        return
    for ep in eps:
        if ep.name in registry:
            continue
        try:
            obj = ep.load()
            registry[ep.name] = obj
            log.debug("entry-point loaded: %s / %s", group, ep.name)
        except Exception as exc:
            log.warning("entry-point %s/%s load failed: %s", group, ep.name, exc)


# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------


def register_provider(name: str) -> Callable[[T], T]:
    """Register a MemoryProvider factory under ``name``."""

    def decorator(cls: T) -> T:
        _providers[name] = cls
        return cls

    return decorator


def register_benchmark(name: str) -> Callable[[T], T]:
    """Register a Benchmark factory under ``name``."""

    def decorator(cls: T) -> T:
        _benchmarks[name] = cls
        return cls

    return decorator


def register_judge(name: str) -> Callable[[T], T]:
    """Register a Judge factory under ``name``."""

    def decorator(cls: T) -> T:
        _judges[name] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------


def get_provider(name: str) -> Any:
    """Return the registered provider factory for ``name``.

    Raises KeyError with a helpful message listing available names.
    """
    _load_entry_points("memory_bench.providers", _providers)
    if name not in _providers:
        available = ", ".join(sorted(_providers)) or "(none)"
        raise KeyError(
            f"unknown provider '{name}'; available: {available}\n"
            "Register via @register_provider(name) or a memory_bench.providers entry-point."
        )
    return _providers[name]


def get_benchmark(name: str) -> Any:
    _load_entry_points("memory_bench.benchmarks", _benchmarks)
    if name not in _benchmarks:
        available = ", ".join(sorted(_benchmarks)) or "(none)"
        raise KeyError(
            f"unknown benchmark '{name}'; available: {available}\n"
            "Register via @register_benchmark(name) or a memory_bench.benchmarks entry-point."
        )
    return _benchmarks[name]


def get_judge(name: str) -> Any:
    _load_entry_points("memory_bench.judges", _judges)
    if name not in _judges:
        available = ", ".join(sorted(_judges)) or "(none)"
        raise KeyError(
            f"unknown judge '{name}'; available: {available}\n"
            "Register via @register_judge(name) or a memory_bench.judges entry-point."
        )
    return _judges[name]


def list_providers() -> list[str]:
    _load_entry_points("memory_bench.providers", _providers)
    return sorted(_providers)


def list_benchmarks() -> list[str]:
    _load_entry_points("memory_bench.benchmarks", _benchmarks)
    return sorted(_benchmarks)


def list_judges() -> list[str]:
    _load_entry_points("memory_bench.judges", _judges)
    return sorted(_judges)


# ---------------------------------------------------------------------------
# Bootstrap: import built-in modules so their decorators fire
# ---------------------------------------------------------------------------


def _bootstrap_builtins() -> None:
    """Import all built-in adapters and benchmarks to trigger decorator registration.

    Called lazily on first registry access via _load_entry_points, but can also
    be called explicitly in tests or CLI startup.
    """
    _modules = [
        "memory_bench.adapters.baselines",
        "memory_bench.adapters.lethyx_provider",
        "memory_bench.benchmarks.agent_quality",
        "memory_bench.benchmarks.locomo",
        "memory_bench.benchmarks.longmemeval",
        "memory_bench.benchmarks.beam",
        "memory_bench.benchmarks.memoryarena",
        "memory_bench.benchmarks.evomembench",
        "memory_bench.adapters.judges",
    ]
    for mod in _modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            log.debug("bootstrap import %s skipped: %s", mod, exc)


def bootstrap() -> None:
    """Call once at CLI startup to ensure all built-ins are registered."""
    _bootstrap_builtins()
