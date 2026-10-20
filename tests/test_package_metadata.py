import tomllib
from pathlib import Path

import lethyx


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_and_mcp_extra_match_supported_runtime() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["version"] == lethyx.__version__ == "0.5.0"
    assert data["project"]["optional-dependencies"]["mcp"] == [
        "mcp>=1.20.0,<2",
        "ruamel.yaml>=0.18,<1",
        "tomlkit>=0.13,<1",
    ]


def test_setuptools_includes_runtime_yaml_package_data() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]

    assert "*.yaml" in package_data["lethyx.memory.blocks"]


def test_hindsight_local_extra_pins_python_313_wheel_compatible_litellm() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    assert extras["hindsight-local"] == [
        "hindsight-all>=0.8.3,<0.9",
        "litellm==1.90.0",
    ]


def test_native_host_hook_entrypoints_are_packaged() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["scripts"]["lethyx-codex-hook"] == (
        "lethyx.integrations.codex_hooks:main"
    )
    assert data["project"]["scripts"]["lethyx-claude-hook"] == (
        "lethyx.integrations.claude_hooks:main"
    )
