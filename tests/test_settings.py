"""Tests for the user-level config and harness onboarding helpers."""

from __future__ import annotations

import argparse
import json
import os

import pytest

from lethyx.memory import cli, settings


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LETHYX_CONFIG",
        "LETHYX_REPO_ROOT",
        "LETHYX_WORKSPACE",
        "LETHYX_PROFILE",
        "LETHYX_BACKEND",
        "LETHYX_SEMANTIC",
        "LETHYX_SEMANTIC_MODEL",
        "LETHYX_SEMANTIC_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_config_round_trip_and_cli_defaults(tmp_path, monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("LETHYX_HOME", str(tmp_path / "home"))
    config = {
        "repo_root": str(tmp_path / "store"),
        "workspace": "project",
        "profile": "team",
        "backend": "mnemosyne",
        "semantic": True,
        "semantic_model": "intfloat/multilingual-e5-small",
        "semantic_cache_dir": str(tmp_path / "cache"),
        "harnesses": ["opencode"],
        "map_output": str(tmp_path / "map.html"),
    }
    path = settings.save_config(config)

    assert path == tmp_path / "home" / "config.json"
    assert settings.load_config()["workspace"] == "project"

    args = argparse.Namespace(
        store="", repo_root="", workspace="", profile="", backend=""
    )
    resolved = settings.resolve_cli_defaults(args)

    assert resolved.repo_root == str(tmp_path / "store")
    assert resolved.workspace == "project"
    assert resolved.profile == "team"
    assert resolved.backend == "mnemosyne"
    assert os.environ["LETHYX_SEMANTIC"] == "1"
    assert os.environ["LETHYX_SEMANTIC_MODEL"] == "intfloat/multilingual-e5-small"


def test_cli_flags_override_config(tmp_path, monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("LETHYX_HOME", str(tmp_path / "home"))
    settings.save_config(
        {
            "repo_root": str(tmp_path / "configured"),
            "workspace": "configured",
            "backend": "mnemosyne",
        }
    )
    args = argparse.Namespace(
        store=str(tmp_path / "flag"),
        repo_root="",
        workspace="flag",
        profile="",
        backend="mempalace",
    )
    resolved = settings.resolve_cli_defaults(args)
    assert resolved.repo_root == str(tmp_path / "flag")
    assert resolved.workspace == "flag"
    assert resolved.backend == "mempalace"


def test_connect_opencode_merges_and_keeps_backup(tmp_path, monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    path = tmp_path / "opencode.json"
    path.write_text(
        json.dumps(
            {
                "mcp": {
                    "servers": {
                        "lethyx": {"type": "local", "command": ["old"], "environment": {"CUSTOM": "keep"}},
                        "other": {"type": "local", "command": ["keep-me"]},
                    }
                },
                "theme": "quiet",
            }
        ),
        encoding="utf-8-sig",
    )
    config = {
        "repo_root": str(tmp_path / "store"),
        "workspace": "project",
        "profile": "default",
        "backend": "mnemosyne",
        "semantic": True,
        "semantic_model": "intfloat/multilingual-e5-small",
        "semantic_cache_dir": str(tmp_path / "cache"),
    }

    result = settings.connect_opencode(config)

    assert result["ok"] is True
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    assert data["theme"] == "quiet"
    assert data["mcp"]["servers"]["other"]["command"] == ["keep-me"]
    entry = data["mcp"]["servers"]["lethyx"]
    assert entry["environment"]["CUSTOM"] == "keep"
    assert entry["environment"]["LETHYX_WORKSPACE"] == "project"
    assert entry["environment"]["LETHYX_BACKEND"] == "mnemosyne"
    assert entry["environment"]["LETHYX_SEMANTIC"] == "1"
    assert entry["environment"]["LETHYX_SEMANTIC_CACHE_DIR"] == str(tmp_path / "cache")
    assert path.with_suffix(".json.bak").exists()
    assert settings.harness_connected("opencode") is True


def test_connect_opencode_refuses_invalid_json(tmp_path, monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    path = tmp_path / "opencode.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        settings.connect_opencode({"repo_root": str(tmp_path)})

    assert path.read_text(encoding="utf-8") == "{not json"


def test_cli_init_writes_a_config_without_prompting(tmp_path, monkeypatch, capsys) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("LETHYX_HOME", str(tmp_path / "home"))

    code = cli.main(
        [
            "init",
            "--yes",
            "--no-connect",
            "--no-semantic",
            "--store",
            str(tmp_path / "store"),
            "--workspace",
            "smoke",
            "--backend",
            "mempalace",
            "--harness",
            "none",
        ]
    )

    assert code == 0
    config = json.loads((tmp_path / "home" / "config.json").read_text(encoding="utf-8"))
    assert config["repo_root"] == str(tmp_path / "store")
    assert config["workspace"] == "smoke"
    assert config["backend"] == "mempalace"
    assert config["semantic"] is False
    assert config["harnesses"] == []
    assert (tmp_path / "store").is_dir()
    assert "Next steps" in capsys.readouterr().out
