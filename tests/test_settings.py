"""Tests for the user-level config and harness onboarding helpers."""

import argparse
import json
import os

import pytest

from um_mem.memory import cli, settings
from um_mem.memory.local_service import build_local_service_identity


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "UM_MEM_CONFIG",
        "UM_MEM_REPO_ROOT",
        "UM_MEM_WORKSPACE",
        "UM_MEM_WORKSPACE_MODE",
        "UM_MEM_PROJECT_ROOT",
        "UM_MEM_PROFILE",
        "UM_MEM_BACKEND",
        "UM_MEM_SEMANTIC",
        "UM_MEM_SEMANTIC_MODEL",
        "UM_MEM_SEMANTIC_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_config_round_trip_and_cli_defaults(tmp_path, monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("UM_MEM_HOME", str(tmp_path / "home"))
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

    args = argparse.Namespace(store="", repo_root="", workspace="", profile="", backend="")
    resolved = settings.resolve_cli_defaults(args)

    assert resolved.repo_root == str(tmp_path / "store")
    assert resolved.workspace == "project"
    assert resolved.profile == "team"
    assert resolved.backend == "mnemosyne"
    assert os.environ["UM_MEM_SEMANTIC"] == "1"
    assert os.environ["UM_MEM_SEMANTIC_MODEL"] == "intfloat/multilingual-e5-small"


def test_cli_flags_override_config(tmp_path, monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("UM_MEM_HOME", str(tmp_path / "home"))
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
                        "um_mem": {
                            "type": "local",
                            "command": ["old"],
                            "environment": {"CUSTOM": "keep"},
                        },
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
    assert "servers" not in data["mcp"]
    assert data["mcp"]["other"]["command"] == ["keep-me"]
    entry = data["mcp"]["um_mem"]
    assert entry["environment"]["CUSTOM"] == "keep"
    assert entry["environment"]["UM_MEM_WORKSPACE"] == "project"
    assert entry["environment"]["UM_MEM_BACKEND"] == "mnemosyne"
    assert entry["environment"]["UM_MEM_SEMANTIC"] == "1"
    assert entry["environment"]["UM_MEM_SEMANTIC_CACHE_DIR"] == str(tmp_path / "cache")
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
    monkeypatch.setenv("UM_MEM_HOME", str(tmp_path / "home"))

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


def test_connect_local_writes_a_project_workspace(tmp_path, monkeypatch, capsys) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("UM_MEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    project = tmp_path / "demo-repo"
    project.mkdir()
    monkeypatch.chdir(project)

    code = cli.main(
        ["connect", "opencode", "--local", "--json", "--store", str(tmp_path / "store")]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["project_workspace"] == "demo-repo"
    assert payload["local_settings"] == str(project / "opencode.json")
    data = json.loads((project / "opencode.json").read_text(encoding="utf-8"))
    environment = data["mcp"]["um_mem"]["environment"]
    assert environment["UM_MEM_WORKSPACE"] == "demo-repo"
    assert environment["UM_MEM_REPO_ROOT"] == str(tmp_path / "store")
    assert not (tmp_path / "config" / "opencode" / "opencode.json").exists()


def test_connect_local_honours_an_explicit_workspace(tmp_path, monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("UM_MEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    project = tmp_path / "firmware"
    project.mkdir()
    monkeypatch.chdir(project)

    code = cli.main(
        [
            "connect",
            "opencode",
            "--local",
            "--json",
            "--workspace",
            "parser",
            "--store",
            str(tmp_path / "store"),
        ]
    )

    assert code == 0
    data = json.loads((project / "opencode.json").read_text(encoding="utf-8"))
    assert data["mcp"]["um_mem"]["environment"]["UM_MEM_WORKSPACE"] == "parser"


def test_connect_local_refuses_hosts_without_project_settings(tmp_path, monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("UM_MEM_HOME", str(tmp_path / "home"))
    project = tmp_path / "demo-repo"
    project.mkdir()
    monkeypatch.chdir(project)

    code = cli.main(["connect", "claude", "--local", "--json", "--store", str(tmp_path / "store")])

    assert code == 1
    assert not (project / "opencode.json").exists()


def test_workspace_mode_environment_does_not_override_the_pinned_workspace(
    tmp_path, monkeypatch
) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("UM_MEM_WORKSPACE", "alpha")
    monkeypatch.setenv("UM_MEM_WORKSPACE_MODE", "project")
    monkeypatch.setenv("UM_MEM_PROJECT_ROOT", str(tmp_path / "alpha"))

    identity = build_local_service_identity(
        repo_root=tmp_path / "store",
        workspace_id=None,
        profile_id="default",
        backend="mnemosyne",
    )

    assert identity.workspace_id == "alpha"


def test_workspace_falls_back_to_default_without_a_pinned_identity(tmp_path, monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("UM_MEM_WORKSPACE_MODE", "project")
    monkeypatch.setenv("UM_MEM_PROJECT_ROOT", str(tmp_path / "alpha"))

    identity = build_local_service_identity(
        repo_root=tmp_path / "store",
        workspace_id=None,
        profile_id="default",
        backend="mnemosyne",
    )

    assert identity.workspace_id == "default"
