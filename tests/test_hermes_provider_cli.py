import json
import sys
from pathlib import Path
from typing import Any

import pytest
import lethyx.memory.cli as memory_cli
from lethyx.memory.cli import main
from lethyx.integrations import mcp_settings, hermes_provider


def test_hermes_provider_install_writes_thin_wrapper(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "hermes-provider-install",
            "--hermes-home",
            str(tmp_path),
            "--code-root",
            str(Path(__file__).resolve().parents[1]),
            "--hermes-python",
            sys.executable,
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    wrapper = tmp_path / "plugins" / "lethyx" / "__init__.py"
    assert payload["ok"] is True
    assert payload["provider_path"] == str(wrapper)
    assert payload["staged"] is True
    assert payload["active"] is False
    assert payload["live_callback_proven"] is False
    assert "lethyx.integrations.hermes_provider" in wrapper.read_text(encoding="utf-8")


def test_hermes_provider_status_reports_install_state(tmp_path: Path, capsys) -> None:
    main(
        [
            "hermes-provider-install",
            "--hermes-home",
            str(tmp_path),
            "--code-root",
            str(Path(__file__).resolve().parents[1]),
            "--hermes-python",
            sys.executable,
            "--json",
        ]
    )
    capsys.readouterr()

    rc = main(["hermes-provider-status", "--hermes-home", str(tmp_path), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["staged"] is True
    assert payload["active"] is False


def test_hermes_provider_install_defaults_to_hermes_home_env(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    rc = main(
        [
            "hermes-provider-install",
            "--hermes-python",
            sys.executable,
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider_path"] == str(tmp_path / "plugins" / "lethyx" / "__init__.py")


def test_hermes_mcp_install_and_status_use_profile_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_settings, "_require_mcp_runtime", lambda: None)
    monkeypatch.setattr(mcp_settings, "_require_backend_runtime", lambda backend: None)

    assert (
        main(
            [
                "hermes-mcp-install",
                "--repo-root",
                str(tmp_path / "store"),
                "--workspace",
                "workspace-h",
                "--profile",
                "profile-h",
                "--backend",
                "mempalace",
                "--json",
            ]
        )
        == 0
    )
    installed = json.loads(capsys.readouterr().out)
    assert installed["configured"] is True
    assert installed["identity"]["workspace"] == "workspace-h"
    assert installed["active_mask_delivery"] is False
    assert Path(installed["path"]) == tmp_path / "config.yaml"

    assert main(["hermes-mcp-status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["configured"] is True
    assert status["identity"]["profile"] == "profile-h"


def test_native_hermes_install_refuses_python_below_lethyx_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hermes_provider, "_hermes_python_version", lambda executable: (3, 11))

    with pytest.raises(RuntimeError, match="Python 3.12 or newer"):
        hermes_provider.install_hermes_provider(
            tmp_path,
            code_root=Path(__file__).resolve().parents[1],
            hermes_python=sys.executable,
        )

    assert not (tmp_path / "plugins").exists()


def test_native_hermes_status_explains_unsupported_python_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hermes_provider, "_hermes_python_version", lambda executable: (3, 11))

    status = hermes_provider.hermes_runtime_import_status(
        sys.executable,
        code_root=Path(__file__).resolve().parents[1],
    )

    assert status["importable"] is False
    assert status["native_runtime_compatible"] is False
    assert status["python_version"] == "3.11"
    assert status["required_python"] == ">=3.12"
    assert status["reason"] == "hermes_python_version_unsupported"


def test_hermes_hard_cutover_cli_is_rejected_before_changes(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "hermes-provider-install",
                "--hermes-home",
                str(tmp_path),
                "--replacement",
                "--json",
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "config.yaml").exists()
    assert not (tmp_path / "plugins").exists()


def test_native_hook_install_and_status_commands_use_project_local_settings(
    tmp_path: Path,
    capsys,
) -> None:
    for host in ("codex", "claude"):
        rc = main([f"{host}-hooks-install", "--project-dir", str(tmp_path), "--json"])
        assert rc == 0
        installed = json.loads(capsys.readouterr().out)
        assert installed["ok"] is True
        assert installed["installed"] is True

        rc = main([f"{host}-hooks-status", "--project-dir", str(tmp_path), "--json"])
        assert rc == 0
        status = json.loads(capsys.readouterr().out)
        assert status["installed"] is True
        assert status["missing_events"] == []


def test_combined_host_install_uses_one_identity_and_global_settings_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    codex_home = tmp_path / "custom-codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    for host in ("claude", "codex"):
        install_args = [
            f"{host}-install",
            "--repo-root",
            str(tmp_path / "memory-store"),
            "--workspace",
            "w-a",
            "--profile",
            "p-a",
            "--backend",
            "mempalace",
            "--json",
        ]
        assert main(install_args) == 0
        installed = json.loads(capsys.readouterr().out)
        assert installed["identity"]["workspace"] == "w-a"
        assert installed["identity"]["profile"] == "p-a"
        assert installed["active_mask_delivery"] is False

        status_command = f"{host}-status"
        assert main([status_command, "--workspace", "w-a", "--profile", "p-a", "--json"]) == 0
        status = json.loads(capsys.readouterr().out)
        assert status["mcp"]["configured"] is True
        assert status["mcp"]["identity"]["workspace"] == "w-a"
        assert status["hooks"]["installed"] is True
        if host == "claude":
            assert Path(status["mcp"]["path"]) == tmp_path / ".claude.json"
            assert Path(status["hooks"]["path"]) == tmp_path / ".claude" / "settings.json"
        else:
            assert Path(status["mcp"]["path"]) == codex_home / "config.toml"
            assert Path(status["hooks"]["path"]) == codex_home / "hooks.json"


def test_claude_hook_cli_passes_explicit_file_watch_names(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "claude-hooks-install",
            "--project-dir",
            str(tmp_path),
            "--claude-watch-file",
            ".env",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "FileChanged" in payload["installed_events"]


def test_backend_health_default_uses_factory_default_not_removed_canonical(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    seen: dict[str, Any] = {}

    class FakeBackend:
        def health(self) -> dict[str, Any]:
            return {"ok": True, "backend_id": "mempalace"}

    def fake_create(repo_root: Path, *, workspace_id: str = "", mode: str | None = None) -> Any:
        seen.update(repo_root=repo_root, workspace_id=workspace_id, mode=mode)
        return FakeBackend()

    monkeypatch.setattr(memory_cli, "create_durable_backend", fake_create)

    rc = main(["backend-health", "--repo-root", str(tmp_path), "--json"])

    assert rc == 0
    assert seen["mode"] == ""
    assert json.loads(capsys.readouterr().out)["backend_id"] == "mempalace"
