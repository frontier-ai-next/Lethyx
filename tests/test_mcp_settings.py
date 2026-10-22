import json
from pathlib import Path

import pytest

from um_mem.integrations import mcp_settings


def test_claude_mcp_install_is_user_scoped_and_preserves_other_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_path = tmp_path / ".claude.json"
    settings_path.write_text(
        json.dumps({"otherSetting": True, "mcpServers": {"docs": {"command": "docs"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_settings, "_require_mcp_runtime", lambda: None)

    installed = mcp_settings.install_claude_mcp(
        settings_path,
        repo_root=tmp_path / "memory-store",
        workspace_id="workspace-a",
        profile_id="profile-a",
        backend="mempalace",
    )

    config = json.loads(installed.read_text(encoding="utf-8"))
    server = config["mcpServers"]["um_mem"]
    assert config["otherSetting"] is True
    assert config["mcpServers"]["docs"] == {"command": "docs"}
    assert server["args"] == ["-m", "um_mem.memory.mcp_server"]
    assert (
        server["env"]["UM_MEM_REPO_ROOT"].casefold()
        == str((tmp_path / "memory-store").resolve()).casefold()
    )
    assert server["env"]["UM_MEM_WORKSPACE"] == "workspace-a"
    assert server["env"]["UM_MEM_PROFILE"] == "profile-a"
    status = mcp_settings.mcp_server_status(installed, host="claude")
    assert status["configured"] is True
    assert status["integration_mode"] == "mcp_tools_only"
    assert status["active_mask_delivery"] is False
    assert status["host_connection_tested"] is False


def test_codex_mcp_install_preserves_toml_comments_and_uses_prompt_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tomlkit = pytest.importorskip("tomlkit")
    settings_path = tmp_path / "config.toml"
    settings_path.write_text("# keep this comment\nmodel = 'test'\n", encoding="utf-8")
    monkeypatch.setattr(mcp_settings, "_require_mcp_runtime", lambda: None)
    monkeypatch.setattr(mcp_settings, "_require_backend_runtime", lambda backend: None)

    installed = mcp_settings.install_codex_mcp(
        settings_path,
        repo_root=tmp_path / "memory-store",
        workspace_id="workspace-b",
        profile_id="profile-b",
        backend="zvec",
    )

    text = installed.read_text(encoding="utf-8")
    config = tomlkit.parse(text)
    server = config["mcp_servers"]["um_mem"]
    assert "# keep this comment" in text
    assert config["model"] == "test"
    assert server["args"] == ["-m", "um_mem.memory.mcp_server"]
    assert server["env"]["UM_MEM_WORKSPACE"] == "workspace-b"
    assert server["default_tools_approval_mode"] == "prompt"
    assert server["enabled"] is True
    assert mcp_settings.mcp_server_status(installed, host="codex")["configured"] is True


def test_hermes_mcp_install_preserves_yaml_comments_and_unrelated_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("ruamel.yaml")
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(
        "# keep this comment\nmodel:\n  provider: test\nmcp_servers:\n  docs:\n    command: docs\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_settings, "_require_mcp_runtime", lambda: None)
    monkeypatch.setattr(mcp_settings, "_require_backend_runtime", lambda backend: None)

    installed = mcp_settings.install_hermes_mcp(
        settings_path,
        repo_root=tmp_path / "memory-store",
        workspace_id="workspace-c",
        profile_id="profile-c",
        backend="mempalace",
    )

    text = installed.read_text(encoding="utf-8")
    import yaml

    config = yaml.safe_load(text)
    server = config["mcp_servers"]["um_mem"]
    assert "# keep this comment" in text
    assert config["model"] == {"provider": "test"}
    assert config["mcp_servers"]["docs"] == {"command": "docs"}
    assert server["args"] == ["-m", "um_mem.memory.mcp_server"]
    assert server["env"]["UM_MEM_WORKSPACE"] == "workspace-c"
    assert server["timeout"] == 120
    status = mcp_settings.hermes_mcp_server_status(installed)
    assert status["configured"] is True
    assert status["integration_mode"] == "mcp_tools_only"
    assert status["active_mask_delivery"] is False
    assert status["host_connection_tested"] is False


def test_hermes_mcp_install_refuses_to_overwrite_unmanaged_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("ruamel.yaml")
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text("mcp_servers:\n  um_mem:\n    command: custom\n", encoding="utf-8")
    monkeypatch.setattr(mcp_settings, "_require_mcp_runtime", lambda: None)
    monkeypatch.setattr(mcp_settings, "_require_backend_runtime", lambda backend: None)

    with pytest.raises(ValueError, match="unmanaged MCP server"):
        mcp_settings.install_hermes_mcp(
            settings_path,
            repo_root=tmp_path / "memory-store",
            workspace_id="default",
            profile_id="default",
            backend="mempalace",
        )


@pytest.mark.parametrize("host", ["Claude Code", "Codex"])
def test_mcp_install_refuses_to_overwrite_an_unmanaged_server_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
) -> None:
    monkeypatch.setattr(mcp_settings, "_require_mcp_runtime", lambda: None)
    if host == "Claude Code":
        path = tmp_path / ".claude.json"
        path.write_text(json.dumps({"mcpServers": {"um_mem": {"command": "custom"}}}))
        install = mcp_settings.install_claude_mcp
    else:
        pytest.importorskip("tomlkit")
        path = tmp_path / "config.toml"
        path.write_text("[mcp_servers.um_mem]\ncommand = 'custom'\n")
        install = mcp_settings.install_codex_mcp

    with pytest.raises(ValueError, match="unmanaged MCP server"):
        install(
            path,
            repo_root=tmp_path / "memory-store",
            workspace_id="default",
            profile_id="default",
            backend="mempalace",
        )
