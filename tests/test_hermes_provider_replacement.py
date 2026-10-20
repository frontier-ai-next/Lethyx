import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from lethyx.integrations import hermes_provider


def test_standalone_observer_routes_public_hooks_and_tools_to_active_provider(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class Provider:
        def on_pre_tool_call(self, **kwargs) -> None:
            calls.append({"hook": "pre", **kwargs})

        def on_post_tool_call(self, **kwargs) -> None:
            calls.append({"hook": "post", **kwargs})

        def handle_tool_call(self, name, args):
            return json.dumps({"tool": name, "args": args})

    class Context:
        def __init__(self) -> None:
            self.callbacks = {}
            self.tools = {}

        def register_hook(self, name, callback) -> None:
            self.callbacks[name] = callback

        def register_tool(self, *, name, handler, **kwargs) -> None:
            self.tools[name] = {"handler": handler, **kwargs}

    module = types.ModuleType("_hermes_user_memory.lethyx__source_deadbeef12345678")
    module._ACTIVE_PROVIDER = Provider()
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    try:
        namespace: dict[str, object] = {
            "__file__": str(tmp_path / "plugins" / "lethyx_observer" / "__init__.py")
        }
        exec(compile(hermes_provider._OBSERVER_PLUGIN_WRAPPER, "<observer>", "exec"), namespace)
        context = Context()
        namespace["register"](context)
        context.callbacks["pre_tool_call"](tool_name="write_file", args={"path": "a.py"})
        context.callbacks["post_tool_call"](tool_name="write_file", result={"success": True})
        status = json.loads(context.tools["lethyx_provider_status"]["handler"]({"verbose": False}))
    finally:
        if previous is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous

    assert calls == [
        {"hook": "pre", "tool_name": "write_file", "args": {"path": "a.py"}},
        {"hook": "post", "tool_name": "write_file", "result": {"success": True}},
    ]
    assert status == {
        "tool": "lethyx_provider_status",
        "args": {"verbose": False},
    }
    assert all(tool["toolset"] == "lethyx" for tool in context.tools.values())
    assert "lethyx_provider_remember" in context.tools


def _write_profile_with_existing_notes(root: Path) -> tuple[bytes, bytes]:
    config = {
        "model": {"default": "test-model"},
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": True,
            "provider": "legacy-provider",
        },
        "agent": {"disabled_toolsets": ["vision"]},
        "platform_toolsets": {"cli": ["hermes-cli"]},
    }
    config_bytes = yaml.safe_dump(config, sort_keys=False).encode()
    config_path = root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(config_bytes)
    notes = b"# MEMORY (your personal notes)\nlegacy secret\n"
    notes_path = root / "memories" / "MEMORY.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_bytes(notes)
    return config_bytes, notes


def test_hermes_installer_stages_provider_without_disabling_builtin_or_deleting_notes(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    original_config, original_notes = _write_profile_with_existing_notes(profile)
    code_root = Path(__file__).resolve().parents[1]

    result = hermes_provider.install_hermes_provider(
        profile,
        code_root=code_root,
        repo_root=profile / "persistent-store",
        workspace_id="workspace-a",
        profile_id="profile-a",
        backend="mempalace",
        hermes_python=sys.executable,
    )

    provider_path = profile / "plugins" / "lethyx" / "__init__.py"
    manifest_path = provider_path.parent / "plugin.yaml"
    config_path = profile / "lethyx" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert result["staged"] is True
    assert result["active"] is False
    assert result["live_callback_proven"] is False
    assert result["built_in_memory_modified_by_installer"] is False
    assert provider_path.is_file()
    assert "python_dependencies" in manifest
    assert (
        "lethyx" not in yaml.safe_load((profile / "config.yaml").read_text())["memory"]["provider"]
    )
    assert config["repo_root"] == str((profile / "persistent-store").resolve())
    assert config["workspace_id"] == "workspace-a"
    assert config["profile_id"] == "profile-a"
    assert (profile / "config.yaml").read_bytes() == original_config
    assert (profile / "memories" / "MEMORY.md").read_bytes() == original_notes
    assert result["runtime_import"]["importable"] is False


def test_old_hermes_hard_cutover_entrypoint_fails_without_mutation(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    original_config, original_notes = _write_profile_with_existing_notes(profile)

    with pytest.raises(RuntimeError, match="hermes memory setup"):
        hermes_provider.install_hermes_replacement(
            profile,
            hermes_agent_root=tmp_path / "unneeded-old-core-inspection",
        )

    assert (profile / "config.yaml").read_bytes() == original_config
    assert (profile / "memories" / "MEMORY.md").read_bytes() == original_notes
    assert not (profile / "plugins").exists()
