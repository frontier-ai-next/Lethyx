import json
import subprocess
import sys
from pathlib import Path

import pytest

from lethyx.integrations import hermes_provider


def test_hermes_preserves_the_canonical_umbrella_workspace_id() -> None:
    workspace_id = "umbrella-user-0123456789abcdef"

    assert hermes_provider._safe_workspace(workspace_id) == workspace_id
    assert (
        hermes_provider._resolve_workspace(
            {"workspace_id": workspace_id},
            {},
        )
        == workspace_id
    )


def test_host_adapter_import_does_not_eagerly_load_backend_runtime() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
import json
import sys

sys.path.insert(0, sys.argv[1])
import lethyx.integrations.hermes_provider

print(json.dumps({
    "api_loaded": "lethyx.api" in sys.modules,
    "mempalace_loaded": "lethyx.memory.backends.mempalace" in sys.modules,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source_root)],
        check=True,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    assert json.loads(completed.stdout) == {
        "api_loaded": False,
        "mempalace_loaded": False,
    }


def test_writer_import_does_not_require_mempalace_runtime() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
import json
import sys
from importlib.abc import MetaPathFinder

sys.path.insert(0, sys.argv[1])


class BlockMemPalaceImport(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lethyx.memory.backends.mempalace":
            raise ModuleNotFoundError("simulated optional MemPalace dependency", name=fullname)
        return None


sys.meta_path.insert(0, BlockMemPalaceImport())
import lethyx.memory.kernel.writer

print(json.dumps({
    "mempalace_loaded": "lethyx.memory.backends.mempalace" in sys.modules,
}))
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source_root)],
        check=True,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    assert json.loads(completed.stdout) == {"mempalace_loaded": False}


def test_install_replaces_stale_monolith_atomically_and_smoke_loads(tmp_path: Path) -> None:
    target = tmp_path / "plugins" / "lethyx" / "__init__.py"
    target.parent.mkdir(parents=True)
    target.write_text("class UmMemMemoryProvider:\n    stale = True\n", encoding="utf-8")

    installed = hermes_provider.install_hermes_plugin(tmp_path, overwrite=True)
    smoke = hermes_provider.smoke_hermes_plugin_wrapper(installed)

    assert installed == target
    text = target.read_text(encoding="utf-8")
    assert "stale = True" not in text
    assert "lethyx.integrations.hermes_provider" in text
    assert smoke["ok"] is True
    assert smoke["provider_module"] == "lethyx.integrations.hermes_provider"
    assert smoke["block_first_supported"] is True
    assert not list(target.parent.glob("*.tmp"))


def test_install_rolls_back_previous_plugin_when_smoke_fails(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "plugins" / "lethyx" / "__init__.py"
    target.parent.mkdir(parents=True)
    original = "# previous working plugin\n"
    target.write_text(original, encoding="utf-8")

    def fail_smoke(_path: Path) -> dict:
        raise RuntimeError("isolated import failed")

    monkeypatch.setattr(hermes_provider, "smoke_hermes_plugin_wrapper", fail_smoke)

    with pytest.raises(RuntimeError, match="isolated import failed"):
        hermes_provider.install_hermes_plugin(tmp_path, overwrite=True)

    assert target.read_text(encoding="utf-8") == original
    assert not list(target.parent.glob("*.tmp"))


def test_thin_wrapper_bootstraps_configured_source_when_package_is_not_installed(
    tmp_path: Path,
) -> None:
    from lethyx.memory import mcp_facade

    project_root = Path(__file__).resolve().parents[1]
    config = tmp_path / "lethyx" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"code_root": str(project_root)}), encoding="utf-8")
    installed = hermes_provider.install_hermes_plugin(tmp_path, overwrite=True)
    assert sys.modules["lethyx.memory.mcp_facade"] is mcp_facade
    script = """
import importlib.util
import json
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

plugin = Path(sys.argv[1])
source_root = Path(sys.argv[2]).resolve()
sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != source_root]


class BlockUmMemUntilSourceBootstrap(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "lethyx" or fullname.startswith("lethyx.")) and str(source_root) not in sys.path:
            raise ModuleNotFoundError("simulated missing Lethyx package", name=fullname)
        return None


sys.meta_path.insert(0, BlockUmMemUntilSourceBootstrap())
try:
    importlib.util.find_spec("lethyx")
except ModuleNotFoundError:
    package_was_missing = True
else:
    package_was_missing = False
spec = importlib.util.spec_from_file_location(
    "_hermes_user_plugins.lethyx",
    plugin,
    submodule_search_locations=[str(plugin.parent)],
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({
    "package_was_missing": package_was_missing,
    "provider_module": module.UmMemMemoryProvider.__module__,
    "wrapper_version": module.WRAPPER_VERSION,
}))
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(installed),
            str(project_root / "src"),
        ],
        check=True,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    payload = json.loads(completed.stdout)
    assert payload == {
        "package_was_missing": True,
        "provider_module": "lethyx.integrations.hermes_provider",
        "wrapper_version": 4,
    }


def test_thin_wrapper_prefers_configured_source_over_cached_installed_package(
    tmp_path: Path,
) -> None:
    installed = hermes_provider.install_hermes_plugin(tmp_path, overwrite=True)
    config = tmp_path / "lethyx" / "config.json"
    configured_root = tmp_path / "configured-source"
    configured_provider = configured_root / "src" / "lethyx" / "integrations" / "hermes_provider.py"
    configured_provider.parent.mkdir(parents=True)
    configured_provider.parents[1].joinpath("__init__.py").write_text("", encoding="utf-8")
    configured_provider.parent.joinpath("__init__.py").write_text("", encoding="utf-8")
    configured_provider.write_text(
        """
class UmMemMemoryProvider:
    MARKER = "configured-source"


def install_hermes_plugin(*_args, **_kwargs):
    return None


def register(_ctx):
    return None
""".lstrip(),
        encoding="utf-8",
    )
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"code_root": str(configured_root)}), encoding="utf-8")

    installed_root = tmp_path / "fake-installed"
    installed_provider = installed_root / "lethyx" / "integrations" / "hermes_provider.py"
    installed_provider.parent.mkdir(parents=True)
    installed_provider.parents[1].joinpath("__init__.py").write_text("", encoding="utf-8")
    installed_provider.parent.joinpath("__init__.py").write_text("", encoding="utf-8")
    installed_provider.write_text(
        """
class UmMemMemoryProvider:
    MARKER = "stale-installed"


def install_hermes_plugin(*_args, **_kwargs):
    return None


def register(_ctx):
    return None
""".lstrip(),
        encoding="utf-8",
    )
    script = """
import importlib.util
import json
import sys
from pathlib import Path

plugin = Path(sys.argv[1])
installed_root = Path(sys.argv[2]).resolve()
configured_src = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(installed_root))
import lethyx.integrations.hermes_provider as stale

spec = importlib.util.spec_from_file_location(
    "_hermes_user_plugins.lethyx",
    plugin,
    submodule_search_locations=[str(plugin.parent)],
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
loaded = sys.modules["lethyx.integrations.hermes_provider"]
print(json.dumps({
    "preloaded_marker": stale.UmMemMemoryProvider.MARKER,
    "provider_marker": module.UmMemMemoryProvider.MARKER,
    "loaded_marker": loaded.UmMemMemoryProvider.MARKER,
    "configured_src_first": str(configured_src) == sys.path[0],
}))
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(installed),
            str(installed_root),
            str(configured_root / "src"),
        ],
        check=True,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    assert json.loads(completed.stdout) == {
        "preloaded_marker": "stale-installed",
        "provider_marker": "configured-source",
        "loaded_marker": "configured-source",
        "configured_src_first": True,
    }
