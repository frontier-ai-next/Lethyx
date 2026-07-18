import json
from pathlib import Path

from lethyx.memory.cli import main


def test_hermes_provider_install_writes_thin_wrapper(tmp_path: Path, capsys) -> None:
    rc = main(["hermes-provider-install", "--hermes-home", str(tmp_path), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    wrapper = tmp_path / "plugins" / "lethyx" / "__init__.py"
    assert payload["ok"] is True
    assert payload["path"] == str(wrapper)
    assert "lethyx.integrations.hermes_provider" in wrapper.read_text(encoding="utf-8")


def test_hermes_provider_status_reports_install_state(tmp_path: Path, capsys) -> None:
    main(["hermes-provider-install", "--hermes-home", str(tmp_path), "--json"])
    capsys.readouterr()

    rc = main(["hermes-provider-status", "--hermes-home", str(tmp_path), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["installed"] is True
    assert payload["implementation"] == "packaged-wrapper"


def test_hermes_provider_install_defaults_to_hermes_home_env(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    rc = main(["hermes-provider-install", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == str(tmp_path / "plugins" / "lethyx" / "__init__.py")
