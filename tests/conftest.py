"""Test-only lifecycle hygiene for listeners rooted in pytest temporary directories."""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from lethyx.api import MemoryClient
from lethyx.contracts.ledger import append_candidate_semantic_verification_ledger_event

from lethyx.memory.local_service import (
    _process_is_alive,
    build_local_service_identity,
    local_service_registry,
    shutdown_local_service_listener,
)


@pytest.fixture
def candidate_semantic_report():
    def build(
        client: MemoryClient,
        memory_id: str,
        *,
        authority: str = "advisory",
        target_block_id: str = "",
        changed_files: tuple[str, ...] = (),
    ):
        record = client.backend.get_block(memory_id)
        subject = client.verification_subject(
            memory_id,
            expected_source_revision_id=record.revision_id,
            authority=authority,
            target_block_id=target_block_id,
        )
        return append_candidate_semantic_verification_ledger_event(
            repo_root=client.repo_root,
            subject=subject,
            verifier_id="verifier:semantic-fixture",
            verifier_model_version="reviewer-fixture",
            proposer_model_version="author-fixture",
            coverage=True,
            preservation=True,
            faithfulness=True,
            instruction_safety=True,
            reasons=("Fixture reviewed this exact candidate and publication subject.",),
            changed_files=changed_files,
        )

    return build


def _windows_process_command_line(pid: int) -> str:
    if os.name != "nt":
        return ""
    probe = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                f"$process = Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}'; "
                "if ($null -ne $process) { [Console]::Out.Write($process.CommandLine) }"
            ),
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=5,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return probe.stdout.strip()


def _force_reap_verified_pytest_listener(pid: int, repo_root: Path) -> None:
    """Force-reap only a known pytest-temp listener after authenticated shutdown fails."""

    if os.name != "nt" or not _process_is_alive(pid):
        return
    command_line = _windows_process_command_line(pid).replace("/", "\\").casefold()
    expected_root = str(repo_root.resolve()).replace("/", "\\").casefold()
    if "--local-service-listener" not in command_line or expected_root not in command_line:
        return
    subprocess.run(
        ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@pytest.fixture(autouse=True)
def _reap_pytest_tmp_listeners(request: pytest.FixtureRequest):
    """Never leave a listener owned by a completed pytest temporary test root."""

    yield
    tmp_path = request.node.funcargs.get("tmp_path")
    if not isinstance(tmp_path, Path):
        return
    failures: list[str] = []
    for registry_path in tmp_path.rglob("local_service_*.json"):
        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            workspace_id = payload.get("workspace_id")
            profile_id = payload.get("profile_id")
            backend = payload.get("backend")
            pid = payload.get("pid")
            if (
                not isinstance(workspace_id, str)
                or not isinstance(profile_id, str)
                or not isinstance(backend, str)
                or not isinstance(pid, int)
                or pid <= 0
            ):
                continue
            repo_root = registry_path.parents[2]
            identity = build_local_service_identity(
                repo_root=repo_root,
                workspace_id=workspace_id,
                profile_id=profile_id,
                backend=backend,
                caller_role="operator",
            )
            shutdown_local_service_listener(identity)
            deadline = time.monotonic() + 5
            while (
                local_service_registry(identity) is not None or _process_is_alive(pid)
            ) and time.monotonic() < deadline:
                time.sleep(0.05)
            _force_reap_verified_pytest_listener(pid, repo_root)
            deadline = time.monotonic() + 5
            while _process_is_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if local_service_registry(identity) is not None or _process_is_alive(pid):
                failures.append(f"{registry_path} (pid {pid})")
        except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            failures.append(f"{registry_path} ({type(exc).__name__})")
    if failures:
        pytest.fail(
            "pytest temporary local-service listener survived teardown: " + ", ".join(failures)
        )
