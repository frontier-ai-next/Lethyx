"""Memory diagnostics CLI."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lethyx.memory.backends.factory import create_durable_backend
from lethyx.api import MemoryClient
from lethyx.memory.obsidian_export import export_backend_to_obsidian
from lethyx.memory.kernel.telemetry import memory_profile_report


def _drive(repo: Path, workspace: str) -> Path:
    return repo / "workspaces" / (workspace or "default") / ".memory" / "drive"


def _print(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lethyx.memory.cli")
    parser.add_argument(
        "command",
        choices=[
            "reconcile",
            "backend-health",
            "obsidian-export",
            "hermes-provider-install",
            "hermes-provider-status",
            "hermes-provider-restore",
            "hermes-mcp-install",
            "hermes-mcp-status",
            "codex-hooks-install",
            "codex-hooks-status",
            "claude-hooks-install",
            "claude-hooks-status",
            "codex-install",
            "codex-status",
            "claude-install",
            "claude-status",
            "explain-recall",
            "profile-report",
        ],
    )

    parser.add_argument("--workspace", default="default")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--drive-root", default="")
    parser.add_argument("--hermes-home", default="")
    parser.add_argument("--hermes-agent-root", default="")
    parser.add_argument("--hermes-python", default="")
    parser.add_argument("--code-root", default="")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--claude-watch-file", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--backend", default="")
    parser.add_argument("--vault-dir", default="")
    parser.add_argument("--project-title", default="")
    parser.add_argument(
        "--source-horizon",
        choices=("working", "durable", "archive"),
        default="durable",
    )
    parser.add_argument("--scope", action="append", default=[])
    parser.add_argument("--user-id", default="")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--query", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--logical-block-id", action="append", default=[])
    reconcile_group = parser.add_mutually_exclusive_group()
    reconcile_group.add_argument("--dry-run", dest="reconcile_dry_run", action="store_true")
    reconcile_group.add_argument("--apply", dest="reconcile_dry_run", action="store_false")
    parser.set_defaults(reconcile_dry_run=True)
    args = parser.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    drive = _drive(repo, args.workspace)
    if args.command == "reconcile":
        backend = create_durable_backend(
            repo,
            workspace_id=args.workspace,
            mode=args.backend,
            reconcile_pending_writes=False,
        )
        try:
            report = backend.reconcile(dry_run=args.reconcile_dry_run)
        finally:
            backend.close()
        payload = report.to_dict()
        reports = drive / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        stamp = report.timestamp.replace("+00:00", "Z").replace(":", "")
        report_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        (reports / f"memory_reconcile_report_{stamp}.json").write_text(
            report_text, encoding="utf-8"
        )
        (reports / "memory_reconcile_report_latest.json").write_text(report_text, encoding="utf-8")
    elif args.command == "explain-recall":
        if not args.query.strip():
            parser.error("--query is required for explain-recall")
        backend = create_durable_backend(repo, workspace_id=args.workspace, mode=args.backend)
        try:
            explanation = MemoryClient(
                repo,
                workspace_id=args.workspace,
                user_id=args.user_id,
                backend_instance=backend,
            ).explain_recall(
                args.query,
                limit=args.limit,
                max_tokens=args.max_tokens,
                source_horizon=args.source_horizon,
                include_candidates=args.include_candidates,
                scope_chain=args.scope or [f"workspace:{args.workspace}"],
                logical_block_ids=args.logical_block_id,
            )
            payload = explanation.to_mapping()
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
    elif args.command == "hermes-provider-install":
        from lethyx.integrations.hermes_provider import _home, install_hermes_provider

        if args.replacement:
            parser.error(
                "--replacement is obsolete and is not safe for current Hermes; "
                "stage the provider, then select it with `hermes memory setup`"
            )
        code_root = (
            Path(args.code_root).expanduser().resolve()
            if args.code_root
            else Path(__file__).resolve().parents[3]
        )
        payload = install_hermes_provider(
            args.hermes_home or _home(),
            code_root=code_root,
            repo_root=args.repo_root or None,
            workspace_id=args.workspace,
            profile_id=args.profile,
            backend=args.backend or "mempalace",
            hermes_python=args.hermes_python or None,
        )
    elif args.command == "hermes-provider-status":
        from lethyx.integrations.hermes_provider import _home, hermes_provider_status

        if args.replacement:
            parser.error("--replacement status belongs to the legacy restore flow")
        payload = hermes_provider_status(
            args.hermes_home or _home(),
            hermes_python=args.hermes_python or None,
        )
    elif args.command in {"hermes-mcp-install", "hermes-mcp-status"}:
        from lethyx.integrations.mcp_settings import (
            default_repo_root,
            hermes_mcp_server_status,
            install_hermes_mcp,
            validate_identity,
        )
        from lethyx.integrations.hermes_provider import _home

        hermes_config = Path(args.hermes_home or _home()).expanduser().resolve() / "config.yaml"
        if args.command == "hermes-mcp-install":
            identity = validate_identity(
                args.repo_root or default_repo_root(),
                args.workspace,
                args.profile,
                args.backend or "mempalace",
            )
            path = install_hermes_mcp(
                hermes_config,
                repo_root=identity["repo_root"],
                workspace_id=identity["workspace"],
                profile_id=identity["profile"],
                backend=identity["backend"],
            )
            payload = {
                "ok": True,
                "host": "hermes",
                "identity": identity,
                **hermes_mcp_server_status(path),
            }
        else:
            payload = hermes_mcp_server_status(hermes_config)
    elif args.command == "hermes-provider-restore":
        from lethyx.integrations.hermes_provider import _home, restore_hermes_replacement

        payload = restore_hermes_replacement(args.hermes_home or _home())
    elif args.command == "codex-hooks-install":
        from lethyx.integrations.codex_hooks import codex_hooks_status, install_codex_hooks

        install_codex_hooks(args.project_dir)
        payload = {"ok": True, **codex_hooks_status(args.project_dir)}
    elif args.command == "codex-hooks-status":
        from lethyx.integrations.codex_hooks import codex_hooks_status

        payload = {"ok": True, **codex_hooks_status(args.project_dir)}
    elif args.command == "claude-hooks-install":
        from lethyx.integrations.claude_hooks import claude_hooks_status, install_claude_hooks

        install_claude_hooks(args.project_dir, watch_files=args.claude_watch_file)
        payload = {
            "ok": True,
            **claude_hooks_status(args.project_dir, watch_files=args.claude_watch_file),
        }
    elif args.command == "claude-hooks-status":
        from lethyx.integrations.claude_hooks import claude_hooks_status

        payload = {
            "ok": True,
            **claude_hooks_status(args.project_dir, watch_files=args.claude_watch_file),
        }
    elif args.command in {"claude-install", "codex-install", "claude-status", "codex-status"}:
        from lethyx.integrations.mcp_settings import (
            default_repo_root,
            install_claude_mcp,
            install_codex_mcp,
            mcp_server_status,
            validate_identity,
        )

        is_claude = args.command.startswith("claude-")
        host = "claude" if is_claude else "codex"
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
        mcp_path = Path.home() / ".claude.json" if is_claude else codex_home / "config.toml"
        identity = validate_identity(
            args.repo_root or default_repo_root(),
            args.workspace,
            args.profile,
            args.backend or "mempalace",
        )
        module_arguments = [
            "--repo-root",
            identity["repo_root"],
            "--workspace",
            identity["workspace"],
            "--profile",
            identity["profile"],
            "--backend",
            identity["backend"],
        ]
        if args.command.endswith("-install"):
            if is_claude:
                from lethyx.integrations.claude_hooks import install_claude_hooks

                hook_target = install_claude_hooks(
                    Path.home(),
                    module_arguments=module_arguments,
                    settings_path=Path.home() / ".claude" / "settings.json",
                )
                mcp_target = install_claude_mcp(
                    mcp_path,
                    repo_root=identity["repo_root"],
                    workspace_id=identity["workspace"],
                    profile_id=identity["profile"],
                    backend=identity["backend"],
                )
            else:
                from lethyx.integrations.codex_hooks import install_codex_hooks

                hook_target = install_codex_hooks(
                    Path.home(),
                    module_arguments=module_arguments,
                    settings_path=codex_home / "hooks.json",
                )
                mcp_target = install_codex_mcp(
                    mcp_path,
                    repo_root=identity["repo_root"],
                    workspace_id=identity["workspace"],
                    profile_id=identity["profile"],
                    backend=identity["backend"],
                )
            payload = {
                "ok": True,
                "host": host,
                "identity": identity,
                "mcp_settings": str(mcp_target),
                "hook_settings": str(hook_target),
                "integration_mode": "hooks_plus_mcp_tools",
                "active_mask_delivery": False,
                "host_connection_tested": False,
            }
        else:
            if is_claude:
                from lethyx.integrations.claude_hooks import claude_hooks_status

                hooks_status = claude_hooks_status(
                    Path.home(), settings_path=Path.home() / ".claude" / "settings.json"
                )
            else:
                from lethyx.integrations.codex_hooks import codex_hooks_status

                hooks_status = codex_hooks_status(
                    Path.home(), settings_path=codex_home / "hooks.json"
                )
            payload = {
                "ok": True,
                "host": host,
                "identity": identity,
                "mcp": mcp_server_status(mcp_path, host=host),
                "hooks": hooks_status,
                "integration_mode": "hooks_plus_mcp_tools",
                "active_mask_delivery": False,
                "host_connection_tested": False,
            }
    elif args.command == "obsidian-export":
        if not args.vault_dir:
            parser.error("--vault-dir is required for obsidian-export")
        backend = create_durable_backend(repo, workspace_id=args.workspace, mode=args.backend)
        result = export_backend_to_obsidian(
            backend=backend,
            vault_dir=Path(args.vault_dir),
            workspace_id=args.workspace,
            project_title=args.project_title,
            scope_chain=tuple(args.scope or (f"workspace:{args.workspace}",)),
        )
        payload = {
            "workspace": args.workspace,
            "backend": args.backend,
            "vault_dir": str(result.vault_dir),
            "index_path": str(result.index_path),
            "memory_overview_path": str(result.memory_overview_path),
            "blocks_exported": len(result.block_paths),
            "files_written": [str(path) for path in result.written_files],
        }
    elif args.command == "profile-report":
        payload = memory_profile_report(
            repo,
            drive_root=Path(args.drive_root).resolve() if args.drive_root else None,
            limit=args.limit,
        )
    else:
        if args.backend == "mnemosyne":
            from lethyx.memory.backends.mnemosyne import MnemosyneBackend

            payload = MnemosyneBackend.from_env(repo, args.workspace).health()
        else:
            backend = create_durable_backend(repo, workspace_id=args.workspace, mode=args.backend)
            payload = backend.health()
    _print(payload, args.json)
    if args.command == "reconcile" and not args.reconcile_dry_run:
        failures = payload.get("failures", 0)
        if type(failures) is not int:
            raise ValueError("reconcile failure count is invalid")
        return 1 if failures > 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
