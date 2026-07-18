"""Memory diagnostics CLI."""

import argparse
import json
from pathlib import Path
from typing import Any

from lethyx.memory.backends.factory import create_durable_backend
from lethyx.memory.evals.ablation import run_static_ablation
from lethyx.memory.evals.report import write_ablation_report
from lethyx.memory.graph.export import export_graph_json
from lethyx.memory.kernel.value import load_memory_value_stats
from lethyx.memory.lifecycle.manager import evaluate_lifecycle
from lethyx.memory.lifecycle.reports import write_lifecycle_report
from lethyx.memory.obsidian_export import export_backend_to_obsidian
from lethyx.memory.research.benchmark_runner import write_research_reports


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
            "stats",
            "eval",
            "graph-export",
            "tribunal",
            "value-report",
            "lifecycle-report",
            "research-report",
            "backend-health",
            "obsidian-export",
            "hermes-provider-install",
            "hermes-provider-status",
        ],
    )

    parser.add_argument("--workspace", default="default")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--hermes-home", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--backend", default="canonical")
    parser.add_argument("--condition", default="")
    parser.add_argument("--mode", default="static")
    parser.add_argument("--format", default="json")
    parser.add_argument("--vault-dir", default="")
    parser.add_argument("--project-title", default="")
    parser.add_argument("--include-global", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    drive = _drive(repo, args.workspace)
    if args.command == "stats":
        payload = {
            "workspace": args.workspace,
            "value_rules": len(load_memory_value_stats(drive)),
            "drive_root": str(drive),
        }
    elif args.command == "eval":
        payload = run_static_ablation()
        write_ablation_report(drive, payload)
    elif args.command == "graph-export":
        path = export_graph_json(drive, drive / "reports" / "memory_graph_export.json")
        payload = {"path": str(path)}
    elif args.command == "tribunal":
        path = drive / "state" / "tribunal_reports" / f"{args.candidate_id}.json"
        payload = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"candidate_id": args.candidate_id, "found": False}
        )
    elif args.command == "value-report":
        payload = {"rules": {k: v.__dict__ for k, v in load_memory_value_stats(drive).items()}}
    elif args.command == "lifecycle-report":
        payload = write_lifecycle_report(drive, evaluate_lifecycle(repo, drive, args.workspace))
    elif args.command == "research-report":
        payload = write_research_reports(drive)
    elif args.command == "hermes-provider-install":
        from lethyx.integrations.hermes_provider import _home, install_hermes_plugin

        path = install_hermes_plugin(args.hermes_home or _home(), overwrite=True)
        payload = {"ok": True, "path": str(path), "implementation": "packaged-wrapper"}
    elif args.command == "hermes-provider-status":
        from lethyx.integrations.hermes_provider import _home

        path = Path(args.hermes_home or _home()) / "plugins" / "lethyx" / "__init__.py"
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        packaged = "lethyx.integrations.hermes_provider" in text
        payload = {
            "ok": True,
            "installed": path.exists(),
            "path": str(path),
            "implementation": "packaged-wrapper"
            if packaged
            else ("local-plugin" if path.exists() else "missing"),
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
            include_global=args.include_global,
        )
        payload = {
            "workspace": args.workspace,
            "backend": args.backend,
            "vault_dir": str(result.vault_dir),
            "index_path": str(result.index_path),
            "memory_overview_path": str(result.notebooklm_source_path),
            "blocks_exported": len(result.block_paths),
            "files_written": [str(path) for path in result.written_files],
        }
    else:
        if args.backend == "mnemosyne":
            from lethyx.memory.backends.mnemosyne import MnemosyneBackend

            payload = MnemosyneBackend.from_env(repo, args.workspace).health()
        else:
            backend = create_durable_backend(repo, workspace_id=args.workspace, mode=args.backend)
            payload = backend.health()
    _print(payload, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
