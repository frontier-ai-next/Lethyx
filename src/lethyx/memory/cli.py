"""Memory diagnostics CLI."""

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

from lethyx.memory.backends.factory import create_durable_backend
from lethyx.api import MemoryClient
from lethyx.memory.obsidian_export import export_backend_to_obsidian
from lethyx.memory.kernel.telemetry import memory_profile_report
from lethyx.memory.settings import (
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    DEFAULT_PROFILE,
    DEFAULT_WORKSPACE,
    HARNESSES,
    backend_available,
    connect_harness,
    config_path,
    default_cache_dir,
    default_map_path,
    default_store,
    detect_harnesses,
    harness_connected,
    has_module,
    load_config,
    resolve_cli_defaults,
    save_config,
    semantic_available,
)


def _drive(repo: Path, workspace: str) -> Path:
    return repo / "workspaces" / (workspace or "default") / ".memory" / "drive"


def _print(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(payload)


def _split_harness_values(values: list[str] | None) -> list[str]:
    found: list[str] = []
    for value in values or []:
        for token in str(value).replace(" ", ",").split(","):
            name = token.strip().lower()
            if name and name != "none" and name not in found:
                found.append(name)
    return found


def _ask(prompt: str, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return value or default


def _ask_yes(prompt: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    value = _ask(f"{prompt} ({hint})", "y" if default else "n").lower()
    return value in {"y", "yes", "1", "true", "on", "да"}


def _ask_backend(default: str) -> str:
    options = [
        ("mempalace", "zero setup, keyword recall"),
        ("mnemosyne", "local durable memory + native recall (recommended)"),
        ("hindsight", "external service, needs the hindsight extra"),
        ("zvec", "experimental vector backend, needs the zvec extra"),
    ]
    print("\nBackend:")
    for index, (name, note) in enumerate(options, 1):
        marker = " *" if name == default else ""
        print(f"  {index}. {name:<10} {note}{marker}")
    value = _ask("Choose a backend (number or name)", default).lower()
    if value.isdigit() and 1 <= int(value) <= len(options):
        return options[int(value) - 1][0]
    allowed = {name for name, _ in options}
    return value if value in allowed else default


def _ask_harnesses(default: list[str]) -> list[str]:
    detected = detect_harnesses()
    print("\nHarness:")
    for name, label in HARNESSES.items():
        marker = " *" if name == default else ""
        state = "detected" if name in detected else "not detected"
        print(f"  {name:<9} {label:<14} {state}{marker}")
    print("  none      do not connect a harness now")
    raw = _ask("Connect (comma-separated ids, or none)", ",".join(default) or "none")
    if raw.strip().lower() in {"none", "-", ""}:
        return []
    return [name for name in _split_harness_values([raw]) if name in HARNESSES]


def _run_init(args: argparse.Namespace) -> int:
    config = load_config()
    interactive = not args.yes and sys.stdin.isatty()
    default_backend = (
        args.backend or config.get("backend") or
        (DEFAULT_BACKEND if has_module("mnemosyne") else "mempalace")
    )
    store_default = str(
        args.store or args.repo_root or config.get("repo_root") or default_store()
    )
    default_workspace = args.workspace or config.get("workspace") or DEFAULT_WORKSPACE
    default_profile = args.profile or config.get("profile") or DEFAULT_PROFILE
    default_harness = [
        name for name in _split_harness_values(
            args.harness if args.harness
            else (config.get("harnesses") if "harnesses" in config else detect_harnesses()[:1])
        )
        if name in HARNESSES
    ]
    default_semantic = (
        args.semantic if args.semantic is not None
        else bool(config.get("semantic", semantic_available()))
    )

    print("Lethyx setup")
    print(f"Config will be written to {config_path()}")
    if not interactive and not args.yes:
        print("Non-interactive session - using defaults. Pass --yes to silence this notice.")

    store = store_default
    workspace = default_workspace
    profile = default_profile
    backend = default_backend
    semantic = default_semantic
    model = args.model or str(config.get("semantic_model") or DEFAULT_MODEL)
    cache = str(config.get("semantic_cache_dir") or default_cache_dir())
    harnesses = list(default_harness)

    if interactive:
        store = _ask("Store folder (memory lives here)", store_default)
        workspace = _ask("Workspace name (a namespace inside the store)", default_workspace)
        profile = _ask("Profile name", default_profile)
        backend = _ask_backend(default_backend)
        semantic = _ask_yes(
            "Enable multilingual embeddings for semantic links? "
            "The model downloads on first use",
            default_semantic,
        )
        if semantic:
            model = _ask("Embedding model", model)
            cache = _ask("Model cache folder", cache)
        harnesses = _ask_harnesses(list(default_harness))
        print("\nSummary:")
        print(f"  store      {Path(store).expanduser()}")
        print(f"  workspace  {workspace}")
        print(f"  profile    {profile}")
        print(f"  backend    {backend}")
        print(f"  semantic   {'on' if semantic else 'off'}" + (f" ({model})" if semantic else ""))
        print(f"  harness    {', '.join(harnesses) if harnesses else 'none'}")
        if not args.force and not _ask_yes("Write this configuration?", True):
            print("Cancelled; nothing was written.")
            return 0

    store_path = Path(store).expanduser()
    if not store_path.is_absolute():
        store_path = (Path.cwd() / store_path).resolve()
    store_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache).expanduser()
    if not cache_path.is_absolute():
        cache_path = (Path.cwd() / cache_path).resolve()
    if semantic:
        cache_path.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "repo_root": str(store_path),
        "workspace": workspace.strip() or DEFAULT_WORKSPACE,
        "profile": profile.strip() or DEFAULT_PROFILE,
        "backend": backend.strip().lower() or DEFAULT_BACKEND,
        "semantic": bool(semantic),
        "semantic_model": model.strip() or DEFAULT_MODEL,
        "semantic_cache_dir": str(cache_path),
        "harnesses": harnesses,
        "map_output": str(config.get("map_output") or default_map_path()),
    }
    path = save_config(config)
    print(f"\nWrote {path}")

    if semantic and not semantic_available():
        print("\nEmbeddings are enabled in the config, but fastembed is not installed here.")
        print('  python -m pip install "lethyx[semantic]"')
        print("  or reinstall through npm with: lethyx install --semantic --force")

    backend_name = config["backend"]
    if not backend_available(backend_name):
        print(f"\nThe '{backend_name}' backend needs an optional dependency that is not installed.")
        print("See docs/lethyx/install.md; 'lethyx doctor' will report it as a failure.")

    connections: list[dict[str, Any]] = []
    failed = False
    if harnesses and not args.no_connect:
        for harness in harnesses:
            try:
                result = connect_harness(harness, config)
                connections.append(result)
                print(f"Connected {harness}: {result.get('mcp_settings', 'ok')}")
            except Exception as error:  # noqa: BLE001 - report and continue
                failed = True
                print(f"Could not connect {harness}: {error}")
    elif not harnesses:
        print("No harness connected. Run 'lethyx connect opencode' (or claude/codex/hermes) later.")

    payload = {"ok": not failed, "config": config, "config_path": str(path), "connections": connections}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print("\nNext steps:")
        print("  lethyx doctor     check the store and backend")
        print("  lethyx map        render and open the memory map")
        print("  restart your harness so it reloads the MCP server")
    return 1 if failed else 0


def _effective_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    return {
        **config,
        "repo_root": str(args.repo_root or config.get("repo_root") or ""),
        "workspace": str(args.workspace or config.get("workspace") or DEFAULT_WORKSPACE),
        "profile": str(args.profile or config.get("profile") or DEFAULT_PROFILE),
        "backend": str(args.backend or config.get("backend") or DEFAULT_BACKEND),
    }


def _run_status(args: argparse.Namespace) -> int:
    config = _effective_config(args)
    configured = [name for name in (config.get("harnesses") or []) if name in HARNESSES]
    harnesses = configured or [
        name for name in detect_harnesses() if name in HARNESSES
    ]
    payload = {
        "ok": True,
        "config_path": str(config_path()),
        "config_exists": config_path().exists(),
        "store": config.get("repo_root") or "",
        "workspace": config.get("workspace"),
        "profile": config.get("profile"),
        "backend": config.get("backend"),
        "semantic": (
            bool(config.get("semantic")) if "semantic" in config
            else semantic_available()
        ),
        "semantic_model": config.get("semantic_model") or DEFAULT_MODEL,
        "semantic_available": semantic_available(),
        "harnesses": harnesses,
        "map_output": config.get("map_output") or str(default_map_path()),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    print("Lethyx status")
    print(f"  config     {config_path()}{'' if payload['config_exists'] else ' (missing - run lethyx init)'}")
    print(f"  store      {payload['store'] or '(not configured)'}")
    print(f"  workspace  {payload['workspace']}")
    print(f"  profile    {payload['profile']}")
    print(f"  backend    {payload['backend']}")
    print(f"  semantic   {'on' if payload['semantic'] else 'off'} ({payload['semantic_model']})" +
          ("" if semantic_available() else "  [fastembed not installed]"))
    print(f"  harness    {', '.join(harnesses) if harnesses else 'none'}" +
          ("" if configured else "  [detected, not configured]"))
    for harness in harnesses:
        print(f"             {harness}: {'connected' if harness_connected(harness) else 'not connected'}")
    print(f"  map        {payload['map_output']}")
    return 0


def _backend_health(repo: Path, workspace: str, backend: str) -> dict[str, Any]:
    if backend == "mnemosyne":
        from lethyx.memory.backends.mnemosyne import MnemosyneBackend

        return MnemosyneBackend.from_env(repo, workspace).health()
    instance = create_durable_backend(repo, workspace_id=workspace, mode=backend)
    try:
        return instance.health()
    finally:
        close = getattr(instance, "close", None)
        if callable(close):
            close()


def _run_doctor(args: argparse.Namespace) -> int:
    config = _effective_config(args)
    backend = str(config.get("backend") or DEFAULT_BACKEND)
    repo_raw = str(config.get("repo_root") or "").strip()
    repo = Path(repo_raw).expanduser() if repo_raw else None
    checks: list[dict[str, Any]] = []

    checks.append({
        "name": "python",
        "ok": sys.version_info >= (3, 12),
        "required": True,
        "detail": sys.version.split()[0],
    })
    checks.append({
        "name": "config",
        "ok": config_path().exists(),
        "required": False,
        "detail": str(config_path()) if config_path().exists() else "run lethyx init",
    })
    checks.append({
        "name": "store",
        "ok": bool(repo and repo.is_dir()),
        "required": True,
        "detail": str(repo) if repo else "not configured - run lethyx init",
    })
    checks.append({
        "name": "mcp runtime",
        "ok": has_module("mcp"),
        "required": bool(config.get("harnesses")),
        "detail": "mcp installed" if has_module("mcp") else "pip install 'lethyx[mcp]'",
    })
    checks.append({
        "name": "semantic runtime",
        "ok": semantic_available(),
        "required": False,
        "detail": (
            ("fastembed installed; " + (
                "enabled" if (
                    config.get("semantic") if "semantic" in config else semantic_available()
                ) else "disabled in config"
            ))
            if semantic_available()
            else "optional: pip install 'lethyx[semantic]'"
        ),
    })
    if repo and repo.is_dir():
        try:
            health = _backend_health(repo, str(config.get("workspace") or DEFAULT_WORKSPACE), backend)
            checks.append({
                "name": f"backend:{backend}",
                "ok": bool(health.get("ok", True)),
                "required": True,
                "detail": health.get("reason") or health.get("native_block_scoping_reason") or "healthy",
            })
        except Exception as error:  # noqa: BLE001 - doctor reports failures
            checks.append({
                "name": f"backend:{backend}",
                "ok": False,
                "required": True,
                "detail": f"{type(error).__name__}: {error}",
            })
    else:
        checks.append({
            "name": f"backend:{backend}",
            "ok": False,
            "required": True,
            "detail": "store folder is not configured",
        })

    failed = [check for check in checks if check["required"] and not check["ok"]]
    payload = {"ok": not failed, "checks": checks}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print("Lethyx doctor")
        for check in checks:
            mark = "ok" if check["ok"] else ("!!" if check["required"] else "--")
            print(f"  [{mark}] {check['name']:<18} {check['detail']}")
        if failed:
            print("\nFix the required items above, then run 'lethyx doctor' again.")
        else:
            print("\nAll required checks passed.")
    return 1 if failed else 0


def _run_connect(args: argparse.Namespace) -> int:
    config = _effective_config(args)
    if not config.get("repo_root"):
        config["repo_root"] = default_store()
    requested = _split_harness_values(
        list(getattr(args, "targets", []) or []) + list(args.harness or [])
    )
    if not requested:
        requested = _split_harness_values(config.get("harnesses") or [])
    if not requested:
        requested = _split_harness_values(detect_harnesses()[:1])
    if not requested:
        print("No harness selected. Use 'lethyx connect opencode' (or claude/codex/hermes).")
        return 1
    failed = False
    results = []
    for harness in requested:
        if harness not in HARNESSES:
            if not args.json:
                print(f"Unknown harness: {harness}")
            failed = True
            continue
        try:
            result = connect_harness(harness, config)
            results.append(result)
            if not args.json:
                print(f"Connected {harness}: {result.get('mcp_settings', 'ok')}")
        except Exception as error:  # noqa: BLE001 - report and continue
            failed = True
            if not args.json:
                print(f"Could not connect {harness}: {error}")
    if not failed:
        existing = load_config()
        if existing:
            existing["harnesses"] = requested
            save_config(existing)
    if args.json:
        print(json.dumps({"ok": not failed, "connections": results}, ensure_ascii=False, indent=2, default=str))
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lethyx")
    parser.add_argument(
        "command",
        choices=[
            "init",
            "setup",
            "connect",
            "status",
            "doctor",
            "map",
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
            "semantic-model",
            "semantic-index",
            "semantic-calibrate",
            "memory-map",
        ],
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="optional harness names for connect, for example: lethyx connect opencode",
    )

    parser.add_argument("--workspace", default="")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--store", default="", help="alias for --repo-root, used by init")
    parser.add_argument("--drive-root", default="")
    parser.add_argument("--hermes-home", default="")
    parser.add_argument("--hermes-agent-root", default="")
    parser.add_argument("--hermes-python", default="")
    parser.add_argument("--code-root", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--claude-watch-file", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--backend", default="")
    parser.add_argument("--harness", action="append", default=[], help="harness id(s) for init/connect")
    parser.add_argument("--semantic", dest="semantic", action="store_true", default=None)
    parser.add_argument("--no-semantic", dest="semantic", action="store_false")
    parser.add_argument("--model", default="", help="embedding model for init")
    parser.add_argument("--yes", action="store_true", help="accept defaults without prompting")
    parser.add_argument("--force", action="store_true", help="skip the setup confirmation prompt")
    parser.add_argument("--no-connect", action="store_true", help="do not touch harness configs during init")
    parser.add_argument("--no-open", action="store_true", help="do not open the generated map")
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
    parser.add_argument("--output", default="")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--logical-block-id", action="append", default=[])
    reconcile_group = parser.add_mutually_exclusive_group()
    reconcile_group.add_argument("--dry-run", dest="reconcile_dry_run", action="store_true")
    reconcile_group.add_argument("--apply", dest="reconcile_dry_run", action="store_false")
    parser.set_defaults(reconcile_dry_run=True)
    args = parser.parse_args(argv)
    args = resolve_cli_defaults(args)
    if args.command in {"init", "setup"}:
        return _run_init(args)
    if args.command == "connect":
        return _run_connect(args)
    if args.command == "status":
        return _run_status(args)
    if args.command == "doctor":
        return _run_doctor(args)

    repo = Path(args.repo_root or ".").resolve()
    drive = _drive(repo, args.workspace)
    printed = False
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
    elif args.command == "semantic-model":
        from lethyx.memory import semantic

        embedder = semantic.get_embedder(
            cache_dir=semantic.default_cache_dir(repo),
        )
        if embedder is None:
            payload = {
                "ok": False,
                "error": "fastembed is not installed",
                "hint": 'python -m pip install "lethyx[semantic]"',
            }
        else:
            vectors = embedder.embed(["lethyx semantic model check"], kind="query")
            payload = {
                "ok": bool(vectors),
                "model": embedder.model,
                "dimensions": len(vectors[0]) if vectors else 0,
                "cache_dir": str(embedder.cache_dir),
                "load_seconds": round(embedder.load_seconds, 2),
            }
    elif args.command == "semantic-index":
        from lethyx.memory import semantic as semantic_tier
        from lethyx.memory.mcp_facade import lethyx_block_map

        embedder = semantic_tier.get_embedder(
            cache_dir=semantic_tier.default_cache_dir(repo),
        )
        if embedder is None:
            payload = {
                "ok": False,
                "error": "fastembed is not installed",
                "hint": 'python -m pip install "lethyx[semantic]"',
            }
        else:
            client = MemoryClient(
                repo,
                workspace_id=args.workspace,
                profile_id=args.profile,
                backend=args.backend,
                session_id="semantic-index",
            )
            try:
                index = semantic_tier.SemanticIndex(
                    semantic_tier.semantic_index_path(repo, args.workspace)
                )
                tree = lethyx_block_map(
                    workspace_id=args.workspace,
                    repo_root=str(repo),
                    backend=args.backend,
                    depth=4,
                )
                leaves: list[dict[str, object]] = []

                def collect(nodes: object) -> None:
                    for node in nodes or []:
                        if not isinstance(node, dict):
                            continue
                        children = node.get("children") or []
                        if children:
                            collect(children)
                        elif not node.get("index_only"):
                            leaves.append(node)

                collect(tree.get("roots") or [])
                indexed = 0
                skipped = 0
                for leaf in leaves:
                    block_id = str(leaf.get("block_id") or "").strip()
                    if not block_id:
                        continue
                    cursor = ""
                    while True:
                        page = client.list_records(
                            scope="workspace",
                            scope_id=args.workspace,
                            logical_block_id=block_id,
                            limit=128,
                            cursor=cursor,
                        )
                        identifiers = [
                            str(item.get("record_id") or "")
                            for item in page.get("records") or []
                            if str(item.get("record_id") or "")
                        ]
                        for start in range(0, len(identifiers), 128):
                            for record in client.read_exact_records(
                                identifiers[start : start + 128]
                            ):
                                if semantic_tier.embed_record(
                                    index,
                                    embedder,
                                    profile=args.profile,
                                    record_id=record.block_id,
                                    title=record.title or "",
                                    content=record.content or "",
                                    kind=record.kind or "",
                                ):
                                    indexed += 1
                                else:
                                    skipped += 1
                        cursor = str(page.get("next_cursor") or "")
                        if not cursor:
                            break
                payload = {
                    "ok": True,
                    "model": embedder.model,
                    "indexed": indexed,
                    "skipped": skipped,
                    "indexed_records": index.count(embedder.model, args.profile),
                    "cache_dir": str(embedder.cache_dir),
                }
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
    elif args.command == "semantic-calibrate":
        from lethyx.memory import semantic as semantic_tier

        embedder = semantic_tier.get_embedder(
            cache_dir=semantic_tier.default_cache_dir(repo),
        )
        if embedder is None:
            payload = {
                "ok": False,
                "error": "fastembed is not installed",
                "hint": 'python -m pip install "lethyx[semantic]"',
            }
        else:
            client = MemoryClient(
                repo,
                workspace_id=args.workspace,
                profile_id=args.profile,
                backend=args.backend,
                session_id="semantic-calibrate",
            )
            try:
                records = client.recent_records(
                    limit=128,
                    scope="workspace",
                    scope_id=args.workspace,
                    include_candidates=True,
                )
                window = [
                    {
                        "record_id": str(record.block_id or ""),
                        "title": str(record.title or ""),
                        "content": str(record.content or ""),
                    }
                    for record in records
                ]
                index = semantic_tier.SemanticIndex(
                    semantic_tier.semantic_index_path(repo, args.workspace)
                )
                measured = semantic_tier.calibrate(
                    window, index, embedder, profile=args.profile
                )
                if measured.get("ok"):
                    semantic_tier.store_calibration(
                        repo, args.workspace, measured
                    )
                payload = measured
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
    elif args.command in {"map", "memory-map"}:
        from lethyx.memory.memory_map import (
            collect_memory_map,
            memory_map_summary,
            write_memory_map,
        )

        config = load_config()
        target = str(args.output or "").strip()
        if not target and args.command == "map":
            target = str(config.get("map_output") or default_map_path())
        if not target:
            target = str(Path(repo) / "memory-map.html")
        if args.command == "map" and not args.json and not config_path().exists() and not args.repo_root:
            print("No Lethyx config found; using the current folder. Run 'lethyx init' once to set defaults.")
        written = write_memory_map(
            repo,
            target,
            workspace_id=args.workspace,
            profile_id=args.profile,
            backend=args.backend,
        )
        payload = {
            "ok": True,
            "path": str(written),
            "summary": memory_map_summary(
                collect_memory_map(
                    repo,
                    workspace_id=args.workspace,
                    profile_id=args.profile,
                    backend=args.backend,
                )
            ),
        }
        if args.command == "map" and not args.json:
            print(f"Map written to {written}")
            print(f"  {payload['summary']}")
            if not args.no_open:
                try:
                    webbrowser.open(written.resolve().as_uri())
                    print("Opened in your browser. Use --no-open to only write the file.")
                except Exception:  # noqa: BLE001 - a headless session still gets the file
                    pass
            printed = True
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
    if not printed:
        _print(payload, args.json)
    if args.command == "reconcile" and not args.reconcile_dry_run:
        failures = payload.get("failures", 0)
        if type(failures) is not int:
            raise ValueError("reconcile failure count is invalid")
        return 1 if failures > 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
