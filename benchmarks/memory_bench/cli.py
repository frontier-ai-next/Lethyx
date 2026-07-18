"""CLI for the memory_bench framework.

Commands:
  run        Run a benchmark suite or single provider/benchmark pair.
  list       List registered providers, benchmarks, or judges.
  report     Re-render reports from an existing run directory.
  datasets   Check dataset availability and integrity.
"""

import argparse
import logging
import sys
from pathlib import Path


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=level,
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from memory_bench.core import registry
    from memory_bench.core.runner import run_suite
    from memory_bench.core.types import (
        DatasetSpec,
        JudgeConfig,
        LimitsSpec,
        ProviderSpec,
        ResponderSpec,
        RunSpec,
    )

    registry.bootstrap()

    if args.suite:
        from memory_bench.core.config import load_suite

        specs = load_suite(args.suite)
        if not specs:
            print("No run specs found in suite file.", file=sys.stderr)
            return 1
        spec = specs[0]
        # CLI overrides
        if args.out:
            spec = RunSpec(
                providers=spec.providers,
                benchmarks=spec.benchmarks,
                dataset=spec.dataset,
                responder=spec.responder,
                judge=spec.judge,
                limits=spec.limits,
                out_dir=args.out,
                cache_dir=spec.cache_dir,
            )
    else:
        # Build spec from CLI flags
        providers = []
        if args.provider:
            for p in args.provider:
                parts = p.split(":", 1)
                adapter = parts[0]
                backend = parts[1] if len(parts) > 1 else ""
                params: dict = {}
                if backend:
                    params["backend"] = backend
                providers.append(ProviderSpec(id=p, adapter=adapter, params=params))
        if not providers:
            print("Specify --suite or at least one --provider", file=sys.stderr)
            return 1

        benchmarks: tuple[str, ...] = tuple(args.benchmark or [])
        if not benchmarks:
            benchmarks = ("locomo",)

        spec = RunSpec(
            providers=tuple(providers),
            benchmarks=benchmarks,
            dataset=DatasetSpec(
                name="",
                max_episodes=args.max_episodes or 0,
                max_probes_per_episode=args.max_probes or 0,
            ),
            responder=ResponderSpec(model=args.model or ""),
            judge=JudgeConfig(use_llm_judge=not args.no_llm_judge),
            limits=LimitsSpec(
                recall_k=args.recall_k or 10,
                seed=args.seed or 0,
                redo_existing=args.redo,
            ),
            out_dir=args.out or "runs/memory_bench",
        )

    repo_root = Path(__file__).resolve().parents[2]
    result = run_suite(spec, repo_root=repo_root)
    print(f"\nRun complete. Output: {result['out_dir']}")
    return 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    from memory_bench.core import registry

    registry.bootstrap()

    what = args.what
    if what == "providers" or what == "all":
        print("Providers:", ", ".join(registry.list_providers()) or "(none)")
    if what == "benchmarks" or what == "all":
        print("Benchmarks:", ", ".join(registry.list_benchmarks()) or "(none)")
    if what == "judges" or what == "all":
        print("Judges:", ", ".join(registry.list_judges()) or "(none)")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    from memory_bench.core.report import build_report, write_report
    from memory_bench.core.runner import read_results_from_dir

    out_dir = Path(args.run_dir)
    results = read_results_from_dir(out_dir)
    if not results:
        print(f"No results found in {out_dir}", file=sys.stderr)
        return 1
    report = build_report(results)
    json_p, md_p, csv_p = write_report(report, out_dir=out_dir)
    print(f"Report written: {md_p}")
    return 0


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


def cmd_datasets(args: argparse.Namespace) -> int:
    from memory_bench.core.datasets import check_all, print_check_report

    raw_names = list(args.names or [])
    # Compatibility with the README and common CLI muscle memory:
    # `memory_bench datasets check` means "check all datasets", not a dataset
    # literally named "check".
    names = None if raw_names in ([], ["check"]) else raw_names
    report = check_all(names)
    print_check_report(report)
    missing = [n for n, info in report.items() if not info["present"]]
    return 1 if missing else 0


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory_bench",
        description="memory_bench — any memory × any benchmark evaluation framework",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Run a benchmark suite")
    run_p.add_argument("--suite", "-s", help="Path to suite.toml")
    run_p.add_argument(
        "--provider",
        "-p",
        action="append",
        help="Provider spec: adapter[:backend]  (e.g. lethyx:mempalace, bm25)",
    )
    run_p.add_argument("--benchmark", "-b", action="append", help="Benchmark name(s)")
    run_p.add_argument("--max-episodes", type=int, default=0)
    run_p.add_argument("--max-probes", type=int, default=0)
    run_p.add_argument("--model", help="Reader LLM model override")
    run_p.add_argument("--recall-k", type=int, default=10)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--no-llm-judge", action="store_true")
    run_p.add_argument("--redo", action="store_true", help="Re-evaluate already-done probes")
    run_p.add_argument("--out", "-o", help="Output directory")

    # list
    list_p = sub.add_parser("list", help="List registered components")
    list_p.add_argument(
        "what",
        choices=["providers", "benchmarks", "judges", "all"],
        nargs="?",
        default="all",
    )

    # report
    rep_p = sub.add_parser("report", help="Re-render report from run directory")
    rep_p.add_argument("run_dir", help="Path to a run output directory")

    # datasets
    ds_p = sub.add_parser("datasets", help="Check dataset availability")
    ds_p.add_argument("names", nargs="*", help="Dataset names to check (default: all)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    handlers = {
        "run": cmd_run,
        "list": cmd_list,
        "report": cmd_report,
        "datasets": cmd_datasets,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
