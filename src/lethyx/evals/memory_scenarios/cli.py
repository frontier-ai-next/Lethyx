"""CLI for Memory Scenario Harness."""

import argparse
import json
import sys
from pathlib import Path

from lethyx.evals.memory_scenarios.fixtures import REPO_ROOT
from lethyx.evals.memory_scenarios.reports import print_terminal_summary
from lethyx.evals.memory_scenarios.runner import (
    MemoryScenarioRunner,
    run_all_scenarios,
    run_scenario_by_id,
)
from lethyx.evals.memory_scenarios.scenario_loader import (
    LLM_SCENARIOS_DIR,
    list_scenario_paths,
    load_all_llm_scenarios,
    load_scenario,
)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", help="Scenario id or yaml stem")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--phase", help="Run scenarios that include this phase")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--update-golden", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--report-dir",
        default=str(REPO_ROOT / ".mrt" / "memory_scenarios"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Memory Scenario Harness / Audit Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List deterministic scenarios")
    sub.add_parser("list-llm", help="List LLM behavioral scenarios")

    run_p = sub.add_parser("run", help="Run deterministic scenario(s)")
    _add_run_args(run_p)

    run_llm_p = sub.add_parser("run-llm", help="Run opt-in LLM behavioral scenarios")
    _add_run_args(run_llm_p)

    inspect_p = sub.add_parser("inspect", help="Show last report path")
    inspect_p.add_argument("--scenario", required=True)
    inspect_p.add_argument("--open-report", action="store_true")

    args = parser.parse_args(argv)
    default_report = REPO_ROOT / ".mrt" / "memory_scenarios"
    report_root = Path(getattr(args, "report_dir", None) or default_report)

    if args.command == "list":
        for path in list_scenario_paths(include_llm=False):
            sc = load_scenario(path)
            print(f"{path.stem}\t{sc.id}\t{sc.mode}")
        return 0

    if args.command == "list-llm":
        if not LLM_SCENARIOS_DIR.is_dir():
            return 0
        for path in sorted(LLM_SCENARIOS_DIR.glob("*.yaml")):
            sc = load_scenario(path)
            print(f"{path.stem}\t{sc.id}\tllm")
        return 0

    if args.command == "inspect":
        report = report_root / args.scenario / "report.md"
        if not report.is_file():
            print(f"no report: {report}", file=sys.stderr)
            return 1
        print(report)
        if args.open_report:
            print(report.read_text(encoding="utf-8"))
        return 0

    if args.command in {"run", "run-llm"}:
        kwargs = {
            "verbose": args.verbose,
            "keep_tmp": args.keep_tmp,
            "update_golden": args.update_golden,
        }
        if args.command == "run-llm":
            from lethyx.evals.memory_scenarios.llm_judge import judge_memory_behavior

            runner = MemoryScenarioRunner(**kwargs)
            results = []
            for scenario in load_all_llm_scenarios():
                result = runner.run(scenario)
                if scenario.llm:
                    step_id = str(scenario.llm.get("step_id") or "")
                    step = next((s for s in result.step_results if s.step_id == step_id), None)
                    if step and step.prompt:
                        verdict = judge_memory_behavior(
                            phase_prompt=step.prompt,
                            task_question=str(scenario.llm.get("task_question") or ""),
                            expect_provenance=bool(scenario.llm.get("expect_provenance")),
                            expect_antipattern=bool(scenario.llm.get("expect_antipattern")),
                            forbid_research_rule=bool(scenario.llm.get("forbid_research_rule")),
                        )
                        if not verdict.followed_directive or verdict.violated_candidate:
                            result.ok = False
                            result.invariant_failures.append(verdict.reasoning)
                results.append(result)
                if args.fail_fast and not result.ok:
                    break
            ok = all(r.ok for r in results)
            if args.json:
                print(json.dumps({"ok": ok}, indent=2))
            else:
                print_terminal_summary(results)
            return 0 if ok else 1

        if args.all:
            result = run_all_scenarios(
                fail_fast=args.fail_fast,
                phase_filter=args.phase,
                include_llm=False,
                report_root=report_root,
                **kwargs,
            )
            if args.json:
                print(json.dumps(result.dashboard, indent=2))
            return 0 if result.ok else 1
        if not args.scenario:
            print("specify --scenario or --all", file=sys.stderr)
            return 2
        result = run_scenario_by_id(args.scenario, report_root=report_root, **kwargs)
        if args.json:
            print(json.dumps({"ok": result.ok, "failures": result.invariant_failures}, indent=2))
        else:
            print(result.summary_text)
        return 0 if result.ok else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
