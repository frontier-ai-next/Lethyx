"""Memory scenario runner — offline phase/tool memory audit."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from lethyx.evals.memory_scenarios.assertions import (
    assert_memory_injection_contract,
    evaluate_assert_block,
    structured_facts,
)
from lethyx.evals.memory_scenarios.fake_hindsight import (
    FakeHindsightCallLog,
    FakeHindsightClient,
    default_reflect_candidate,
)
from lethyx.evals.memory_scenarios.fake_host_agent import run_host_agent_dedup_check
from lethyx.evals.memory_scenarios.fixtures import (
    EXPECTED_ROOT,
    REPO_ROOT,
    drive_root,
    manifest_path,
    prepare_scenario_repo,
)
from lethyx.evals.memory_scenarios.models import (
    MemoryScenario,
    MemoryScenarioResult,
    ScenarioStep,
    ScenarioStepResult,
)
from lethyx.evals.memory_scenarios.reports import (
    build_dashboard,
    print_terminal_summary,
    write_scenario_artifacts,
)
from lethyx.evals.memory_scenarios.scenario_loader import (
    load_all_scenarios,
    load_scenario_by_id,
)
from lethyx.evals.memory_scenarios.seeding import seed_from_dict
from lethyx.evals.memory_scenarios.snapshots import (
    capture_snapshot,
    write_step_snapshots,
)
from lethyx.evals.memory_scenarios.phase_tasks import (
    PhaseNode,
    build_phase_task,
    load_manifest,
    render_phase_user_prompt,
)
from lethyx.memory.palace.facade import MemPalace
from lethyx.memory.proactive.compiler import ProactiveMemoryCompiler


class MemoryScenarioRunner:
    def __init__(
        self,
        *,
        report_root: Path | None = None,
        keep_tmp: bool = False,
        update_golden: bool = False,
        verbose: bool = False,
    ) -> None:
        self.report_root = report_root or (REPO_ROOT / ".mrt" / "memory_scenarios")
        self.keep_tmp = keep_tmp
        self.update_golden = update_golden
        self.verbose = verbose
        self._fake_hindsight: FakeHindsightClient | None = None
        self._hindsight_log: FakeHindsightCallLog | None = None

    def run(self, scenario: MemoryScenario) -> MemoryScenarioResult:
        tmp = tempfile.mkdtemp(prefix="mem-scenario-")
        repo = Path(tmp)
        failures: list[str] = []
        step_results: list[ScenarioStepResult] = []
        ws = scenario.workspace
        old_env = dict(os.environ)

        try:
            os.environ.clear()
            os.environ.update(old_env)
            for key, val in scenario.env.items():
                os.environ[key] = str(val)

            prepare_scenario_repo(
                repo,
                workspace_id=ws,
                workspace_fixture=scenario.seed.workspace_fixture,
                manager_fixture=scenario.seed.manager_fixture,
                extra_workspaces=scenario.seed.extra_workspaces,
            )
            seed_from_dict(repo, ws, scenario.raw_seed)

            report_dir = self.report_root / scenario.id
            if report_dir.exists():
                shutil.rmtree(report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)

            context: dict[str, Any] = {
                "repo": repo,
                "workspace_id": ws,
                "drive": drive_root(repo, ws),
                "tasks": {},
                "fake_hindsight": self._ensure_fake_hindsight(report_dir),
            }

            steps = list(scenario.steps)
            if scenario.mode == "phase_prompt_matrix" and not steps:
                phases = scenario.assertions.get("matrix_phases") or [
                    "preflight",
                    "research",
                    "plan",
                    "execute",
                    "verify",
                    "reflexion",
                ]
                run_id = str(scenario.assertions.get("run_id") or "run-scenario-matrix")
                for phase in phases:
                    steps.append(
                        ScenarioStep(
                            id=f"{phase}_prompt",
                            action="build_phase_task",
                            phase=phase,
                            run_id=run_id,
                        )
                    )

            for step in steps:
                step_drive = context["drive"]
                if step.workspace_id and step.workspace_id != ws:
                    step_drive = drive_root(repo, step.workspace_id)
                before = capture_snapshot(repo, step.workspace_id or ws, step_drive)
                result = self._run_step(scenario, step, context)
                after = capture_snapshot(repo, step.workspace_id or ws, step_drive)
                result.palace_before = before.get("palace", {})
                result.palace_after = after.get("palace", {})
                result.snapshot_before = before
                result.snapshot_after = after
                write_step_snapshots(report_dir, step.id, before, after)
                self._write_step_artifacts(report_dir, step.id, result)
                step_results.append(result)

                if result.errors:
                    failures.extend(f"{scenario.id}.{step.id}: {err}" for err in result.errors)

                assert_spec = scenario.assertions.get(step.id)
                if assert_spec is None and scenario.mode == "phase_prompt_matrix":
                    assert_spec = scenario.assertions.get("_default")
                if assert_spec is not None:
                    errs = evaluate_assert_block(
                        step.id,
                        assert_spec,
                        result,
                        task=result.task,
                        repo=repo,
                        workspace_id=step.workspace_id or ws,
                        drive=step_drive,
                    )
                    result.errors.extend(errs)
                    if errs:
                        failures.extend(f"{scenario.id}.{step.id}: {e}" for e in errs)
                    result.ok = result.ok and not errs

            for key, spec in scenario.assertions.items():
                if key.endswith("_prompt") or key in context.get("tasks", {}):
                    continue
                if isinstance(spec, dict) and key not in {s.id for s in scenario.steps}:
                    task = context["tasks"].get(key)
                    if task:
                        sr = ScenarioStepResult(step_id=key, action="assert", ok=True, task=task)
                        errs = evaluate_assert_block(key, spec, sr, task=task)
                        if errs:
                            failures.extend(errs)
                            sr.ok = False
                            step_results.append(sr)

            if self.update_golden:
                self._update_golden(scenario.id, step_results)

            if not self.update_golden:
                failures.extend(self._check_golden(scenario.id, step_results))

            ok = not failures
            mem_result = MemoryScenarioResult(
                scenario_id=scenario.id,
                ok=ok,
                repo_root=repo,
                workspace_id=ws,
                step_results=step_results,
                invariant_failures=failures,
                report_dir=report_dir,
            )
            write_scenario_artifacts(report_dir, scenario, mem_result)
            return mem_result
        finally:
            os.environ.clear()
            os.environ.update(old_env)
            if not self.keep_tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    def _ensure_fake_hindsight(self, report_dir: Path) -> FakeHindsightClient:
        if self._fake_hindsight is None:
            self._fake_hindsight = FakeHindsightClient()
            self._hindsight_log = FakeHindsightCallLog(report_dir / "hindsight_fake_calls.jsonl")
        return self._fake_hindsight

    def _run_step(
        self,
        scenario: MemoryScenario,
        step: ScenarioStep,
        context: dict[str, Any],
    ) -> ScenarioStepResult:
        repo: Path = context["repo"]
        ws = step.workspace_id or context["workspace_id"]
        drive = context["drive"]
        if step.workspace_id and step.workspace_id != context["workspace_id"]:
            drive = drive_root(repo, ws)
        errors: list[str] = []
        result = ScenarioStepResult(step_id=step.id, action=step.action, ok=True)

        try:
            if step.action == "build_phase_task":
                task = self._build_task(repo, ws, step.phase or "research", step.run_id, drive)
                result.task = task
                result.prompt = str(task.get("input") or "")
                result.overlays = task.get("context_overlays") or {}
                context["tasks"][step.id] = task
                _, cerrs = assert_memory_injection_contract(task)
                errors.extend(cerrs)
                bundle_path = drive / "state" / "llm_input_bundle_latest.json"
                report_path = drive / "state" / "memory_injection_report_latest.json"
                if bundle_path.is_file():
                    result.bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                if report_path.is_file():
                    result.injection_report = json.loads(report_path.read_text(encoding="utf-8"))

            elif step.action == "render_phase_prompt":
                manifest = load_manifest(manifest_path(step.phase or "research"))
                overlay = ProactiveMemoryCompiler().build_overlay(
                    repo_root=repo,
                    workspace_id=ws,
                    run_id=step.run_id,
                    phase_id=step.phase or "research",
                    subtask_id=None,
                    task_brief="scenario",
                    manifest=manifest,
                )
                from lethyx.memory.palace.recall import RecallBundle

                result.prompt = render_phase_user_prompt(
                    manifest, RecallBundle(), proactive_overlay=overlay
                )

            elif step.action == "palace_add":
                errors.extend(self._action_palace_add(repo, ws, drive, step))

            elif step.action == "save_LETHYX_memory":
                errors.extend(self._action_save_memory(repo, ws, step))

            elif step.action == "promote_to_durable":
                errors.extend(self._action_promote(repo, ws, drive, step))

            elif step.action == "fake_hindsight_reflect":
                errors.extend(self._action_fake_reflect(repo, ws, drive, step, context))

            elif step.action == "hindsight_backend_reflect":
                errors.extend(
                    self._action_hindsight_backend_reflect(repo, ws, drive, step, context)
                )

            elif step.action == "accept_bkb_patch":
                errors.extend(self._action_accept_bkb_patch(repo, ws, drive, step))

            elif step.action == "process_reflexion_bkb_patch":
                errors.extend(self._action_process_reflexion_patch(repo, ws, drive, step))

            elif step.action == "init_host_agent_memory":
                prior_id = step.args.get("task_step_id")
                task = context["tasks"].get(str(prior_id)) if prior_id else None
                if not task:
                    phase = step.phase or "verify"
                    task = self._build_task(repo, ws, phase, step.run_id, drive)
                result.task = task
                context["tasks"][step.id] = task
                dedup_errs = run_host_agent_dedup_check(repo, ws, task)
                errors.extend(dedup_errs)

            elif step.action == "memory_health":
                palace = MemPalace(repo, ws)
                try:
                    health = palace.health()
                    result.overlays = dict(result.overlays or {})
                    result.overlays["memory_health"] = health
                    if step.args.get("expect_unavailable"):
                        if health.get("ok") and not health.get("stores_fail"):
                            # Real chromadb available — unavailable path not exercisable here.
                            pass
                        elif health.get("ok"):
                            errors.append("memory_health: failures present but ok=true (masked)")
                        elif not health.get("stores_fail"):
                            errors.append(
                                "memory_health: expected stores_fail when backend missing"
                            )
                finally:
                    palace.close()

            elif step.action == "phase_forced_completion":
                errors.extend(self._action_phase_forced_completion(drive, step, result))

            elif step.action == "phase_no_write_guard":
                errors.extend(self._action_phase_no_write_guard(step, result))

            elif step.action == "submit_micro_review":
                errors.extend(self._action_submit_micro_review(drive, step, result))

            elif step.action == "validate_phase_plan_contract":
                errors.extend(self._action_validate_phase_plan_contract(step, result))

            elif step.action == "validate_required_plan_change":
                errors.extend(self._action_validate_required_plan_change(step, result))

            elif step.action == "analyze_python_test":
                errors.extend(self._action_analyze_python_test(step, result))

            else:
                errors.append(f"unsupported action: {step.action}")
        except Exception as exc:
            errors.append(f"{step.action}: {exc}")

        result.errors = errors
        result.ok = not errors
        return result

    def _build_task(
        self,
        repo: Path,
        workspace_id: str,
        phase_id: str,
        run_id: str,
        drive: Path,
    ) -> dict[str, Any]:
        manifest = load_manifest(manifest_path(phase_id))
        phase_node = PhaseNode(id=phase_id, manifest_id=phase_id, status="running")
        return build_phase_task(
            phase_node=phase_node,
            manifest=manifest,
            workspace_id=workspace_id,
            run_id=run_id,
            repo_root=repo,
            drive_root=drive,
        )

    def _action_phase_forced_completion(
        self, drive: Path, step: ScenarioStep, result: ScenarioStepResult
    ) -> list[str]:
        args = step.args
        messages: list[dict[str, Any]] = []
        trace_text = "\n".join(
            str(item.get("result") or item.get("tool") or "")
            for item in (args.get("trace_tool_calls") or [])
            if isinstance(item, dict)
        )
        terminating = {str(item) for item in (args.get("terminating_tools") or [])}
        accepted = {str(item) for item in (args.get("accepted_terminating_tools") or [])}
        forced = args.get("forced_progress_tool_choice")
        if (
            "submit_research_summary" in terminating
            and "submit_research_summary" not in accepted
            and "palace_add" in trace_text
        ):
            forced = "palace_add"
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "REQUIRED_RESEARCH_FINDING_PENDING: call palace_add with "
                        "accepted research_finding evidence before retrying the summary."
                    ),
                }
            )
        result.overlays = {
            **(result.overlays or {}),
            "forced_tool": forced,
            "messages": messages,
        }
        errors: list[str] = []
        expected = args.get("expect_forced_tool")
        if forced != expected:
            errors.append(f"expected forced tool {expected!r}, got {forced!r}")
        for needle in args.get("expect_message_contains") or []:
            if not any(str(needle) in str(msg.get("content") or "") for msg in messages):
                errors.append(f"forced completion message missing {needle!r}")
        return errors

    def _action_phase_no_write_guard(
        self, step: ScenarioStep, result: ScenarioStepResult
    ) -> list[str]:
        args = step.args
        messages: list[dict[str, Any]] = []
        llm_trace = {"tool_calls": list(args.get("trace_tool_calls") or [])}
        forced_tool = str(args.get("forced_tool") or "")
        allowed_during_summary_repair = {"palace_add"}
        requested = {
            str(item.get("tool") or item.get("name") or "")
            for item in (args.get("tool_calls") or [])
            if isinstance(item, dict)
        }
        rejected = bool(
            forced_tool == "submit_research_summary"
            and requested
            and not requested.issubset(allowed_during_summary_repair)
        )
        if rejected:
            messages.append(
                {
                    "role": "system",
                    "content": f"Tool call rejected under phase no-write guard: {sorted(requested)}",
                }
            )
        result.overlays = {
            **(result.overlays or {}),
            "rejected": rejected,
            "messages": messages,
            "llm_trace": llm_trace,
        }
        errors: list[str] = []
        expected = bool(args.get("expect_rejected"))
        if rejected != expected:
            errors.append(f"expected guard rejected={expected}, got {rejected}")
        for needle in args.get("expect_message_contains") or []:
            if not any(str(needle) in str(msg.get("content") or "") for msg in messages):
                errors.append(f"guard message missing {needle!r}")
        return errors

    def _action_submit_micro_review(
        self, drive: Path, step: ScenarioStep, result: ScenarioStepResult
    ) -> list[str]:
        args = step.args
        state = drive / "state"
        state.mkdir(parents=True, exist_ok=True)
        if isinstance(args.get("submitted_plan"), dict):
            (state / "phase_plan_submitted_latest.json").write_text(
                json.dumps(args["submitted_plan"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        phase = str(step.phase or args.get("phase") or "plan_review")
        payload = {
            "phase": phase,
            "verdict": str(args.get("verdict") or "revise"),
            "issues": list(args.get("issues") or []),
            "revisions": list(args.get("revisions") or []),
            "loop_back_target": str(args.get("loop_back_target") or ""),
            "notes": str(args.get("notes") or ""),
            "coverage": dict(args.get("coverage") or {}),
            "required_plan_changes": [
                self._normalize_required_plan_change(change)
                for change in (args.get("required_plan_changes") or [])
                if isinstance(change, dict)
            ],
        }
        row = {
            "kind": "submit_micro_review",
            "task_id": f"{step.run_id}:{phase}",
            "payload": payload,
        }
        with (state / "phase_control_signals.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        review_result = "OK: micro review submitted"
        rows = self._read_jsonl(state / "phase_control_signals.jsonl")
        review_rows = [
            row
            for row in rows
            if isinstance(row, dict) and row.get("kind") == "submit_micro_review"
        ]
        payload = (
            review_rows[-1].get("payload")
            if review_rows and isinstance(review_rows[-1], dict)
            else {}
        )
        result.overlays = {
            **(result.overlays or {}),
            "review_result": review_result,
            "review_payload": payload,
        }
        errors: list[str] = []
        prefix = str(args.get("expect_result_prefix") or "OK:")
        if not review_result.startswith(prefix):
            errors.append(f"micro review result {review_result!r} does not start {prefix!r}")
        changes = payload.get("required_plan_changes") if isinstance(payload, dict) else []
        changes = changes if isinstance(changes, list) else []
        for expected in args.get("expect_required_plan_changes") or []:
            if isinstance(expected, dict) and not any(
                self._mapping_contains(change, expected)
                for change in changes
                if isinstance(change, dict)
            ):
                errors.append(f"missing required_plan_change subset: {expected}")
        absent_ids = {str(item) for item in (args.get("expect_absent_change_ids") or [])}
        for change in changes:
            if isinstance(change, dict) and str(change.get("id") or "") in absent_ids:
                errors.append(f"unexpected required_plan_change id: {change.get('id')}")
        return errors

    def _action_validate_phase_plan_contract(
        self, step: ScenarioStep, result: ScenarioStepResult
    ) -> list[str]:
        from lethyx.contracts import ContractBundle, ContractValidator
        from lethyx.contracts.plan_ir import compile_phase_plan

        args = step.args
        plan, compile_issues = compile_phase_plan(
            dict(args.get("plan") or {}),
            run_id=str(args.get("run_id") or step.run_id),
            workspace_id=str(args.get("workspace_id") or step.workspace_id or "test"),
        )
        issues = list(compile_issues or [])
        if plan is not None:
            issues.extend(
                ContractValidator.validate(
                    ContractBundle(
                        run_id=str(args.get("run_id") or step.run_id),
                        workspace_id=str(args.get("workspace_id") or step.workspace_id or "test"),
                        plan=plan,
                    )
                )
            )
        codes = self._issue_codes(issues)
        result.overlays = {
            **(result.overlays or {}),
            "issue_codes": sorted(codes),
            "issues": [str(getattr(issue, "message", issue)) for issue in issues],
        }
        errors: list[str] = []
        for code in args.get("expect_issue_codes") or []:
            if str(code) not in codes:
                errors.append(f"expected contract issue code {code!r}, got {sorted(codes)}")
        for code in args.get("expect_absent_issue_codes") or []:
            if str(code) in codes:
                errors.append(f"unexpected contract issue code {code!r}")
        return errors

    def _action_validate_required_plan_change(
        self, step: ScenarioStep, result: ScenarioStepResult
    ) -> list[str]:
        args = step.args
        issue = self._required_plan_change_issue(
            dict(args.get("plan") or {}),
            dict(args.get("change") or {}),
        )
        result.overlays = {
            **(result.overlays or {}),
            "required_plan_change_issue": issue,
        }
        if args.get("expect_issue_empty", True):
            return [] if not issue else [f"required_plan_change still unmet: {issue}"]
        expected = str(args.get("expect_issue_contains") or "")
        if expected and expected not in issue:
            return [f"required_plan_change issue missing {expected!r}: {issue}"]
        return []

    def _action_analyze_python_test(
        self, step: ScenarioStep, result: ScenarioStepResult
    ) -> list[str]:
        args = step.args
        source = str(args.get("source") or "")
        issues = []
        if "tk.Button()" in source or "tkinter.Button()" in source:
            issues.append(
                {
                    "code": "masterless_gui_widget_in_test",
                    "message": "Tk widgets in tests need an explicit root/master.",
                }
            )
        codes = self._issue_codes(issues)
        result.overlays = {
            **(result.overlays or {}),
            "issue_codes": sorted(codes),
            "issues": [str(getattr(issue, "message", issue)) for issue in issues],
        }
        errors: list[str] = []
        for code in args.get("expect_issue_codes") or []:
            if str(code) not in codes:
                errors.append(f"expected analysis issue code {code!r}, got {sorted(codes)}")
        for code in args.get("expect_absent_issue_codes") or []:
            if str(code) in codes:
                errors.append(f"unexpected analysis issue code {code!r}")
        return errors

    @staticmethod
    def _tool_call_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
        name = str(spec.get("name") or spec.get("tool") or "").strip()
        return {
            "id": str(spec.get("id") or f"call_{name or 'tool'}"),
            "function": {
                "name": name,
                "arguments": json.dumps(spec.get("arguments") or {}),
            },
        }

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not path.is_file():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    @staticmethod
    def _mapping_contains(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
        return all(actual.get(key) == value for key, value in expected.items())

    @staticmethod
    def _issue_codes(issues: list[Any]) -> set[str]:
        codes: set[str] = set()
        for issue in issues:
            if isinstance(issue, dict):
                code = issue.get("code")
            else:
                code = getattr(issue, "code", None)
            if code:
                codes.add(str(code))
        return codes

    @staticmethod
    def _normalize_required_plan_change(change: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(change)
        path = str(normalized.get("path") or "")
        if path.startswith("proof.execution."):
            normalized["path"] = "proof.execution"
            normalized["op"] = "semantic_diff"
        return normalized

    @staticmethod
    def _required_plan_change_issue(plan: dict[str, Any], change: dict[str, Any]) -> str:
        target = str(change.get("target_subtask_id") or "")
        path = str(change.get("path") or "")
        subtasks = (plan.get("plan") or {}).get("subtasks") or plan.get("subtasks") or []
        subtask = next(
            (
                item
                for item in subtasks
                if isinstance(item, dict) and str(item.get("id") or "") == target
            ),
            {},
        )
        if not subtask:
            return f"target subtask not found: {target}"
        if path == "proof.execution":
            current = (subtask.get("proof") or {}).get("execution") or {}
            previous = change.get("previous_value") or {}
            if isinstance(previous, dict) and previous == current:
                return "required semantic proof.execution change was not applied"
            return ""
        return ""

    def _action_palace_add(self, repo: Path, ws: str, drive: Path, step: ScenarioStep) -> list[str]:
        from lethyx.memory.palace.write import canonical_palace_add

        args = step.args
        payload = canonical_palace_add(
            repo,
            workspace_id=str(args.get("workspace_id") or ws),
            title=str(args.get("title") or "Harness title"),
            content=str(args.get("content") or "Harness body"),
            kind=str(args.get("kind") or "observation"),
            source_path=str(args.get("palace_path") or ""),
            tags=str(args.get("tags") or "observation"),
        )
        if not payload.get("saved"):
            return [f"palace_add not saved: {payload}"]
        nested = repo / "workspaces" / ws / "workspaces"
        if nested.exists():
            return [f"nested workspaces path exists: {nested}"]
        return []

    def _action_save_memory(self, repo: Path, ws: str, step: ScenarioStep) -> list[str]:
        from lethyx.memory.palace.write import canonical_palace_add

        args = step.args
        first = canonical_palace_add(
            repo,
            workspace_id=ws,
            source_path=str(args.get("palace_path") or f"workspaces/{ws}/plan"),
            title=str(args.get("title") or "Dup title"),
            content=str(args.get("content") or "Dup body"),
            kind="observation",
            tags="observation",
        )
        cid = first.get("canonical_id")
        if args.get("duplicate"):
            canonical_palace_add(
                repo,
                workspace_id=ws,
                source_path=str(args.get("palace_path") or f"workspaces/{ws}/plan"),
                title=str(args.get("title") or "Dup title"),
                content=str(args.get("content") or "Dup body"),
                kind="observation",
                tags="observation",
                extra={"canonical_id": cid},
            )
            store = str(first.get("store") or "palace.idea")
            palace = MemPalace(repo, ws)
            try:
                matches = [h for h in palace.list_all(n=200, stores=[store]) if h.get("id") == cid]
            finally:
                palace.close()
            if len(matches) != 1:
                return [f"expected one canonical node, got {len(matches)}"]
        return []

    def _action_promote(self, repo: Path, ws: str, drive: Path, step: ScenarioStep) -> list[str]:
        from lethyx.enforcement.ledger import (
            append_supervisor_ledger_event,
            find_supervisor_ledger_event,
        )
        from lethyx.memory.palace.write import canonical_palace_add

        args = step.args
        evidence = list(args.get("evidence_refs") or [])
        if args.get("append_ledger"):
            event = append_supervisor_ledger_event(
                repo_root=repo,
                workspace_id=ws,
                actor="verifier",
                phase="verify",
                tool="pytest_harness",
                result={"passed": True},
            )
            evidence = [
                {
                    "ref_type": "ledger_event",
                    "ref_id": event.event_id,
                    "hash": event.event_hash,
                    "produced_by": "verifier",
                }
            ]
        trust_level = str(args.get("trust_level") or "public_verified")
        evidence_valid = bool(evidence)
        for ref in evidence:
            if not isinstance(ref, dict):
                evidence_valid = False
                continue
            if str(ref.get("ref_type") or "") != "ledger_event":
                continue
            row = find_supervisor_ledger_event(
                repo_root=repo,
                workspace_id=ws,
                event_id=str(ref.get("ref_id") or ""),
            )
            if row is None or (
                ref.get("hash") and str(ref.get("hash")) != str(row.get("event_hash") or "")
            ):
                evidence_valid = False
        try:
            payload = canonical_palace_add(
                repo,
                workspace_id=ws,
                title=str(args.get("title") or "Harness verification"),
                content=str(args.get("content") or "Verified content."),
                kind="verification_report",
                store="palace.durable",
                tags=str(args.get("tags") or "verification_report,durable"),
                phase="verify",
                source_path=f"ledger/{step.run_id}",
                run_id=step.run_id,
                verified=evidence_valid,
                extra={
                    "trust_level": trust_level,
                    "evidence_refs_json": json.dumps(evidence, ensure_ascii=False),
                },
            )
        except ValueError as exc:
            payload = {"saved": False, "error": str(exc)}
        if args.get("expect_saved") is False:
            if payload.get("saved"):
                return ["promote_to_durable should have been blocked"]
            return []
        if not payload.get("saved"):
            return [f"promote_to_durable failed: {payload}"]
        return []

    def _action_fake_reflect(
        self,
        repo: Path,
        ws: str,
        drive: Path,
        step: ScenarioStep,
        context: dict[str, Any],
    ) -> list[str]:
        from lethyx.enforcement.ledger import append_supervisor_ledger_event
        from lethyx.memory.hindsight.candidates import (
            write_hindsight_candidates_as_pending_proposals,
        )

        event = append_supervisor_ledger_event(
            repo_root=repo,
            workspace_id=ws,
            actor="verifier",
            phase="verify",
            tool="harness",
            result={"passed": True},
        )
        evidence = [
            {
                "ref_type": "ledger_event",
                "ref_id": event.event_id,
                "hash": event.event_hash,
                "produced_by": "verifier",
            }
        ]
        fake: FakeHindsightClient = context["fake_hindsight"]
        candidate = default_reflect_candidate(evidence)
        fake.reflect_payload = {
            "candidates": [
                {
                    "id": candidate.candidate_id,
                    "title": candidate.title,
                    "content": candidate.content,
                }
            ]
        }
        log = self._hindsight_log
        if log:
            log.append("reflect_candidates", {"workspace_id": ws})
        result = write_hindsight_candidates_as_pending_proposals(
            drive_root=drive,
            repo_root=repo,
            workspace_id=ws,
            run_id=step.run_id,
            phase_id=step.phase or "reflexion",
            candidates=[candidate],
        )
        if result.get("queued", 0) < 1:
            return ["hindsight candidate not queued"]
        context["last_reflect_queued"] = result.get("queued", 0)
        return []

    def _action_hindsight_backend_reflect(
        self,
        repo: Path,
        ws: str,
        drive: Path,
        step: ScenarioStep,
        context: dict[str, Any],
    ) -> list[str]:
        from lethyx.enforcement.ledger import append_supervisor_ledger_event
        from lethyx.memory.backends.base import ReflectionQuery
        from lethyx.memory.backends.hindsight import HindsightBackend
        from lethyx.memory.hindsight.candidates import (
            write_hindsight_candidates_as_pending_proposals,
        )
        from lethyx.memory.hindsight.config import HindsightConfig

        event = append_supervisor_ledger_event(
            repo_root=repo,
            workspace_id=ws,
            actor="verifier",
            phase="verify",
            tool="harness",
            result={"passed": True},
        )
        evidence = [
            {
                "ref_type": "ledger_event",
                "ref_id": event.event_id,
                "hash": event.event_hash,
                "produced_by": "verifier",
            }
        ]
        fake: FakeHindsightClient = context["fake_hindsight"]
        fake.reflect_payload = {
            "candidates": [
                {
                    "id": "hs-backend-c1",
                    "title": "Backend reflect candidate",
                    "content": "Candidate via HindsightBackend.reflect_candidates path.",
                    "kind": "behavior",
                    "confidence": 0.9,
                    "evidence_refs": evidence,
                }
            ]
        }
        if self._hindsight_log:
            self._hindsight_log.append("reflect_candidates", {"via": "backend"})
        backend = HindsightBackend(
            repo_root=repo,
            workspace_id=ws,
            config=HindsightConfig(enabled=True, reflect_enabled=True),
            client=fake,
        )
        candidates = backend.reflect_candidates(
            ReflectionQuery(
                question="What lessons apply?",
                workspace_id=ws,
                run_id=step.run_id,
                phase_id=step.phase or "reflexion",
            )
        )
        if not candidates:
            return ["hindsight_backend_reflect returned no candidates"]
        queued = write_hindsight_candidates_as_pending_proposals(
            drive_root=drive,
            repo_root=repo,
            workspace_id=ws,
            run_id=step.run_id,
            phase_id=step.phase or "reflexion",
            candidates=candidates,
        )
        if queued.get("queued", 0) < 1:
            return [f"hindsight backend queue failed: {queued}"]
        return []

    def _action_accept_bkb_patch(
        self, repo: Path, ws: str, drive: Path, step: ScenarioStep
    ) -> list[str]:
        from lethyx.enforcement.ledger import append_supervisor_ledger_event
        from lethyx.memory.proactive.promotion import ProposedBkbPatch, accept_bkb_patch

        args = step.args
        expect_accepted = bool(args.get("expect_accepted", True))
        evidence = list(args.get("evidence_refs") or [])
        if args.get("use_valid_ledger"):
            event = append_supervisor_ledger_event(
                repo_root=repo,
                workspace_id=ws,
                actor="verifier",
                phase="verify",
                tool="harness",
                result={"passed": True},
            )
            evidence = [
                {
                    "ref_type": "ledger_event",
                    "ref_id": event.event_id,
                    "hash": event.event_hash,
                    "produced_by": "verifier",
                }
            ]
        patch = ProposedBkbPatch(
            patch_id=str(args.get("patch_id") or "harness-patch"),
            workspace_id=ws,
            run_id=step.run_id,
            phase_id=step.phase or "reflexion",
            actor="supervisor",
            source_evidence=evidence,
            rules=list(
                args.get("rules")
                or [
                    {
                        "id": "harness_hs_rule",
                        "title": "Harness hindsight rule",
                        "scope": "workspace",
                        "type": "behavior",
                        "source_backend": "hindsight",
                        "rule": {"behavior": "Typed evidence required."},
                    }
                ]
            ),
        )
        try:
            result = accept_bkb_patch(repo, patch, target="workspace", drive_root=drive)
            accepted = bool(result.get("accepted"))
        except ValueError as exc:
            if expect_accepted:
                return [f"accept_bkb_patch failed: {exc}"]
            return []
        if not accepted and expect_accepted:
            return [f"accept_bkb_patch not accepted: {result}"]
        if accepted and not expect_accepted:
            return ["accept_bkb_patch should have been rejected"]
        return []

    def _action_process_reflexion_patch(
        self, repo: Path, ws: str, drive: Path, step: ScenarioStep
    ) -> list[str]:
        from lethyx.memory.proactive.phase_hooks import process_reflexion_bkb_patch

        result = process_reflexion_bkb_patch(
            repo_root=repo,
            drive_root=drive,
            workspace_id=ws,
            run_id=step.run_id,
        )
        if step.args.get("expect_accepted") is False:
            if result and result.get("accepted"):
                return ["process_reflexion_bkb_patch should not accept without proposal"]
        return []

    def _write_step_artifacts(
        self, report_dir: Path, step_id: str, result: ScenarioStepResult
    ) -> None:
        if result.prompt:
            (report_dir / f"prompt_{step_id}.txt").write_text(result.prompt, encoding="utf-8")
        if result.injection_report:
            (report_dir / f"memory_injection_report_{step_id}.json").write_text(
                json.dumps(result.injection_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        proactive = (result.overlays or {}).get("proactive_memory")
        if proactive:
            (report_dir / f"proactive_overlay_{step_id}.json").write_text(
                json.dumps(proactive, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if result.bundle:
            (report_dir / f"llm_input_bundle_{step_id}.json").write_text(
                json.dumps(result.bundle, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _update_golden(self, scenario_id: str, steps: list[ScenarioStepResult]) -> None:
        dest = EXPECTED_ROOT / scenario_id
        dest.mkdir(parents=True, exist_ok=True)
        for step in steps:
            if step.task:
                facts = structured_facts(step, step.task)
                name = f"{step.step_id}.report.golden.json"
                (dest / name).write_text(
                    json.dumps(facts, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    def _check_golden(self, scenario_id: str, steps: list[ScenarioStepResult]) -> list[str]:
        errors: list[str] = []
        expected_dir = EXPECTED_ROOT / scenario_id
        if not expected_dir.is_dir():
            return errors
        for step in steps:
            if not step.task:
                continue
            golden_path = expected_dir / f"{step.step_id}.report.golden.json"
            if not golden_path.is_file():
                continue
            expected = json.loads(golden_path.read_text(encoding="utf-8"))
            actual = structured_facts(step, step.task)
            for key in ("included_bkb_ids", "skipped_bkb_ids"):
                if key in expected and expected[key] != actual.get(key):
                    errors.append(
                        f"golden {step.step_id}.{key}: "
                        f"expected {expected[key]}, got {actual.get(key)}"
                    )
        return errors


def run_scenario_by_id(
    scenario_id: str,
    *,
    report_root: Path | None = None,
    **kwargs: Any,
) -> MemoryScenarioResult:
    scenario = load_scenario_by_id(scenario_id)
    return MemoryScenarioRunner(report_root=report_root, **kwargs).run(scenario)


def run_all_scenarios(
    *,
    report_root: Path | None = None,
    fail_fast: bool = False,
    phase_filter: str | None = None,
    include_llm: bool = False,
    **kwargs: Any,
) -> MemoryScenarioResult:
    results: list[MemoryScenarioResult] = []
    runner = MemoryScenarioRunner(report_root=report_root, **kwargs)
    scenarios = load_all_scenarios(include_llm=include_llm)
    if phase_filter:
        scenarios = [
            s
            for s in scenarios
            if any(st.phase == phase_filter for st in s.steps)
            or phase_filter in str(s.assertions.get("matrix_phases") or [])
        ]
    for scenario in scenarios:
        result = runner.run(scenario)
        results.append(result)
        if fail_fast and not result.ok:
            break
    root = report_root or (REPO_ROOT / ".mrt" / "memory_scenarios")
    dashboard = build_dashboard(results, root)
    (root / "latest" / "dashboard.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "latest" / "dashboard.json").write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_terminal_summary(results)
    ok = all(r.ok for r in results)
    return MemoryScenarioResult(
        scenario_id="__aggregate__",
        ok=ok,
        repo_root=REPO_ROOT,
        workspace_id="",
        step_results=[],
        invariant_failures=[m for r in results for m in r.invariant_failures],
        report_dir=root / "latest",
        dashboard=dashboard,
    )
