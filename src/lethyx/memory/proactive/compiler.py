"""ProactiveMemoryCompiler — assemble always-loaded overlay before agent action."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lethyx.memory.paths import manager_core_root, workspace_core_root
from lethyx.memory.proactive.bkb import (
    ensure_platform_bkb_rules,
    format_bkb_section,
    load_bkb_rules,
    partition_bkb_rules,
    resolve_bkb_conflicts,
)
from lethyx.memory.proactive.budget import (
    estimate_tokens,
    resolve_proactive_budget,
    trim_sections_to_budget,
)
from lethyx.memory.proactive.core_files import (
    ensure_core_seed_files,
    load_manager_core,
    load_workspace_core,
)
from lethyx.memory.proactive.models import OverlaySection, ProactiveMemoryOverlay
from lethyx.memory.kernel.value import (
    load_memory_value_stats,
    rank_directives_by_value,
    record_directive_injection,
    update_memory_value_stats,
)
from lethyx.memory.proactive.triggers import build_trigger_context, trigger_matches
from lethyx.memory.proactive.overlays import (
    build_phase_state_section,
    phase_policy,
    select_sections_by_policy,
)


class ProactiveMemoryCompiler:
    def build_overlay(
        self,
        *,
        repo_root: Path,
        workspace_id: str,
        run_id: str,
        phase_id: str,
        subtask_id: str | None,
        task_brief: str,
        manifest: Any,
        token_budget: int | None = None,
        drive_root: Path | None = None,
    ) -> ProactiveMemoryOverlay:
        manifest_budget = int(getattr(getattr(manifest, "budgets", None), "max_tokens", 0) or 0)
        budget = token_budget or resolve_proactive_budget(
            phase=phase_id,
            manifest_budget=manifest_budget,
            env_override=os.environ.get("LETHYX_PROACTIVE_MEMORY_BUDGET"),
        )

        manager_root = manager_core_root(repo_root)
        ensure_core_seed_files(manager_root, kind="manager")
        ensure_platform_bkb_rules(manager_root)
        sections: list[OverlaySection] = []
        sections.extend(load_manager_core(manager_root, budget_tokens=min(1500, budget // 3)))

        if workspace_id.strip():
            ws_root = workspace_core_root(repo_root, workspace_id)
            ensure_core_seed_files(ws_root, kind="workspace")
            sections.extend(load_workspace_core(ws_root, budget_tokens=min(2000, budget // 2)))

        bkb_rules = load_bkb_rules(manager_root / "bkb.yaml")
        if workspace_id.strip():
            bkb_rules.extend(
                load_bkb_rules(workspace_core_root(repo_root, workspace_id) / "bkb.yaml")
            )

        active, skipped_bkb = partition_bkb_rules(
            bkb_rules,
            workspace_id=workspace_id,
            phase_id=phase_id,
        )
        manifest_tools = getattr(manifest, "tools", None)
        tools_available = tuple(
            str(tool)
            for tool in (
                getattr(manifest, "allowed_tools", None)
                or getattr(manifest_tools, "allowed", None)
                or getattr(manifest_tools, "available", None)
                or ()
            )
        )
        trigger_context = build_trigger_context(
            drive_root=drive_root,
            phase_id=phase_id,
            workspace_id=workspace_id,
            run_id=run_id,
            tools_available=tools_available,
        )
        trigger_audit: list[dict[str, Any]] = []
        trigger_filtered = []
        for rule in active:
            result = trigger_matches(rule, trigger_context)
            if result.matched:
                trigger_filtered.append(rule)
                if rule.rule_type == "prospective_trigger":
                    rule.metadata["trigger_match_reason"] = result.reason
                    rule.metadata["expected_first_action_any"] = list(result.expected_first_actions)
                    trigger_audit.append({"id": rule.id, "matched": True, "reason": result.reason})
            else:
                skipped_bkb.append(
                    {
                        "id": rule.id,
                        "title": rule.title,
                        "reason": f"trigger_{result.reason}",
                        "surface": "directive",
                        "source_backend": "bkb",
                    }
                )
                trigger_audit.append({"id": rule.id, "matched": False, "reason": result.reason})
        active = trigger_filtered
        resolved, conflicts = resolve_bkb_conflicts(active)
        value_stats = load_memory_value_stats(drive_root) if drive_root is not None else {}
        resolved = rank_directives_by_value(resolved, value_stats, budget)
        value_records = []
        for rule in resolved:
            token_cost = estimate_tokens(str((rule.rule or {}).get("behavior") or rule.title))
            record = record_directive_injection(
                drive_root,
                rule,
                {
                    "workspace_id": workspace_id,
                    "run_id": run_id,
                    "phase_id": phase_id,
                    "subtask_id": subtask_id or "",
                },
                token_cost=token_cost,
                reason=rule.metadata.get("trigger_match_reason", "eligible_directive"),
            )
            value_records.append(record)
            rule.metadata["injection_id"] = record.injection_id
            stat = value_stats.get(rule.id)
            rule.metadata["estimated_value"] = stat.estimated_value if stat else 0.0
            rule.metadata["token_cost"] = token_cost
        if drive_root is not None and value_records:
            update_memory_value_stats(drive_root, records=value_records)
            value_stats = load_memory_value_stats(drive_root)
        bkb_text, bkb_refs = format_bkb_section(resolved, max_chars=min(2000, budget // 2))
        if bkb_text:
            sections.append(
                OverlaySection(
                    name="BKB (verified behavior rules)",
                    content=bkb_text,
                    source_refs=bkb_refs,
                    trust="verified",
                    token_count=estimate_tokens(bkb_text),
                )
            )

        run_state = self._build_run_state(
            drive_root=drive_root,
            run_id=run_id,
            workspace_id=workspace_id,
            max_tokens=min(800, budget // 4),
        )
        if run_state:
            sections.append(
                OverlaySection(
                    name="Current run state",
                    content=run_state,
                    source_refs=[f"run:{run_id}"] if run_id else [],
                    trust="summarized",
                    token_count=estimate_tokens(run_state),
                )
            )

        phase_text = build_phase_state_section(
            phase_id=phase_id,
            manifest_description=str(getattr(manifest, "description", "") or ""),
            task_brief=task_brief,
            active_risks=self._extract_risks(sections),
            forbidden_repeats=self._extract_forbidden(resolved),
            open_threads=self._extract_open_threads(sections),
            max_tokens=min(600, budget // 5),
        )
        sections.append(
            OverlaySection(
                name="Phase commitments",
                content=phase_text,
                source_refs=[f"phase:{phase_id}"],
                trust="curated",
                token_count=estimate_tokens(phase_text),
            )
        )

        policy = phase_policy(phase_id)
        sections = select_sections_by_policy(sections, policy)
        sections = trim_sections_to_budget(sections, budget, phase_id=phase_id)

        archive_hints = self._archive_hints(
            phase_id=phase_id,
            workspace_id=workspace_id,
            verification_count=self._count_durable_verification_reports(repo_root, workspace_id),
        )

        total_tokens = sum(s.token_count for s in sections)
        overlay = ProactiveMemoryOverlay(
            sections=sections,
            conflicts=conflicts,
            archive_hints=archive_hints,
            telemetry={
                "memory_overlay_tokens": total_tokens,
                "core_memory_tokens": total_tokens,
                "bkb_rules_injected": len(resolved),
                "memory_overlay_conflicts_count": len(conflicts),
                "phase_policy": policy,
                "proactive_budget": budget,
                "injection_audit": {
                    "included_bkb_ids": [rule.id for rule in resolved if rule.id],
                    "included_bkb": [
                        {
                            "id": rule.id,
                            "surface": "directive",
                            "source_backend": "bkb",
                            "token_cost": rule.metadata.get("token_cost", 0),
                            "estimated_value": rule.metadata.get("estimated_value", 0.0),
                            "injection_id": rule.metadata.get("injection_id", ""),
                            "reason": rule.metadata.get(
                                "trigger_match_reason", "eligible_directive"
                            ),
                        }
                        for rule in resolved
                    ],
                    "skipped_bkb": skipped_bkb,
                    "trigger_audit": trigger_audit,
                },
            },
        )
        return overlay

    def build_minimal_overlay(
        self,
        *,
        repo_root: Path,
        workspace_id: str = "",
        phase_id: str = "",
    ) -> ProactiveMemoryOverlay:
        """Task-start overlay: identity + optional workspace + BKB only."""
        budget = resolve_proactive_budget(phase=phase_id, manifest_budget=0)
        return self.build_overlay(
            repo_root=repo_root,
            workspace_id=workspace_id,
            run_id="",
            phase_id=phase_id or "task_start",
            subtask_id=None,
            task_brief="",
            manifest=_MinimalManifest(),
            token_budget=min(budget, 3000),
        )

    @staticmethod
    def _build_run_state(
        *,
        drive_root: Path | None,
        run_id: str,
        workspace_id: str,
        max_tokens: int,
    ) -> str:
        if drive_root is None or not drive_root.is_dir():
            return ""

        def _read_json(rel: str) -> Any:
            path = drive_root / rel
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None

        def _brief(value: Any, *, limit: int = 300) -> str:
            if isinstance(value, str):
                text = value
            else:
                try:
                    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
                except TypeError:
                    text = str(value)
            text = " ".join(text.split())
            if len(text) > limit:
                return text[: limit - 18].rstrip() + " ...[trimmed]"
            return text

        def _summarize_phase_plan(raw: Any) -> dict[str, Any]:
            if not isinstance(raw, dict):
                return {}
            raw_nodes = raw.get("nodes")
            nodes = raw_nodes if isinstance(raw_nodes, list) else []
            counts: dict[str, int] = {}
            rows: list[dict[str, str]] = []
            execute_summary: dict[str, Any] = {}
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                status = str(node.get("status") or "")
                counts[status or "unknown"] = counts.get(status or "unknown", 0) + 1
                rows.append(
                    {
                        "id": str(node.get("id") or ""),
                        "manifest_id": str(node.get("manifest_id") or ""),
                        "status": status,
                    }
                )
                if str(node.get("manifest_id") or node.get("id") or "") == "execute":
                    subtasks = [
                        item for item in (node.get("subtasks") or []) if isinstance(item, dict)
                    ]
                    sub_counts: dict[str, int] = {}
                    active = None
                    for subtask in subtasks:
                        sub_status = str(subtask.get("status") or "pending")
                        sub_counts[sub_status] = sub_counts.get(sub_status, 0) + 1
                        if active is None and sub_status != "done":
                            active = {
                                "id": str(subtask.get("id") or ""),
                                "status": sub_status,
                                "title": _brief(subtask.get("title") or "", limit=120),
                            }
                    execute_summary = {
                        "status": status,
                        "subtask_counts": sub_counts,
                        "active_subtask": active,
                        "all_subtasks_done": bool(subtasks)
                        and all(str(subtask.get("status") or "") == "done" for subtask in subtasks),
                    }
            return {
                "run_id": str(raw.get("run_id") or ""),
                "plan_id": str(raw.get("plan_id") or ""),
                "node_counts": counts,
                "nodes": rows[:16],
                "execute": execute_summary,
            }

        def _summarize_work_item(raw: Any) -> dict[str, Any]:
            if not isinstance(raw, dict):
                return {}
            changes = [
                _brief(item, limit=220)
                for item in (raw.get("required_changes") or [])
                if isinstance(item, dict)
            ][:3]
            proof = raw.get("proof_contract") if isinstance(raw.get("proof_contract"), dict) else {}
            execution = proof.get("execution") if isinstance(proof, dict) else {}
            command = execution.get("command") if isinstance(execution, dict) else None
            blockers = []
            for item in raw.get("required_changes") or []:
                if isinstance(item, dict) and item.get("blocker_fingerprint"):
                    blockers.append(str(item.get("blocker_fingerprint")))
            return {
                "id": str(raw.get("id") or ""),
                "kind": str(raw.get("kind") or ""),
                "active_subtask_id": str(raw.get("active_subtask_id") or ""),
                "allowed_files": list(raw.get("allowed_files") or [])[:12],
                "forbidden_files": list(raw.get("forbidden_files") or [])[:12],
                "blocker_fingerprints": blockers[:3],
                "required_changes": changes,
                "proof": {
                    "harness_profile": str(proof.get("harness_profile") or "")
                    if isinstance(proof, dict)
                    else "",
                    "execution_kind": str(execution.get("kind") or "")
                    if isinstance(execution, dict)
                    else "",
                    "command": _brief(command, limit=220) if command else "",
                },
            }

        def _active_work_item_is_current(
            raw: Any,
            phase_plan: dict[str, Any],
        ) -> tuple[bool, str]:
            if not isinstance(raw, dict):
                return False, "missing"
            execute = phase_plan.get("execute") if isinstance(phase_plan, dict) else {}
            if not isinstance(execute, dict) or not execute:
                return True, ""
            if execute.get("all_subtasks_done"):
                return False, "execute_all_subtasks_done"
            if str(execute.get("status") or "").strip() in {
                "done",
                "skipped",
                "failed",
            }:
                return False, "execute_phase_terminal"
            active_subtask = execute.get("active_subtask")
            if not isinstance(active_subtask, dict):
                counts = execute.get("subtask_counts")
                if isinstance(counts, dict) and counts:
                    return False, "no_active_execute_subtask"
                return True, ""
            work_item_subtask_id = str(raw.get("active_subtask_id") or "").strip()
            plan_subtask_id = str(active_subtask.get("id") or "").strip()
            if work_item_subtask_id and plan_subtask_id != work_item_subtask_id:
                return False, "active_subtask_mismatch"
            if str(active_subtask.get("status") or "").strip() == "done":
                return False, "active_subtask_done"
            return True, ""

        def _summarize_decision(raw: Any) -> dict[str, Any]:
            if not isinstance(raw, dict):
                return {}
            decision_id = str(raw.get("source_tool_call_id") or "")
            blocker = ""
            try:
                from lethyx.contracts.work_items import (
                    loopback_blocker_fingerprint,
                    phase_exit_decision_id,
                )

                decision_id = phase_exit_decision_id(raw)
                blocker = loopback_blocker_fingerprint(raw)
            except Exception:
                pass
            completed = _read_json("state/completed_work_items.json")
            completed_items = completed.get("work_items") if isinstance(completed, dict) else []
            completed_decisions = {
                str(item.get("created_from_decision_id") or "")
                for item in completed_items or []
                if isinstance(item, dict)
            }
            completed_blockers = set()
            for item in completed_items or []:
                if not isinstance(item, dict):
                    continue
                for change in item.get("required_changes") or []:
                    if isinstance(change, dict) and change.get("blocker_fingerprint"):
                        completed_blockers.add(str(change.get("blocker_fingerprint")))
            issues = [
                _brief(item, limit=220)
                for item in (raw.get("issues") or [])
                if isinstance(item, dict)
            ][:3]
            changes = [
                _brief(item, limit=220)
                for item in (raw.get("required_changes") or [])
                if isinstance(item, dict)
            ][:3]
            return {
                "decision_id": decision_id,
                "blocker_fingerprint": blocker,
                "outcome": str(raw.get("outcome") or ""),
                "phase_id": str(raw.get("phase_id") or ""),
                "target_phase": str(raw.get("target_phase") or ""),
                "completed_or_superseded": bool(
                    (decision_id and decision_id in completed_decisions)
                    or (blocker and blocker in completed_blockers)
                ),
                "issues": issues,
                "required_changes": changes,
            }

        phase_plan_summary = _summarize_phase_plan(_read_json("state/phase_plan.json"))
        active_work_item_raw = _read_json("state/active_work_item.json")
        active_current, active_suppressed_reason = _active_work_item_is_current(
            active_work_item_raw,
            phase_plan_summary,
        )
        active_work_item_summary = (
            _summarize_work_item(active_work_item_raw) if active_current else {}
        )

        control: dict[str, Any] = {
            "run_id": run_id,
            "workspace_id": workspace_id,
            "phase_plan": phase_plan_summary,
            "active_work_item": active_work_item_summary,
            "latest_loopback_decision": _summarize_decision(
                _read_json("state/phase_exit_decision_latest.json")
            ),
            "soft_exit": _read_json("state/soft_phase_exit_latest.json") or {},
            "phase_impasse": _read_json("state/phase_impasse.json") or {},
            "artifact_refs": {
                "current_control_verdict": ".memory/drive/state/current_control_verdict.json",
                "phase_plan": ".memory/drive/state/phase_plan.json",
                "active_work_item": ".memory/drive/state/active_work_item.json",
                "work_items": ".memory/drive/state/work_items.json",
                "completed_work_items": ".memory/drive/state/completed_work_items.json",
                "latest_loopback_decision": ".memory/drive/state/phase_exit_decision_latest.json",
            },
        }
        if active_suppressed_reason and isinstance(active_work_item_raw, dict):
            control["active_work_item_suppressed"] = {
                "id": str(active_work_item_raw.get("id") or ""),
                "active_subtask_id": str(active_work_item_raw.get("active_subtask_id") or ""),
                "reason": active_suppressed_reason,
            }
        task_path = drive_root / "TASK_MAIN.md"
        if task_path.is_file():
            try:
                control["task_brief"] = _brief(
                    task_path.read_text(encoding="utf-8", errors="replace"),
                    limit=500,
                )
            except OSError:
                pass
        try:
            state_dir = drive_root / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "current_control_verdict.json").write_text(
                json.dumps(control, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
        text = (
            "Active Lethyx control state (compact; raw artifacts are refs, not "
            "prompt truth):\n```json\n"
            + json.dumps(control, ensure_ascii=False, indent=2)
            + "\n```"
        )
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[: max_chars - 20].rstrip() + "\n...[run state truncated]"
        return text

    @staticmethod
    def _count_durable_verification_reports(repo_root: Path, workspace_id: str) -> int:
        try:
            from lethyx.memory.palace.facade import MemPalace

            palace = MemPalace(repo_root, workspace_id or None)
            try:
                hits = palace.search(
                    "verification",
                    stores=["palace.durable"],
                    tags_any=["verification_report"],
                    n=50,
                )
                return len(hits)
            finally:
                palace.close()
        except Exception:
            return 0

    @staticmethod
    def _archive_hints(
        *,
        phase_id: str,
        workspace_id: str,
        verification_count: int,
    ) -> list[str]:
        hints: list[str] = []
        if verification_count > 2:
            hints.append(
                f"{verification_count} verification reports in palace.durable; "
                "only promoted BKB/core lessons appear above."
            )
        if phase_id == "research":
            hints.append("Use palace_search for prior research findings (archive, not directive).")
        if workspace_id:
            hints.append(f"Workspace archive: workspaces/{workspace_id}/.memory/palace/")
        return hints[:5]

    @staticmethod
    def _extract_risks(sections: list[OverlaySection]) -> list[str]:
        for section in sections:
            if "risk" in section.name.lower():
                return [
                    line.strip("- ").strip()
                    for line in section.content.splitlines()
                    if line.strip().startswith("-")
                ][:8]
        return []

    @staticmethod
    def _extract_forbidden(rules: list[Any]) -> list[str]:
        out: list[str] = []
        for rule in rules:
            if getattr(rule, "rule_type", "") == "anti_pattern":
                forbidden = (getattr(rule, "rule", None) or {}).get("forbidden")
                if forbidden:
                    out.append(str(forbidden))
        return out[:8]

    @staticmethod
    def _extract_open_threads(sections: list[OverlaySection]) -> list[str]:
        for section in sections:
            if "open thread" in section.name.lower():
                return [
                    line.strip("- ").strip()
                    for line in section.content.splitlines()
                    if line.strip().startswith("-")
                ][:6]
        return []


class _MinimalManifest:
    description = ""
    budgets = SimpleNamespace(max_tokens=8000)
