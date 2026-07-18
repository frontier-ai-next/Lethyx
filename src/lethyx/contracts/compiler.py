"""Compile persisted run artifacts into a ContractBundle."""

import json
from pathlib import Path
from typing import Any

from lethyx.contracts.capability_declaration import (
    declaration_effective_capabilities,
    load_capability_declaration,
    proof_required_capabilities,
)
from lethyx.contracts.models import (
    CompletionContract,
    ContractBundle,
    ContractIssue,
    EvidenceRef,
    PlanIR,
    ResearchSummaryContract,
    ReviewContract,
    TaskRiskProfile,
    VerificationReportRef,
)
from lethyx.contracts.plan_ir import compile_phase_plan
from lethyx.contracts.subtask_recovery import (
    review_superseded_by_recovery,
    subtask_passing_recovery_at,
)
from lethyx.contracts.validators import _path_looks_like_test

_LLM_RISK_CAPABILITIES = frozenset(
    {
        "llm_api",
        "llm_runtime",
        "openai",
        "openrouter",
        "anthropic",
        "gmas",
        "multi_agent_gmas",
    }
)


class ContractCompiler:
    """Build contract bundles from the current Lethyx drive layout."""

    def __init__(self, *, repo_root: str | Path, drive_root: str | Path, workspace_id: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.drive_root = Path(drive_root).resolve()
        self.workspace_id = workspace_id

    @classmethod
    def from_run(
        cls,
        *,
        repo_root: str | Path,
        drive_root: str | Path,
        workspace_id: str,
        run_id: str = "",
    ) -> ContractBundle:
        return cls(
            repo_root=repo_root,
            drive_root=drive_root,
            workspace_id=workspace_id,
        ).compile(run_id=run_id)

    def _read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return None

    def _tool_signal_rows(self, *, run_id: str = "") -> list[dict[str, Any]]:
        path = self.drive_root / "state" / "phase_control_signals.jsonl"
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if run_id and str(row.get("run_id") or "") not in {"", run_id}:
                    continue
                rows.append(row)
        except OSError:
            return rows
        return rows

    @staticmethod
    def _signal_created_at(row: dict[str, Any]) -> float:
        try:
            return float(row.get("created_at") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _latest_signal_time(
        cls,
        rows: list[dict[str, Any]],
        *,
        kind: str,
        phase: str = "",
    ) -> float:
        latest = 0.0
        for row in rows:
            if str(row.get("kind") or "") != kind:
                continue
            if phase and str(row.get("phase") or "") != phase:
                continue
            latest = max(latest, cls._signal_created_at(row))
        return latest

    @staticmethod
    def _research_summary_from_payload(
        payload: dict[str, Any],
    ) -> ResearchSummaryContract | None:
        architecture_id = str(payload.get("architecture_id") or "").strip()
        if not architecture_id:
            return None
        raw_status = str(payload.get("coverage_status") or "complete").strip().lower()
        coverage_status = (
            raw_status if raw_status in {"complete", "source_scarce", "blocked"} else "complete"
        )
        findings_ids = tuple(
            str(item) for item in payload.get("findings_ids") or () if str(item).strip()
        )
        return ResearchSummaryContract(
            architecture_id=architecture_id,
            findings_ids=findings_ids,
            coverage_status=coverage_status,
            source_scarcity_reason=str(payload.get("source_scarcity_reason") or ""),
            evidence_refs=tuple(
                EvidenceRef.from_mapping(item)
                for item in payload.get("evidence_refs") or ()
                if isinstance(item, dict)
            ),
        )

    def _compile_plan(self, *, run_id: str) -> tuple[Any, list[ContractIssue]]:
        for name in (
            "phase_plan_submitted_latest.json",
            "phase_plan_proposal_latest.json",
        ):
            payload = self._read_json(self.drive_root / "state" / name)
            if not isinstance(payload, dict):
                continue
            payload_run_id = str(payload.get("run_id") or "").strip()
            if run_id and payload_run_id and payload_run_id != run_id:
                continue
            raw_plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
            return compile_phase_plan(
                raw_plan,
                run_id=str(payload.get("run_id") or run_id or ""),
                workspace_id=str(payload.get("workspace_id") or self.workspace_id),
            )
        return None, []

    def _build_risk_profile(self, plan: PlanIR | None) -> TaskRiskProfile:
        if plan is None:
            return TaskRiskProfile()
        proofs = [subtask.proof for subtask in plan.subtasks if subtask.proof is not None]
        kinds = {proof.execution.kind for proof in proofs}
        paths: list[str] = []
        for subtask in plan.subtasks:
            paths.extend(subtask.files_to_change)
            paths.extend(subtask.files_to_create)
            affected = getattr(subtask, "files_affected", ())
            if affected:
                paths.extend(affected)
        tests_changed = any(_path_looks_like_test(path) for path in paths)
        code_changed = bool(paths)
        declaration = load_capability_declaration(self.drive_root)
        caps = declaration_effective_capabilities(declaration)
        required_capability_sets = [proof_required_capabilities(proof) for proof in proofs]
        llm_caps = any(
            bool(required & _LLM_RISK_CAPABILITIES) for required in required_capability_sets
        )
        uses_network = any("network" in required for required in required_capability_sets)
        uses_browser_ui = any("browser_ui" in required for required in required_capability_sets)
        return TaskRiskProfile(
            code_changed=code_changed,
            tests_changed=tests_changed,
            external_api=bool(caps.get("external_api") or uses_network),
            llm_or_prompt_logic=bool(llm_caps),
            web_or_http_runtime=bool({"http_boot", "behavioral_http"} & kinds or uses_browser_ui),
            high_stub_risk=bool({"mutation_smoke", "input_sensitivity", "metamorphic"} & kinds),
        )

    def compile(self, *, run_id: str = "") -> ContractBundle:
        plan, issues = self._compile_plan(run_id=run_id)
        risk = self._build_risk_profile(plan if isinstance(plan, PlanIR) else None)
        latest_review: ReviewContract | None = None
        completions: list[CompletionContract] = []
        reports: list[VerificationReportRef] = []
        research_summary = None
        signal_rows = self._tool_signal_rows(run_id=run_id)
        latest_plan_submit_at = self._latest_signal_time(
            signal_rows,
            kind="submit_phase_plan",
            phase="plan",
        )
        latest_research_summary_at = self._latest_signal_time(
            signal_rows,
            kind="submit_research_summary",
            phase="research",
        )
        plan_data = self._read_json(self.drive_root / "state" / "phase_plan.json")
        recovery_at = subtask_passing_recovery_at(
            plan_data=plan_data if isinstance(plan_data, dict) else None,
            signal_rows=signal_rows,
        )
        for row in signal_rows:
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            kind = str(row.get("kind") or "")
            if kind == "submit_micro_review":
                row_phase = str(row.get("phase") or "")
                if (
                    row_phase == "plan_review"
                    and latest_plan_submit_at
                    and self._signal_created_at(row) < latest_plan_submit_at
                ):
                    continue
                if (
                    row_phase == "research_review"
                    and latest_research_summary_at
                    and self._signal_created_at(row) < latest_research_summary_at
                ):
                    continue
                candidate = ReviewContract.from_mapping(payload)
                if review_superseded_by_recovery(
                    candidate,
                    recovery_at=recovery_at,
                    review_created_at=self._signal_created_at(row),
                ):
                    continue
                latest_review = candidate
            elif kind == "mark_subtask_complete":
                contract = payload.get("completion_contract")
                if isinstance(contract, dict):
                    completions.append(CompletionContract.from_mapping(contract))
            elif kind == "submit_verification":
                ref = payload.get("verification_report_ref")
                if isinstance(ref, dict):
                    reports.append(VerificationReportRef.from_mapping(ref))
            elif kind == "submit_research_summary":
                candidate_summary = self._research_summary_from_payload(payload)
                if candidate_summary is not None:
                    research_summary = candidate_summary
        latest_completion_by_subtask = {
            completion.subtask_id: completion for completion in completions if completion.subtask_id
        }
        completions = [
            completion
            for completion in completions
            if not completion.subtask_id
            or latest_completion_by_subtask.get(completion.subtask_id) is completion
        ]
        return ContractBundle(
            run_id=run_id,
            workspace_id=self.workspace_id,
            plan=plan,
            reviews=((latest_review,) if latest_review is not None else ()),
            research_summary=research_summary,
            completions=tuple(completions),
            verification_reports=tuple(reports),
            issues=tuple(issues),
            risk=risk,
        )
