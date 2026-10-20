"""Evidence resolution against supervisor-owned ledgers and workspace hashes."""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

from lethyx.contracts.hashing import diff_hash, hash_value, workspace_hash
from lethyx.contracts.models import (
    Actor,
    ContractIssue,
    EvidenceRef,
    EvidenceRefType,
    VerificationReportRef,
    WorkspaceContext,
)
from lethyx.contracts.ledger import (
    read_supervisor_ledger_events,
)

ALLOWED_EVIDENCE_REF_TYPES = frozenset(str(item) for item in get_args(EvidenceRefType))
ALLOWED_EVIDENCE_ACTORS = frozenset(str(item) for item in get_args(Actor))
LEDGER_BACKED_EVIDENCE_REF_TYPES = frozenset(
    {
        "ledger_event",
        "verification_report",
        "test_run",
    }
)


@dataclass(frozen=True)
class EvidenceResolver:
    context: WorkspaceContext
    _rows: tuple[dict, ...] = field(init=False, repr=False, compare=False)
    _by_id: dict[str, dict] = field(init=False, repr=False, compare=False)
    _positions: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rows = tuple(
            read_supervisor_ledger_events(
                repo_root=self.repo_root, workspace_id=self.context.workspace_id
            )
        )
        by_id: dict[str, dict] = {}
        positions: dict[str, int] = {}
        for position, row in enumerate(rows):
            for key in (str(row.get("event_id") or ""), str(row.get("event_hash") or "")):
                if key:
                    if key in by_id:
                        raise ValueError("supervisor ledger contains duplicate event identities")
                    by_id[key] = row
                    positions[key] = position
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(self, "_positions", positions)

    @property
    def repo_root(self) -> Path:
        return Path(self.context.repo_root).resolve()

    @property
    def workspace_root(self) -> Path:
        return Path(self.context.workspace_root).resolve()

    def ledger_rows(self) -> list[dict]:
        return deepcopy(list(self._rows))

    def ledger_row(self, ref_id: str) -> dict | None:
        return deepcopy(self._by_id.get(ref_id))

    def _ledger_index(self, event_id: str) -> int:
        return self._positions.get(event_id, -1)

    def validate_ref(
        self, ref: EvidenceRef, *, stage_id: str = "", subtask_id: str = ""
    ) -> list[ContractIssue]:
        issues: list[ContractIssue] = []
        if ref.ref_type not in ALLOWED_EVIDENCE_REF_TYPES:
            issues.append(
                ContractIssue(
                    code="invalid_evidence_ref",
                    severity="blocking",
                    stage_id=stage_id,
                    subtask_id=subtask_id,
                    message=(
                        "EvidenceRef.ref_type must be one of the typed evidence "
                        f"ref types, got `{ref.ref_type or '<missing>'}`."
                    ),
                    evidence_refs=(ref,),
                )
            )
        if ref.produced_by not in ALLOWED_EVIDENCE_ACTORS:
            issues.append(
                ContractIssue(
                    code="invalid_evidence_ref",
                    severity="blocking",
                    stage_id=stage_id,
                    subtask_id=subtask_id,
                    message=(
                        "EvidenceRef.produced_by must be a valid contract actor, "
                        f"got `{ref.produced_by or '<missing>'}`."
                    ),
                    evidence_refs=(ref,),
                )
            )
        if not ref.ref_id:
            issues.append(
                ContractIssue(
                    code="fake_evidence_ref",
                    severity="blocking",
                    stage_id=stage_id,
                    subtask_id=subtask_id,
                    message="EvidenceRef.ref_id is required.",
                    evidence_refs=(ref,),
                )
            )
        if issues:
            return issues
        if ref.ref_type in LEDGER_BACKED_EVIDENCE_REF_TYPES:
            row = self.ledger_row(ref.ref_id)
            if row is None:
                issues.append(
                    ContractIssue(
                        code="fake_evidence_ref",
                        severity="blocking",
                        stage_id=stage_id,
                        subtask_id=subtask_id,
                        message=f"Evidence ref `{ref.ref_id}` does not exist in supervisor ledger.",
                        evidence_refs=(ref,),
                    )
                )
                return issues
            if ref.hash and ref.hash != str(row.get("event_hash") or ""):
                issues.append(
                    ContractIssue(
                        code="evidence_hash_mismatch",
                        severity="blocking",
                        stage_id=stage_id,
                        subtask_id=subtask_id,
                        message=f"Evidence ref `{ref.ref_id}` hash does not match ledger event hash.",
                        evidence_refs=(ref,),
                    )
                )
            if ref.produced_by != "agent" and ref.produced_by != str(row.get("actor") or ""):
                issues.append(
                    ContractIssue(
                        code="wrong_evidence_producer",
                        severity="blocking",
                        stage_id=stage_id,
                        subtask_id=subtask_id,
                        message=(
                            f"Evidence ref `{ref.ref_id}` was produced by "
                            f"`{row.get('actor')}`, not `{ref.produced_by}`."
                        ),
                        evidence_refs=(ref,),
                    )
                )
            if ref.created_after_event:
                ref_idx = self._ledger_index(ref.ref_id)
                after_idx = self._ledger_index(ref.created_after_event)
                if after_idx < 0:
                    issues.append(
                        ContractIssue(
                            code="fake_evidence_ref",
                            severity="blocking",
                            stage_id=stage_id,
                            subtask_id=subtask_id,
                            message=(
                                f"created_after_event `{ref.created_after_event}` "
                                "does not exist in supervisor ledger."
                            ),
                            evidence_refs=(ref,),
                        )
                    )
                elif ref_idx <= after_idx:
                    issues.append(
                        ContractIssue(
                            code="stale_proof_ref",
                            severity="blocking",
                            stage_id=stage_id,
                            subtask_id=subtask_id,
                            message=(
                                f"Evidence ref `{ref.ref_id}` is not newer than "
                                f"required event `{ref.created_after_event}`."
                            ),
                            evidence_refs=(ref,),
                        )
                    )
        return issues

    def validate_refs(
        self, refs: tuple[EvidenceRef, ...], *, stage_id: str = "", subtask_id: str = ""
    ) -> list[ContractIssue]:
        issues: list[ContractIssue] = []
        for ref in refs:
            issues.extend(self.validate_ref(ref, stage_id=stage_id, subtask_id=subtask_id))
        return issues

    def validate_verification_report_ref(
        self,
        ref: VerificationReportRef,
        *,
        stage_id: str = "",
        subtask_id: str = "",
        allow_workspace_hash_mismatch_if_diff_matches: bool = False,
    ) -> list[ContractIssue]:
        evidence_ref = ref.evidence_ref(stage_id=stage_id, subtask_id=subtask_id)
        issues = self.validate_ref(evidence_ref, stage_id=stage_id, subtask_id=subtask_id)
        row = self.ledger_row(ref.report_id)
        if row is None:
            return issues
        result = row.get("result")
        if not isinstance(result, dict) or result.get("verifier_id") != ref.verifier_id:
            issues.append(
                ContractIssue(
                    code="verification_report_verifier_mismatch",
                    severity="blocking",
                    stage_id=stage_id,
                    subtask_id=subtask_id,
                    message="VerificationReportRef verifier identity does not match the ledger.",
                    evidence_refs=(evidence_ref,),
                )
            )
        expected_result_hash = hash_value(
            {
                "report_hash": ref.report_hash,
                "passed": ref.passed,
                "workspace_hash": ref.workspace_hash,
                "diff_hash": ref.diff_hash,
            }
        )
        if expected_result_hash != str(row.get("result_hash") or ""):
            issues.append(
                ContractIssue(
                    code="verification_report_hash_mismatch",
                    severity="blocking",
                    stage_id=stage_id,
                    subtask_id=subtask_id,
                    message="VerificationReportRef does not match the ledger result hash.",
                    evidence_refs=(evidence_ref,),
                )
            )
        current_workspace_hash = self.context.current_workspace_hash
        if not current_workspace_hash and not allow_workspace_hash_mismatch_if_diff_matches:
            current_workspace_hash = workspace_hash(self.workspace_root)
        current_diff_hash = self.context.current_diff_hash
        scoped_diff_matches = (
            allow_workspace_hash_mismatch_if_diff_matches
            and bool(current_diff_hash)
            and bool(ref.diff_hash)
            and ref.diff_hash == current_diff_hash
        )
        if (
            ref.workspace_hash
            and ref.workspace_hash != current_workspace_hash
            and not scoped_diff_matches
        ):
            issues.append(
                ContractIssue(
                    code="proof_stale_rerun_required",
                    severity="blocking",
                    stage_id=stage_id,
                    subtask_id=subtask_id,
                    message=(
                        "Verification report was produced for a different "
                        "workspace hash; rerun proof instead of rolling back a stage."
                    ),
                    evidence_refs=(evidence_ref,),
                )
            )
        workspace_matches = (
            bool(ref.workspace_hash) and ref.workspace_hash == current_workspace_hash
        )
        if (
            not workspace_matches
            and current_diff_hash
            and ref.diff_hash
            and ref.diff_hash != current_diff_hash
            and ref.diff_hash != current_workspace_hash
        ):
            issues.append(
                ContractIssue(
                    code="proof_stale_rerun_required",
                    severity="blocking",
                    stage_id=stage_id,
                    subtask_id=subtask_id,
                    message=(
                        "Verification report was produced for a different diff "
                        "hash; rerun proof instead of rolling back a stage."
                    ),
                    evidence_refs=(evidence_ref,),
                )
            )
        if not ref.passed:
            issues.append(
                ContractIssue(
                    code="verification_report_not_passed",
                    severity="blocking",
                    stage_id=stage_id,
                    subtask_id=subtask_id,
                    message="VerificationReportRef.passed must be true for pass/complete gates.",
                    evidence_refs=(evidence_ref,),
                )
            )
        return issues


def build_workspace_context(
    *,
    repo_root: str | Path,
    workspace_root: str | Path,
    workspace_id: str,
    changed_files: tuple[str, ...] = (),
    last_patch_event_id: str = "",
    compute_workspace_hash: bool = True,
) -> WorkspaceContext:
    root = Path(workspace_root).resolve()
    return WorkspaceContext(
        repo_root=str(Path(repo_root).resolve()),
        workspace_root=str(root),
        workspace_id=workspace_id,
        current_workspace_hash=workspace_hash(root) if compute_workspace_hash else "",
        current_diff_hash=diff_hash(root, changed_files),
        last_patch_event_id=last_patch_event_id,
    )
