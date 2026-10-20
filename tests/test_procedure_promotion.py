from dataclasses import replace

import pytest

from lethyx import MemoryClient
from lethyx.contracts import (
    EvidenceRef,
    EvidenceResolver,
    MemoryDecisionEvent,
    MemoryUsageEvent,
    OutcomeCreditEvent,
    build_workspace_context,
)
from lethyx.contracts.ledger import (
    append_candidate_semantic_verification_ledger_event,
    append_memory_application_verification_ledger_event,
    append_verification_report_ledger_event,
)
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.refinement import verified_procedure_promotion_gate


def _observation(name: str) -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="session_event",
            ref_id=name,
            produced_by="agent",
            producer_id="agent:test",
        ),
    )


def _application(
    client: MemoryClient,
    memory_id: str,
    suffix: str,
    *,
    verified: bool,
    outcome: str = "helped",
) -> None:
    decision_id = f"decision-{suffix}"
    trace_id = f"trace-{suffix}"
    usage_id = f"usage-{suffix}"
    client.record_memory_decision(
        MemoryDecisionEvent(
            decision_id=decision_id,
            workspace_id="ws",
            run_id=f"run-{suffix}",
            task_signature="repair:python",
            action_trace_id=trace_id,
            action="recall",
            canonical_snapshot_sha256=f"sha256:{'a' * 64}",
            context_snapshot_sha256=f"sha256:{'b' * 64}",
            live_state_snapshot_sha256=f"sha256:{'c' * 64}",
            live_trigger="repair",
            candidate_ids=(memory_id,),
            selected_candidate_ids=(memory_id,),
            policy_revision="policy-1",
            catalog_revision="catalog-1",
            cost_tokens=1,
            cost_latency_ms=1,
            downstream_outcome="verified",
            evidence_refs=_observation(f"decision-observation-{suffix}"),
        )
    )
    client.record_memory_usage_event(
        MemoryUsageEvent(
            event_id=usage_id,
            decision_id=decision_id,
            memory_id=memory_id,
            workspace_id="ws",
            task_signature="repair:python",
            action_trace_id=trace_id,
            usage_type="applied",
            sequence=0,
            evidence_refs=_observation(f"usage-observation-{suffix}"),
        )
    )
    if verified:
        evidence = (
            append_memory_application_verification_ledger_event(
                repo_root=client.repo_root,
                workspace_id="ws",
                verifier_id=f"verifier:{suffix}",
                decision_id=decision_id,
                action_trace_id=trace_id,
                memory_id=memory_id,
                usage_event_id=usage_id,
                task_signature="repair:python",
                outcome=outcome,
                passed=True,
            ),
        )
    else:
        report = append_verification_report_ledger_event(
            repo_root=client.repo_root,
            workspace_id="ws",
            verifier_id=f"verifier:{suffix}",
            report_hash=f"successful-application-{suffix}",
            passed=True,
        )
        evidence = (report.evidence_ref(),)
    client.record_outcome_credit(
        OutcomeCreditEvent(
            credit_id=f"credit-{suffix}",
            decision_id=decision_id,
            workspace_id="ws",
            task_signature="repair:python",
            action_trace_id=trace_id,
            memory_ids=(memory_id,),
            applied_usage_event_ids=(usage_id,),
            outcome=outcome,
            attribution_basis=(
                "deterministic_verifier" if outcome == "helped" else "proven_regression"
            ),
            verifier_result="passed" if outcome == "helped" else "failed",
            approval_principal_id="",
            task_family="repair",
            stage="verification",
            project_id="project",
            host_id="host",
            profile_id="default",
            independence_domain=f"run-{suffix}",
            evidence_refs=evidence,
            reason="verified application",
        )
    )


def test_procedure_requires_two_host_verified_applications(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    try:
        client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
        source_root = tmp_path / "source"
        source_root.mkdir()
        episode = client.remember(
            "The parser failed on empty input.", requested_block_id="stable_project_facts"
        )
        candidate = client.remember(
            "Check empty input before parsing.",
            kind="procedure",
            metadata={
                "procedure_applicability": "repair:python parser input",
                "source_task_signature": "repair:python",
                "source_episode_ids": [episode.memory_id],
            },
            requested_block_id="stable_project_facts",
        )
        record = index.get_block(candidate.memory_id)
        assert record is not None
        subject = client.verification_subject(
            candidate.memory_id, expected_source_revision_id=record.revision_id
        )
        semantic = append_candidate_semantic_verification_ledger_event(
            repo_root=tmp_path,
            subject=subject,
            workspace_root=source_root,
            verifier_id="verifier:semantic",
            verifier_model_version="reviewer",
            proposer_model_version="author",
            coverage=True,
            preservation=True,
            faithfulness=True,
            instruction_safety=True,
            reasons=("Compared the exact episode and procedure.",),
        )
        arguments = {
            "reason": "publish verified procedure",
            "evidence_refs": [semantic],
            "expected_source_revision_id": record.revision_id,
            "workspace_root": source_root,
        }
        try:
            client.promote(candidate.memory_id, **arguments)
        except ValueError as exc:
            assert "two independent verified applications" in str(exc)
        else:
            raise AssertionError("unapplied procedure was promoted")
        _application(client, candidate.memory_id, "one", verified=True)
        _application(client, candidate.memory_id, "forged", verified=False)
        assert len(index.outcome_credits_for_memory(candidate.memory_id)) == 2
        try:
            client.promote(candidate.memory_id, **arguments)
        except ValueError as exc:
            assert "two independent verified applications" in str(exc)
        else:
            raise AssertionError("unverified outcome was counted")
        _application(client, candidate.memory_id, "two", verified=True)
        result = client.promote(candidate.memory_id, **arguments)
        published = index.get_block(result["block_id"])
        assert published is not None
        assert published.metadata["procedure_promotion_gate"]["allowed"] is True
        assert published.metadata["procedure_promotion_gate"]["verified_outcome_ids"] == [
            "credit-one",
            "credit-two",
        ]
        resolver = EvidenceResolver(
            build_workspace_context(
                repo_root=tmp_path, workspace_root=source_root, workspace_id="ws"
            )
        )
        credits = index.outcome_credits_for_memory(candidate.memory_id)
        duplicate_domain = tuple(
            replace(credit, independence_domain="run-one")
            if credit.credit_id == "credit-two"
            else credit
            for credit in credits
        )
        assert not verified_procedure_promotion_gate(
            memory_id=candidate.memory_id,
            profile_id="default",
            task_signature="repair:python",
            outcome_credits=duplicate_domain,
            resolver=resolver,
        ).allowed
        wrong_application = tuple(
            replace(credit, decision_id="unrelated-decision")
            if credit.credit_id == "credit-two"
            else credit
            for credit in credits
        )
        assert not verified_procedure_promotion_gate(
            memory_id=candidate.memory_id,
            profile_id="default",
            task_signature="repair:python",
            outcome_credits=wrong_application,
            resolver=resolver,
        ).allowed
        forged_credit = next(credit for credit in credits if credit.credit_id == "credit-forged")
        assert verified_procedure_promotion_gate(
            memory_id=candidate.memory_id,
            profile_id="default",
            task_signature="repair:python",
            outcome_credits=(
                *credits,
                replace(
                    forged_credit,
                    credit_id="forged-harm",
                    outcome="hurt",
                    attribution_basis="proven_regression",
                    verifier_result="failed",
                ),
            ),
            resolver=resolver,
        ).allowed
        _application(client, candidate.memory_id, "harm", verified=True, outcome="hurt")
        with pytest.raises(ValueError, match="harmful application"):
            verified_procedure_promotion_gate(
                memory_id=candidate.memory_id,
                profile_id="default",
                task_signature="repair:python",
                outcome_credits=index.outcome_credits_for_memory(candidate.memory_id),
                resolver=EvidenceResolver(
                    build_workspace_context(
                        repo_root=tmp_path,
                        workspace_root=source_root,
                        workspace_id="ws",
                    )
                ),
            )
    finally:
        index.close()


def test_procedure_rejects_stale_source_episode(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    try:
        client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
        source_root = tmp_path / "source"
        source_root.mkdir()
        episode = client.remember("Initial episode.", requested_block_id="stable_project_facts")
        candidate = client.remember(
            "Repeat the checked step.",
            kind="procedure",
            metadata={
                "procedure_applicability": "repair:python",
                "source_task_signature": "repair:python",
                "source_episode_ids": [episode.memory_id],
            },
            requested_block_id="stable_project_facts",
        )
        candidate_record = index.get_block(candidate.memory_id)
        episode_record = index.get_block(episode.memory_id)
        assert candidate_record is not None and episode_record is not None
        subject = client.verification_subject(
            candidate.memory_id, expected_source_revision_id=candidate_record.revision_id
        )
        semantic = append_candidate_semantic_verification_ledger_event(
            repo_root=tmp_path,
            subject=subject,
            workspace_root=source_root,
            verifier_id="verifier:semantic",
            verifier_model_version="reviewer",
            proposer_model_version="author",
            coverage=True,
            preservation=True,
            faithfulness=True,
            instruction_safety=True,
            reasons=("Reviewed the original episode.",),
        )
        client.amend(
            episode.memory_id,
            expected_revision=episode_record.revision_id,
            content="Corrected episode.",
            reason="source correction",
        )
        with pytest.raises(ValueError, match="source episode is no longer current"):
            client.promote(
                candidate.memory_id,
                reason="publish",
                evidence_refs=[semantic],
                expected_source_revision_id=candidate_record.revision_id,
                workspace_root=source_root,
            )
    finally:
        index.close()
