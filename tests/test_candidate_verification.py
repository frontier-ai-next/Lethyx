import json
from dataclasses import asdict, replace

import pytest

from um_mem import MemoryClient
from um_mem.contracts import EvidenceResolver, VerificationReportRef, build_workspace_context
from um_mem.contracts.ledger import (
    append_candidate_semantic_verification_ledger_event,
    append_verification_report_ledger_event,
)
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.tribunal.models import TribunalInput
from um_mem.memory.tribunal.runner import run_tribunal


def test_generic_test_report_does_not_verify_candidate_content(tmp_path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / ".UM-MEM" / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        candidate = client.remember(
            "The parser accepts 500.", requested_block_id="stable_project_facts"
        )
        report = append_verification_report_ledger_event(
            repo_root=tmp_path,
            workspace_id="alpha",
            verifier_id="tests:parser",
            report_hash="passed-tests",
            passed=True,
        )
        record = index.get_block(candidate.memory_id)
        with pytest.raises(ValueError, match="semantic"):
            client.promote(
                candidate.memory_id,
                reason="publish",
                evidence_refs=[report],
                expected_source_revision_id=record.revision_id,
            )
        assert index.get_block(candidate.memory_id).status == "candidate"
    finally:
        index.close()


def test_failed_native_amend_cannot_recover_a_generic_copied_witness(tmp_path, monkeypatch):
    from um_mem.contracts import EvidenceRef, HostObservation
    from um_mem.memory import mcp_call, mcp_facade
    from um_mem.memory.local_service import build_local_service_identity
    from um_mem.memory.service_receipts import (
        host_tool_mutation_correlation_id,
        read_committed_mutation_receipt,
    )

    index = BackendSQLiteIndex(
        tmp_path / "index.sqlite", backend_id="mempalace", workspace_id="alpha"
    )
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        session_id="session-a",
        task_id="task-a",
        backend="mempalace",
        backend_instance=index,
    )
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda value: None)
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="parent"
    )
    correlation_id = host_tool_mutation_correlation_id("hermes:session-a", "failed-call")
    try:
        created = client.remember("Original value.", requested_block_id="stable_project_facts")
        original = index.get_block(created.memory_id)
        native_arguments = {
            "memory_id": original.block_id,
            "expected_revision": original.revision_id,
            "content": "",
            "requested_block_id": "stable_project_facts",
            "reason": "native proposal",
        }
        witness = client.observe(
            HostObservation(
                event_id="failed-call",
                event_kind="ambiguous",
                raw_evidence=json.dumps(
                    {
                        "tool": "um_mem_provider_amend",
                        "arguments": native_arguments,
                        "native_tool_call_id": "failed-call",
                        "session_id": "session-a",
                        "author_model_version": "native-author",
                    }
                ),
                observed_at="2026-10-01T12:00:00Z",
                producer_id="hermes:provider-tool",
            )
        )
        metadata = {
            "raw_witness_id": witness["witness_id"],
            "native_tool_call_id": "failed-call",
            "source_id": "hermes:session-a",
        }
        request = json.dumps(
            {
                "protocol_version": identity.protocol_version,
                "request_id": "failed-call",
                "lease": "test-lease",
                "caller_role": "parent",
                "tool": "amend",
                "arguments": {
                    **native_arguments,
                    "evidence_refs": witness["evidence_refs"],
                    "metadata_patch": metadata,
                },
                "mutation_correlation_id": correlation_id,
            }
        )
        first = mcp_call._dispatch_local_service_request(
            identity, request, client=client, expected_lease="test-lease"
        )["payload"]
        assert first["ok"] is False
        assert "content cannot be empty" in first["error"]

        generic = client.amend(
            original.block_id,
            expected_revision=original.revision_id,
            content="Unrelated generic payload.",
            reason="generic edit",
            requested_block_id="stable_project_facts",
            metadata_patch=metadata,
            evidence_refs=[EvidenceRef.from_mapping(ref) for ref in witness["evidence_refs"]],
        )
        assert generic["saved"] is True
        record = index.get_block(generic["memory_id"])
        retry = mcp_call._dispatch_local_service_request(
            identity, request, client=client, expected_lease="test-lease"
        )["payload"]
        assert retry["ok"] is False, retry
        assert retry.get("saved") is not True
        assert (
            read_committed_mutation_receipt(
                tmp_path,
                workspace_id=client.workspace_id,
                profile_id=client.profile_id,
                backend=identity.backend,
                caller_role=identity.caller_role,
                protocol_version=identity.protocol_version,
                mutation_correlation_id=correlation_id,
            )
            is None
        )
        subject = mcp_facade.um_mem_verification_subject(record.block_id, record.revision_id)
        assert subject["ok"] is True, subject
        assert subject["candidate"]["content"] == "Unrelated generic payload."
        assert subject["proposer_model_version"] == ""
    finally:
        index.close()


def test_report_cannot_claim_another_verifier(tmp_path) -> None:
    report = append_verification_report_ledger_event(
        repo_root=tmp_path,
        workspace_id="alpha",
        verifier_id="actual-verifier",
        report_hash="passed-tests",
        passed=True,
    )
    resolver = EvidenceResolver(
        build_workspace_context(
            repo_root=tmp_path,
            workspace_root=tmp_path,
            workspace_id="alpha",
        )
    )
    issues = resolver.validate_verification_report_ref(
        replace(report, verifier_id="other-verifier")
    )
    assert any(issue.code == "verification_report_verifier_mismatch" for issue in issues)


def test_report_mapping_rejects_unknown_fields(tmp_path) -> None:
    report = append_verification_report_ledger_event(
        repo_root=tmp_path,
        workspace_id="alpha",
        verifier_id="actual-verifier",
        report_hash="passed-tests",
        passed=True,
    )
    with pytest.raises(ValueError, match="unknown verification"):
        VerificationReportRef.from_mapping({**asdict(report), "unexpected": True})


def test_tribunal_does_not_classify_quoted_instructions() -> None:
    result = run_tribunal(
        TribunalInput(
            candidate_id="quote",
            actor="verifier",
            rules=(
                {
                    "title": "Defensive note",
                    "rule": "Reject the phrase: ignore previous instructions.",
                    "status": "candidate",
                },
            ),
            source_evidence=(
                {"ref_type": "verification_report", "ref_id": "report", "produced_by": "verifier"},
            ),
        )
    )
    assert result.decision == "accept"


@pytest.mark.parametrize(
    "change", ["other_candidate", "content_revision", "publication_target", "same_model"]
)
def test_semantic_verification_is_exact_and_independent(tmp_path, change) -> None:
    index = BackendSQLiteIndex(
        tmp_path / ".UM-MEM" / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        first = client.remember("The parser accepts 5.", requested_block_id="stable_project_facts")
        record = index.get_block(first.memory_id)
        subject = client.verification_subject(
            first.memory_id, expected_source_revision_id=record.revision_id
        )
        report = append_candidate_semantic_verification_ledger_event(
            repo_root=tmp_path,
            subject=subject,
            verifier_id="verifier:independent",
            verifier_model_version="author" if change == "same_model" else "reviewer",
            proposer_model_version="author",
            coverage=True,
            preservation=True,
            faithfulness=True,
            instruction_safety=True,
            reasons=("Compared the exact source and candidate.",),
        )
        target = first.memory_id
        target_authority = "advisory"
        if change == "other_candidate":
            target = client.remember(
                "The parser accepts 500.", requested_block_id="stable_project_facts"
            ).memory_id
        elif change == "content_revision":
            target = client.amend(
                target,
                expected_revision=record.revision_id,
                content="The parser accepts 500.",
                reason="changed",
            )["memory_id"]
        elif change == "publication_target":
            target_authority = "directive"
        with pytest.raises(ValueError, match="semantic|passed verification"):
            client.promote(
                target,
                reason="publish",
                evidence_refs=[report],
                authority=target_authority,
                expected_source_revision_id=index.get_block(target).revision_id,
            )
        assert index.get_block(target).status == "candidate"
    finally:
        index.close()


def test_exact_semantic_report_can_publish_a_candidate(tmp_path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / ".UM-MEM" / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        candidate = client.remember(
            "The parser accepts 5.", requested_block_id="stable_project_facts"
        )
        record = index.get_block(candidate.memory_id)
        subject = client.verification_subject(
            candidate.memory_id, expected_source_revision_id=record.revision_id
        )
        report = append_candidate_semantic_verification_ledger_event(
            repo_root=tmp_path,
            subject=subject,
            verifier_id="verifier:independent",
            verifier_model_version="reviewer",
            proposer_model_version="author",
            coverage=True,
            preservation=True,
            faithfulness=True,
            instruction_safety=True,
            reasons=("Compared the exact source and candidate.",),
        )
        result = client.promote(
            candidate.memory_id,
            reason="publish",
            evidence_refs=[report],
            expected_source_revision_id=record.revision_id,
        )
        assert result["ok"] is True
        published = index.get_block(result["block_id"])
        assert published.content == record.content
        assert published.status == "accepted"
        assert published.verification_status == "verified"
        assert published.revision_of == record.block_id
    finally:
        index.close()


def test_publication_operation_replays_after_newer_revision_and_workspace_change(
    tmp_path, candidate_semantic_report
) -> None:
    index = BackendSQLiteIndex(
        tmp_path / ".UM-MEM" / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        saved = client.remember("Reviewed finding.", requested_block_id="stable_project_facts")
        source = index.get_block(saved.memory_id)
        arguments = {
            "reason": "publish",
            "evidence_refs": [candidate_semantic_report(client, saved.memory_id)],
            "expected_source_revision_id": source.revision_id,
            "operation_id": "publication-one",
        }
        first = client.promote(saved.memory_id, **arguments)
        retracted = client.forget(first["block_id"], reason="later retraction")
        (tmp_path / "changed.txt").write_text("new workspace version")
        replay = client.promote(saved.memory_id, **arguments)
        assert replay["replayed"] is True
        assert replay["block_id"] == first["block_id"]
        assert index.get_block(retracted["block_id"]).status == "retracted"
        assert len(index.export_blocks()) == 3
        with pytest.raises(ValueError, match="receipt conflict"):
            client.promote(saved.memory_id, **{**arguments, "authority": "directive"})
    finally:
        index.close()


@pytest.mark.parametrize("materialized_copy", [False, True])
def test_pending_publication_receipt_recovers_the_committed_native_write(
    tmp_path, monkeypatch, candidate_semantic_report, materialized_copy
) -> None:
    from um_mem import api

    index = BackendSQLiteIndex(
        tmp_path / ".UM-MEM" / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        saved = client.remember(
            "Reviewed finding.",
            run_id="run-one",
            requested_block_id="reflection_candidates"
            if materialized_copy
            else "stable_project_facts",
        )
        source = index.get_block(saved.memory_id)
        arguments = {
            "reason": "publish",
            "evidence_refs": [
                candidate_semantic_report(
                    client,
                    saved.memory_id,
                    target_block_id="verified_lessons" if materialized_copy else "",
                )
            ],
            "target_block_id": "verified_lessons" if materialized_copy else "",
            "expected_source_revision_id": source.revision_id,
            "operation_id": "publication-one",
        }
        complete = api.complete_remember_receipt

        def fail_receipt(*args, **kwargs):
            raise OSError("simulated lost acknowledgement")

        monkeypatch.setattr(api, "complete_remember_receipt", fail_receipt)
        with pytest.raises(OSError, match="lost acknowledgement"):
            client.promote(saved.memory_id, **arguments)
        monkeypatch.setattr(api, "complete_remember_receipt", complete)
        committed = next(record for record in index.export_blocks() if record.status == "accepted")
        client.forget(committed.block_id, reason="later retraction")
        (tmp_path / "changed.txt").write_text("changed before receipt recovery")
        replay = client.promote(saved.memory_id, **arguments)
        assert replay["replayed"] is True
        assert replay["block_id"] == committed.block_id
        assert len(index.export_blocks()) == 3
    finally:
        index.close()


def test_semantic_verifier_rejects_disguised_same_model(tmp_path) -> None:
    from um_mem.contracts import CandidateVerificationSubject

    subject = CandidateVerificationSubject(
        workspace_id="alpha",
        profile_id="default",
        memory_id="candidate",
        revision_id="rev",
        payload_hash="hash",
        source_evidence_hash="hash",
        target_block_id="leaf",
        authority="advisory",
        scope="workspace",
        scope_id="alpha",
    )
    with pytest.raises(ValueError, match="identities"):
        append_candidate_semantic_verification_ledger_event(
            repo_root=tmp_path,
            subject=subject,
            verifier_id="reviewer",
            verifier_model_version="same-model ",
            proposer_model_version="same-model",
            coverage=True,
            preservation=True,
            faithfulness=True,
            instruction_safety=True,
            reasons=("reviewed",),
        )


def test_generic_leaf_publication_replay_keeps_the_original_verified_revision(
    tmp_path, candidate_semantic_report
):
    index = BackendSQLiteIndex(
        tmp_path / ".UM-MEM" / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    try:
        client = MemoryClient(
            tmp_path, workspace_id="alpha", project_id="project-a", backend_instance=index
        )
        root_id = client.block_map()[0]["block_id"]
        leaf = client.create_leaf_block(
            name="Reviewed measurements",
            synopsis="Verified measurements",
            parent_id=root_id,
            reason="model chosen purpose",
            scope="project",
        )
        target = leaf["block_id"]
        source = client.remember(
            "Original approved payload.", requested_block_id="stable_project_facts"
        )
        report = candidate_semantic_report(client, source.memory_id, target_block_id=target)
        arguments = {
            "reason": "publish",
            "evidence_refs": [report],
            "expected_source_revision_id": index.get_block(source.memory_id).revision_id,
            "target_block_id": target,
        }
        published = client.promote(source.memory_id, operation_id="first-publication", **arguments)
        original = index.get_block(published["block_id"])
        assert original.scope == "project" and original.project_id == "project-a"
        corrected = client.correction(
            original.block_id,
            content="Different approved payload.",
            reason="new measurement",
            requested_block_id=target,
        )
        corrected_id = corrected["block_id"]
        client.promote(
            corrected_id,
            reason="publish correction",
            evidence_refs=[candidate_semantic_report(client, corrected_id)],
            expected_source_revision_id=index.get_block(corrected_id).revision_id,
        )
        replay = client.promote(
            source.memory_id, operation_id="another-acknowledgement", **arguments
        )
        assert replay["replayed"] is True
        assert replay["block_id"] == original.block_id
        assert replay["revision_id"] == original.revision_id
        assert index.get_block(replay["block_id"]).content == "Original approved payload."
    finally:
        index.close()


def test_publication_subject_rejects_scope_broader_than_signed_target(tmp_path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / ".UM-MEM" / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    try:
        client = MemoryClient(
            tmp_path, workspace_id="alpha", project_id="project-one", backend_instance=index
        )
        saved = client.remember("Scoped finding.", requested_block_id="reflection_candidates")
        with pytest.raises(ValueError, match="target.*scope"):
            client.verification_subject(
                saved.memory_id,
                expected_source_revision_id=index.get_block(saved.memory_id).revision_id,
                target_block_id="verified_lessons",
                target_scope="project",
            )
    finally:
        index.close()


def test_native_authorship_requires_the_exact_committed_operation(tmp_path, monkeypatch):
    from um_mem.contracts import HostObservation
    from um_mem.memory import mcp_call, mcp_facade
    from um_mem.memory.local_service import build_local_service_identity
    from um_mem.memory.service_receipts import host_tool_mutation_correlation_id

    index = BackendSQLiteIndex(
        tmp_path / "index.sqlite", backend_id="mempalace", workspace_id="alpha"
    )
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        session_id="session-a",
        task_id="task-a",
        backend="mempalace",
        backend_instance=index,
    )
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda value: None)
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="parent"
    )

    def native(tool, arguments, invocation, author, *, lose_ack=False):
        witness = client.observe(
            HostObservation(
                event_id=invocation,
                event_kind="ambiguous",
                raw_evidence=json.dumps(
                    {
                        "tool": "um_mem_provider_" + tool,
                        "arguments": arguments,
                        "native_tool_call_id": invocation,
                        "session_id": "session-a",
                        "author_model_version": author,
                    }
                ),
                observed_at="2026-10-01T12:00:00Z",
                producer_id="hermes:provider-tool",
            )
        )
        metadata = {
            "raw_witness_id": witness["witness_id"],
            "native_tool_call_id": invocation,
            "source_id": "hermes:session-a",
        }
        request_arguments = {
            **arguments,
            "evidence_refs": witness["evidence_refs"],
            "metadata_patch" if tool == "amend" else "metadata": metadata,
        }
        if tool == "remember":
            request_arguments["source_id"] = "hermes:session-a"
        request = json.dumps(
            {
                "protocol_version": identity.protocol_version,
                "request_id": invocation,
                "lease": "test-lease",
                "caller_role": "parent",
                "tool": tool,
                "arguments": request_arguments,
                "mutation_correlation_id": host_tool_mutation_correlation_id(
                    "hermes:session-a", invocation
                ),
            }
        )
        if lose_ack:
            completed = mcp_call.complete_remember_receipt
            with monkeypatch.context() as failing_ack:
                failing_ack.setattr(
                    mcp_call,
                    "complete_remember_receipt",
                    lambda *args, **kwargs: (_ for _ in ()).throw(OSError("acknowledgement lost")),
                )
                failed = mcp_call._dispatch_local_service_request(
                    identity, request, client=client, expected_lease="test-lease"
                )
                assert failed["payload"]["ok"] is False
            assert mcp_call.complete_remember_receipt is completed
        response = mcp_call._dispatch_local_service_request(
            identity, request, client=client, expected_lease="test-lease"
        )
        assert response["payload"]["ok"] is True, response
        if lose_ack:
            assert response["payload"]["recovered"] is True
        result = response["payload"]
        identifier = result.get("memory_id") or result.get("new_id")
        return index.get_block(identifier)

    def author_of(record):
        result = mcp_facade.um_mem_verification_subject(
            record.block_id, record.revision_id, target_block_id="stable_project_facts"
        )
        assert result["ok"] is True, result
        return result["proposer_model_version"]

    try:
        original = native(
            "remember",
            {
                "operation": "add",
                "content": "Exact value 005.",
                "requested_block_id": "stable_project_facts",
            },
            "call-a",
            "author-a",
        )
        assert author_of(original) == "author-a"
        duplicate = client.remember(
            original.content,
            requested_block_id="stable_project_facts",
            metadata={**original.metadata, "another_interpretation": True},
            evidence_refs=original.evidence_refs,
        )
        assert duplicate.memory_id != original.block_id
        assert author_of(index.get_block(duplicate.memory_id)) == ""
        amended = client.amend(
            original.block_id,
            expected_revision=original.revision_id,
            title="Edited by another adapter",
            reason="title changed",
        )
        assert author_of(index.get_block(amended["memory_id"])) == ""
        corrected = native(
            "remember",
            {
                "operation": "correct",
                "memory_id": original.block_id,
                "content": "Exact value 007.",
                "requested_block_id": "stable_project_facts",
            },
            "call-b",
            "author-b",
        )
        assert author_of(corrected) == "author-b"
        native_amend = native(
            "amend",
            {
                "memory_id": corrected.block_id,
                "expected_revision": corrected.revision_id,
                "title": "CAS title",
                "requested_block_id": "stable_project_facts",
                "reason": "edit title",
            },
            "call-c",
            "author-c",
            lose_ack=True,
        )
        assert author_of(native_amend) == "author-c"
    finally:
        index.close()


@pytest.mark.parametrize("corrupt", ["none", "report", "reference", "scope", "source"])
def test_published_candidate_assurance_is_bound_on_read(
    tmp_path, candidate_semantic_report, corrupt
):
    from copy import deepcopy

    from um_mem.memory.backends.contracts import block_record_to_memory_hit
    from um_mem.memory.context_payload import project_memory_assurance

    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        saved = client.remember(
            "Synthetic checked fact.", requested_block_id="stable_project_facts"
        )
        record = index.get_block(saved.memory_id)
        report = candidate_semantic_report(client, record.block_id, authority="directive")
        published = client.promote(
            record.block_id,
            expected_source_revision_id=record.revision_id,
            evidence_refs=[report],
            authority="directive",
            reason="Exact review",
        )
        metadata = deepcopy(
            block_record_to_memory_hit(index.get_block(published["block_id"])).metadata
        )
        if corrupt == "report":
            metadata["candidate_semantic_verification"]["result"]["coverage"] = False
        elif corrupt == "reference":
            metadata["evidence_refs"][0]["hash"] = "unbound"
        elif corrupt == "scope":
            metadata["profile_id"] = "other"
        elif corrupt == "source":
            metadata["promotion_source_revision_id"] = "other"
        projected = project_memory_assurance(metadata)
        assert projected["verification_status"] == (
            "verified" if corrupt == "none" else "unverified"
        )
        assert projected["authority"] == ("directive" if corrupt == "none" else "advisory")
    finally:
        index.close()


@pytest.mark.parametrize("scope", ["project", "user_global", "session", "task", "subtask"])
@pytest.mark.parametrize("copy_to_leaf", [False, True])
def test_publication_binds_destination_scope_identity(tmp_path, scope, copy_to_leaf):
    from um_mem.memory.backends.contracts import block_record_to_memory_hit
    from um_mem.memory.context_payload import project_memory_assurance

    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="alpha")
    try:
        identities = ("project_id", "user_id", "session_id", "task_id", "subtask_id")
        writer = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            backend_instance=index,
            **{key: key + "-a" for key in identities},
        )
        publisher = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            backend_instance=index,
            **{key: key + "-b" for key in identities},
        )
        candidate = writer.remember(
            "Synthetic scoped fact.", requested_block_id="stable_project_facts"
        )
        record = index.get_block(candidate.memory_id)
        target = ""
        if copy_to_leaf:
            target = publisher.create_leaf_block(
                name="Reviewed scoped facts",
                synopsis="Exact destination scope",
                parent_id=publisher.block_map()[0]["block_id"],
                reason="Explicit publication",
                scope=scope,
            )["block_id"]
        subject = publisher.verification_subject(
            record.block_id,
            expected_source_revision_id=record.revision_id,
            target_block_id=target,
            target_scope=scope,
        )
        report = append_candidate_semantic_verification_ledger_event(
            repo_root=tmp_path,
            subject=subject,
            verifier_id="reviewer",
            verifier_model_version="review-model",
            proposer_model_version="author-model",
            coverage=True,
            preservation=True,
            faithfulness=True,
            instruction_safety=True,
            reasons=("Synthetic exact destination review.",),
        )
        result = publisher.promote(
            record.block_id,
            expected_source_revision_id=record.revision_id,
            target_block_id=target,
            target_scope=scope,
            evidence_refs=[report],
            reason="Exact review",
        )
        assert result["ok"]
        published = index.get_block(result["block_id"])
        identity = "user_id" if scope == "user_global" else scope + "_id"
        assert getattr(published, identity) == identity + "-b"
        projected = project_memory_assurance(block_record_to_memory_hit(published).metadata)
        assert projected["verification_status"] == "verified"
    finally:
        index.close()
