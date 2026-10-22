from pathlib import Path

from um_mem.contracts import EvidenceResolver, build_workspace_context, evidence
from um_mem.contracts.hashing import SKIP_DIRS, file_sha256, hash_value, workspace_hash
from um_mem.contracts.ledger import append_verification_report_ledger_event


def test_evidence_resolver_reads_one_operation_snapshot(tmp_path, monkeypatch) -> None:
    first = append_verification_report_ledger_event(
        repo_root=tmp_path,
        workspace_id="alpha",
        verifier_id="reviewer",
        report_hash="one",
        passed=True,
    )
    read = evidence.read_supervisor_ledger_events
    calls = []

    def counted(**kwargs):
        calls.append(kwargs)
        return read(**kwargs)

    monkeypatch.setattr(evidence, "read_supervisor_ledger_events", counted)
    context = build_workspace_context(
        repo_root=tmp_path, workspace_root=tmp_path, workspace_id="alpha"
    )
    resolver = EvidenceResolver(context)
    second = append_verification_report_ledger_event(
        repo_root=tmp_path,
        workspace_id="alpha",
        verifier_id="reviewer",
        report_hash="two",
        passed=True,
    )
    assert resolver.validate_verification_report_ref(first) == []
    assert resolver.validate_verification_report_ref(first) == []
    assert resolver.ledger_row(second.report_id) is None
    assert len(calls) == 1
    assert EvidenceResolver(context).validate_verification_report_ref(second) == []
    assert len(calls) == 2


def test_workspace_hash_preserves_manifest_and_prunes_excluded_trees(tmp_path, monkeypatch) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "file.py").write_text("value = +5\n")
    ignored = tmp_path / ".venv" / "nested"
    ignored.mkdir(parents=True)
    (ignored / "large.bin").write_bytes(b"irrelevant" * 1000)
    expected = hash_value(
        [
            (path.relative_to(tmp_path).as_posix(), file_sha256(path))
            for path in sorted(tmp_path.rglob("*"))
            if path.is_file()
            and not any(part in SKIP_DIRS for part in path.relative_to(tmp_path).parts)
        ]
    )

    def reject_rglob(*args, **kwargs):
        raise AssertionError("must prune directories before traversal")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rglob", reject_rglob)
        assert workspace_hash(tmp_path) == expected
