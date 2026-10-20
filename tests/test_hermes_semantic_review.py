from dataclasses import replace

import pytest

from lethyx.contracts import CandidateVerificationSubject
from lethyx.integrations.hermes_provider import UmMemMemoryProvider, _load_config


def test_working_context_configuration_requires_an_independent_verifier(tmp_path):
    import json

    directory = tmp_path / "lethyx"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"working_context_enabled": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="independent"):
        _load_config(tmp_path)
    (directory / "config.json").write_text(
        json.dumps({"working_context_enabled": "false"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bool"):
        _load_config(tmp_path)


def test_saved_working_configuration_is_validated_before_writing(tmp_path):
    provider = UmMemMemoryProvider()
    with pytest.raises(ValueError, match="independent"):
        provider.save_config({"working_context_enabled": True}, str(tmp_path))
    assert not (tmp_path / "lethyx" / "config.json").exists()


def test_semantic_review_reuses_only_the_same_fresh_subject(tmp_path, monkeypatch):
    provider = UmMemMemoryProvider()
    provider._config = {"repo_root": str(tmp_path), "verification_model": "reviewer"}
    calls = []

    def review(**kwargs):
        calls.append(kwargs)
        return {
            "coverage": True,
            "preservation": True,
            "faithfulness": True,
            "instruction_safety": True,
            "reasons": ["Compared full immutable source evidence."],
        }, "reviewer"

    monkeypatch.setattr(provider, "_model_json", review)
    subject = CandidateVerificationSubject(
        workspace_id="alpha",
        profile_id="profile-a",
        memory_id="candidate-a",
        revision_id="revision-a",
        payload_hash="payload-a",
        source_evidence_hash="evidence-a",
        target_block_id="leaf-a",
        authority="advisory",
        scope="workspace",
        scope_id="alpha",
    )
    first = provider._semantic_review(
        subject, payload={"content": "5"}, sources=[{"content": "5"}], proposer_model="author"
    )
    replay = provider._semantic_review(
        subject, payload={"content": "5"}, sources=[{"content": "5"}], proposer_model="author"
    )
    assert first == replay
    assert len(calls) == 1
    other = provider._semantic_review(
        replace(subject, memory_id="candidate-b"),
        payload={"content": "5"},
        sources=[{"content": "5"}],
        proposer_model="author",
    )
    assert other.report_id != first.report_id
    assert len(calls) == 2
    (tmp_path / "evidence.txt").write_text("changed", encoding="utf-8")
    fresh = provider._semantic_review(
        subject, payload={"content": "5"}, sources=[{"content": "5"}], proposer_model="author"
    )
    assert fresh.report_id != first.report_id
    assert len(calls) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision_id", "revision-new"),
        ("payload_hash", "payload-new"),
        ("source_evidence_hash", "evidence-new"),
        ("target_block_id", "leaf-new"),
    ],
)
def test_subject_changes_require_new_review(tmp_path, monkeypatch, field, value):
    provider = UmMemMemoryProvider()
    provider._config = {"repo_root": str(tmp_path), "verification_model": "reviewer"}
    calls = []

    def review(**kwargs):
        calls.append(kwargs)
        return {
            "coverage": True,
            "preservation": True,
            "faithfulness": True,
            "instruction_safety": True,
            "reasons": ["Exact sources reviewed."],
        }, "reviewer"

    monkeypatch.setattr(provider, "_model_json", review)
    subject = CandidateVerificationSubject(
        workspace_id="alpha",
        profile_id="default",
        memory_id="candidate",
        revision_id="revision",
        payload_hash="payload",
        source_evidence_hash="evidence",
        target_block_id="leaf",
        authority="advisory",
        scope="workspace",
        scope_id="alpha",
    )
    first = provider._semantic_review(subject, payload={}, sources=[], proposer_model="author")
    second = provider._semantic_review(
        replace(subject, **{field: value}), payload={}, sources=[], proposer_model="author"
    )
    assert second.report_id != first.report_id
    assert len(calls) == 2


def test_host_verifier_fallback_to_author_is_rejected(tmp_path, monkeypatch):
    provider = UmMemMemoryProvider()
    provider._config = {"repo_root": str(tmp_path), "verification_model": "reviewer"}
    monkeypatch.setattr(provider, "_model_json", lambda **kwargs: ({}, "author"))
    subject = CandidateVerificationSubject(
        workspace_id="alpha",
        profile_id="profile-a",
        memory_id="candidate-a",
        revision_id="revision-a",
        payload_hash="payload-a",
        source_evidence_hash="evidence-a",
        target_block_id="leaf-a",
        authority="advisory",
        scope="workspace",
        scope_id="alpha",
    )
    with pytest.raises(ValueError, match="author model"):
        provider._semantic_review(subject, payload={}, sources=[], proposer_model="author")
