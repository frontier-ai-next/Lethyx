from pathlib import Path

from um_mem.contracts import EvidenceRef, MemoryBlockRecord
from um_mem.memory import cli
from um_mem.memory.backends.contracts import MemoryEdgeRecord
from um_mem.memory.obsidian_export import export_records_to_obsidian, note_name_for_block


def _block(
    block_id: str,
    content: str,
    *,
    title: str = "",
    workspace_id: str = "demo",
    status: str = "accepted",
    trust_level: str = "workspace_verified",
    sensitivity: str = "internal",
    content_mode: str = "full",
    operational_risk: str = "low",
    tags: tuple[str, ...] = ("testing",),
    kind: str = "lesson",
    scope: str = "workspace",
    user_id: str = "",
) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        title=title or block_id,
        content=content,
        kind=kind,
        status=status,
        retention="durable",
        delivery_policy="on_demand",
        temperature="warm",
        authority="advisory",
        trust_level=trust_level,
        scope=scope,
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
        workspace_id=workspace_id,
        user_id=user_id,
        source_backend="mempalace",
        tags=tags,
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"tests::{block_id}",
                produced_by="verifier",
                producer_id=f"verifier:{block_id}",
            ),
        ),
        sensitivity=sensitivity,
        content_mode=content_mode,
        operational_risk=operational_risk,
    )


def test_obsidian_export_writes_project_index_notes_and_graph_links(tmp_path: Path) -> None:
    first = _block("mem-a", "Use scoped recall for Project Alpha.", title="Scoped Recall")
    second = _block("mem-b", "Do not load unrelated project memory.", title="Isolation Rule")
    edge = MemoryEdgeRecord(
        src_id="mem-a",
        dst_id="mem-b",
        edge_type="supports",
        workspace_id="demo",
        reason="Scoped recall supports isolation.",
    )

    result = export_records_to_obsidian(
        blocks=[first, second],
        edges=[edge],
        vault_dir=tmp_path,
        workspace_id="demo",
        project_title="Demo Project",
    )

    assert result.index_path == tmp_path / "UM-MEM" / "Projects" / "demo" / "Index.md"
    assert result.index_path.exists()
    assert result.memory_overview_path.exists()

    first_note = result.block_paths["mem-a"]
    second_note = result.block_paths["mem-b"]
    assert first_note.name == "Scoped Recall - mem-a.md"
    assert "Core" in first_note.parts
    assert "Project knowledge" in first_note.parts
    assert second_note.name == "Isolation Rule - mem-b.md"

    first_text = first_note.read_text(encoding="utf-8")
    assert "workspace_id: demo" in first_text
    assert "status: accepted" in first_text
    assert "authority: advisory" in first_text
    assert "trust_level: workspace_verified" in first_text
    assert "Supports: [[Isolation Rule - mem-b]]" in first_text
    assert "Scoped recall supports isolation." in first_text

    index_text = result.index_path.read_text(encoding="utf-8")
    assert "# Demo Project memory" in index_text
    assert "[[Core/Core Index|Core memory]]" in index_text
    assert "[[Evidence/Evidence Index|Evidence / session recall]]" in index_text
    assert "[[Scoped Recall - mem-a]]" not in index_text

    core_index_text = (
        tmp_path / "UM-MEM" / "Projects" / "demo" / "Core" / "Core Index.md"
    ).read_text(encoding="utf-8")
    assert "[[Core/Project knowledge/Project knowledge Index|Project knowledge]]" in core_index_text
    area_hub_text = (
        tmp_path
        / "UM-MEM"
        / "Projects"
        / "demo"
        / "Core"
        / "Project knowledge"
        / "Project knowledge Index.md"
    ).read_text(encoding="utf-8")
    assert "[[Core/Project knowledge/Scoped Recall - mem-a|Scoped Recall - mem-a]]" in area_hub_text
    assert (
        "[[Core/Project knowledge/Isolation Rule - mem-b|Isolation Rule - mem-b]]" in area_hub_text
    )

    overview_text = result.memory_overview_path.read_text(encoding="utf-8")
    assert "# Demo Project memory overview" in overview_text
    assert "Use scoped recall for Project Alpha." in overview_text


def test_obsidian_export_is_workspace_scoped_by_default(tmp_path: Path) -> None:
    demo = _block("demo-memory", "Demo workspace memory.", workspace_id="demo")
    other = _block("other-memory", "Other workspace memory.", workspace_id="other")
    global_memory = _block(
        "global-memory",
        "Global user memory.",
        workspace_id="demo",
        scope="user_global",
        user_id="user-1",
    )

    result = export_records_to_obsidian(
        blocks=[demo, other, global_memory],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
    )

    assert set(result.block_paths) == {"demo-memory"}
    text = result.index_path.read_text(encoding="utf-8")
    assert "Demo workspace memory." in (
        tmp_path
        / "UM-MEM"
        / "Projects"
        / "demo"
        / "Core"
        / "Project knowledge"
        / "demo-memory - demo-memory.md"
    ).read_text(encoding="utf-8")
    assert "Other workspace memory" not in text
    assert "Global user memory" not in text


def test_obsidian_export_keeps_session_evidence_out_of_core_memory_tree(tmp_path: Path) -> None:
    core = _block("core-rule", "Always use scoped recall for this project.", title="Core Rule")
    session = _block(
        "session_summary_20260627_210403_045200",
        "Raw-ish session summary should remain recall evidence, not a top-level core memory note.",
        title="Session Summary",
        kind="session_summary",
        tags=("session-summary",),
    )

    result = export_records_to_obsidian(
        blocks=[core, session],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
    )

    assert set(result.block_paths) == {"core-rule"}
    assert not (tmp_path / "UM-MEM" / "Projects" / "demo" / "Core" / "Session summaries").exists()
    index_text = result.index_path.read_text(encoding="utf-8")
    assert "Core memory blocks: **1**" in index_text
    assert "Evidence/archive blocks kept in backend: **1**" in index_text
    archive_text = (tmp_path / "UM-MEM" / "Projects" / "demo" / "Evidence Archive.md").read_text(
        encoding="utf-8"
    )
    assert "[[Evidence/Evidence Index|Evidence Index]]" in archive_text
    assert "[[Session Summary - session_summary_20260627_210403_045200]]" not in archive_text

    evidence_index_text = (
        tmp_path / "UM-MEM" / "Projects" / "demo" / "Evidence" / "Evidence Index.md"
    ).read_text(encoding="utf-8")
    assert "[[Evidence/Sessions/Evidence Sessions Index|Sessions]]" in evidence_index_text
    sessions_hub_text = (
        tmp_path
        / "UM-MEM"
        / "Projects"
        / "demo"
        / "Evidence"
        / "Sessions"
        / "Evidence Sessions Index.md"
    ).read_text(encoding="utf-8")
    assert (
        "[[Evidence/Sessions/Session 20260627 210403 045200|Session 20260627 210403 045200]]"
        in sessions_hub_text
    )
    session_hub_text = (
        tmp_path
        / "UM-MEM"
        / "Projects"
        / "demo"
        / "Evidence"
        / "Sessions"
        / "Session 20260627 210403 045200.md"
    ).read_text(encoding="utf-8")
    assert (
        "[[Evidence/Sessions/Session Summary - session_summary_20260627_210403_045200|Session Summary - session_summary_20260627_210403_045200]]"
        in session_hub_text
    )
    evidence_note = (
        tmp_path
        / "UM-MEM"
        / "Projects"
        / "demo"
        / "Evidence"
        / "Sessions"
        / "Session Summary - session_summary_20260627_210403_045200.md"
    )
    assert evidence_note.exists()
    evidence_text = evidence_note.read_text(encoding="utf-8")
    assert "area: Evidence/Sessions" in evidence_text
    assert "not a directive or stable project fact" in evidence_text


def test_obsidian_export_uses_universal_evidence_categories_without_ideas_parent(
    tmp_path: Path,
) -> None:
    user_intent = _block(
        "turn_20260627_213118_a95a57_intent",
        "User asked the agent to recall prior project context.",
        title="User intent signal",
        kind="user_intent",
        status="candidate",
    )
    candidate = _block(
        "candidate_feature_note",
        "Potential project feature inferred from session evidence.",
        title="Candidate feature note",
        kind="hypothesis",
        status="candidate",
    )

    export_records_to_obsidian(
        blocks=[user_intent, candidate],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
    )

    root = tmp_path / "UM-MEM" / "Projects" / "demo"
    assert (
        root
        / "Evidence"
        / "User intents"
        / "User intent signal - turn_20260627_213118_a95a57_intent.md"
    ).exists()
    assert (
        root / "Evidence" / "Candidate ideas" / "Candidate feature note - candidate_feature_note.md"
    ).exists()
    assert not (root / "Evidence" / "Ideas").exists()


def test_obsidian_export_does_not_mutate_unowned_project_state(tmp_path: Path) -> None:
    unowned = tmp_path / "UM-MEM" / "Projects" / "demo" / ".obsidian" / "workspace.json"
    unowned.parent.mkdir(parents=True, exist_ok=True)
    unowned.write_text("{}", encoding="utf-8")

    export_records_to_obsidian(
        blocks=[_block("core-rule", "Core memory only.", title="Core Rule")],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
    )

    assert unowned.read_text(encoding="utf-8") == "{}"


def test_obsidian_export_can_include_user_global_scope_explicitly(tmp_path: Path) -> None:
    demo = _block("demo-memory", "Demo workspace memory.", workspace_id="demo")
    global_memory = _block(
        "global-memory",
        "Global user memory.",
        workspace_id="demo",
        scope="user_global",
        user_id="user-1",
    )

    result = export_records_to_obsidian(
        blocks=[demo, global_memory],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
        scope_chain=("workspace:demo", "user_global:user-1"),
    )

    assert set(result.block_paths) == {"demo-memory", "global-memory"}
    assert "Global user memory." in result.block_paths["global-memory"].read_text(encoding="utf-8")


def test_obsidian_export_excludes_protected_canonical_safety_records(tmp_path: Path) -> None:
    public = _block("public-memory", "Public memory can be exported.")
    secret = _block(
        "secret-memory",
        "Secret pointer must not leave local storage.",
        sensitivity="secret",
        content_mode="pointer_only",
    )
    local_only = _block(
        "local-only-memory",
        "Non-injectable memory must stay out of the bundle.",
        content_mode="non_injectable",
    )

    result = export_records_to_obsidian(
        blocks=[public, secret, local_only],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
    )

    assert set(result.block_paths) == {"public-memory"}
    bundle = result.memory_overview_path.read_text(encoding="utf-8")
    assert "Public memory can be exported." in bundle
    assert "Secret pointer must not leave local storage." not in bundle
    assert "Non-injectable memory must stay out of the bundle." not in bundle


def test_note_name_for_block_is_stable_and_filesystem_safe() -> None:
    block = _block("mem:weird/id", "body", title="Bad / Windows: Name?")

    assert note_name_for_block(block) == "Bad Windows Name - mem-weird-id"


def test_cli_obsidian_export_writes_vault_from_selected_backend(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.block = _block("cli-memory", "CLI export memory.", title="CLI Memory")

        def export_blocks(self) -> list[MemoryBlockRecord]:
            return [self.block]

        def export_edges(self) -> list[MemoryEdgeRecord]:
            return []

    def fake_create_backend(
        repo: Path, *, workspace_id: str, mode: str | None = None
    ) -> FakeBackend:
        assert repo == tmp_path.resolve()
        assert workspace_id == "demo"
        assert mode == "mempalace"
        return FakeBackend()

    monkeypatch.setattr(cli, "create_durable_backend", fake_create_backend)

    code = cli.main(
        [
            "obsidian-export",
            "--repo-root",
            str(tmp_path),
            "--workspace",
            "demo",
            "--backend",
            "mempalace",
            "--vault-dir",
            str(tmp_path / "vault"),
            "--project-title",
            "Demo CLI",
            "--json",
        ]
    )

    assert code == 0
    assert (tmp_path / "vault" / "UM-MEM" / "Projects" / "demo" / "Index.md").exists()
    assert (tmp_path / "vault" / "UM-MEM" / "Projects" / "demo" / "Memory Overview.md").exists()
    assert not (tmp_path / "vault" / "UM-MEM" / "Projects" / "demo" / "NotebookLM").exists()
