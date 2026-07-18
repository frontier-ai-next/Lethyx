from pathlib import Path

from lethyx.contracts import EvidenceRef
from lethyx.memory import cli
from lethyx.memory.backends.contracts import MemoryBlockRecord, MemoryEdgeRecord
from lethyx.memory.obsidian_export import export_records_to_obsidian, note_name_for_block


def _block(
    block_id: str,
    content: str,
    *,
    title: str = "",
    workspace_id: str = "demo",
    lifecycle: str = "active",
    surface: str = "supplemental_evidence",
    trust_level: str = "workspace_verified",
    privacy_class: str = "normal",
    tags: tuple[str, ...] = ("testing",),
    kind: str = "lesson",
) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        title=title or block_id,
        content=content,
        kind=kind,
        lifecycle=lifecycle,
        trust_level=trust_level,
        scope="workspace" if workspace_id else "manager",
        surface=surface,
        workspace_id=workspace_id,
        source_backend="mempalace",
        tags=tags,
        evidence_refs=(
            EvidenceRef(ref_type="test", ref_id=f"tests::{block_id}", produced_by="pytest"),
        ),
        privacy_class=privacy_class,
        verified=True,
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

    assert result.index_path == tmp_path / "Lethyx" / "Projects" / "demo" / "Index.md"
    assert result.index_path.exists()
    assert result.notebooklm_source_path.exists()

    first_note = result.block_paths["mem-a"]
    second_note = result.block_paths["mem-b"]
    assert first_note.name == "Scoped Recall - mem-a.md"
    assert "Core" in first_note.parts
    assert "Project knowledge" in first_note.parts
    assert second_note.name == "Isolation Rule - mem-b.md"

    first_text = first_note.read_text(encoding="utf-8")
    assert "workspace_id: demo" in first_text
    assert "surface: supplemental_evidence" in first_text
    assert "trust_level: workspace_verified" in first_text
    assert "Supports: [[Isolation Rule - mem-b]]" in first_text
    assert "Scoped recall supports isolation." in first_text

    index_text = result.index_path.read_text(encoding="utf-8")
    assert "# Demo Project memory" in index_text
    assert "[[Core/Core Index|Core memory]]" in index_text
    assert "[[Evidence/Evidence Index|Evidence / session recall]]" in index_text
    assert "[[Scoped Recall - mem-a]]" not in index_text

    core_index_text = (
        tmp_path / "Lethyx" / "Projects" / "demo" / "Core" / "Core Index.md"
    ).read_text(encoding="utf-8")
    assert "[[Core/Project knowledge/Project knowledge Index|Project knowledge]]" in core_index_text
    area_hub_text = (
        tmp_path
        / "Lethyx"
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

    notebooklm_text = result.notebooklm_source_path.read_text(encoding="utf-8")
    assert "# Demo Project memory overview" in notebooklm_text
    assert "Use scoped recall for Project Alpha." in notebooklm_text


def test_obsidian_export_is_workspace_scoped_by_default(tmp_path: Path) -> None:
    demo = _block("demo-memory", "Demo workspace memory.", workspace_id="demo")
    other = _block("other-memory", "Other workspace memory.", workspace_id="other")
    manager = _block("manager-memory", "Global manager memory.", workspace_id="")

    result = export_records_to_obsidian(
        blocks=[demo, other, manager],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
    )

    assert set(result.block_paths) == {"demo-memory"}
    text = result.index_path.read_text(encoding="utf-8")
    assert "Demo workspace memory." in (
        tmp_path
        / "Lethyx"
        / "Projects"
        / "demo"
        / "Core"
        / "Project knowledge"
        / "demo-memory - demo-memory.md"
    ).read_text(encoding="utf-8")
    assert "Other workspace memory" not in text
    assert "Global manager memory" not in text


def test_obsidian_export_keeps_session_evidence_out_of_core_memory_tree(tmp_path: Path) -> None:
    core = _block("core-rule", "Always use scoped recall for this project.", title="Core Rule")
    session = _block(
        "session_summary_20260627_210403_045200",
        "Raw-ish session summary should remain recall evidence, not a top-level core memory note.",
        title="Session Summary",
        kind="session_summary",
        tags=("session-migration", "tier:hot"),
    )

    result = export_records_to_obsidian(
        blocks=[core, session],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
    )

    assert set(result.block_paths) == {"core-rule"}
    assert not (tmp_path / "Lethyx" / "Projects" / "demo" / "Core" / "Session summaries").exists()
    index_text = result.index_path.read_text(encoding="utf-8")
    assert "Core memory blocks: **1**" in index_text
    assert "Evidence/archive blocks kept in backend: **1**" in index_text
    archive_text = (tmp_path / "Lethyx" / "Projects" / "demo" / "Evidence Archive.md").read_text(
        encoding="utf-8"
    )
    assert "[[Evidence/Evidence Index|Evidence Index]]" in archive_text
    assert "[[Session Summary - session_summary_20260627_210403_045200]]" not in archive_text

    evidence_index_text = (
        tmp_path / "Lethyx" / "Projects" / "demo" / "Evidence" / "Evidence Index.md"
    ).read_text(encoding="utf-8")
    assert "[[Evidence/Sessions/Evidence Sessions Index|Sessions]]" in evidence_index_text
    sessions_hub_text = (
        tmp_path
        / "Lethyx"
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
        / "Lethyx"
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
        / "Lethyx"
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
        lifecycle="candidate",
    )
    candidate = _block(
        "candidate_feature_note",
        "Potential project feature inferred from session evidence.",
        title="Candidate feature note",
        kind="hypothesis",
        lifecycle="candidate",
    )

    export_records_to_obsidian(
        blocks=[user_intent, candidate],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
    )

    root = tmp_path / "Lethyx" / "Projects" / "demo"
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


def test_obsidian_export_removes_nested_project_obsidian_state(tmp_path: Path) -> None:
    stale = tmp_path / "Lethyx" / "Projects" / "demo" / ".obsidian" / "workspace.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("{}", encoding="utf-8")

    export_records_to_obsidian(
        blocks=[_block("core-rule", "Core memory only.", title="Core Rule")],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
    )

    assert not (tmp_path / "Lethyx" / "Projects" / "demo" / ".obsidian").exists()


def test_obsidian_export_can_include_global_blocks_explicitly(tmp_path: Path) -> None:
    demo = _block("demo-memory", "Demo workspace memory.", workspace_id="demo")
    manager = _block("manager-memory", "Global manager memory.", workspace_id="")

    result = export_records_to_obsidian(
        blocks=[demo, manager],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
        include_global=True,
    )

    assert set(result.block_paths) == {"demo-memory", "manager-memory"}
    assert "Global manager memory." in result.block_paths["manager-memory"].read_text(
        encoding="utf-8"
    )


def test_obsidian_export_excludes_non_exportable_privacy_classes(tmp_path: Path) -> None:
    public = _block("public-memory", "Public memory can be exported.")
    secret = _block(
        "secret-memory",
        "Secret pointer must not leave local storage.",
        privacy_class="secret_pointer",
    )
    local_only = _block(
        "local-only-memory",
        "Non-injectable memory must stay out of the bundle.",
        privacy_class="non-injectable",
    )

    result = export_records_to_obsidian(
        blocks=[public, secret, local_only],
        edges=[],
        vault_dir=tmp_path,
        workspace_id="demo",
    )

    assert set(result.block_paths) == {"public-memory"}
    bundle = result.notebooklm_source_path.read_text(encoding="utf-8")
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

        def edges_for(self, block_id: str) -> list[MemoryEdgeRecord]:
            assert block_id == "cli-memory"
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
    assert (tmp_path / "vault" / "Lethyx" / "Projects" / "demo" / "Index.md").exists()
    assert (tmp_path / "vault" / "Lethyx" / "Projects" / "demo" / "Memory Overview.md").exists()
    assert not (tmp_path / "vault" / "Lethyx" / "Projects" / "demo" / "NotebookLM").exists()
