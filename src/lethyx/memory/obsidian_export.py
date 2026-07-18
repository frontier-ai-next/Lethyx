"""Export Lethyx memory records to an Obsidian-compatible live vault.

The Obsidian vault is the human-facing surface of the real Lethyx runtime
memory. It must not look like a migration log or a backend dump: project folders
contain readable memory areas, typed notes, and a compact index. Runtime
authority still lives in the Lethyx metadata preserved in YAML frontmatter.
"""

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from lethyx.memory.backends.contracts import MemoryBlockRecord, MemoryEdgeRecord

_HIDDEN_EXPORT_LIFECYCLES = {"retracted", "quarantined", "superseded", "deprecated"}
_HIDDEN_EXPORT_KINDS = {"source_index"}
_HIDDEN_EXPORT_TAGS = {"source-node"}
_HIDDEN_EXPORT_PRIVACY_CLASSES = {"secret", "secret_pointer", "non-injectable", "non_injectable"}
_SERVICE_DIRS = {"Graphs", "NotebookLM", "Notes", ".obsidian"}
_CANONICAL_CORE_DIR = "Core"
_LEGACY_CORE_DIRS = {"Memory"}
_LEGACY_ROOT_FILES = {"Core Memory Index.md", "Untitled.canvas"}
_EVIDENCE_AREAS = {"Session summaries", "Candidate ideas", "Archive"}
_CANONICAL_CORE_AREAS = (
    "Project knowledge",
    "User profile",
    "Agent profile memory",
    "Skills",
    "Tools and MCP",
    "Project documents",
    "Other memory",
)
_CANONICAL_EVIDENCE_CATEGORIES = (
    "Sessions",
    "Turns",
    "User intents",
    "Candidate ideas",
    "Outcomes",
    "Questions",
    "Archive",
)

_AREA_ORDER = {
    "User profile": 10,
    "Agent profile memory": 20,
    "Project knowledge": 30,
    "Session summaries": 40,
    "Skills": 50,
    "Tools and MCP": 60,
    "Project documents": 70,
    "Candidate ideas": 80,
    "Archive": 90,
    "Other memory": 100,
}


@dataclass(frozen=True)
class ObsidianExportResult:
    """Files written by an Obsidian export run."""

    vault_dir: Path
    workspace_id: str
    index_path: Path
    # Kept for API compatibility with older callers/tests; this is now a clean
    # project overview note, not a NotebookLM/service export.
    notebooklm_source_path: Path
    block_paths: dict[str, Path]
    written_files: tuple[Path, ...]


def _safe_segment(value: str, *, fallback: str = "untitled") -> str:
    """Return a Windows/Obsidian-friendly path segment."""

    text = str(value or "").strip()
    out: list[str] = []
    previous_space = False
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            out.append(char)
            previous_space = False
        elif char.isspace():
            if not previous_space:
                out.append(" ")
                previous_space = True
        else:
            if not previous_space:
                out.append(" ")
                previous_space = True
    clean = "".join(out).strip(" ._-")
    while "  " in clean:
        clean = clean.replace("  ", " ")
    return clean or fallback


def _safe_id(value: str) -> str:
    return _safe_segment(value, fallback="memory").replace(" ", "-")


def _wikilink(target: str, alias: str | None = None) -> str:
    clean = str(target).replace("\\", "/").removesuffix(".md")
    if alias and alias != clean:
        return f"[[{clean}|{alias}]]"
    return f"[[{clean}]]"


def _note_target(path: Path, *, project_root: Path) -> str:
    rel = path.relative_to(project_root).with_suffix("")
    return rel.as_posix()


def _session_key(block: MemoryBlockRecord) -> str:
    block_id = str(block.block_id or "")
    if block_id.startswith("session_summary_"):
        return block_id.removeprefix("session_summary_")
    if block_id.startswith("session_"):
        return block_id.removeprefix("session_")
    if block_id.startswith("turn_"):
        parts = block_id.split("_")
        if len(parts) >= 4:
            return "_".join(parts[1:4])
    for value in (
        block.run_id,
        str((block.metadata or {}).get("session_id") or ""),
        str((block.metadata or {}).get("source_session_id") or ""),
    ):
        match = re.search(r"(\d{8}_\d{6}_[A-Za-z0-9]+)", value)
        if match:
            return match.group(1)
    return ""


def _session_hub_name(session_key: str) -> str:
    return f"Session {_safe_segment(session_key.replace('_', ' '), fallback='unknown')}"


def _core_area_hub_name(area: str) -> str:
    return f"{_safe_segment(area)} Index"


def _evidence_category_hub_name(category: str) -> str:
    return f"Evidence {_safe_segment(category.replace('/', ' '))} Index"


def _tag_set(block: MemoryBlockRecord) -> set[str]:
    return {str(tag).strip().lower() for tag in (block.tags or ()) if str(tag).strip()}


def _metadata_text(block: MemoryBlockRecord, key: str) -> str:
    return str((block.metadata or {}).get(key) or "").strip().lower()


def _effective_tier(block: MemoryBlockRecord) -> str:
    meta_tier = _metadata_text(block, "tier")
    if meta_tier:
        return meta_tier
    for tag in block.tags or ():
        text = str(tag or "").strip().lower()
        if text.startswith("tier:") and text.split(":", 1)[1].strip():
            return text.split(":", 1)[1].strip()
    return str(block.tier or "warm").strip().lower() or "warm"


def memory_area_for_block(block: MemoryBlockRecord) -> str:
    """Return the human-facing memory area for a record."""

    tags = _tag_set(block)
    block_id = str(block.block_id or "").lower()
    kind = str(block.kind or "").lower()
    surface = str(block.surface or "").lower()
    lifecycle = str(block.lifecycle or "").lower()
    tier = _effective_tier(block)

    if block_id.startswith("user_profile_") or "target:user" in tags:
        return "User profile"
    if block_id.startswith("profile_memory_") or "target:memory" in tags:
        return "Agent profile memory"
    if "session" in kind or "session-migration" in tags or block_id.startswith("session_"):
        return "Session summaries"
    if "skill" in kind or "skill-index" in tags or "skill" in tags:
        return "Skills"
    if "mcp" in kind or "mcp-inventory" in tags or block_id.startswith("mcp_inventory"):
        return "Tools and MCP"
    if "doc" in kind or "project-doc" in tags or "project-doc-pointer" in tags:
        return "Project documents"
    if (
        lifecycle == "candidate"
        or surface == "reflection_candidate"
        or kind
        in {
            "idea",
            "hypothesis",
            "research_lead",
        }
    ):
        return "Candidate ideas"
    if lifecycle == "archived" or surface == "archive_hint" or tier == "cold":
        return "Archive"
    if kind in {
        "lesson",
        "durable",
        "preference",
        "fact",
        "project_fact",
        "pattern",
        "architecture",
        "decision",
    }:
        return "Project knowledge"
    return "Other memory"


def _is_evidence_archive_block(block: MemoryBlockRecord) -> bool:
    area = memory_area_for_block(block)
    if area in _EVIDENCE_AREAS:
        return True
    kind = str(block.kind or "").lower()
    surface = str(block.surface or "").lower()
    lifecycle = str(block.lifecycle or "").lower()
    return (
        "session" in kind
        or lifecycle in {"candidate", "archived"}
        or surface in {"archive_hint", "reflection_candidate"}
        or _effective_tier(block) in {"cold", "transient"}
    )


def note_name_for_block(block: MemoryBlockRecord) -> str:
    """Return a stable Obsidian note stem for a memory block."""

    block_id = _safe_id(block.block_id)
    title = _safe_segment(block.title or block.block_id, fallback=block_id)
    if title == block_id:
        return f"{block_id} - {block_id}"
    return f"{title} - {block_id}"


def _yaml_scalar(value: object) -> str:
    text = str(value or "")
    if not text:
        return ""
    if any(char in text for char in [":", "#", "[", "]", "{", "}", "\n", "\r", '"']):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _frontmatter(block: MemoryBlockRecord, *, area: str) -> str:
    tier = _effective_tier(block)
    lines = [
        "---",
        f"id: {_yaml_scalar(block.block_id)}",
        f"title: {_yaml_scalar(block.title)}",
        f"workspace_id: {_yaml_scalar(block.workspace_id)}",
        f"area: {_yaml_scalar(area)}",
        f"kind: {_yaml_scalar(block.kind)}",
        f"tier: {_yaml_scalar(tier)}",
        f"scope: {_yaml_scalar(block.scope)}",
        f"surface: {_yaml_scalar(block.surface)}",
        f"lifecycle: {_yaml_scalar(block.lifecycle)}",
        f"trust_level: {_yaml_scalar(block.trust_level)}",
        f"source_backend: {_yaml_scalar(block.source_backend)}",
        f"privacy_class: {_yaml_scalar(block.privacy_class)}",
        f"verified: {str(bool(block.verified)).lower()}",
        "tags:",
    ]
    if block.tags:
        lines.extend(f"  - {_yaml_scalar(tag)}" for tag in block.tags)
    else:
        lines.append("  []")
    if block.run_id:
        lines.append(f"run_id: {_yaml_scalar(block.run_id)}")
    if block.phase_id:
        lines.append(f"phase_id: {_yaml_scalar(block.phase_id)}")
    if block.subtask_id:
        lines.append(f"subtask_id: {_yaml_scalar(block.subtask_id)}")
    if block.evidence_refs:
        lines.append("evidence_refs:")
        for ref in block.evidence_refs:
            lines.append(f"  - type: {_yaml_scalar(getattr(ref, 'ref_type', ''))}")
            lines.append(f"    id: {_yaml_scalar(getattr(ref, 'ref_id', ''))}")
            produced_by = getattr(ref, "produced_by", "")
            if produced_by:
                lines.append(f"    produced_by: {_yaml_scalar(produced_by)}")
    lines.append("---")
    return "\n".join(lines)


def _edge_label(edge_type: str) -> str:
    return str(edge_type or "related").replace("_", " ").strip().title() or "Related"


def _block_in_scope(block: MemoryBlockRecord, *, workspace_id: str, include_global: bool) -> bool:
    if str(block.lifecycle or "").lower() in _HIDDEN_EXPORT_LIFECYCLES:
        return False
    if str(block.kind or "").lower() in _HIDDEN_EXPORT_KINDS:
        return False
    if str(block.privacy_class or "").strip().lower() in _HIDDEN_EXPORT_PRIVACY_CLASSES:
        return False
    if any(str(tag).lower() in _HIDDEN_EXPORT_TAGS for tag in (block.tags or ())):
        return False
    if block.workspace_id == workspace_id:
        return True
    if include_global and (
        (not block.workspace_id and block.scope in {"manager", "cross_run_durable"})
        or block.workspace_id in {"global", "shared"}
    ):
        return True
    return False


def _block_preference_key(block: MemoryBlockRecord) -> tuple[int, int, int, str]:
    lifecycle_rank = {"active": 0, "candidate": 1, "archived": 2}.get(block.lifecycle, 3)
    verified_rank = 0 if block.verified else 1
    backend_rank = {"mempalace": 0, "hindsight": 1, "mnemosyne": 2}.get(block.source_backend, 3)
    return (lifecycle_rank, verified_rank, backend_rank, block.updated_at or block.created_at or "")


def _dedupe_blocks(blocks: Iterable[MemoryBlockRecord]) -> list[MemoryBlockRecord]:
    by_id: dict[str, MemoryBlockRecord] = {}
    for block in blocks:
        key = block.block_id or block.content
        current = by_id.get(key)
        if current is None or _block_preference_key(block) < _block_preference_key(current):
            by_id[key] = block
    return list(by_id.values())


def _scoped_blocks(
    blocks: Iterable[MemoryBlockRecord], *, workspace_id: str, include_global: bool
) -> list[MemoryBlockRecord]:
    selected = [
        block
        for block in blocks
        if _block_in_scope(block, workspace_id=workspace_id, include_global=include_global)
    ]
    return _dedupe_blocks(selected)


def _dedupe_edges(edges: Sequence[MemoryEdgeRecord]) -> list[MemoryEdgeRecord]:
    seen: set[tuple[str, str, str, str]] = set()
    rows: list[MemoryEdgeRecord] = []
    for edge in edges:
        key = (str(edge.src_id), str(edge.dst_id), str(edge.edge_type), str(edge.reason or ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(edge)
    return rows


def _edges_for_block(
    block_id: str,
    edges: Sequence[MemoryEdgeRecord],
    note_names: dict[str, str],
) -> tuple[list[MemoryEdgeRecord], list[MemoryEdgeRecord]]:
    unique_edges = _dedupe_edges(edges)
    outgoing = [
        edge for edge in unique_edges if edge.src_id == block_id and edge.dst_id in note_names
    ]
    incoming = [
        edge for edge in unique_edges if edge.dst_id == block_id and edge.src_id in note_names
    ]
    return outgoing, incoming


def _backend_sentence(block: MemoryBlockRecord) -> str:
    role = _metadata_text(block, "memory_block_role") or "active"
    physical = block.source_backend or _metadata_text(block, "memory_block_backend") or "unknown"
    logical = _metadata_text(block, "memory_block_id") or physical
    return f"{physical} / {logical} / {role}"


def _render_block_note(
    block: MemoryBlockRecord,
    *,
    edges: Sequence[MemoryEdgeRecord],
    note_names: dict[str, str],
) -> str:
    area = memory_area_for_block(block)
    title = block.title or block.block_id
    tier = _effective_tier(block)
    lines = [
        _frontmatter(block, area=area),
        "",
        f"# {title}",
        "",
        f"> Area: **{area}** · Tier: `{tier}` · Lifecycle: `{block.lifecycle}` · Trust: `{block.trust_level}`",
        "",
        "## Memory",
        "",
        block.content.strip(),
        "",
        "## Runtime semantics",
        "",
        f"- Workspace: `{block.workspace_id or 'global/manager'}`",
        f"- Kind: `{block.kind}`",
        f"- Scope: `{block.scope}`",
        f"- Surface: `{block.surface}`",
        f"- Backend block: `{_backend_sentence(block)}`",
        f"- Verified: `{str(bool(block.verified)).lower()}`",
    ]
    if block.evidence_refs:
        lines.extend(["", "## Evidence", ""])
        for ref in block.evidence_refs:
            lines.append(
                "- "
                + ":".join(
                    part
                    for part in [
                        str(getattr(ref, "ref_type", "") or "evidence"),
                        str(getattr(ref, "ref_id", "") or ""),
                    ]
                    if part
                )
            )
    outgoing, incoming = _edges_for_block(block.block_id, edges, note_names)
    if outgoing or incoming:
        lines.extend(["", "## Related memory", ""])
        for edge in outgoing:
            line = f"- {_edge_label(edge.edge_type)}: [[{note_names[edge.dst_id]}]]"
            if edge.reason:
                line += f" — {edge.reason}"
            lines.append(line)
        for edge in incoming:
            line = f"- {_edge_label(edge.edge_type)} by: [[{note_names[edge.src_id]}]]"
            if edge.reason:
                line += f" — {edge.reason}"
            lines.append(line)
    public_metadata = {
        key: value
        for key, value in sorted((block.metadata or {}).items())
        if value not in (None, "")
        and key
        not in {
            "memory_block_rank",
            "memory_block_rrf_score",
            "raw_backend_score",
            "retrieval_score",
            "native_score",
        }
    }
    if public_metadata:
        lines.extend(["", "## Details", ""])
        for key, value in public_metadata.items():
            lines.append(f"- `{key}`: {value}")
    return "\n".join(lines).rstrip() + "\n"


def _area_groups(blocks: Sequence[MemoryBlockRecord]) -> dict[str, list[MemoryBlockRecord]]:
    groups: dict[str, list[MemoryBlockRecord]] = {}
    for block in blocks:
        groups.setdefault(memory_area_for_block(block), []).append(block)
    return groups


def _sort_blocks_for_index(
    blocks: Sequence[MemoryBlockRecord], note_names: dict[str, str]
) -> list[MemoryBlockRecord]:
    return sorted(
        blocks,
        key=lambda block: (
            {"always_on": 0, "hot": 1, "warm": 2, "cold": 3, "transient": 4}.get(
                _effective_tier(block), 9
            ),
            {"active": 0, "candidate": 1, "archived": 2}.get(block.lifecycle, 9),
            note_names.get(block.block_id, block.block_id).lower(),
        ),
    )


def _render_index(
    *,
    project_title: str,
    blocks: Sequence[MemoryBlockRecord],
    note_names: dict[str, str],
    evidence_blocks: Sequence[MemoryBlockRecord] = (),
) -> str:
    tier_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    lifecycle_counts: dict[str, int] = {}
    for block in blocks:
        tier = _effective_tier(block)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        backend_counts[block.source_backend or "unknown"] = (
            backend_counts.get(block.source_backend or "unknown", 0) + 1
        )
        lifecycle_counts[block.lifecycle] = lifecycle_counts.get(block.lifecycle, 0) + 1

    lines = [
        f"# {project_title} memory",
        "",
        "This is the workspace entry point. It links only to hub notes and the canvas so the Obsidian graph remains hierarchical.",
        "",
        "## Navigation",
        "",
        f"- {_wikilink('Core/Core Index', 'Core memory')}",
        f"- {_wikilink('Evidence/Evidence Index', 'Evidence / session recall')}",
        f"- {_wikilink('Memory Overview', 'Readable overview')}",
        f"- {_wikilink('Memory Map.canvas', 'Colored memory canvas')}",
        "",
        "## Snapshot",
        "",
        f"- Core memory blocks: **{len(blocks)}**",
        f"- Evidence/archive blocks kept in backend: **{len(evidence_blocks)}**",
        f"- Tiers: {', '.join(f'{key}={value}' for key, value in sorted(tier_counts.items())) or 'none'}",
        f"- Lifecycles: {', '.join(f'{key}={value}' for key, value in sorted(lifecycle_counts.items())) or 'none'}",
        f"- Backends: {', '.join(f'{key}={value}' for key, value in sorted(backend_counts.items())) or 'none'}",
        "",
        "## Authority rule",
        "",
        "Evidence/session notes are recall and audit material only. Stable project facts should appear under Core memory after promotion/upsert.",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _render_project_overview(
    *,
    project_title: str,
    blocks: Sequence[MemoryBlockRecord],
    note_names: dict[str, str],
    evidence_blocks: Sequence[MemoryBlockRecord] = (),
) -> str:
    lines = [
        f"# {project_title} memory overview",
        "",
        "A compact human-readable snapshot of the current Lethyx workspace memory.",
        "",
    ]
    for area in sorted(_area_groups(blocks), key=lambda item: (_AREA_ORDER.get(item, 999), item)):
        lines.extend([f"## {area}", ""])
        for block in _sort_blocks_for_index(_area_groups(blocks)[area], note_names)[:30]:
            text = " ".join(block.content.strip().split())
            if len(text) > 240:
                text = text[:237].rstrip() + "..."
            lines.append(f"- **{note_names[block.block_id]}** — {text}")
        lines.append("")
    if evidence_blocks:
        lines.extend(
            [
                "## Evidence archive",
                "",
                f"{len(evidence_blocks)} source-backed session/candidate/archive records are kept in the backend for recall and audit. They are summarized in [[Evidence Archive]] instead of being listed as core memory.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _evidence_category(block: MemoryBlockRecord) -> str:
    kind = str(block.kind or "").lower()
    title = str(block.title or "").lower()
    block_id = str(block.block_id or "").lower()
    if (
        "session_summary" in kind
        or kind == "session_summary"
        or block_id.startswith("session_summary")
        or block_id.startswith("session_")
    ):
        return "Sessions"
    if kind == "session_turn" or title.startswith("conversation turn"):
        return "Turns"
    if kind == "user_intent":
        return "User intents"
    if kind == "assistant_outcome":
        return "Outcomes"
    if kind == "open_question":
        return "Questions"
    if (
        str(block.lifecycle or "").lower() == "archived"
        or str(block.surface or "").lower() == "archive_hint"
    ):
        return "Archive"
    return "Candidate ideas"


def _render_evidence_note(block: MemoryBlockRecord, *, session_hub_target: str = "") -> str:
    area = f"Evidence/{_evidence_category(block)}"
    lines = [
        _frontmatter(block, area=area),
        "",
        f"# {block.title or block.block_id}",
        "",
        f"> Evidence category: **{_evidence_category(block)}** · Tier: `{_effective_tier(block)}` · Lifecycle: `{block.lifecycle}` · Surface: `{block.surface}` · Backend: `{block.source_backend or 'unknown'}`",
        "",
        "This note is source/candidate/session evidence. It can help recall and audit, but it is not a directive or stable project fact until promoted into core memory.",
        "",
        "## Content",
        "",
        block.content.strip() or "_(empty)_",
        "",
    ]
    if session_hub_target:
        lines.extend(["## Navigation", "", f"- Session hub: {_wikilink(session_hub_target)}", ""])
    if block.metadata:
        visible = {k: v for k, v in block.metadata.items() if k not in {"raw", "payload"}}
        if visible:
            lines.extend(["## Metadata", ""])
            for key in sorted(visible):
                lines.append(f"- **{key}**: `{visible[key]}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _canonical_core_areas(groups: dict[str, list[MemoryBlockRecord]]) -> list[str]:
    seen = list(_CANONICAL_CORE_AREAS)
    for area in groups:
        if area not in seen:
            seen.append(area)
    return sorted(seen, key=lambda item: (_AREA_ORDER.get(item, 999), item))


def _canonical_evidence_categories(groups: dict[str, list[MemoryBlockRecord]]) -> list[str]:
    seen = list(_CANONICAL_EVIDENCE_CATEGORIES)
    for category in groups:
        if category not in seen:
            seen.append(category)
    return sorted(seen)


def _render_core_index(
    *,
    project_title: str,
    blocks: Sequence[MemoryBlockRecord],
    note_names: dict[str, str],
    block_targets: dict[str, str],
    area_hub_targets: dict[str, str],
) -> str:
    groups = _area_groups(blocks)
    lines = [
        f"# {project_title} core memory index",
        "",
        "Core memory is the compact, stable project/profile memory surface. It is separated from raw session evidence to keep the graph readable and authority boundaries clear.",
        "",
    ]
    if not blocks:
        lines.extend(
            [
                "No promoted/core memory blocks for this workspace yet. The canonical categories are still listed below so the vault has a stable visual shape.",
                "",
            ]
        )
    for area in _canonical_core_areas(groups):
        lines.append(f"- {_wikilink(area_hub_targets[area], area)} ({len(groups.get(area, []))})")
    return "\n".join(lines).rstrip() + "\n"


def _render_core_area_hub(
    *,
    area: str,
    blocks: Sequence[MemoryBlockRecord],
    note_names: dict[str, str],
    block_targets: dict[str, str],
) -> str:
    lines = [
        f"# {area}",
        "",
        f"Core area hub for **{area}**. Leaf notes are linked below; project indexes link here instead of linking to every leaf directly.",
        "",
    ]
    if not blocks:
        lines.append("No blocks in this category yet.")
        return "\n".join(lines).rstrip() + "\n"
    for block in _sort_blocks_for_index(blocks, note_names):
        suffix = f"`{_effective_tier(block)}` / `{block.lifecycle}` / `{block.source_backend or 'unknown'}`"
        lines.append(
            f"- {_wikilink(block_targets[block.block_id], note_names[block.block_id])} — {suffix}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_evidence_index(
    *,
    project_title: str,
    blocks: Sequence[MemoryBlockRecord],
    category_hub_targets: dict[str, str],
) -> str:
    groups: dict[str, list[MemoryBlockRecord]] = {}
    for block in blocks:
        groups.setdefault(_evidence_category(block), []).append(block)
    lines = [
        f"# {project_title} evidence index",
        "",
        "Evidence is source/candidate/session material used for recall and audit. It is non-directive until promoted into core memory.",
        "",
        f"Total evidence blocks: **{len(blocks)}**",
        "",
    ]
    if not blocks:
        lines.extend(
            [
                "No evidence/archive records for this workspace yet. The canonical evidence categories are still listed below so the vault has a stable visual shape.",
                "",
            ]
        )
    for category in _canonical_evidence_categories(groups):
        lines.append(
            f"- {_wikilink(category_hub_targets[category], category)} ({len(groups.get(category, []))})"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_evidence_category_hub(
    *,
    category: str,
    blocks: Sequence[MemoryBlockRecord],
    note_names: dict[str, str],
    block_targets: dict[str, str],
    session_hub_targets: dict[str, str],
) -> str:
    by_session: dict[str, list[MemoryBlockRecord]] = {}
    direct: list[MemoryBlockRecord] = []
    for block in blocks:
        session = _session_key(block)
        if session:
            by_session.setdefault(session, []).append(block)
        else:
            direct.append(block)
    lines = [
        f"# {category}",
        "",
        "Category hub. It links to session hubs first so the graph stays hierarchical instead of becoming a leaf dandelion.",
        "",
    ]
    for session in sorted(by_session):
        lines.append(
            f"- {_wikilink(session_hub_targets[session], _session_hub_name(session))} ({len(by_session[session])})"
        )
    if not by_session and not direct:
        lines.append("No evidence in this category yet.")
    if direct:
        lines.extend(["", "## Direct evidence", ""])
        for block in _sort_blocks_for_index(direct, note_names):
            lines.append(
                f"- {_wikilink(block_targets[block.block_id], note_names[block.block_id])}"
            )
    return "\n".join(lines).rstrip() + "\n"


def _render_session_hub(
    *,
    session_key: str,
    blocks: Sequence[MemoryBlockRecord],
    note_names: dict[str, str],
    block_targets: dict[str, str],
) -> str:
    groups: dict[str, list[MemoryBlockRecord]] = {}
    for block in blocks:
        groups.setdefault(_evidence_category(block), []).append(block)
    lines = [
        f"# {_session_hub_name(session_key)}",
        "",
        "Session hub. Open this first for a compact view of one agent-memory episode; leaf evidence hangs from this node, not from the project index.",
        "",
    ]
    for category in sorted(groups):
        lines.extend([f"## {category}", ""])
        for block in _sort_blocks_for_index(groups[category], note_names):
            text = " ".join(block.content.strip().split())
            if len(text) > 160:
                text = text[:157].rstrip() + "..."
            lines.append(
                f"- {_wikilink(block_targets[block.block_id], note_names[block.block_id])} — {text}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_evidence_archive(
    *,
    project_title: str,
    blocks: Sequence[MemoryBlockRecord],
    evidence_index_target: str,
) -> str:
    lines = [
        f"# {project_title} evidence archive",
        "",
        "This legacy-facing note intentionally links only to the evidence index, not to every evidence leaf. Use the hub hierarchy to avoid an Obsidian graph dandelion.",
        "",
        f"- {_wikilink(evidence_index_target, 'Evidence Index')}",
        f"- Total evidence blocks: **{len(blocks)}**",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _render_memory_canvas(
    *,
    project_title: str,
    core_count: int,
    core_counts: dict[str, int],
    evidence_counts: dict[str, int],
) -> str:
    nodes = [
        {
            "id": "project",
            "type": "text",
            "text": f"{project_title}\nLethyx workspace",
            "x": 0,
            "y": 0,
            "width": 260,
            "height": 120,
            "color": "1",
        },
        {
            "id": "core",
            "type": "text",
            "text": f"Core memory\n{core_count} blocks",
            "x": -340,
            "y": 190,
            "width": 260,
            "height": 120,
            "color": "4",
        },
        {
            "id": "evidence",
            "type": "text",
            "text": f"Evidence / recall audit\n{sum(evidence_counts.values())} blocks",
            "x": 340,
            "y": 190,
            "width": 300,
            "height": 120,
            "color": "2",
        },
    ]
    edges = [
        {
            "id": "project-core",
            "fromNode": "project",
            "fromSide": "bottom",
            "toNode": "core",
            "toSide": "top",
            "color": "4",
        },
        {
            "id": "project-evidence",
            "fromNode": "project",
            "fromSide": "bottom",
            "toNode": "evidence",
            "toSide": "top",
            "color": "2",
        },
    ]
    x = -620
    for idx, (area, count) in enumerate(
        sorted(core_counts.items(), key=lambda item: (_AREA_ORDER.get(item[0], 999), item[0]))
    ):
        node_id = f"core-{idx}"
        nodes.append(
            {
                "id": node_id,
                "type": "text",
                "text": f"{area}\n{count}",
                "x": x + idx * 260,
                "y": 390,
                "width": 230,
                "height": 100,
                "color": "4",
            }
        )
        edges.append(
            {
                "id": f"core-{idx}-edge",
                "fromNode": "core",
                "fromSide": "bottom",
                "toNode": node_id,
                "toSide": "top",
                "color": "4",
            }
        )
    x = 120
    for idx, (category, count) in enumerate(sorted(evidence_counts.items())):
        node_id = f"evidence-{idx}"
        nodes.append(
            {
                "id": node_id,
                "type": "text",
                "text": f"{category}\n{count}",
                "x": x + idx * 260,
                "y": 390,
                "width": 230,
                "height": 100,
                "color": str((idx % 5) + 1),
            }
        )
        edges.append(
            {
                "id": f"evidence-{idx}-edge",
                "fromNode": "evidence",
                "fromSide": "bottom",
                "toNode": node_id,
                "toSide": "top",
                "color": str((idx % 5) + 1),
            }
        )
    return json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2) + "\n"


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _render_projects_index(projects_root: Path) -> str:
    projects = (
        [
            path.name
            for path in sorted(projects_root.iterdir(), key=lambda p: p.name.lower())
            if path.is_dir() and not path.name.startswith(".") and (path / "Index.md").exists()
        ]
        if projects_root.exists()
        else []
    )
    lines = [
        "# Lethyx workspaces",
        "",
        "Open a workspace index first, then drill into Core or Evidence.",
        "",
    ]
    if not projects:
        lines.append("No exported workspaces yet.")
    else:
        for project in projects:
            lines.append(f"- [[Lethyx/Projects/{project}/Index|{project}]]")
    return "\n".join(lines).rstrip() + "\n"


def export_records_to_obsidian(
    *,
    blocks: Iterable[MemoryBlockRecord],
    edges: Iterable[MemoryEdgeRecord] = (),
    vault_dir: str | Path,
    workspace_id: str,
    project_title: str = "",
    include_global: bool = False,
) -> ObsidianExportResult:
    """Write scoped Lethyx records into a clean Obsidian project folder.

    `include_global` is intentionally false by default so manager/global memory
    never leaks into a project export without explicit caller intent.
    """

    vault = Path(vault_dir)
    workspace = _safe_id(workspace_id or "default")
    title = project_title or workspace_id or "Lethyx"
    lethyx_root = vault / "Lethyx"
    project_root = lethyx_root / "Projects" / workspace
    core_dir = project_root / _CANONICAL_CORE_DIR
    evidence_dir = project_root / "Evidence"

    # Remove old experimental/service surfaces before writing the canonical view.
    _remove_tree(lethyx_root / "Graphs")
    for dirname in _SERVICE_DIRS:
        _remove_tree(project_root / dirname)
    for dirname in _LEGACY_CORE_DIRS:
        _remove_tree(project_root / dirname)
    for filename in _LEGACY_ROOT_FILES:
        (project_root / filename).unlink(missing_ok=True)
    _remove_tree(core_dir)
    _remove_tree(evidence_dir)

    core_dir.mkdir(parents=True, exist_ok=True)

    selected_blocks = _scoped_blocks(
        list(blocks), workspace_id=workspace_id, include_global=include_global
    )
    core_blocks = [block for block in selected_blocks if not _is_evidence_archive_block(block)]
    evidence_blocks = [block for block in selected_blocks if _is_evidence_archive_block(block)]
    selected_edges = list(edges)
    core_note_names = {block.block_id: note_name_for_block(block) for block in core_blocks}
    evidence_note_names = {block.block_id: note_name_for_block(block) for block in evidence_blocks}
    note_names = {**core_note_names, **evidence_note_names}

    block_paths: dict[str, Path] = {}
    written: list[Path] = []

    core_block_targets: dict[str, str] = {}
    core_area_hub_targets: dict[str, str] = {}
    core_groups_for_targets = _area_groups(core_blocks)
    for area in _canonical_core_areas(core_groups_for_targets):
        hub_path = core_dir / _safe_segment(area) / f"{_core_area_hub_name(area)}.md"
        core_area_hub_targets[area] = _note_target(hub_path, project_root=project_root)
    for block in core_blocks:
        area = _safe_segment(memory_area_for_block(block))
        path = core_dir / area / f"{note_names[block.block_id]}.md"
        core_block_targets[block.block_id] = _note_target(path, project_root=project_root)

    evidence_block_targets: dict[str, str] = {}
    evidence_category_groups: dict[str, list[MemoryBlockRecord]] = {}
    evidence_session_groups: dict[str, list[MemoryBlockRecord]] = {}
    category_hub_targets: dict[str, str] = {}
    session_hub_targets: dict[str, str] = {}
    for block in evidence_blocks:
        category = _evidence_category(block)
        evidence_category_groups.setdefault(category, []).append(block)
        if "/" in category:
            parts = [_safe_segment(part) for part in category.split("/") if part]
            path = evidence_dir.joinpath(*parts) / f"{note_names[block.block_id]}.md"
        else:
            path = evidence_dir / _safe_segment(category) / f"{note_names[block.block_id]}.md"
        evidence_block_targets[block.block_id] = _note_target(path, project_root=project_root)
        session = _session_key(block)
        if session:
            evidence_session_groups.setdefault(session, []).append(block)
    for category in _canonical_evidence_categories(evidence_category_groups):
        if "/" in category:
            parts = [_safe_segment(part) for part in category.split("/") if part]
            hub_path = evidence_dir.joinpath(*parts) / f"{_evidence_category_hub_name(category)}.md"
        else:
            hub_path = (
                evidence_dir
                / _safe_segment(category)
                / f"{_evidence_category_hub_name(category)}.md"
            )
        category_hub_targets[category] = _note_target(hub_path, project_root=project_root)
    for session in evidence_session_groups:
        hub_path = evidence_dir / "Sessions" / f"{_session_hub_name(session)}.md"
        session_hub_targets[session] = _note_target(hub_path, project_root=project_root)

    stale_note_paths = {path for path in core_dir.rglob("*.md") if path.is_file()}
    for block in core_blocks:
        path = (project_root / core_block_targets[block.block_id]).with_suffix(".md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _render_block_note(block, edges=selected_edges, note_names=note_names),
            encoding="utf-8",
        )
        block_paths[block.block_id] = path
        written.append(path)
        stale_note_paths.discard(path)
    core_groups = _area_groups(core_blocks)
    for area in _canonical_core_areas(core_groups):
        items = core_groups.get(area, [])
        hub_path = (project_root / core_area_hub_targets[area]).with_suffix(".md")
        hub_path.parent.mkdir(parents=True, exist_ok=True)
        hub_path.write_text(
            _render_core_area_hub(
                area=area,
                blocks=items,
                note_names=note_names,
                block_targets=core_block_targets,
            ),
            encoding="utf-8",
        )
        written.append(hub_path)
        stale_note_paths.discard(hub_path)
    for stale_path in stale_note_paths:
        stale_path.unlink(missing_ok=True)
    for child in sorted(core_dir.rglob("*"), reverse=True):
        if child.is_dir():
            try:
                child.rmdir()
            except OSError:
                pass

    for block in evidence_blocks:
        path = (project_root / evidence_block_targets[block.block_id]).with_suffix(".md")
        path.parent.mkdir(parents=True, exist_ok=True)
        session = _session_key(block)
        path.write_text(
            _render_evidence_note(
                block,
                session_hub_target=session_hub_targets.get(session, "") if session else "",
            ),
            encoding="utf-8",
        )
        written.append(path)
    for category in _canonical_evidence_categories(evidence_category_groups):
        items = evidence_category_groups.get(category, [])
        hub_path = (project_root / category_hub_targets[category]).with_suffix(".md")
        hub_path.parent.mkdir(parents=True, exist_ok=True)
        hub_path.write_text(
            _render_evidence_category_hub(
                category=category,
                blocks=items,
                note_names=note_names,
                block_targets=evidence_block_targets,
                session_hub_targets=session_hub_targets,
            ),
            encoding="utf-8",
        )
        written.append(hub_path)
    for session, items in evidence_session_groups.items():
        hub_path = (project_root / session_hub_targets[session]).with_suffix(".md")
        hub_path.parent.mkdir(parents=True, exist_ok=True)
        hub_path.write_text(
            _render_session_hub(
                session_key=session,
                blocks=items,
                note_names=note_names,
                block_targets=evidence_block_targets,
            ),
            encoding="utf-8",
        )
        written.append(hub_path)

    core_index_path = core_dir / "Core Index.md"
    core_index_path.write_text(
        _render_core_index(
            project_title=title,
            blocks=core_blocks,
            note_names=note_names,
            block_targets=core_block_targets,
            area_hub_targets=core_area_hub_targets,
        ),
        encoding="utf-8",
    )
    written.append(core_index_path)

    evidence_index_path = evidence_dir / "Evidence Index.md"
    evidence_index_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_index_path.write_text(
        _render_evidence_index(
            project_title=title,
            blocks=evidence_blocks,
            category_hub_targets=category_hub_targets,
        ),
        encoding="utf-8",
    )
    written.append(evidence_index_path)

    canvas_path = project_root / "Memory Map.canvas"
    evidence_counts = {
        category: len(evidence_category_groups.get(category, []))
        for category in _canonical_evidence_categories(evidence_category_groups)
    }
    core_counts = {
        area: len(core_groups.get(area, [])) for area in _canonical_core_areas(core_groups)
    }
    canvas_path.write_text(
        _render_memory_canvas(
            project_title=title,
            core_count=len(core_blocks),
            core_counts=core_counts,
            evidence_counts=evidence_counts,
        ),
        encoding="utf-8",
    )
    written.append(canvas_path)

    index_path = project_root / "Index.md"
    index_path.write_text(
        _render_index(
            project_title=title,
            blocks=core_blocks,
            note_names=note_names,
            evidence_blocks=evidence_blocks,
        ),
        encoding="utf-8",
    )
    written.append(index_path)

    projects_index_path = lethyx_root / "Projects" / "Index.md"
    projects_index_path.parent.mkdir(parents=True, exist_ok=True)
    projects_index_path.write_text(
        _render_projects_index(projects_index_path.parent), encoding="utf-8"
    )
    written.append(projects_index_path)

    overview_path = project_root / "Memory Overview.md"
    overview_path.write_text(
        _render_project_overview(
            project_title=title,
            blocks=core_blocks,
            note_names=note_names,
            evidence_blocks=evidence_blocks,
        ),
        encoding="utf-8",
    )
    written.append(overview_path)

    evidence_archive_path = project_root / "Evidence Archive.md"
    evidence_archive_path.write_text(
        _render_evidence_archive(
            project_title=title,
            blocks=evidence_blocks,
            evidence_index_target=_note_target(evidence_index_path, project_root=project_root),
        ),
        encoding="utf-8",
    )
    written.append(evidence_archive_path)

    return ObsidianExportResult(
        vault_dir=vault,
        workspace_id=workspace_id,
        index_path=index_path,
        notebooklm_source_path=overview_path,
        block_paths=block_paths,
        written_files=tuple(written),
    )


def export_backend_to_obsidian(
    *,
    backend: object,
    vault_dir: str | Path,
    workspace_id: str,
    project_title: str = "",
    include_global: bool = False,
) -> ObsidianExportResult:
    """Export blocks and backend graph edges from a durable Lethyx backend."""

    blocks = list(getattr(backend, "export_blocks")())
    edges: list[MemoryEdgeRecord] = []
    edges_for = getattr(backend, "edges_for", None)
    if callable(edges_for):
        for block in blocks:
            edges.extend(edges_for(block.block_id) or [])
    return export_records_to_obsidian(
        blocks=blocks,
        edges=edges,
        vault_dir=vault_dir,
        workspace_id=workspace_id,
        project_title=project_title,
        include_global=include_global,
    )
