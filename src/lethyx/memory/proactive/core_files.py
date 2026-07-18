"""Load and truncate core MD/YAML files for proactive overlay."""

import hashlib
from pathlib import Path

from lethyx.memory.proactive.budget import estimate_tokens
from lethyx.memory.proactive.models import OverlaySection

_MANAGER_FILES = [
    ("Identity / constitution", "00_identity.md"),
    ("Manager operating principles", "10_operating_principles.md"),
    ("Manager lessons", "20_manager_lessons.md"),
    ("Failure patterns", "30_failure_patterns.md"),
    ("Active risks", "40_active_risks.md"),
    ("Open threads", "50_open_threads.md"),
]

_WORKSPACE_FILES = [
    ("Workspace charter", "00_workspace_charter.md"),
    ("Workspace state", "10_workspace_state.md"),
    ("Workspace lessons", "20_workspace_lessons.md"),
    ("Workspace antipatterns", "30_workspace_antipatterns.md"),
    ("Current strategy", "40_current_strategy.md"),
    ("Open threads", "50_open_threads.md"),
]


def _sha256_ref(path: Path) -> str:
    if not path.is_file():
        return "sha256:missing"
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _truncate_text(text: str, max_tokens: int) -> str:
    max_chars = max(80, max_tokens * 4)
    if len(text) <= max_chars:
        return text.strip()
    return text[: max_chars - 24].rstrip() + "\n...[section truncated]"


def load_core_sections(
    core_root: Path,
    file_specs: list[tuple[str, str]],
    *,
    prefix: str,
    budget_tokens: int,
    per_section_cap: int | None = None,
) -> list[OverlaySection]:
    if not core_root.is_dir():
        return []
    cap = per_section_cap or max(200, budget_tokens // max(1, len(file_specs)))
    sections: list[OverlaySection] = []
    for title, filename in file_specs:
        path = core_root / filename
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        content = _truncate_text(raw, cap)
        sections.append(
            OverlaySection(
                name=title,
                content=content,
                source_refs=[f"core:{prefix}:{filename}"],
                source_hashes=[_sha256_ref(path)],
                trust="curated",
                token_count=estimate_tokens(content),
            )
        )
    return sections


def load_manager_core(core_root: Path, *, budget_tokens: int = 1500) -> list[OverlaySection]:
    return load_core_sections(
        core_root,
        _MANAGER_FILES,
        prefix="manager",
        budget_tokens=budget_tokens,
        per_section_cap=400,
    )


def load_workspace_core(core_root: Path, *, budget_tokens: int = 2000) -> list[OverlaySection]:
    return load_core_sections(
        core_root,
        _WORKSPACE_FILES,
        prefix="workspace",
        budget_tokens=budget_tokens,
        per_section_cap=450,
    )


def ensure_core_seed_files(core_root: Path, *, kind: str = "manager") -> None:
    """Create minimal seed files if core directory is empty."""
    core_root.mkdir(parents=True, exist_ok=True)
    specs = _MANAGER_FILES if kind == "manager" else _WORKSPACE_FILES
    for _title, filename in specs:
        path = core_root / filename
        if path.is_file():
            continue
        if filename == "bkb.yaml":
            continue
        seeds = {
            "00_identity.md": "# Identity\nVerify outcomes before claiming success.\n",
            "10_operating_principles.md": "# Principles\nPrefer evidence-backed changes over assumptions.\n",
            "20_manager_lessons.md": "# Lessons\nPromote only verified lessons via BKB gate.\n",
            "30_failure_patterns.md": (
                "# Anti-patterns\n"
                "Do not repeat known failure modes without mitigation.\n\n"
                "## Phase exit evidence\n"
                "- False diff_hash_mismatch on phase exit: use completion "
                "changed_files for contract validation.\n\n"
                "## Loop-back routing\n"
                "- Stale loop_back_to after mark_subtask_complete or "
                "mutate_phase_plan is superseded.\n"
            ),
            "40_active_risks.md": "# Risks\nTrack active risks until closed or accepted.\n",
            "50_open_threads.md": "# Open threads\nUnresolved work stays explicit until resolved.\n",
            "00_workspace_charter.md": "# Charter\nScope and constraints for this workspace.\n",
            "10_workspace_state.md": "# State\nCurrent workspace status and blockers.\n",
            "20_workspace_lessons.md": "# Workspace lessons\nWorkspace-specific verified lessons only.\n",
            "30_workspace_antipatterns.md": "# Antipatterns\nForbidden repeats for this workspace.\n",
            "40_current_strategy.md": "# Strategy\nCurrent execution strategy for this workspace.\n",
        }
        path.write_text(
            seeds.get(filename, f"# {filename}\n\n"),
            encoding="utf-8",
        )
    bkb_path = core_root / "bkb.yaml"
    if not bkb_path.is_file():
        bkb_path.write_text("rules: []\n", encoding="utf-8")
