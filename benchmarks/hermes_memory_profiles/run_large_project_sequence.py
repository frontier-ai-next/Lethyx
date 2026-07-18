"""Large real-project sequential Hermes memory benchmark.

This benchmark stress-tests memory providers on a popular, noisy repository rather
than compact synthetic facts. It builds a fact corpus from a cloned project,
seeds clean Hermes profiles, asks memory-only questions in order, then injects a
late update card and asks follow-up questions. The goal is to expose quality
separation on volume, multi-hop repo facts, negative leakage, and sequential
profile memory updates.
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sqlite3
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_profile_matrix as base  # noqa: E402


@dataclass(frozen=True)
class FactCard:
    id: str
    title: str
    content: str


@dataclass(frozen=True)
class AskResult:
    answer: str
    elapsed: float
    stderr: str
    returncode: int
    session_id: str
    root_session_id: str
    session_metrics: dict[str, Any]


LANGCHAIN_UPDATE_CARD = FactCard(
    id="langchain_update_openai_reasoning_summary",
    title="Late update: OpenAI reasoning summary unit test",
    content=(
        "Late benchmark update for langchain-ai/langchain: when touching the OpenAI "
        "reasoning-summary behavior in langchain-openai, use package path "
        "libs/partners/openai and run the focused unit command "
        "uv run --group test pytest tests/unit_tests/chat_models/test_base.py::test_openai_reasoning_summary. "
        "This update supersedes any generic instruction to run the entire OpenAI package test suite for that narrow change."
    ),
)

LANGCHAIN_EXAMPLE_QUESTIONS: tuple[base.Question, ...] = (
    base.Question(
        "lc_monorepo_map",
        "In langchain-ai/langchain, what are the core, classic legacy, active v1, partner integration, and standard-tests directories?",
        (
            "libs/core",
            "libs/langchain",
            "libs/langchain_v1",
            "libs/partners",
            "libs/standard-tests",
        ),
    ),
    base.Question(
        "lc_active_vs_legacy",
        "Which LangChain package directory is actively maintained and which one is legacy/classic?",
        (
            "libs/langchain_v1",
            "actively maintained",
            "libs/langchain",
            "legacy",
            "langchain-classic",
        ),
    ),
    base.Question(
        "lc_core_pyproject",
        "For langchain-core, name the package version, Python range, and several key runtime dependencies.",
        ("langchain-core", "1.4.8", ">=3.10.0,<4.0.0", "langsmith", "pydantic", "uuid-utils"),
    ),
    base.Question(
        "lc_openai_integration",
        "For the OpenAI partner integration, what is the package name, local path, key runtime deps, and editable source links?",
        (
            "langchain-openai",
            "libs/partners/openai",
            "openai>=2.26.0,<3.0.0",
            "tiktoken",
            "../../core",
            "../../standard-tests",
        ),
    ),
    base.Question(
        "lc_anthropic_mypy_reason",
        "Why does the Anthropic partner package enable pydantic.mypy while most other partner packages do not?",
        (
            "langchain-anthropic",
            "pydantic.mypy",
            "false-flags",
            "public alias-based constructor API",
        ),
    ),
    base.Question(
        "lc_standard_tests_plugin",
        "What package provides LangChain standard tests and which pytest plugin entry point does it expose?",
        ("langchain-tests", "langsmith_ci", "langchain_tests._langsmith_plugin"),
    ),
    base.Question(
        "lc_negative_core_openai",
        "Does langchain-core directly depend on openai or anthropic? Answer from subject memory only.",
        ("langchain-core", "langsmith", "pydantic"),
        ("openai", "anthropic"),
    ),
    base.Question(
        "lc_issue_like_openai_test_plan",
        "A PR changes OpenAI chat model behavior. Which package path and local test command style should the developer use before broader checks?",
        ("libs/partners/openai", "langchain-openai", "uv run --group test", "pytest"),
    ),
    base.Question(
        "lc_late_update_recall",
        "After the late benchmark update, what focused command should be used for OpenAI reasoning-summary behavior?",
        (
            "uv run --group test pytest tests/unit_tests/chat_models/test_base.py::test_openai_reasoning_summary",
            "libs/partners/openai",
        ),
    ),
)

LANGCHAIN_QUESTIONS = LANGCHAIN_EXAMPLE_QUESTIONS
LATE_UPDATE_QUESTION_IDS = {"lc_late_update_recall"}
DEFAULT_QUESTIONS_PER_MEMORY = 67


def _run_git(repo: Path, args: list[str], timeout: int = 120) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr[-1000:] or proc.stdout[-1000:]
        raise RuntimeError(f"git {' '.join(args)} failed: {msg}")
    return proc.stdout.strip()


def _git_files(repo: Path) -> list[str]:
    return [line for line in _run_git(repo, ["ls-files"]).splitlines() if line]


def _read_limited(path: Path, max_chars: int = 6000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n[...middle truncated for benchmark card...]\n" + tail


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={val}" for key, val in value.items())
    return str(value)


def _pyproject_card(repo: Path, rel: str) -> FactCard:
    path = repo / rel
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    groups = data.get("dependency-groups", {})
    uv_sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    deps = project.get("dependencies", [])
    lines = [
        f"LangChain pyproject card for {rel}.",
        f"Project name: {project.get('name')}.",
        f"Version: {project.get('version')}.",
        f"Description: {project.get('description')}.",
        f"Requires Python: {project.get('requires-python')}.",
        f"Runtime dependencies: {_one_line(deps)}.",
        f"Dependency groups present: {', '.join(sorted(groups))}.",
    ]
    for group_name in sorted(groups):
        lines.append(f"Dependency group {group_name}: {_one_line(groups[group_name])}.")
    if uv_sources:
        source_items = [
            f"{name} -> {_one_line(value)}" for name, value in sorted(uv_sources.items())
        ]
        lines.append(f"Editable/local uv sources: {'; '.join(source_items)}.")
    entry_points = project.get("entry-points", {})
    if entry_points:
        lines.append(f"Project entry points: {_one_line(entry_points)}.")
    pytest_entry = data.get("project", {}).get("entry-points", {}).get("pytest11")
    if pytest_entry:
        lines.append(f"Pytest plugin entry points: {_one_line(pytest_entry)}.")
    lines.append("Raw pyproject excerpt including comments:")
    lines.append(_read_limited(path, max_chars=7000))
    return FactCard(
        id="pyproject_" + re.sub(r"[^a-zA-Z0-9]+", "_", rel).strip("_").lower(),
        title=f"LangChain pyproject: {rel}",
        content="\n".join(lines),
    )


def _agents_card(repo: Path) -> FactCard:
    text = _read_limited(repo / "AGENTS.md", max_chars=12000)
    return FactCard(
        id="langchain_agents_guidelines",
        title="LangChain AGENTS.md development guidelines",
        content=(
            "LangChain repo development-guidelines card from AGENTS.md. This card is evidence for "
            "monorepo structure, package roles, commands, PR conventions, public API rules, and testing rules.\n\n"
            + text
        ),
    )


def _readme_card(repo: Path) -> FactCard:
    text = _read_limited(repo / "README.md", max_chars=9000)
    return FactCard(
        id="langchain_readme_overview",
        title="LangChain README overview",
        content="LangChain repo README overview card.\n\n" + text,
    )


def _repo_inventory_card(repo: Path, files: list[str]) -> FactCard:
    py_files = [path for path in files if path.endswith(".py")]
    md_files = [path for path in files if path.endswith(".md")]
    pyprojects = [path for path in files if path.endswith("pyproject.toml")]
    package_roots = sorted(
        {"/".join(path.split("/")[:2]) for path in pyprojects if path.startswith("libs/")}
    )
    commit = _run_git(repo, ["rev-parse", "--short", "HEAD"])
    lines = [
        "LangChain repository inventory card.",
        f"Git commit: {commit}.",
        f"Tracked files: {len(files)}.",
        f"Python files: {len(py_files)}.",
        f"Markdown files: {len(md_files)}.",
        f"Package pyprojects: {len(pyprojects)}.",
        f"Package roots with pyprojects: {', '.join(package_roots)}.",
        "Important monorepo roots: libs/core, libs/langchain, libs/langchain_v1, libs/partners, libs/text-splitters, libs/standard-tests, libs/model-profiles.",
    ]
    return FactCard("langchain_repo_inventory", "LangChain repository inventory", "\n".join(lines))


def _symbols_from_python(path: Path) -> tuple[list[str], list[str], list[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return [], [], []
    classes: list[str] = []
    functions: list[str] = []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names[:3])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return classes[:12], functions[:18], sorted(set(imports))[:18]


def _source_card(repo: Path, rel: str) -> FactCard:
    path = repo / rel
    classes, functions, imports = _symbols_from_python(path)
    text = _read_limited(path, max_chars=2400)
    lines = [
        f"LangChain source file card for {rel}.",
        f"Classes: {', '.join(classes) if classes else 'none'}.",
        f"Functions: {', '.join(functions) if functions else 'none'}.",
        f"Imports/modules: {', '.join(imports) if imports else 'none'}.",
        "Excerpt:",
        text,
    ]
    return FactCard(
        id="source_" + re.sub(r"[^a-zA-Z0-9]+", "_", rel).strip("_").lower(),
        title=f"LangChain source: {rel}",
        content="\n".join(lines),
    )


def _important_source_files(files: list[str]) -> list[str]:
    priority_patterns = (
        "libs/core/langchain_core/runnables/base.py",
        "libs/core/langchain_core/language_models/chat_models.py",
        "libs/core/langchain_core/tools/base.py",
        "libs/langchain_v1/langchain/agents/factory.py",
        "libs/langchain_v1/langchain/agents/middleware/types.py",
        "libs/partners/openai/langchain_openai/chat_models/base.py",
        "libs/partners/anthropic/langchain_anthropic/chat_models.py",
        "libs/standard-tests/langchain_tests/unit_tests/chat_models.py",
        "libs/text-splitters/langchain_text_splitters/base.py",
    )
    selected = [path for path in priority_patterns if path in files]
    py_files = [path for path in files if path.endswith(".py") and path.startswith("libs/")]
    for path in py_files:
        if path not in selected and len(selected) < 220:
            selected.append(path)
    return selected


def build_langchain_cards(repo: Path, *, max_cards: int = 80) -> list[FactCard]:
    files = _git_files(repo)
    cards: list[FactCard] = [
        _repo_inventory_card(repo, files),
        _agents_card(repo),
        _readme_card(repo),
    ]
    pyprojects = sorted(
        path for path in files if path.endswith("pyproject.toml") and path.startswith("libs/")
    )
    priority_pyprojects = [
        "libs/core/pyproject.toml",
        "libs/langchain/pyproject.toml",
        "libs/langchain_v1/pyproject.toml",
        "libs/standard-tests/pyproject.toml",
        "libs/partners/openai/pyproject.toml",
        "libs/partners/anthropic/pyproject.toml",
        "libs/text-splitters/pyproject.toml",
        "libs/model-profiles/pyproject.toml",
    ]
    ordered_pyprojects = [path for path in priority_pyprojects if path in pyprojects]
    ordered_pyprojects.extend(path for path in pyprojects if path not in ordered_pyprojects)
    for rel in ordered_pyprojects:
        cards.append(_pyproject_card(repo, rel))
    for rel in _important_source_files(files):
        if len(cards) >= max_cards:
            break
        cards.append(_source_card(repo, rel))
    return cards[:max_cards]


LANGCHAIN_PACKAGE_EXTRA_QUESTIONS: tuple[base.Question, ...] = (
    base.Question(
        "lc_classic_package_sources",
        "For the legacy/classic LangChain package, name its package path, package name, and several runtime/local source dependencies.",
        (
            "libs/langchain",
            "langchain-classic",
            "SQLAlchemy",
            "langchain-text-splitters",
            "../text-splitters",
        ),
    ),
    base.Question(
        "lc_v1_package_deps",
        "For the active LangChain v1 package, name its path, package name, and key runtime dependencies/local sources.",
        ("libs/langchain_v1", "langchain", "langgraph", "pydantic", "langchain-anthropic"),
    ),
    base.Question(
        "lc_text_splitters_package",
        "For LangChain text splitters, name the package path, package name, core dependency, and the spaCy model source.",
        ("libs/text-splitters", "langchain-text-splitters", "langchain-core", "en-core-web-sm"),
    ),
    base.Question(
        "lc_model_profiles_package",
        "For LangChain model profiles, name the package path, package name, and key dependencies.",
        ("libs/model-profiles", "langchain-model-profiles", "httpx", "typing-extensions"),
    ),
    base.Question(
        "lc_chroma_partner_package",
        "For the Chroma partner package, name its path, package name, and important runtime dependencies.",
        ("libs/partners/chroma", "langchain-chroma", "chromadb", "numpy"),
    ),
    base.Question(
        "lc_deepseek_partner_package",
        "For the DeepSeek partner package, name its path, package name, and the local partner dependency it reuses.",
        ("libs/partners/deepseek", "langchain-deepseek", "langchain-openai", "../openai"),
    ),
    base.Question(
        "lc_exa_partner_package",
        "For the Exa partner package, name its path, package name, and key runtime dependency.",
        ("libs/partners/exa", "langchain-exa", "exa-py"),
    ),
    base.Question(
        "lc_fireworks_partner_package",
        "For the Fireworks partner package, name its path, package name, and key runtime dependencies.",
        ("libs/partners/fireworks", "langchain-fireworks", "fireworks-ai", "openai", "aiohttp"),
    ),
    base.Question(
        "lc_groq_partner_package",
        "For the Groq partner package, name its path, package name, and key runtime dependency.",
        ("libs/partners/groq", "langchain-groq", "groq"),
    ),
    base.Question(
        "lc_huggingface_partner_package",
        "For the HuggingFace partner package, name its path, package name, and key runtime/local dependencies.",
        (
            "libs/partners/huggingface",
            "langchain-huggingface",
            "tokenizers",
            "huggingface-hub",
            "langchain_v1",
        ),
    ),
    base.Question(
        "lc_mistralai_partner_package",
        "For the MistralAI partner package, name its path, package name, and key runtime dependencies.",
        ("libs/partners/mistralai", "langchain-mistralai", "tokenizers", "httpx-sse", "pydantic"),
    ),
    base.Question(
        "lc_nomic_partner_package",
        "For the Nomic partner package, name its path, package name, and key runtime dependencies.",
        ("libs/partners/nomic", "langchain-nomic", "nomic", "pillow"),
    ),
    base.Question(
        "lc_ollama_partner_package",
        "For the Ollama partner package, name its path, package name, and key runtime dependencies.",
        ("libs/partners/ollama", "langchain-ollama", "ollama", "langchain-core"),
    ),
    base.Question(
        "lc_openrouter_partner_package",
        "For the OpenRouter partner package, name its path, package name, and key runtime dependency.",
        ("libs/partners/openrouter", "langchain-openrouter", "openrouter"),
    ),
    base.Question(
        "lc_perplexity_partner_package",
        "For the Perplexity partner package, name its path, package name, and key runtime dependency.",
        ("libs/partners/perplexity", "langchain-perplexity", "perplexityai"),
    ),
    base.Question(
        "lc_qdrant_partner_package",
        "For the Qdrant partner package, name its path, package name, and key runtime dependencies.",
        ("libs/partners/qdrant", "langchain-qdrant", "qdrant-client", "pydantic"),
    ),
    base.Question(
        "lc_xai_partner_package",
        "For the xAI partner package, name its path, package name, and key runtime/local dependencies.",
        ("libs/partners/xai", "langchain-xai", "langchain-openai", "../openai", "aiohttp"),
    ),
)


LANGCHAIN_SESSION_RECALL_QUESTIONS: tuple[base.Question, ...] = (
    base.Question(
        "lc_session_user_codename_recall",
        "Earlier in this same benchmark session, what codename did the user assign to the run, and how many rows per memory/profile did they request?",
        ("LC67-ORION", "67"),
    ),
    base.Question(
        "lc_session_profile_ack_recall",
        "Earlier in this same benchmark session, what exact assistant acknowledgment marker did the Hermes profile write for the codename note?",
        ("PROFILE-ACK-ORION",),
    ),
    base.Question(
        "lc_session_clean_profile_recall",
        "Earlier in this same benchmark session, what clean-profile phrase did the user ask the profile to remember, and what marker did the profile acknowledge with?",
        ("clean-profile-single-session", "PROFILE-ACK-CLEAN"),
    ),
    base.Question(
        "lc_session_metrics_request_recall",
        "Earlier in this same benchmark session, which operational counts did the user ask the harness to track?",
        ("tool call", "context", "summary"),
    ),
)


SESSION_PROMPT_NOTES: dict[str, str] = {
    "lc_monorepo_map": "Session note for later recall: the user calls this expanded run LC67-ORION and wants 67 rows per memory/profile. In your answer, include the exact assistant acknowledgment marker PROFILE-ACK-ORION.",
    "lc_core_pyproject": "Session note for later recall: the user wants clean profiles and one resumed session per profile; the phrase is clean-profile-single-session. In your answer, include the exact assistant acknowledgment marker PROFILE-ACK-CLEAN.",
    "lc_openai_integration": "Session note for later recall: the user asked the harness to track tool call count and context summary count. In your answer, include the exact assistant acknowledgment marker PROFILE-ACK-METRICS.",
}


def _extract_source_terms(card: FactCard) -> tuple[str, str, tuple[str, ...]] | None:
    if not card.id.startswith("source_"):
        return None
    rel_match = re.search(r"LangChain source file card for ([^.]+\.py)\.", card.content)
    rel = rel_match.group(1) if rel_match else card.title.replace("LangChain source: ", "")
    terms: list[str] = [rel]
    for prefix in ("Classes:", "Functions:", "Imports/modules:"):
        match = re.search(rf"^{re.escape(prefix)}\s*(.+?)\.$", card.content, flags=re.MULTILINE)
        if not match:
            continue
        for item in (part.strip() for part in match.group(1).split(",")):
            if item and item != "none" and item not in terms:
                terms.append(item)
            if len(terms) >= 4:
                break
        if len(terms) >= 4:
            break
    if len(terms) < 3:
        return None
    qid = "lc_" + re.sub(r"[^a-zA-Z0-9]+", "_", card.id).strip("_").lower()
    return qid, rel, tuple(terms[:4])


def _source_question_from_card(card: FactCard) -> base.Question | None:
    extracted = _extract_source_terms(card)
    if extracted is None:
        return None
    qid, rel, terms = extracted
    return base.Question(
        qid,
        f"For LangChain source file {rel}, name the file path and the main classes/functions/imports remembered for it.",
        terms,
    )


def build_langchain_questions(
    cards: list[FactCard], *, target_count: int = DEFAULT_QUESTIONS_PER_MEMORY
) -> tuple[base.Question, ...]:
    if target_count <= 0:
        target_count = DEFAULT_QUESTIONS_PER_MEMORY
    if target_count <= len(LANGCHAIN_EXAMPLE_QUESTIONS):
        return LANGCHAIN_EXAMPLE_QUESTIONS[:target_count]

    late_update = LANGCHAIN_EXAMPLE_QUESTIONS[-1]
    questions: list[base.Question] = list(LANGCHAIN_EXAMPLE_QUESTIONS[:8])
    seen = {question.id for question in questions}

    def add(question: base.Question) -> None:
        if len(questions) >= target_count - 1:
            return
        if question.id in seen or question.id == late_update.id:
            return
        questions.append(question)
        seen.add(question.id)

    for question in LANGCHAIN_PACKAGE_EXTRA_QUESTIONS:
        add(question)
    for question in LANGCHAIN_SESSION_RECALL_QUESTIONS:
        add(question)
    for card in cards:
        source_question = _source_question_from_card(card)
        if source_question is not None:
            add(source_question)
        if len(questions) >= target_count - 1:
            break
    if len(questions) < target_count - 1:
        raise RuntimeError(
            f"could only build {len(questions) + 1} LangChain questions, requested {target_count}"
        )
    questions.append(late_update)
    return tuple(questions)


def _cards_to_scenarios(cards: list[FactCard]) -> tuple[base.Scenario, ...]:
    scenarios = []
    for card in cards:
        scenarios.append(
            base.Scenario(
                id=card.id,
                repo_name="langchain-ai/langchain",
                facts=f"{card.title}\n\n{card.content}",
                questions=(),
            )
        )
    return tuple(scenarios)


def _seed_profile(
    profile: str, spec: base.ProviderSpec, cards: list[FactCard], store: Path, bank_id: str
) -> None:
    scenarios = _cards_to_scenarios(cards)
    if spec.id == "builtin":
        base._seed_builtin(profile, scenarios)
    elif spec.id == "hindsight":
        base._seed_hindsight(profile, scenarios, bank_id=bank_id)
    elif spec.hermes_provider == "lethyx":
        base._seed_lethyx(profile, scenarios, store, spec)
    else:
        raise ValueError(f"Unsupported provider spec: {spec}")


def _append_card(
    profile: str, spec: base.ProviderSpec, card: FactCard, store: Path, bank_id: str
) -> None:
    if spec.id == "builtin":
        mem_path = base._profile_home(profile) / "memories" / "MEMORY.md"
        with mem_path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + card.title + "\n\n" + card.content + "\n")
        return
    _seed_profile(profile, spec, [card], store, bank_id)


def _memory_prompt(question: base.Question) -> str:
    session_note = SESSION_PROMPT_NOTES.get(question.id, "")
    note_text = f"{session_note}\n\n" if session_note else ""
    return (
        "Do not read files, terminal, web, or session_search. Use only this profile's connected memory/provider tools "
        "plus earlier messages in this same resumed benchmark session. "
        "Use positive memory evidence to answer negative/comparative questions: if memory identifies the subject "
        "but has no evidence for named alternatives, say no memory-backed fact supports those alternatives and include the subject name. "
        "Do not separately recall named alternatives or list unrelated facts about alternatives unless memory ties them directly to the subject. "
        "Only say 'not in memory' when no relevant subject memory is connected. "
        "Answer concisely but include relevant exact paths, commands, names, and any requested session markers.\n\n"
        f"{note_text}Question: {question.text}"
    )


def _hermes_chat_command(
    profile: str, prompt: str, *, session_id: str | None, source: str
) -> list[str]:
    cmd = ["hermes", "-p", profile, "chat", "-Q", "--toolsets", "memory", "--source", source]
    if session_id:
        cmd.extend(["--resume", session_id])
    cmd.extend(["-q", prompt])
    return cmd


def _parse_session_id(stderr: str, fallback: str | None = None) -> str:
    matches = re.findall(r"session_id:\s*([^\s]+)", stderr or "")
    if matches:
        return matches[-1]
    return fallback or ""


def _session_lineage_metrics(profile: str, session_id: str) -> dict[str, Any]:
    empty = {
        "session_id": session_id,
        "root_session_id": session_id,
        "lineage_ids": [session_id] if session_id else [],
        "message_count": 0,
        "tool_call_count": 0,
        "unique_tool_call_count": 0,
        "tool_message_count": 0,
        "api_call_count": 0,
        "compression_count": 0,
        "compacted_message_count": 0,
    }
    if not session_id:
        return empty
    db_path = base._profile_home(profile) / "state.db"
    if not db_path.exists():
        return empty
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        lineage: list[str] = []
        compression_count = 0
        current = session_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            lineage.append(current)
            row = con.execute(
                "select id, parent_session_id, end_reason from sessions where id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            if row["end_reason"] == "compression":
                compression_count += 1
            current = row["parent_session_id"]
        if not lineage:
            return empty
        placeholders = ",".join("?" for _ in lineage)
        session_row = con.execute(
            f"select coalesce(sum(message_count),0), coalesce(sum(tool_call_count),0), coalesce(sum(api_call_count),0) from sessions where id in ({placeholders})",
            lineage,
        ).fetchone()
        message_count, tool_call_count, api_call_count = (
            int(session_row[0]),
            int(session_row[1]),
            int(session_row[2]),
        )
        tool_message_count = int(
            con.execute(
                f"select count(*) from messages where session_id in ({placeholders}) and (tool_name is not null or role = 'tool')",
                lineage,
            ).fetchone()[0]
        )
        tool_call_ids: set[str] = set()
        for msg in con.execute(
            f"""
            select id, role, tool_call_id, tool_calls, tool_name
            from messages
            where session_id in ({placeholders})
              and (tool_name is not null or role = 'tool' or tool_calls is not null)
            """,
            lineage,
        ):
            if msg["tool_call_id"]:
                tool_call_ids.add(str(msg["tool_call_id"]))
            elif msg["role"] == "tool" or msg["tool_name"]:
                tool_call_ids.add(f"tool-message:{msg['id']}")
            if msg["tool_calls"]:
                try:
                    parsed_calls = json.loads(msg["tool_calls"])
                except Exception:
                    parsed_calls = []
                for idx, call in enumerate(parsed_calls or []):
                    if isinstance(call, dict):
                        call_id = call.get("id") or call.get("tool_call_id")
                    else:
                        call_id = None
                    tool_call_ids.add(str(call_id or f"assistant-message:{msg['id']}:{idx}"))
        unique_tool_call_count = len(tool_call_ids)
        compacted_message_count = int(
            con.execute(
                f"select count(*) from messages where session_id in ({placeholders}) and compacted = 1",
                lineage,
            ).fetchone()[0]
        )
        parent_compressions = int(
            con.execute(
                f"select count(*) from sessions where id in ({placeholders}) and end_reason = 'compression'",
                lineage,
            ).fetchone()[0]
        )
        compression_count = max(compression_count, parent_compressions, max(0, len(lineage) - 1))
        return {
            "session_id": session_id,
            "root_session_id": lineage[-1],
            "lineage_ids": lineage,
            "message_count": message_count,
            "tool_call_count": tool_call_count,
            "unique_tool_call_count": unique_tool_call_count,
            "tool_message_count": tool_message_count,
            "api_call_count": api_call_count,
            "compression_count": compression_count,
            "compacted_message_count": compacted_message_count,
        }
    except Exception as exc:
        empty["metrics_error"] = str(exc)
        return empty
    finally:
        try:
            con.close()  # type: ignore[name-defined]
        except Exception:
            pass


def _ask_profile_session(
    profile: str,
    question: base.Question,
    *,
    session_id: str | None,
    source: str,
    timeout: int,
) -> AskResult:
    prompt = _memory_prompt(question)
    start = time.perf_counter()
    proc = base._run(
        _hermes_chat_command(profile, prompt, session_id=session_id, source=source), timeout=timeout
    )
    elapsed = time.perf_counter() - start
    head_session_id = _parse_session_id(proc.stderr, fallback=session_id)
    metrics = _session_lineage_metrics(profile, head_session_id)
    return AskResult(
        answer=proc.stdout.strip(),
        elapsed=elapsed,
        stderr=proc.stderr.strip(),
        returncode=proc.returncode,
        session_id=head_session_id,
        root_session_id=str(metrics.get("root_session_id") or head_session_id),
        session_metrics=metrics,
    )


def _write_large_reports(
    out_dir: Path, rows: list[dict[str, Any]], providers: list[str], cards: list[FactCard]
) -> tuple[Path, Path, Path]:
    jsonl = out_dir / "results.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    summary: dict[str, Any] = {}
    for provider in providers:
        provider_rows = [row for row in rows if row["provider"] == provider]
        total_tool_calls = sum(int(row.get("tool_call_count_delta") or 0) for row in provider_rows)
        max_context_summaries = max(
            [int(row.get("context_summary_count_total") or 0) for row in provider_rows], default=0
        )
        summary[provider] = {
            "rows": len(provider_rows),
            "avg_score": round(mean([row["score"]["score"] for row in provider_rows]), 4)
            if provider_rows
            else 0,
            "avg_seconds": round(mean([row["elapsed_seconds"] for row in provider_rows]), 3)
            if provider_rows
            else 0,
            "max_seconds": max([row["elapsed_seconds"] for row in provider_rows], default=0),
            "failures": sum(1 for row in provider_rows if row["returncode"] != 0),
            "tool_calls": total_tool_calls,
            "avg_tool_calls_per_row": round(total_tool_calls / len(provider_rows), 3)
            if provider_rows
            else 0,
            "context_summaries": max_context_summaries,
            "session_roots": sorted(
                {
                    row.get("session_root_id", "")
                    for row in provider_rows
                    if row.get("session_root_id")
                }
            ),
        }
    summary["_corpus"] = {
        "cards": len(cards),
        "approx_chars": sum(len(card.content) for card in cards),
        "late_update_cards": 1,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = out_dir / "report.md"
    lines = [
        "# Large Project Sequential Hermes Memory Benchmark",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Project: langchain-ai/langchain",
        f"Seed cards: {len(cards)}",
        f"Approx seed chars: {sum(len(card.content) for card in cards)}",
        "",
        "## Provider summary",
        "",
        "| Provider | Rows | Avg score | Avg seconds | Max seconds | Failures | Tool calls | Avg tools/row | Context summaries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in providers:
        item = summary[provider]
        lines.append(
            f"| {provider} | {item['rows']} | {item['avg_score']:.4f} | {item['avg_seconds']:.3f} | {item['max_seconds']:.3f} | {item['failures']} | {item['tool_calls']} | {item['avg_tool_calls_per_row']:.3f} | {item['context_summaries']} |"
        )
    lines += [
        "",
        "## Sequential question results",
        "",
        "| Provider | # | Question | Score | Seconds | Tool calls Δ | Context summaries | Session head |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['provider']} | {row.get('provider_row_index', '')} | {row['question']} | {row['score']['score']:.2f} | {row['elapsed_seconds']:.3f} | {row.get('tool_call_count_delta', 0)} | {row.get('context_summary_count_total', 0)} | {row.get('session_id', '')} |"
        )
    lines += ["", "## Corpus cards", ""]
    for card in cards[:80]:
        lines.append(f"- `{card.id}` — {card.title} ({len(card.content)} chars)")
    if len(cards) > 80:
        lines.append(f"- ... {len(cards) - 80} more cards")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl, summary_path, report


def run_large_sequence(
    providers: list[str],
    out_dir: Path,
    *,
    repo_path: Path,
    max_cards: int,
    max_questions: int,
    provider_workers: int = 1,
    query_timeout: int = 720,
    source: str = "large-langchain-bench",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    progress_lock = threading.Lock()
    progress_path = out_dir / "progress.json"
    incremental_jsonl = out_dir / "results.jsonl"
    incremental_jsonl.write_text("", encoding="utf-8")

    cards = build_langchain_cards(repo_path, max_cards=max_cards)
    target_questions = max_questions if max_questions > 0 else DEFAULT_QUESTIONS_PER_MEMORY
    questions = list(build_langchain_questions(cards, target_count=target_questions))

    specs = [base.PROVIDER_SPECS[base._normalize_provider(provider)] for provider in providers]
    normalized_ids = [spec.id for spec in specs]
    total_rows = len(specs) * len(questions)
    prefix = "hmp_large_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    provider_status: dict[str, dict[str, Any]] = {}

    def write_progress(*, provider: str = "", question: str = "", stage: str = "") -> None:
        with progress_lock:
            progress_path.write_text(
                json.dumps(
                    {
                        "rows_done": len(rows),
                        "rows_total": total_rows,
                        "current_provider": provider,
                        "current_question": question,
                        "stage": stage,
                        "provider_workers": provider_workers,
                        "questions_per_provider": len(questions),
                        "provider_status": provider_status,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    write_progress(stage="start")

    def append_row(row: dict[str, Any]) -> None:
        with rows_lock:
            rows.append(row)
            with incremental_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            provider_status.setdefault(row["provider"], {})["rows_done"] = (
                provider_status.get(row["provider"], {}).get("rows_done", 0) + 1
            )
        write_progress(provider=row["provider"], question=row["question"], stage=row["stage"])

    def run_provider(spec: base.ProviderSpec) -> dict[str, Any]:
        profile = f"{prefix}_{base._safe_profile_suffix(spec.id)}"
        provider_store = out_dir / "stores" / base._safe_profile_suffix(spec.id)
        bank_id = f"{prefix}_{spec.id}"
        provider_status[spec.id] = {"stage": "seed_start", "profile": profile, "rows_done": 0}
        write_progress(provider=spec.id, stage="seed_start")
        print(
            json.dumps(
                {"stage": "seed_start", "provider": spec.id, "profile": profile}, ensure_ascii=False
            ),
            flush=True,
        )
        base._create_clean_profile(profile, spec.hermes_provider)
        _seed_profile(profile, spec, cards, provider_store, bank_id=bank_id)
        provider_status[spec.id]["stage"] = "seed_done"
        write_progress(provider=spec.id, stage="seed_done")
        print(
            json.dumps(
                {"stage": "seed_done", "provider": spec.id, "profile": profile}, ensure_ascii=False
            ),
            flush=True,
        )
        if spec.id == "hindsight" or spec.hermes_provider == "lethyx":
            time.sleep(8)

        session_id: str | None = None
        previous_tool_calls = 0
        previous_context_summaries = 0
        late_update_done = False
        for provider_row_index, question in enumerate(questions, start=1):
            if question.id in LATE_UPDATE_QUESTION_IDS and not late_update_done:
                provider_status[spec.id]["stage"] = "late_update_seed"
                write_progress(provider=spec.id, question=question.id, stage="late_update_seed")
                _append_card(profile, spec, LANGCHAIN_UPDATE_CARD, provider_store, bank_id=bank_id)
                late_update_done = True
                if spec.id == "hindsight" or spec.hermes_provider == "lethyx":
                    time.sleep(3)
            stage = "late_update" if question.id in LATE_UPDATE_QUESTION_IDS else "initial"
            provider_status[spec.id].update({"stage": stage, "current_question": question.id})
            write_progress(provider=spec.id, question=question.id, stage=stage)
            result = _ask_profile_session(
                profile,
                question,
                session_id=session_id,
                source=source,
                timeout=query_timeout,
            )
            if result.session_id:
                session_id = result.session_id
            metrics = result.session_metrics
            if metrics.get("metrics_error"):
                tool_total = previous_tool_calls
                context_total = previous_context_summaries
            else:
                tool_total = int(
                    metrics.get("unique_tool_call_count")
                    or metrics.get("tool_call_count")
                    or metrics.get("tool_message_count")
                    or 0
                )
                context_total = int(metrics.get("compression_count") or 0)
            row = {
                "provider": spec.id,
                "profile": profile,
                "hermes_provider": spec.hermes_provider,
                "lethyx_backend": spec.lethyx_backend,
                "include_passive": spec.include_passive,
                "provider_row_index": provider_row_index,
                "question": question.id,
                "question_text": question.text,
                "stage": stage,
                "session_id": result.session_id,
                "session_root_id": result.root_session_id,
                "elapsed_seconds": round(result.elapsed, 3),
                "score": base._score(result.answer, question),
                "answer": result.answer,
                "stderr": result.stderr[-1000:],
                "returncode": result.returncode,
                "session_metrics": metrics,
                "tool_call_count_total": tool_total,
                "tool_call_count_delta": max(0, tool_total - previous_tool_calls),
                "stored_tool_call_count_total": int(metrics.get("tool_call_count") or 0),
                "context_summary_count_total": context_total,
                "context_summary_count_delta": max(0, context_total - previous_context_summaries),
                "message_count_total": int(metrics.get("message_count") or 0),
                "api_call_count_total": int(metrics.get("api_call_count") or 0),
            }
            previous_tool_calls = tool_total
            previous_context_summaries = context_total
            append_row(row)
        provider_status[spec.id]["stage"] = "done"
        write_progress(provider=spec.id, stage="done")
        return {"provider": spec.id, "profile": profile, "session_id": session_id}

    provider_workers = max(1, min(provider_workers, len(specs)))
    if provider_workers == 1:
        provider_results = [run_provider(spec) for spec in specs]
    else:
        provider_results = []
        with ThreadPoolExecutor(max_workers=provider_workers) as pool:
            futures = {pool.submit(run_provider, spec): spec for spec in specs}
            for future in as_completed(futures):
                provider_results.append(future.result())

    rows.sort(
        key=lambda row: (normalized_ids.index(row["provider"]), row.get("provider_row_index", 0))
    )
    jsonl, summary_path, report = _write_large_reports(out_dir, rows, normalized_ids, cards)
    return {
        "out_dir": str(out_dir),
        "project": "langchain-ai/langchain",
        "seed_cards": len(cards),
        "seed_chars": sum(len(card.content) for card in cards),
        "rows": len(rows),
        "questions_per_provider": len(questions),
        "provider_workers": provider_workers,
        "providers": normalized_ids,
        "provider_sessions": provider_results,
        "jsonl": str(jsonl),
        "summary": str(summary_path),
        "report": str(report),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--providers",
        default="builtin,lethyx-hindsight",
        help=f"Comma-separated providers. Known: {', '.join(base.PROVIDER_SPECS)}",
    )
    parser.add_argument("--repo-path", default=".external/langchain")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--max-cards", type=int, default=80)
    parser.add_argument(
        "--max-questions", type=int, default=0, help="Questions per provider; default 0 means 67"
    )
    parser.add_argument(
        "--provider-workers",
        type=int,
        default=1,
        help="Run independent provider sessions in parallel",
    )
    parser.add_argument(
        "--query-timeout", type=int, default=720, help="Seconds per Hermes question turn"
    )
    parser.add_argument(
        "--source", default="large-langchain-bench", help="Hermes session source tag"
    )
    args = parser.parse_args(argv)

    repo_path = Path(args.repo_path)
    if not repo_path.is_absolute():
        repo_path = Path.cwd() / repo_path
    if not (repo_path / ".git").exists():
        raise SystemExit(f"repo path is not a git checkout: {repo_path}")
    providers = [provider.strip() for provider in args.providers.split(",") if provider.strip()]
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("runs/hermes_memory_profiles")
        / ("large_langchain_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    )
    result = run_large_sequence(
        providers,
        out_dir,
        repo_path=repo_path,
        max_cards=args.max_cards,
        max_questions=args.max_questions,
        provider_workers=args.provider_workers,
        query_timeout=args.query_timeout,
        source=args.source,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
