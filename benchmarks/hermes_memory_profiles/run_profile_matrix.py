"""Hermes memory-provider profile matrix benchmark.

Creates clean temporary Hermes profiles, seeds each memory provider/backend config with
the same project facts, asks memory-only questions, and writes JSONL + Markdown
reports. The benchmark is intentionally small enough to run before release while
still separating direct provider quality from Lethyx backend configuration quality.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

HERMES_HOME = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
HERMES_AGENT_ROOT = Path(os.environ.get("HERMES_AGENT_ROOT") or HERMES_HOME / "hermes-agent")
LETHYX_ROOT = Path(os.environ.get("LETHYX_CODE_ROOT") or Path.cwd())


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    id: str
    repo_name: str
    facts: str
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    hermes_provider: str
    lethyx_backend: str = ""
    include_passive: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="api_tasks",
        repo_name="synthetic_fastapi_tasks",
        facts=(
            "Synthetic repo synthetic_fastapi_tasks: FastAPI task tracker. Repository path "
            "/tmp/synthetic_fastapi_tasks. App package is app/. Routes live in app/routes/tasks.py, "
            "persistence uses SQLite file data/tasks.db, migrations are managed by alembic under "
            "migrations/. Critical pitfall: tests must use TASKS_DB_URL=sqlite+aiosqlite:///:memory: "
            "and must not touch data/tasks.db."
        ),
        questions=(
            Question(
                "api_architecture",
                "What is the package, route file, database file, and migration location for synthetic_fastapi_tasks?",
                ("app", "app/routes/tasks.py", "data/tasks.db", "migrations"),
            ),
            Question(
                "api_test_pitfall",
                "What is the testing pitfall for synthetic_fastapi_tasks?",
                ("TASKS_DB_URL", "sqlite+aiosqlite:///:memory:", "data/tasks.db"),
            ),
            Question(
                "api_entry_domain",
                "Which repo is the task tracker and what persistence technology does it use?",
                ("synthetic_fastapi_tasks", "FastAPI", "SQLite"),
            ),
            Question(
                "api_negative_leak",
                "Does synthetic_fastapi_tasks use reportkit or Zustand? Answer from memory only.",
                ("synthetic_fastapi_tasks",),
                ("reportkit", "Zustand", "zustand"),
            ),
        ),
    ),
    Scenario(
        id="cli_reports",
        repo_name="synthetic_typer_reports",
        facts=(
            "Synthetic repo synthetic_typer_reports: Typer/Rich reporting CLI. Package is reportkit, "
            "CLI entrypoint is reportkit.cli:app, templates live in templates/, golden snapshots live "
            "in tests/golden/. Critical pitfall: rich color output must be disabled with NO_COLOR=1 "
            "before comparing snapshots."
        ),
        questions=(
            Question(
                "cli_layout",
                "For synthetic_typer_reports, what is the package, CLI entrypoint, template directory, and golden snapshot directory?",
                ("reportkit", "reportkit.cli:app", "templates", "tests/golden"),
            ),
            Question(
                "cli_snapshot_pitfall",
                "What environment variable matters for snapshot tests in synthetic_typer_reports, and why?",
                ("NO_COLOR", "rich", "snapshot"),
            ),
            Question(
                "cli_command_shape",
                "What kind of project is synthetic_typer_reports and which UI/output libraries are involved?",
                ("Typer", "Rich", "CLI"),
            ),
            Question(
                "cli_negative_leak",
                "Does synthetic_typer_reports store tasks in data/tasks.db or use alembic migrations?",
                ("synthetic_typer_reports",),
                ("data/tasks.db", "alembic", "migrations"),
            ),
        ),
    ),
    Scenario(
        id="vite_dashboard",
        repo_name="synthetic_vite_dashboard",
        facts=(
            "Synthetic repo synthetic_vite_dashboard: React/Vite dashboard. Stack is React 19, "
            "TypeScript, Vite, Tailwind CSS, ShadCn UI, and Zustand. Routes live in src/AppRoutes.tsx, "
            "global state lives in src/store/globalStore.ts, layout components live under "
            "src/components/layout/. Dev command is pnpm dev; production build command is pnpm build; "
            "dist/ is generated output and should not be edited."
        ),
        questions=(
            Question(
                "vite_stack",
                "What frontend stack does synthetic_vite_dashboard use?",
                ("React", "TypeScript", "Vite", "Tailwind", "ShadCn", "Zustand"),
            ),
            Question(
                "vite_paths",
                "Where are routes, global store, and layout components in synthetic_vite_dashboard?",
                ("src/AppRoutes.tsx", "src/store/globalStore.ts", "src/components/layout"),
            ),
            Question(
                "vite_commands",
                "What are the dev and production build commands for synthetic_vite_dashboard?",
                ("pnpm dev", "pnpm build"),
            ),
            Question(
                "vite_generated_output",
                "Which generated output directory should not be edited in synthetic_vite_dashboard?",
                ("dist",),
            ),
        ),
    ),
    Scenario(
        id="public_fastapi_todo",
        repo_name="lymanny/FastAPI-CRUD-Todo",
        facts=(
            "Public repo lymanny/FastAPI-CRUD-Todo: simple FastAPI CRUD Todo application with "
            "SQLite, SQLAlchemy, Pydantic, and Uvicorn. Run command is uvicorn main:app --reload. "
            "Interactive docs are at http://127.0.0.1:8000/docs. Project structure includes "
            "database.py, models.py, schemas.py, routers/todo.py, main.py, and requirements.txt."
        ),
        questions=(
            Question(
                "public_fastapi_stack",
                "What stack does lymanny/FastAPI-CRUD-Todo use?",
                ("FastAPI", "SQLite", "SQLAlchemy", "Pydantic", "Uvicorn"),
            ),
            Question(
                "public_fastapi_run",
                "How do you run lymanny/FastAPI-CRUD-Todo and where are the interactive docs?",
                ("uvicorn main:app --reload", "http://127.0.0.1:8000/docs"),
            ),
            Question(
                "public_fastapi_files",
                "Name key source files in lymanny/FastAPI-CRUD-Todo.",
                ("database.py", "models.py", "schemas.py", "routers/todo.py", "main.py"),
            ),
            Question(
                "public_fastapi_negative_leak",
                "Does lymanny/FastAPI-CRUD-Todo use Zustand or ShadCn?",
                ("lymanny/FastAPI-CRUD-Todo",),
                ("Zustand", "zustand", "ShadCn"),
            ),
        ),
    ),
    Scenario(
        id="public_typer_cli",
        repo_name="alisterburt/minimal-rich-typer-cli",
        facts=(
            "Public repo alisterburt/minimal-rich-typer-cli: minimal pip-installable CLI example using "
            "Typer and Rich. Install in editable mode with pip install -e . Two example entry programs "
            "are single_command and multi_command, each with linked example output HTML."
        ),
        questions=(
            Question(
                "public_typer_stack",
                "What libraries does alisterburt/minimal-rich-typer-cli demonstrate?",
                ("Typer", "Rich"),
            ),
            Question(
                "public_typer_install",
                "How is alisterburt/minimal-rich-typer-cli installed for local development?",
                ("pip install -e .",),
            ),
            Question(
                "public_typer_entries",
                "What are the two example entry programs in alisterburt/minimal-rich-typer-cli?",
                ("single_command", "multi_command"),
            ),
            Question(
                "public_typer_negative_leak",
                "Does alisterburt/minimal-rich-typer-cli run with pnpm dev?",
                ("alisterburt/minimal-rich-typer-cli",),
                ("pnpm dev",),
            ),
        ),
    ),
    Scenario(
        id="public_vite_dashboard",
        repo_name="Daaviddev/vite-dashboard-starter",
        facts=(
            "Public repo Daaviddev/vite-dashboard-starter: modern dashboard template built with React, "
            "TypeScript, Vite, Tailwind CSS, ShadCn, and Zustand. Install with pnpm install, start dev "
            "server with pnpm dev at http://localhost:3000, build with pnpm build into dist. Key files: "
            "src/store/globalStore.ts, src/AppRoutes.tsx, src/lib/utils.ts; utility cn combines clsx with "
            "tailwind-merge."
        ),
        questions=(
            Question(
                "public_vite_stack",
                "What stack does Daaviddev/vite-dashboard-starter use?",
                ("React", "TypeScript", "Vite", "Tailwind", "ShadCn", "Zustand"),
            ),
            Question(
                "public_vite_commands",
                "What are the install, dev, and build commands for Daaviddev/vite-dashboard-starter?",
                ("pnpm install", "pnpm dev", "pnpm build"),
            ),
            Question(
                "public_vite_files",
                "Which files hold the global store, routes, and cn utility in Daaviddev/vite-dashboard-starter?",
                ("src/store/globalStore.ts", "src/AppRoutes.tsx", "src/lib/utils.ts"),
            ),
            Question(
                "public_vite_utility",
                "What does the cn utility combine in Daaviddev/vite-dashboard-starter?",
                ("clsx", "tailwind-merge"),
            ),
        ),
    ),
)

PUBLIC_REPO_CANDIDATES = [
    {
        "domain": "FastAPI + SQLite CRUD",
        "name": "lymanny/FastAPI-CRUD-Todo",
        "url": "https://github.com/lymanny/FastAPI-CRUD-Todo",
        "why": "small CRUD service with clear SQLite/domain facts for architecture recall probes",
    },
    {
        "domain": "Typer/Rich CLI",
        "name": "alisterburt/minimal-rich-typer-cli",
        "url": "https://github.com/alisterburt/minimal-rich-typer-cli",
        "why": "minimal installable CLI; good for entrypoint/layout/snapshot-memory questions",
    },
    {
        "domain": "React/Vite dashboard",
        "name": "Daaviddev/vite-dashboard-starter",
        "url": "https://github.com/Daaviddev/vite-dashboard-starter",
        "why": "frontend stack memory probes: state management, UI kit, routes, build commands",
    },
]

PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "builtin": ProviderSpec("builtin", ""),
    "hindsight": ProviderSpec("hindsight", "hindsight"),
    "lethyx-hindsight": ProviderSpec("lethyx-hindsight", "lethyx", "hindsight"),
    "lethyx-mempalace": ProviderSpec("lethyx-mempalace", "lethyx", "mempalace"),
    "lethyx-mnemosyne": ProviderSpec("lethyx-mnemosyne", "lethyx", "mnemosyne"),
    "lethyx-composite": ProviderSpec(
        "lethyx-composite", "lethyx", "composite", include_passive=True
    ),
}

PROVIDER_ALIASES = {
    "lethyx": "lethyx-hindsight",
    "lethyx_hindsight": "lethyx-hindsight",
    "lethyx_mempalace": "lethyx-mempalace",
    "lethyx_mnemosyne": "lethyx-mnemosyne",
    "lethyx_composite": "lethyx-composite",
}


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 240,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is required for Hermes profile config editing")
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("pyyaml is required for Hermes profile config editing")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _profile_home(profile: str) -> Path:
    return HERMES_HOME / "profiles" / profile


def _safe_profile_suffix(provider_id: str) -> str:
    return provider_id.replace("-", "_").replace(".", "_")


def _install_lethyx_plugin_wrapper(profile: str) -> None:
    """Force clean benchmark profiles to load the current Lethyx source tree."""

    from lethyx.integrations.hermes_provider import hermes_plugin_wrapper_source

    target = _profile_home(profile) / "plugins" / "lethyx" / "__init__.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(hermes_plugin_wrapper_source(), encoding="utf-8")


def _normalize_provider(provider: str) -> str:
    key = provider.strip().lower().replace("_", "-")
    key = PROVIDER_ALIASES.get(provider.strip().lower(), key)
    if key not in PROVIDER_SPECS:
        raise ValueError(f"unknown provider '{provider}'. Known: {', '.join(PROVIDER_SPECS)}")
    return key


def _create_clean_profile(profile: str, provider: str) -> None:
    if _profile_home(profile).exists():
        _run(["hermes", "profile", "delete", profile, "--yes"], timeout=120)
    created = _run(
        ["hermes", "profile", "create", profile, "--clone-all", "--no-alias"], timeout=180
    )
    if created.returncode != 0:
        raise RuntimeError(f"profile create failed for {profile}: {created.stderr[-1000:]}")
    reset = _run(["hermes", "-p", profile, "memory", "reset", "--yes"], timeout=120)
    if reset.returncode != 0:
        raise RuntimeError(f"memory reset failed for {profile}: {reset.stderr[-1000:]}")
    cfg_path = _profile_home(profile) / "config.yaml"
    cfg = _load_yaml(cfg_path)
    cfg.setdefault("memory", {})["memory_enabled"] = True
    cfg.setdefault("memory", {})["user_profile_enabled"] = True
    cfg.setdefault("memory", {})["provider"] = provider
    cfg["toolsets"] = ["memory"]
    cfg.setdefault("platform_toolsets", {})["cli"] = ["memory"]
    _save_yaml(cfg_path, cfg)
    if provider == "lethyx":
        _install_lethyx_plugin_wrapper(profile)


def _seed_builtin(profile: str, scenarios: tuple[Scenario, ...]) -> None:
    mem_dir = _profile_home(profile) / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_dir.joinpath("MEMORY.md").write_text(
        "\n".join(s.facts for s in scenarios) + "\n", encoding="utf-8"
    )
    mem_dir.joinpath("USER.md").write_text("", encoding="utf-8")


def _seed_hindsight(profile: str, scenarios: tuple[Scenario, ...], bank_id: str) -> None:
    hdir = _profile_home(profile) / "hindsight"
    hdir.mkdir(parents=True, exist_ok=True)
    hdir.joinpath("config.json").write_text(
        json.dumps(
            {
                "mode": "local_external",
                "api_url": "http://127.0.0.1:8888",
                "bank_id": bank_id,
                "recall_budget": "mid",
                "memory_mode": "hybrid",
                "auto_retain": False,
                "auto_recall": True,
                "retain_async": False,
                "retain_tags": ["hermes-profile-bench"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(_profile_home(profile))
    env["PYTHONPATH"] = str(HERMES_AGENT_ROOT)
    env["PROFILE"] = profile
    facts_path = hdir / "seed_facts.json"
    facts_path.write_text(
        json.dumps([s.facts for s in scenarios], ensure_ascii=False), encoding="utf-8"
    )
    env["FACTS_PATH"] = str(facts_path)
    code = """
import json, os
from plugins.memory.hindsight import HindsightMemoryProvider
p = HindsightMemoryProvider()
p.initialize('seed', platform='cli', hermes_home=os.environ['HERMES_HOME'], agent_identity=os.environ['PROFILE'], agent_workspace='bench')
try:
    with open(os.environ['FACTS_PATH'], encoding='utf-8') as fh:
        facts = json.load(fh)
    for item in facts:
        result = p.handle_tool_call('hindsight_retain', {'content': item, 'context': 'Hermes memory profile benchmark facts', 'tags': ['hermes-profile-bench']})
        if isinstance(result, dict) and result.get('ok') is False:
            raise SystemExit(result)
finally:
    p.shutdown()
""".strip()
    proc = _run(
        [sys.executable, "-c", code],
        timeout=int(os.environ.get("HERMES_PROFILE_BENCH_SEED_TIMEOUT", "1800")),
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"hindsight seed failed: {proc.stderr[-1000:] or proc.stdout[-1000:]}")


def _lethyx_env(repo_root: Path, backend: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LETHYX_ROOT / "src")
    env["LETHYX_REPO_ROOT"] = str(repo_root)
    env["LETHYX_BACKEND"] = backend
    env["LETHYX_MEMORY_BACKEND"] = backend
    if backend == "composite":
        env["LETHYX_MEMORY_BLOCKS"] = json.dumps(
            [
                {
                    "id": "mempalace.active",
                    "backend": "mempalace",
                    "role": "active",
                    "phases": ["*"],
                    "weight": 1.0,
                },
                {
                    "id": "hindsight.active",
                    "backend": "hindsight",
                    "role": "active",
                    "phases": ["*"],
                    "weight": 0.95,
                },
                {
                    "id": "mnemosyne.active",
                    "backend": "mnemosyne",
                    "role": "active",
                    "phases": ["*"],
                    "weight": 0.90,
                },
            ]
        )
    return env


def _seed_lethyx(
    profile: str, scenarios: tuple[Scenario, ...], repo_root: Path, spec: ProviderSpec
) -> None:
    workspace_id = f"hermes_profile_bench_{_safe_profile_suffix(spec.id)}"
    udir = _profile_home(profile) / "lethyx"
    udir.mkdir(parents=True, exist_ok=True)
    udir.joinpath("config.json").write_text(
        json.dumps(
            {
                "code_root": str(LETHYX_ROOT),
                "repo_root": str(repo_root),
                "workspace_id": workspace_id,
                "backend": spec.lethyx_backend,
                "vault_dir": str(repo_root / "obsidian_vault"),
                "auto_prefetch": True,
                "auto_retain": False,
                "auto_export_obsidian": False,
                "auto_bootstrap": False,
                "include_passive": spec.include_passive,
                "timeout_seconds": 180,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env = _lethyx_env(repo_root, spec.lethyx_backend)
    for scenario in scenarios:
        payload = {
            "workspace_id": workspace_id,
            "backend": spec.lethyx_backend,
            "repo_root": str(repo_root),
            "content": scenario.facts,
            "title": f"Benchmark facts: {scenario.repo_name}",
            "kind": "project_fact",
            "memory_id": f"bench_{scenario.id}",
            "tags": ["hermes-profile-bench", scenario.id, scenario.repo_name],
            "metadata": {"memory_activity": "active", "repo_name": scenario.repo_name},
            "trust_level": "workspace_verified",
            "evidence_refs": [
                {
                    "ref_type": "benchmark_seed",
                    "ref_id": scenario.id,
                    "produced_by": "hermes_profile_matrix",
                }
            ],
        }
        proc = _run(
            [sys.executable, "-m", "lethyx.memory.mcp_call", "remember"],
            input_text=json.dumps(payload),
            timeout=300,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Lethyx seed failed for {spec.id}/{scenario.id}: {proc.stderr[-1000:] or proc.stdout[-1000:]}"
            )
        try:
            result = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Lethyx seed returned non-JSON for {spec.id}: {proc.stdout[-1000:]}"
            ) from exc
        if not result.get("ok"):
            raise RuntimeError(f"Lethyx seed returned not ok for {spec.id}: {result}")


def _ask(profile: str, question: Question) -> tuple[str, float, str, int]:
    prompt = (
        "Do not read files, terminal, web, or session_search. Use only this profile's connected memory/provider tools. "
        "Use positive memory evidence to answer negative/comparative questions: if memory identifies the subject "
        "but has no evidence for named alternatives, say no memory-backed fact supports those alternatives and include the subject name. "
        "Do not separately recall named alternatives or list unrelated facts about alternatives unless memory ties them directly to the subject. "
        "Only say 'not in memory' when no relevant subject memory is connected. "
        "Answer concisely but include relevant exact paths, commands, and names.\n\n"
        f"Question: {question.text}"
    )
    start = time.perf_counter()
    proc = _run(
        ["hermes", "-p", profile, "chat", "-Q", "--toolsets", "memory", "-q", prompt], timeout=720
    )
    elapsed = time.perf_counter() - start
    return proc.stdout.strip(), elapsed, proc.stderr.strip(), proc.returncode


def _term_negated_spans(answer: str, term: str) -> list[tuple[int, int]]:
    low = answer.lower()
    escaped = re.escape(term.lower())
    local_patterns = (
        rf"(?:does|do|did|is|are)\s+not\s+[^.!?\n]{{0,80}}{escaped}",
        rf"(?:doesn't|isn't|aren't)\s+[^.!?\n]{{0,80}}{escaped}",
        rf"not\s+(?:as\s+)?[^.!?\n]{{0,80}}{escaped}",
        rf"no\s+(?:memory[-\s]backed\s+)?(?:fact|evidence|record)\s+(?:says|shows|indicates|supports|lists|names|includes)\s+[^.!?\n]{{0,160}}{escaped}",
        rf"no\s+(?:memory[-\s]backed\s+)?(?:fact|evidence|record)\s+[^.!?\n]{{0,80}}\s+(?:says|shows|indicates|supports|lists|names|includes)\s+[^.!?\n]{{0,160}}{escaped}",
        rf"no\s+memory\s+entries?\s+found\s+[\s\S]{{0,160}}{escaped}",
        rf"not\s+in\s+memory\s+[\s\S]{{0,160}}{escaped}",
        rf"no\s+(?:use of\s+)?{escaped}",
        rf"(?:neither|nor)\s+[^.!?\n]{{0,80}}{escaped}",
        rf"{escaped}\s+(?:is|are)?\s*(?:not|absent|missing|unsupported|not supported)",
        rf"{escaped}[^.!?\n]{{0,80}}(?:is|are)\s+not\s+supported",
        rf"{escaped}[^.!?\n]{{0,80}}not\s+supported\s+by",
        rf"{escaped}[:\s-]+(?:not in memory|absent|missing|no)",
    )
    spans: list[tuple[int, int]] = []
    for pattern in local_patterns:
        spans.extend((match.start(), match.end()) for match in re.finditer(pattern, low))
    return spans


def _term_is_locally_negated(answer: str, term: str) -> bool:
    low = answer.lower()
    occurrences = list(re.finditer(re.escape(term.lower()), low))
    if not occurrences:
        return False
    negated_spans = _term_negated_spans(answer, term)
    if not negated_spans:
        return False
    return all(
        any(
            span_start <= occ.start() and occ.end() <= span_end
            for span_start, span_end in negated_spans
        )
        for occ in occurrences
    )


def _score(answer: str, question: Question) -> dict[str, Any]:
    low = answer.lower()
    required = [term for term in question.required_terms if term.lower() in low]
    forbidden = [term for term in question.forbidden_terms if term.lower() in low]
    negated_forbidden = [term for term in forbidden if _term_is_locally_negated(answer, term)]
    penalized_forbidden = [term for term in forbidden if term not in negated_forbidden]
    return {
        "required_found": required,
        "required_missing": [term for term in question.required_terms if term not in required],
        "required_total": len(question.required_terms),
        "forbidden_found": forbidden,
        "forbidden_penalized": penalized_forbidden,
        "forbidden_negated": negated_forbidden,
        "score": max(
            0.0, (len(required) / max(1, len(question.required_terms))) - len(penalized_forbidden)
        ),
    }


def _select_questions(max_questions: int = 0) -> list[tuple[Scenario, Question]]:
    pairs = [(scenario, question) for scenario in SCENARIOS for question in scenario.questions]
    if max_questions > 0:
        return pairs[:max_questions]
    return pairs


def _write_reports(
    out_dir: Path, rows: list[dict[str, Any]], providers: list[str]
) -> tuple[Path, Path, Path]:
    jsonl = out_dir / "results.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    summary: dict[str, Any] = {}
    for provider in providers:
        provider_rows = [row for row in rows if row["provider"] == provider]
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
        }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = out_dir / "report.md"
    lines = [
        "# Hermes Memory Profile Matrix",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Provider summary",
        "",
        "| Provider | Rows | Avg score | Avg seconds | Max seconds | Failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for provider in providers:
        item = summary[provider]
        lines.append(
            f"| {provider} | {item['rows']} | {item['avg_score']:.4f} | {item['avg_seconds']:.3f} | {item['max_seconds']:.3f} | {item['failures']} |"
        )
    lines += ["", "## Public repo candidates", ""]
    for item in PUBLIC_REPO_CANDIDATES:
        lines.append(f"- **{item['domain']}** — {item['name']} ({item['url']}): {item['why']}")
    lines += [
        "",
        "## Per-question results",
        "",
        "| Provider | Scenario | Question | Score | Seconds |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['provider']} | {row['scenario']} | {row['question']} | {row['score']['score']:.2f} | {row['elapsed_seconds']:.3f} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl, summary_path, report


def run_matrix(providers: list[str], out_dir: Path, *, max_questions: int = 0) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    incremental_jsonl = out_dir / "results.jsonl"
    progress_path = out_dir / "progress.json"
    incremental_jsonl.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    prefix = "hmpb_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    specs = [PROVIDER_SPECS[_normalize_provider(provider)] for provider in providers]
    normalized_ids = [spec.id for spec in specs]
    profile_for_provider: dict[str, str] = {}
    selected_scenarios = tuple(
        {scenario.id: scenario for scenario, _ in _select_questions(max_questions)}.values()
    )
    for spec in specs:
        profile = f"{prefix}_{_safe_profile_suffix(spec.id)}"
        profile_for_provider[spec.id] = profile
        provider_store = out_dir / "stores" / _safe_profile_suffix(spec.id)
        _create_clean_profile(profile, spec.hermes_provider)
        if spec.id == "builtin":
            _seed_builtin(profile, selected_scenarios)
        elif spec.id == "hindsight":
            _seed_hindsight(profile, selected_scenarios, bank_id=f"{prefix}_direct_hindsight")
        elif spec.hermes_provider == "lethyx":
            _seed_lethyx(profile, selected_scenarios, provider_store, spec)
    if any(spec.id == "hindsight" or spec.hermes_provider == "lethyx" for spec in specs):
        time.sleep(6)
    for spec in specs:
        profile = profile_for_provider[spec.id]
        for scenario, question in _select_questions(max_questions):
            answer, elapsed, stderr, returncode = _ask(profile, question)
            row = {
                "provider": spec.id,
                "profile": profile,
                "hermes_provider": spec.hermes_provider,
                "lethyx_backend": spec.lethyx_backend,
                "include_passive": spec.include_passive,
                "scenario": scenario.id,
                "repo_name": scenario.repo_name,
                "question": question.id,
                "elapsed_seconds": round(elapsed, 3),
                "score": _score(answer, question),
                "answer": answer,
                "stderr": stderr[-1000:],
                "returncode": returncode,
            }
            rows.append(row)
            with incremental_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress_path.write_text(
                json.dumps(
                    {
                        "rows_done": len(rows),
                        "rows_total": len(normalized_ids) * len(_select_questions(max_questions)),
                        "current_provider": spec.id,
                        "current_scenario": scenario.id,
                        "current_question": question.id,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    jsonl, summary_path, report = _write_reports(out_dir, rows, normalized_ids)
    return {
        "out_dir": str(out_dir),
        "rows": len(rows),
        "providers": normalized_ids,
        "questions_per_provider": len(_select_questions(max_questions)),
        "jsonl": str(jsonl),
        "summary": str(summary_path),
        "report": str(report),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--providers",
        default="builtin,hindsight,lethyx-hindsight,lethyx-mempalace,lethyx-mnemosyne,lethyx-composite",
        help=f"Comma-separated providers. Known: {', '.join(PROVIDER_SPECS)}",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--max-questions", type=int, default=0, help="Limit questions per provider for smoke runs"
    )
    args = parser.parse_args(argv)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("runs/hermes_memory_profiles")
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    result = run_matrix(providers, out_dir, max_questions=args.max_questions)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
