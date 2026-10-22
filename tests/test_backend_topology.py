import ast
from pathlib import Path
from typing import get_args

from um_mem.contracts import BackendId

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_selected_backend_identity_is_the_only_runtime_backend_axis() -> None:
    assert set(get_args(BackendId)) == {"mempalace", "hindsight", "mnemosyne", "zvec"}


def test_replaced_structured_memory_modules_are_physically_absent() -> None:
    memory_root = REPO_ROOT / "src/um_mem/memory"
    retired_modules = (
        "store.py",
        "models.py",
        "lessons.py",
        "competency.py",
        "summarization.py",
        "context_builder.py",
        "relevance.py",
        "contrastive.py",
        "user_profile.py",
        "reflection.py",
        "harness_stats.py",
    )
    assert [name for name in retired_modules if (memory_root / name).exists()] == []

    retired_imports = {f"um_mem.memory.{name.removesuffix('.py')}" for name in retired_modules}
    imports: set[str] = set()
    for path in (REPO_ROOT / "src/um_mem").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.add(node.module)
    assert retired_imports.isdisjoint(imports)
