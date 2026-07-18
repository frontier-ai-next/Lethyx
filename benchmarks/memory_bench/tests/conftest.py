"""Pytest configuration for memory_bench tests.

memory_bench/ lives at the repo root.  We add the repo root to sys.path so
that `import memory_bench` works without the package being pip-installed.
"""

import sys
from pathlib import Path

# Repo root (contains memory_bench/, lethyx/, etc.)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
