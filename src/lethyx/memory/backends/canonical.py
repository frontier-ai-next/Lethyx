"""Compatibility wrapper for imports renamed to mempalace.

Product-facing selection no longer accepts a ``canonical`` backend mode; this
module only keeps historical Python imports from crashing during migration.
"""

from lethyx.memory.backends.mempalace import MemPalaceBackend

CanonicalMemoryBackend = MemPalaceBackend

__all__ = ["MemPalaceBackend", "CanonicalMemoryBackend"]
