"""Hindsight adapter errors."""


class HindsightError(RuntimeError):
    """Base class for Hindsight archive adapter failures."""


class HindsightPolicyError(HindsightError):
    """Raised when UM-MEM policy forbids retaining or promoting a memory."""


class HindsightUnavailableError(HindsightError):
    """Raised when Hindsight is enabled but the client/server is unavailable."""
