"""Token estimation used by bounded memory payloads."""


def estimate_tokens(text: str) -> int:
    """Heuristic ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(str(text)) // 4)
