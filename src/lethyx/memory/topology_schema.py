"""Shared public split/merge wire schemas; no storage or semantic routing."""

BLOCK_EVOLUTION_TARGET_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "synopsis": {"type": "string", "minLength": 1},
        "record_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
            "description": "Exact current record IDs assigned to this target, not source IDs. Enumerate all pages and cover every source record exactly once across targets.",
        },
        "tags": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "metadata": {"type": "object"},
        "token_budget": {"type": "integer", "minimum": 0, "default": 1000},
    },
    "required": ["name", "synopsis", "record_ids"],
    "additionalProperties": False,
}
BLOCK_MERGE_SOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "block_id": {"type": "string", "minLength": 1},
        "expected_revision": {"type": "integer", "minimum": 1},
    },
    "required": ["block_id", "expected_revision"],
    "additionalProperties": False,
}
