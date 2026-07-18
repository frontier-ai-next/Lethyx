"""Lethyx-owned truncation and preview limits."""

import os


def _int_env(
    name: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raw = str(default)
    try:
        return max(minimum, int(raw))
    except (ValueError, TypeError):
        return max(minimum, default)


TOOL_LOG_PREVIEW_CHARS = _int_env(
    "LETHYX_TOOL_LOG_PREVIEW_CHARS",
    16000,
    minimum=2000,
)
TOOL_RESULT_TO_MODEL_CHARS = _int_env(
    "LETHYX_TOOL_RESULT_TO_MODEL_CHARS",
    48000,
    minimum=4000,
)
TOOL_TRACE_SNIPPET_CHARS = _int_env(
    "LETHYX_TOOL_TRACE_SNIPPET_CHARS",
    2400,
    minimum=400,
)
TOOL_ARGS_LOG_CHARS = _int_env(
    "LETHYX_TOOL_ARGS_LOG_CHARS",
    12000,
    minimum=500,
)
TRUNCATE_FOR_LOG_DEFAULT = _int_env(
    "LETHYX_TRUNCATE_FOR_LOG_DEFAULT",
    16000,
    minimum=1000,
)
TASK_TEXT_LOG_CHARS = _int_env(
    "LETHYX_TASK_TEXT_LOG_CHARS",
    16000,
    minimum=1000,
)
DISCOVERY_CONTENT_CHARS = _int_env(
    "LETHYX_DISCOVERY_CONTENT_CHARS",
    12000,
    minimum=2000,
)
RECALL_DRAWER_PREVIEW_CHARS = _int_env(
    "LETHYX_RECALL_DRAWER_PREVIEW_CHARS",
    1200,
    minimum=200,
)
RECALL_LESSON_PREVIEW_CHARS = _int_env(
    "LETHYX_RECALL_LESSON_PREVIEW_CHARS",
    2000,
    minimum=200,
)
RECALL_BLOCK_MAX_CHARS = _int_env(
    "LETHYX_RECALL_BLOCK_MAX_CHARS",
    8000,
    minimum=1000,
)
TASK_BRIEF_MAX_CHARS = _int_env(
    "LETHYX_TASK_BRIEF_MAX_CHARS",
    8000,
    minimum=500,
)
PHASE_ERROR_ARTIFACT_CHARS = _int_env(
    "LETHYX_PHASE_ERROR_ARTIFACT_CHARS",
    8000,
    minimum=500,
)
WATCHER_TOOL_SNIPPET_CHARS = _int_env(
    "LETHYX_WATCHER_TOOL_SNIPPET_CHARS",
    800,
    minimum=160,
)
WATCHER_TOOL_ARGS_SNIPPET_CHARS = _int_env(
    "LETHYX_WATCHER_TOOL_ARGS_SNIPPET_CHARS",
    480,
    minimum=80,
)
MEMORY_HIT_PREVIEW_CHARS = _int_env(
    "LETHYX_MEMORY_HIT_PREVIEW_CHARS",
    8000,
    minimum=500,
)
