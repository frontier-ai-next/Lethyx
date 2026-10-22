import json

from um_mem.integrations.hermes_provider import UmMemMemoryProvider


def test_compact_recall_preserves_retryable_failure_without_claiming_empty_memory():
    provider = UmMemMemoryProvider()
    result = json.loads(
        provider._compact_recall_tool_result(
            {
                "ok": False,
                "error": "private transport details",
                "reason_code": "native_backend_unavailable",
                "error_type": "ConnectionError",
                "retryable": True,
                "hits": [],
            },
            query_hash="query",
            max_tokens=256,
            context_tokens=128,
        )
    )
    assert result["ok"] is False
    assert result["reason_code"] == "native_backend_unavailable"
    assert result["retryable"] is True
    assert result["error_type"] == "ConnectionError"
    assert "private transport details" not in json.dumps(result)
