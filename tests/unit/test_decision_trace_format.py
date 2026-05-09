"""format_trace produces parseable header value."""

from __future__ import annotations

from slancha_local.proxy.middleware import format_trace


def test_format_trace_contains_all_fields():
    s = format_trace(
        picked="local:ollama:qwen3:8b",
        reason="domain=coding (computer science) — coding-capable model preferred",
        fallbacks=["local:ollama:llama3.3:8b"],
        domain="computer science",
        difficulty="medium",
        jailbreak=False,
        pii=False,
        tool_calling=False,
        confidence=0.85,
        classifier_ms=8.2,
        total_overhead_ms=12.4,
    )
    assert "picked=local:ollama:qwen3:8b" in s
    assert 'reason="domain=coding' in s
    assert "domain=computer science" in s
    assert "difficulty=medium" in s
    assert "jailbreak=no" in s
    assert "confidence=0.85" in s
    assert "classifier_ms=8.2" in s


def test_format_trace_handles_none_confidence():
    s = format_trace(
        picked="cloud:reject:jailbreak",
        reason="classifier flagged jailbreak",
        fallbacks=[],
        domain=None,
        difficulty=None,
        jailbreak=True,
        pii=False,
        tool_calling=False,
        confidence=None,
        classifier_ms=5.0,
        total_overhead_ms=7.0,
    )
    assert "confidence=na" in s
    assert "jailbreak=yes" in s
    assert "domain=unknown" in s
