"""HTTP header values must be latin-1 / ASCII-safe — no em-dashes, no unicode."""

from __future__ import annotations

from slancha_local.proxy.middleware import _ascii_safe, format_trace


def test_ascii_safe_replaces_em_dash():
    assert _ascii_safe("foo — bar") == "foo -- bar"


def test_ascii_safe_handles_smart_quotes():
    assert _ascii_safe("don’t") == "don't"
    assert _ascii_safe("“hello”") == '"hello"'


def test_ascii_safe_drops_unencodable():
    out = _ascii_safe("hello \U0001f600 world")
    assert "?" in out  # the emoji becomes a replacement char
    out.encode("ascii")  # round-trip safe


def test_format_trace_emoji_in_reason_does_not_crash():
    s = format_trace(
        picked="local:ollama:qwen3:8b",
        reason="prompt looked spicy \U0001f525 — escalating to hard model",
        fallbacks=["local:ollama:qwen3:8b"],
        domain="coding",
        difficulty="hard",
        jailbreak=False,
        pii=False,
        tool_calling=False,
        confidence=0.85,
        classifier_ms=10.0,
        total_overhead_ms=15.0,
    )
    s.encode("latin-1")  # must not raise


def test_format_trace_em_dash_in_reason_is_safe():
    s = format_trace(
        picked="local:ollama:codestral:22b",
        reason="domain=computer science — coding-capable model preferred",
        fallbacks=[],
        domain="computer science",
        difficulty="medium",
        jailbreak=False,
        pii=False,
        tool_calling=False,
        confidence=0.85,
        classifier_ms=4.1,
        total_overhead_ms=8.0,
    )
    s.encode("latin-1")
    assert "--" in s  # em-dash was rewritten
    assert "—" not in s
