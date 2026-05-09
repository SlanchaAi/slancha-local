"""Streaming chat E2E: stream=true returns SSE + decision-trace header."""

from __future__ import annotations

import httpx
import respx
from fastapi.testclient import TestClient

SSE_BODY = (
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{"content":" there"}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


def _build_app(monkeypatch, tmp_path):
    monkeypatch.setenv("SLANCHA_CLASSIFIER_KIND", "rules")
    monkeypatch.setenv("SLANCHA_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("SLANCHA_TRACES_ROOT", str(tmp_path / "traces"))
    from slancha_local.proxy.main import build_app

    return build_app()


@respx.mock
def test_stream_returns_sse_with_decision_trace_header(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"model": "qwen3:8b"}]})
    )
    respx.post("http://127.0.0.1:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=SSE_BODY,
            headers={"content-type": "text/event-stream"},
        )
    )

    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "auto",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers.get("slancha-decision-trace"), "missing decision-trace header"
        assert "picked=" in r.headers["slancha-decision-trace"]
        body = b""
        for chunk in r.iter_raw():
            body += chunk

    assert b"Hi" in body
    assert b"[DONE]" in body
    assert b"there" in body
