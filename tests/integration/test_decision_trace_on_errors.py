"""Decision-trace header must be present on EVERY chat response — including errors.

Caught on Spark smoke 2026-05-09: 503 cloud-escalation responses had no
slancha-decision-trace header, defeating the load-bearing differentiator
(gallery / brag / why CLI can't introspect failed decisions otherwise).
"""

from __future__ import annotations

import httpx
import respx
from fastapi.testclient import TestClient


def _build_app(monkeypatch, tmp_path, *, ollama_models: list[dict] | None = None):
    monkeypatch.setenv("SLANCHA_CLASSIFIER_KIND", "rules")
    monkeypatch.setenv("SLANCHA_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("SLANCHA_TRACES_ROOT", str(tmp_path / "traces"))

    # Disable the noisy default-on backends we don't want here.
    monkeypatch.setenv("SLANCHA_LLAMACPP_ENABLED", "false")
    monkeypatch.setenv("SLANCHA_OLLAMA_ENABLED", "true")

    from slancha_local.proxy.main import build_app

    app = build_app()
    return app


@respx.mock
def test_decision_trace_header_present_on_503_cloud_escalation(monkeypatch, tmp_path):
    """Repro of the Spark smoke finding: cloud-escalation 503 must still emit the header."""
    # Empty model catalog → rules fallback escalates to cloud → 503
    respx.get("http://127.0.0.1:11434/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))

    app = _build_app(monkeypatch, tmp_path)
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
        },
    )

    # The bug: 503 came back with no header. The fix: header MUST be present.
    assert r.status_code in {503, 502, 400}, f"unexpected {r.status_code}: {r.text}"
    assert "slancha-decision-trace" in r.headers, (
        f"decision-trace header missing on {r.status_code} response. Headers: {dict(r.headers)}"
    )
    trace = r.headers["slancha-decision-trace"]
    assert "picked=" in trace, f"trace malformed: {trace}"


@respx.mock
def test_decision_trace_header_present_on_502_backend_error(monkeypatch, tmp_path):
    """Backend reachable in /api/tags but errors on the actual chat call → 502 with header."""
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"model": "qwen3:8b"}]})
    )
    respx.post("http://127.0.0.1:11434/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "ollama exploded"})
    )

    app = _build_app(monkeypatch, tmp_path)
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
        },
    )

    assert r.status_code == 502
    assert "slancha-decision-trace" in r.headers, (
        f"header missing on 502 backend-error. Headers: {dict(r.headers)}"
    )
    assert "picked=" in r.headers["slancha-decision-trace"]


@respx.mock
def test_decision_trace_header_on_200_success(monkeypatch, tmp_path):
    """Sanity: success path still has the header (no regression from the fix)."""
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"model": "qwen3:8b"}]})
    )
    respx.post("http://127.0.0.1:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "4"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )
    )

    app = _build_app(monkeypatch, tmp_path)
    client = TestClient(app)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
        },
    )

    assert r.status_code == 200
    assert "slancha-decision-trace" in r.headers
    assert "picked=" in r.headers["slancha-decision-trace"]
