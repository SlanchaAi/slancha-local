"""Mocked-Ollama + rules-classifier E2E. Asserts decision-trace header + zero egress to api.slancha.ai."""

from __future__ import annotations

from pathlib import Path

import httpx
import respx
from fastapi.testclient import TestClient


def _build_app_with_rules_classifier(monkeypatch, tmp_path: Path):
    """Force rules-fallback classifier (no treelite/network); pin trace dir."""
    monkeypatch.setenv("SLANCHA_CLASSIFIER_KIND", "rules")
    monkeypatch.setenv("SLANCHA_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("SLANCHA_API_BASE_URL", "https://api.slancha.ai")
    monkeypatch.setenv("SLANCHA_TRACES_ROOT", str(tmp_path / "traces"))
    monkeypatch.setenv("SLANCHA_BIND_HOST", "127.0.0.1")
    from slancha_local.proxy.main import build_app

    return build_app()


@respx.mock
def test_chat_round_trip_with_decision_trace_header(monkeypatch, tmp_path):
    app = _build_app_with_rules_classifier(monkeypatch, tmp_path)

    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"model": "qwen3:8b", "name": "qwen3:8b"}]},
        )
    )
    respx.post("http://127.0.0.1:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "object": "chat.completion",
                "created": 1715275000,
                "model": "qwen3:8b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
    )

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hello!"

    # Load-bearing: decision-trace header is present and well-formed
    trace = r.headers.get("slancha-decision-trace")
    assert trace, "missing slancha-decision-trace header"
    assert "picked=local:ollama:qwen3:8b" in trace
    assert "reason=" in trace

    # Trace was written to disk
    traces_dir = Path(client.app.state.settings.traces_root)
    files = list(traces_dir.glob("*.jsonl"))
    assert len(files) == 1
    assert len(files[0].read_text().splitlines()) == 1


@respx.mock
def test_default_install_makes_zero_calls_to_api_slancha_ai(monkeypatch, tmp_path):
    """V2 load-bearing: rules+local classifier never calls api.slancha.ai."""
    app = _build_app_with_rules_classifier(monkeypatch, tmp_path)

    no_egress = respx.post("https://api.slancha.ai/v1/classify-routed").mock(
        return_value=httpx.Response(500)
    )
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"model": "qwen3:8b"}]})
    )
    respx.post("http://127.0.0.1:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3:8b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert no_egress.call_count == 0


@respx.mock
def test_decisions_endpoint_returns_recent(monkeypatch, tmp_path):
    app = _build_app_with_rules_classifier(monkeypatch, tmp_path)
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"model": "qwen3:8b"}]})
    )
    respx.post("http://127.0.0.1:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3:8b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    client = TestClient(app)
    client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "ping"}]},
    )
    r = client.get("/v1/decisions/last?n=5")
    assert r.status_code == 200
    assert len(r.json()["decisions"]) >= 1


def test_health_endpoint(monkeypatch, tmp_path):
    app = _build_app_with_rules_classifier(monkeypatch, tmp_path)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
