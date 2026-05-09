"""GET /v1/models returns OpenAI-compat list including 'auto' + backend models."""

from __future__ import annotations

import httpx
import respx
from fastapi.testclient import TestClient


def _build_app(monkeypatch, tmp_path):
    monkeypatch.setenv("SLANCHA_CLASSIFIER_KIND", "rules")
    monkeypatch.setenv("SLANCHA_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("SLANCHA_LLAMACPP_ENABLED", "false")
    monkeypatch.setenv("SLANCHA_TRACES_ROOT", str(tmp_path / "traces"))
    from slancha_local.proxy.main import build_app

    return build_app()


@respx.mock
def test_v1_models_includes_auto_and_backend_models(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"model": "qwen3:8b"},
                    {"model": "codestral:22b"},
                ]
            },
        )
    )
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "auto" in ids
    assert "ollama:qwen3:8b" in ids
    assert "ollama:codestral:22b" in ids

    auto_entry = next(m for m in body["data"] if m["id"] == "auto")
    assert auto_entry["owned_by"] == "slancha-local"
    assert auto_entry["metadata"]["routing"] == "per-prompt"

    codestral = next(m for m in body["data"] if m["id"] == "ollama:codestral:22b")
    assert "coding" in codestral["metadata"]["capabilities"]


@respx.mock
def test_v1_models_with_no_backends_still_returns_auto(monkeypatch, tmp_path):
    app = _build_app(monkeypatch, tmp_path)
    respx.get("http://127.0.0.1:11434/api/tags").mock(return_value=httpx.Response(503))
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    ids = [m["id"] for m in body["data"]]
    assert ids == ["auto"]
