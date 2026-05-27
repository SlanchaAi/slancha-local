"""L1 — in-mesh proxy degrade.

When a self-hosted specialist backend (a LoRA, here `demo-model-v2`) is down
but the proxy + base backend (`demo-model`) are alive, a request for the
specialist must degrade to the base instead of 502ing, stamping
X-Slancha-Fallback so callers can detect the base-no-lora output.

The specialist→base map ships empty and is deployment-configured
(`SLANCHA_MESH_FALLBACK_MAP`); these tests inject a map rather than relying on
any shipped default.

The probe-driven catalog drops a down backend within its TTL, so the
steady-state degrade is pre-dispatch (works for streaming + non-streaming).
The <=TTL transient (backend dies mid-cache) is covered by the
non-streaming post-dispatch retry.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from slancha_local.backends.base import (
    Backend,
    BackendCapability,
    BackendModel,
)
from slancha_local.backends.registry import BackendRegistry
from slancha_local.capability.catalog import LocalCatalog
from slancha_local.proxy.mesh_fallback import (
    MESH_LOCAL_FALLBACK,
    _load_fallback_map,
    resolve_local_fallback_target,
)
from slancha_local.proxy.models import ChatCompletionRequest

# Deployment-configured map the tests inject (the shipped default is empty).
_FB = {"demo-model-v2": "demo-model"}


# ── pure resolver ──────────────────────────────────────────────────────────

def _catalog_with(*model_ids: tuple[str, str]) -> LocalCatalog:
    """Build a catalog from (backend_id, model_id) pairs."""
    caps = tuple(
        BackendCapability(
            id=bid,
            healthy=True,
            base_url="http://x",
            models=(BackendModel(backend_id=bid, model_id=mid, ctx_window=8192),),
        )
        for bid, mid in model_ids
    )
    return LocalCatalog(capabilities=caps)


def test_resolve_returns_base_target_when_base_present():
    cat = _catalog_with(("vllm", "demo-model"))
    assert resolve_local_fallback_target("demo-model-v2", cat, fallback_map=_FB) == "local:vllm:demo-model"


def test_resolve_none_when_base_backend_absent():
    """Base not in the live catalog (both down) → no fallback."""
    cat = _catalog_with(("ollama", "qwen3:8b"))
    assert resolve_local_fallback_target("demo-model-v2", cat, fallback_map=_FB) is None


def test_resolve_none_for_unknown_specialist():
    cat = _catalog_with(("vllm", "demo-model"))
    assert resolve_local_fallback_target("demo-model", cat, fallback_map=_FB) is None
    assert resolve_local_fallback_target("auto", cat, fallback_map=_FB) is None


def test_fallback_map_empty_by_default():
    """Ships empty — no deployment-specific model ids hardcoded in OSS."""
    assert MESH_LOCAL_FALLBACK == {}


def test_load_fallback_map_from_env(monkeypatch):
    monkeypatch.setenv("SLANCHA_MESH_FALLBACK_MAP", '{"spec-v2": "base"}')
    assert _load_fallback_map() == {"spec-v2": "base"}


def test_load_fallback_map_ignores_invalid(monkeypatch):
    monkeypatch.setenv("SLANCHA_MESH_FALLBACK_MAP", "not json")
    assert _load_fallback_map() == {}
    monkeypatch.setenv("SLANCHA_MESH_FALLBACK_MAP", '["not", "an object"]')
    assert _load_fallback_map() == {}


# ── endpoint integration ─────────────────────────────────────────────────────

class _FakeProbe:
    def __init__(self, catalog: LocalCatalog) -> None:
        self._catalog = catalog

    async def get(self) -> LocalCatalog:
        return self._catalog


class _FakeBackend(Backend):
    """OpenAI-compat-shaped fake. `fail` makes chat() raise (transient case)."""

    def __init__(self, backend_id: str, *, content: str = "ok", fail: bool = False) -> None:
        self.id = backend_id
        self._content = content
        self._fail = fail

    async def probe(self) -> BackendCapability:  # not used (probe is faked)
        return BackendCapability(id=self.id, healthy=True, base_url="http://x", models=())

    async def chat(self, model_id: str, request: ChatCompletionRequest) -> dict[str, Any]:
        if self._fail:
            raise ConnectionError("backend down")
        return {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "model": model_id,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": self._content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    async def chat_stream(self, model_id: str, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        if self._fail:
            raise ConnectionError("backend down")
        yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        yield b"data: [DONE]\n\n"


def _build_app(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SLANCHA_CLASSIFIER_KIND", "rules")
    monkeypatch.setenv("SLANCHA_TRACES_ROOT", str(tmp_path / "traces"))
    monkeypatch.setenv("SLANCHA_BIND_HOST", "127.0.0.1")
    # The shipped map is empty; inject the deployment map the dispatch path reads.
    monkeypatch.setattr("slancha_local.proxy.mesh_fallback.MESH_LOCAL_FALLBACK", dict(_FB))
    from slancha_local.proxy.main import build_app

    return build_app()


def test_v8_down_degrades_to_base_with_header(monkeypatch, tmp_path):
    """Steady state: demo-model-v2 absent from catalog (backend down), base
    demo-model on vllm is healthy → degrade + header + correct trace."""
    app = _build_app(monkeypatch, tmp_path)
    catalog = _catalog_with(("vllm", "demo-model"))  # NB: no demo-model-v2
    app.state.probe = _FakeProbe(catalog)
    app.state.registry = BackendRegistry([_FakeBackend("vllm", content="base-voice")])

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "demo-model-v2", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "base-voice"
    assert r.headers.get("x-slancha-fallback") == "base-no-lora; specialist=demo-model-v2"
    assert "picked=local:vllm:demo-model" in r.headers.get("slancha-decision-trace", "")


def test_v8_transient_dispatch_failure_retries_base(monkeypatch, tmp_path):
    """Transient: both backends in catalog (probe fresh) but v8 dispatch
    raises → non-streaming retries the base and degrades."""
    app = _build_app(monkeypatch, tmp_path)
    catalog = _catalog_with(("generic-openai", "demo-model-v2"), ("vllm", "demo-model"))
    app.state.probe = _FakeProbe(catalog)
    app.state.registry = BackendRegistry(
        [
            _FakeBackend("generic-openai", fail=True),  # v8 dies at dispatch
            _FakeBackend("vllm", content="base-voice"),
        ]
    )

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "demo-model-v2", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "base-voice"
    assert r.headers.get("x-slancha-fallback") == "base-no-lora; specialist=demo-model-v2"


def test_both_down_returns_502_no_fallback(monkeypatch, tmp_path):
    """v8 in catalog but dispatch fails AND base also fails → 502 (no silent
    success). Fallback header absent."""
    app = _build_app(monkeypatch, tmp_path)
    catalog = _catalog_with(("generic-openai", "demo-model-v2"), ("vllm", "demo-model"))
    app.state.probe = _FakeProbe(catalog)
    app.state.registry = BackendRegistry(
        [_FakeBackend("generic-openai", fail=True), _FakeBackend("vllm", fail=True)]
    )

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "demo-model-v2", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 502, r.text
    assert "x-slancha-fallback" not in r.headers
