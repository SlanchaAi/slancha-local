"""OpenRouter backend: auth header threading + model probe + opt-in default-off."""

from __future__ import annotations

import httpx
import pytest
import respx

from slancha_local.backends.openai_compat import (
    OpenAICompatBackend,
    OpenRouterBackend,
    _openrouter_capabilities,
    _openrouter_ctx,
)

# ---------- capability + ctx inference ----------


@pytest.mark.parametrize(
    "model,want_caps_subset",
    [
        ("anthropic/claude-3.5-sonnet", {"frontier", "vision", "multilingual"}),
        ("openai/gpt-4o", {"frontier", "vision", "multilingual"}),
        ("google/gemini-2.5-pro", {"frontier", "vision", "multilingual"}),
        ("meta-llama/llama-3.1-8b-instruct", set()),
    ],
)
def test_openrouter_capabilities(model: str, want_caps_subset: set[str]):
    caps = set(_openrouter_capabilities(model))
    assert want_caps_subset.issubset(caps), f"missing {want_caps_subset - caps} in {caps}"


@pytest.mark.parametrize(
    "model,expected_ctx",
    [
        ("anthropic/claude-3.5-sonnet", 200_000),
        ("google/gemini-2.5-pro", 1_000_000),
        ("openai/gpt-4o", 128_000),
        ("meta-llama/llama-3.1-8b-instruct", 8192),  # falls through to _generic_ctx default
    ],
)
def test_openrouter_ctx(model: str, expected_ctx: int):
    assert _openrouter_ctx(model) == expected_ctx


# ---------- auth header threading ----------


def test_compat_backend_omits_auth_when_no_key():
    b = OpenAICompatBackend(base_url="http://x")
    assert "Authorization" not in (b._client.headers or {})


def test_compat_backend_threads_bearer_when_key_set():
    b = OpenAICompatBackend(base_url="http://x", api_key="sk-test")
    assert b._client.headers.get("Authorization") == "Bearer sk-test"


def test_compat_backend_extra_headers_merge():
    b = OpenAICompatBackend(
        base_url="http://x",
        api_key="sk-test",
        extra_headers={"HTTP-Referer": "https://slancha.ai", "X-Title": "test"},
    )
    assert b._client.headers.get("Authorization") == "Bearer sk-test"
    assert b._client.headers.get("HTTP-Referer") == "https://slancha.ai"
    assert b._client.headers.get("X-Title") == "test"


# ---------- probe ----------


@respx.mock
async def test_openrouter_probe_returns_models():
    base = "https://openrouter.ai/api"
    respx.get(f"{base}/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "anthropic/claude-3.5-sonnet"},
                    {"id": "openai/gpt-4o"},
                    {"id": "meta-llama/llama-3.1-70b-instruct"},
                ]
            },
        )
    )

    b = OpenRouterBackend(base_url=base, api_key="sk-or-v1-test")
    cap = await b.probe()
    assert cap.healthy
    assert cap.id == "openrouter"
    ids = {m.model_id for m in cap.models}
    assert "anthropic/claude-3.5-sonnet" in ids
    assert "openai/gpt-4o" in ids
    # 200K ctx for claude
    claude = next(m for m in cap.models if m.model_id == "anthropic/claude-3.5-sonnet")
    assert claude.ctx_window == 200_000


@respx.mock
async def test_openrouter_probe_unhealthy_on_401():
    base = "https://openrouter.ai/api"
    respx.get(f"{base}/v1/models").mock(return_value=httpx.Response(401, json={"error": "missing api key"}))

    b = OpenRouterBackend(base_url=base, api_key="bad")
    cap = await b.probe()
    assert not cap.healthy
    assert cap.models == ()


# ---------- chat round-trip ----------


@respx.mock
async def test_openrouter_chat_threads_auth_and_model():
    from slancha_local.proxy.models import ChatCompletionRequest, ChatMessage

    base = "https://openrouter.ai/api"
    route = respx.post(f"{base}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "or-1",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )

    b = OpenRouterBackend(base_url=base, api_key="sk-or-v1-test")
    req = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")], stream=False)
    body = await b.chat("anthropic/claude-3.5-sonnet", req)
    assert body["choices"][0]["message"]["content"] == "hi"
    sent = route.calls.last.request
    assert sent.headers.get("authorization") == "Bearer sk-or-v1-test"
    payload = httpx_json(sent)
    assert payload["model"] == "anthropic/claude-3.5-sonnet"
    assert payload["stream"] is False


def httpx_json(request: httpx.Request) -> dict:
    import json as _json

    return _json.loads(request.content)


# ---------- proxy wiring (opt-in default-off) ----------


def test_openrouter_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("SLANCHA_TRACES_ROOT", str(tmp_path / "traces"))
    # critical: do NOT set SLANCHA_OPENROUTER_ENABLED
    from slancha_local.config import Settings

    s = Settings()
    assert s.openrouter_enabled is False
    assert s.openrouter_api_key is None


def test_openrouter_enabled_picks_up_key(monkeypatch, tmp_path):
    monkeypatch.setenv("SLANCHA_TRACES_ROOT", str(tmp_path / "traces"))
    monkeypatch.setenv("SLANCHA_OPENROUTER_ENABLED", "true")
    monkeypatch.setenv("SLANCHA_OPENROUTER_API_KEY", "sk-or-v1-test")
    from slancha_local.config import Settings

    s = Settings()
    assert s.openrouter_enabled is True
    assert s.openrouter_api_key == "sk-or-v1-test"
