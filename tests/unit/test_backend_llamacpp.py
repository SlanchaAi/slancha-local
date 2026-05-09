"""LlamaCppBackend probe + chat tests (mocked HTTP)."""

from __future__ import annotations

import httpx
import pytest
import respx

from slancha_local.backends.llamacpp import LlamaCppBackend
from slancha_local.proxy.models import ChatCompletionRequest, ChatMessage

_TAGS_RESPONSE = {
    "object": "list",
    "data": [
        {"id": "qwen3-8b-instruct.Q4_K_M.gguf", "object": "model"},
        {"id": "codestral-22b-v0.1.Q4_K_M.gguf", "object": "model"},
    ],
}


_CHAT_RESPONSE = {
    "id": "chatcmpl-llamacpp-1",
    "object": "chat.completion",
    "created": 1715275000,
    "model": "qwen3-8b-instruct.Q4_K_M.gguf",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi from llamacpp"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
}


@respx.mock
async def test_probe_returns_models():
    respx.get("http://127.0.0.1:8080/v1/models").mock(return_value=httpx.Response(200, json=_TAGS_RESPONSE))
    backend = LlamaCppBackend(base_url="http://127.0.0.1:8080")
    cap = await backend.probe()
    assert cap.healthy is True
    assert {m.model_id for m in cap.models} == {
        "qwen3-8b-instruct.Q4_K_M.gguf",
        "codestral-22b-v0.1.Q4_K_M.gguf",
    }


@respx.mock
async def test_probe_marks_unhealthy_on_5xx():
    respx.get("http://127.0.0.1:8080/v1/models").mock(return_value=httpx.Response(503))
    backend = LlamaCppBackend(base_url="http://127.0.0.1:8080")
    cap = await backend.probe()
    assert cap.healthy is False
    assert cap.models == ()


@respx.mock
async def test_probe_marks_unhealthy_on_connect_error():
    respx.get("http://127.0.0.1:8080/v1/models").mock(side_effect=httpx.ConnectError("nope"))
    backend = LlamaCppBackend(base_url="http://127.0.0.1:8080")
    cap = await backend.probe()
    assert cap.healthy is False


@respx.mock
async def test_chat_returns_openai_compat_dict():
    respx.post("http://127.0.0.1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    backend = LlamaCppBackend(base_url="http://127.0.0.1:8080")
    req = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    resp = await backend.chat("qwen3-8b-instruct.Q4_K_M.gguf", req)
    assert resp["choices"][0]["message"]["content"] == "hi from llamacpp"


@respx.mock
async def test_chat_5xx_raises():
    respx.post("http://127.0.0.1:8080/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )
    backend = LlamaCppBackend(base_url="http://127.0.0.1:8080")
    req = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(httpx.HTTPStatusError):
        await backend.chat("foo", req)


@respx.mock
async def test_capability_inference_for_codestral():
    respx.get("http://127.0.0.1:8080/v1/models").mock(return_value=httpx.Response(200, json=_TAGS_RESPONSE))
    backend = LlamaCppBackend(base_url="http://127.0.0.1:8080")
    cap = await backend.probe()
    coder = next(m for m in cap.models if "codestral" in m.model_id)
    assert "coding" in coder.capabilities
