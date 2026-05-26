"""OpenAICompatBackend variants — vLLM, MLX, LM Studio, generic."""

from __future__ import annotations

import httpx
import pytest
import respx

from slancha_local.backends.openai_compat import (
    GenericOpenAIBackend,
    LMStudioBackend,
    MLXBackend,
    OpenAICompatBackend,
    VLLMBackend,
)
from slancha_local.proxy.models import ChatCompletionRequest, ChatMessage

_MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {"id": "qwen3-8b", "object": "model"},
        {"id": "deepseek-coder-v2-16b", "object": "model"},
    ],
}

_CHAT_RESPONSE = {
    "id": "x",
    "object": "chat.completion",
    "created": 0,
    "model": "qwen3-8b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.mark.parametrize(
    "cls,port",
    [
        (VLLMBackend, 8000),
        (MLXBackend, 8081),
        (LMStudioBackend, 1234),
        (GenericOpenAIBackend, 9999),
    ],
)
@respx.mock
async def test_each_backend_probe_returns_models(cls: type[OpenAICompatBackend], port: int):
    base_url = f"http://127.0.0.1:{port}"
    respx.get(f"{base_url}/v1/models").mock(return_value=httpx.Response(200, json=_MODELS_RESPONSE))
    backend = cls(base_url=base_url)
    cap = await backend.probe()
    assert cap.id == cls.id
    assert cap.healthy is True
    assert {m.model_id for m in cap.models} == {"qwen3-8b", "deepseek-coder-v2-16b"}


@pytest.mark.parametrize("cls", [VLLMBackend, MLXBackend, LMStudioBackend, GenericOpenAIBackend])
@respx.mock
async def test_each_backend_unhealthy_on_error(cls: type[OpenAICompatBackend]):
    base_url = "http://127.0.0.1:9999"
    respx.get(f"{base_url}/v1/models").mock(side_effect=httpx.ConnectError("nope"))
    backend = cls(base_url=base_url)
    cap = await backend.probe()
    assert cap.healthy is False


@pytest.mark.parametrize("exc", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout])
@respx.mock
async def test_probe_unhealthy_on_timeouts_not_crash(exc: type[httpx.HTTPError]):
    """Regression: Windows raises ConnectTimeout connecting to a closed port
    where Linux raises ConnectError. probe() caught ConnectError/ReadTimeout
    but NOT ConnectTimeout → crashed `slancha doctor` on Windows (found on a
    real GTX-1070 Win10 box, 2026-05-26). Catching httpx.HTTPError (the base)
    covers the whole timeout/transport family — probe must never raise."""
    base_url = "http://127.0.0.1:9999"
    respx.get(f"{base_url}/v1/models").mock(side_effect=exc("timed out"))
    cap = await GenericOpenAIBackend(base_url=base_url).probe()
    assert cap.healthy is False


@respx.mock
async def test_vllm_chat_round_trip():
    base_url = "http://127.0.0.1:8000"
    respx.post(f"{base_url}/v1/chat/completions").mock(return_value=httpx.Response(200, json=_CHAT_RESPONSE))
    backend = VLLMBackend(base_url=base_url)
    req = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    resp = await backend.chat("qwen3-8b", req)
    assert resp["choices"][0]["message"]["content"] == "ok"


@respx.mock
async def test_capability_inference_for_coder_model():
    base_url = "http://127.0.0.1:8000"
    respx.get(f"{base_url}/v1/models").mock(return_value=httpx.Response(200, json=_MODELS_RESPONSE))
    backend = VLLMBackend(base_url=base_url)
    cap = await backend.probe()
    coder = next(m for m in cap.models if "deepseek-coder" in m.model_id)
    assert "coding" in coder.capabilities
