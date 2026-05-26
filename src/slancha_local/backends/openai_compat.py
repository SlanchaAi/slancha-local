"""Common base for backends that speak OpenAI-compatible /v1/models + /v1/chat/completions.

llama.cpp server, vLLM, MLX (mlx_lm.server), LM Studio, and any generic OpenAI-compat
endpoint all share this surface; they differ mainly in defaults + capability inference.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from slancha_local.backends.base import Backend, BackendCapability, BackendModel
from slancha_local.proxy.models import ChatCompletionRequest

logger = logging.getLogger(__name__)


class OpenAICompatBackend(Backend):
    """Subclass-and-set ``id`` + ``infer_capabilities`` + ``ctx_default``."""

    id: str = "openai-compat"
    infer_capabilities: Callable[[str], tuple[str, ...]] = staticmethod(lambda name: ("en",))
    ctx_default: Callable[[str], int] = staticmethod(lambda name: 8192)

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 120.0,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers or None)

    async def probe(self) -> BackendCapability:
        try:
            resp = await self._client.get(f"{self._base_url}/v1/models", timeout=2.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:  # base class — covers ConnectTimeout (Windows) too
            logger.warning("%s probe failed: %s", self.id, e)
            return BackendCapability(id=self.id, healthy=False, base_url=self._base_url, models=())

        data = resp.json()
        models: list[BackendModel] = []
        for m in data.get("data", []):
            mid = m.get("id") or m.get("model")
            if not mid:
                continue
            models.append(
                BackendModel(
                    backend_id=self.id,
                    model_id=mid,
                    ctx_window=type(self).ctx_default(mid),
                    capabilities=type(self).infer_capabilities(mid),
                )
            )
        return BackendCapability(id=self.id, healthy=True, base_url=self._base_url, models=tuple(models))

    async def chat(self, model_id: str, request: ChatCompletionRequest) -> dict[str, Any]:
        body = request.model_dump(exclude_none=True, exclude={"model"})
        body["model"] = model_id
        body["stream"] = False
        resp = await self._client.post(f"{self._base_url}/v1/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()

    async def chat_stream(self, model_id: str, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        body = request.model_dump(exclude_none=True, exclude={"model"})
        body["model"] = model_id
        body["stream"] = True
        async with self._client.stream("POST", f"{self._base_url}/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_raw():
                yield chunk

    async def aclose(self) -> None:
        await self._client.aclose()


def _generic_capabilities(name: str) -> tuple[str, ...]:
    n = name.lower()
    caps: list[str] = ["en"]
    if any(k in n for k in ("coder", "starcoder", "codellama", "codestral", "deepseek-coder")):
        caps.append("coding")
    if "tool" in n or "function" in n:
        caps.append("tool_use")
    if any(k in n for k in ("32b", "70b", "deepseek-r1", "qwq", "405b", "405-")):
        caps.append("hard")
    if "vision" in n or "vl" in n or "llava" in n:
        caps.append("vision")
    return tuple(caps)


def _generic_ctx(name: str) -> int:
    n = name.lower()
    if "qwen3" in n or "qwen-3" in n:
        return 32768
    if "deepseek" in n:
        return 16384
    if "llama-3.3" in n or "llama3.3" in n:
        return 131072
    if "codestral" in n:
        return 32768
    return 8192


class VLLMBackend(OpenAICompatBackend):
    id = "vllm"
    infer_capabilities = staticmethod(_generic_capabilities)
    ctx_default = staticmethod(_generic_ctx)


class MLXBackend(OpenAICompatBackend):
    id = "mlx"
    infer_capabilities = staticmethod(_generic_capabilities)
    ctx_default = staticmethod(_generic_ctx)


class LMStudioBackend(OpenAICompatBackend):
    id = "lmstudio"
    infer_capabilities = staticmethod(_generic_capabilities)
    ctx_default = staticmethod(_generic_ctx)


class GenericOpenAIBackend(OpenAICompatBackend):
    id = "generic-openai"
    infer_capabilities = staticmethod(_generic_capabilities)
    ctx_default = staticmethod(_generic_ctx)


def _openrouter_capabilities(name: str) -> tuple[str, ...]:
    """OpenRouter model ids look like 'anthropic/claude-3.5-sonnet' or 'openai/gpt-4o'."""
    n = name.lower()
    caps = list(_generic_capabilities(name))
    # Multilingual baseline for cloud frontier models
    if any(k in n for k in ("claude", "gpt-4", "gpt-5", "gemini", "command-r", "mistral-large")):
        caps.extend(["multilingual", "frontier"])
    if "vision" in n or "vl" in n or "claude" in n or "gpt-4o" in n or "gemini" in n:
        if "vision" not in caps:
            caps.append("vision")
    return tuple(dict.fromkeys(caps))  # de-dupe preserving order


def _openrouter_ctx(name: str) -> int:
    n = name.lower()
    if "claude" in n:
        return 200_000
    if "gemini" in n:
        return 1_000_000
    if "gpt-4o" in n or "gpt-4-turbo" in n:
        return 128_000
    return _generic_ctx(name)


class OpenRouterBackend(OpenAICompatBackend):
    """OpenRouter — fan-out cloud router. NON-LOCAL. Opt-in via SLANCHA_OPENROUTER_ENABLED.

    OpenAI-compat surface at https://openrouter.ai/api. Routes to many model
    providers behind one API + key. Privacy red line: this is a network call
    OUT OF the local box; defaults disabled. When enabled, the user pays
    OpenRouter's per-token cost on each request.
    """

    id = "openrouter"
    infer_capabilities = staticmethod(_openrouter_capabilities)
    ctx_default = staticmethod(_openrouter_ctx)
