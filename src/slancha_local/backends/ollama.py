"""Ollama backend adapter — speaks OpenAI-compat at /v1 + /api/tags for probe."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from slancha_local.backends.base import Backend, BackendCapability, BackendModel
from slancha_local.proxy.models import ChatCompletionRequest

logger = logging.getLogger(__name__)


def _infer_capabilities(model_name: str) -> tuple[str, ...]:
    name = model_name.lower()
    caps: list[str] = ["en"]
    if any(k in name for k in ("coder", "deepseek-coder", "starcoder", "codellama", "codestral")):
        caps.append("coding")
    if "tool" in name or "function" in name:
        caps.append("tool_use")
    if any(k in name for k in ("32b", "70b", "deepseek-r1", "qwq")):
        caps.append("hard")
    if "vision" in name or "vl" in name:
        caps.append("vision")
    return tuple(caps)


def _ctx_window_default(name: str) -> int:
    n = name.lower()
    if "qwen3" in n:
        return 32768
    if "deepseek" in n:
        return 16384
    if "llama-3.3" in n or "llama3.3" in n:
        return 131072
    if "codestral" in n:
        return 32768
    if "tinyllama" in n:
        return 2048
    return 8192


class OllamaBackend(Backend):
    id = "ollama"

    def __init__(self, *, base_url: str, timeout_s: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def probe(self) -> BackendCapability:
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags", timeout=2.0)
            resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
            logger.warning("ollama probe failed: %s", e)
            return BackendCapability(id=self.id, healthy=False, base_url=self._base_url, models=())

        models: list[BackendModel] = []
        for m in resp.json().get("models", []):
            name = m.get("model") or m.get("name")
            if not name:
                continue
            models.append(
                BackendModel(
                    backend_id=self.id,
                    model_id=name,
                    ctx_window=_ctx_window_default(name),
                    capabilities=_infer_capabilities(name),
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
