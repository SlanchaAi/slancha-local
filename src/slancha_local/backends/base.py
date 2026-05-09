"""Backend abstraction. Each adapter speaks OpenAI-compat HTTP to a local LLM server."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from slancha_local.proxy.models import ChatCompletionRequest


@dataclass(frozen=True)
class BackendModel:
    backend_id: str
    model_id: str
    ctx_window: int
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    est_throughput_tps: float | None = None


@dataclass(frozen=True)
class BackendCapability:
    id: str
    healthy: bool
    base_url: str
    models: tuple[BackendModel, ...] = field(default_factory=tuple)


class Backend(ABC):
    id: str

    @abstractmethod
    async def probe(self) -> BackendCapability: ...

    @abstractmethod
    async def chat(self, model_id: str, request: ChatCompletionRequest) -> dict[str, Any]: ...

    @abstractmethod
    async def chat_stream(
        self, model_id: str, request: ChatCompletionRequest
    ) -> AsyncIterator[bytes]: ...
