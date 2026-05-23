"""OpenAI-compatible request/response Pydantic schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    # stream_options controls extra streaming-only chunks. Most usefully:
    # {"include_usage": true} → backend emits a final chunk with prompt/
    # completion/total tokens before [DONE]. Without it, streaming clients
    # see chunk-of-deltas but no usage block, and the telemetry sidecar
    # records tokens=0 for streamed responses. We default it on at the
    # proxy layer when the client didn't ask (see chat.py).
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    n: int | None = 1
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
