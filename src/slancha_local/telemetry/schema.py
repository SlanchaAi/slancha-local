"""Trace schema (schema_version=1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ClassifierBlock(BaseModel):
    domain: str | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    language: str | None = None
    jailbreak: bool = False
    pii: bool = False
    tool_calling: bool = False
    route: str | None = None
    confidence: float | None = None


class DecisionBlock(BaseModel):
    target: str
    fallbacks: list[str] = Field(default_factory=list)
    reason: str


class ExecutionBlock(BaseModel):
    executed_target: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    ttft_ms: int | None = None
    tps: float | None = None
    status: str = "ok"


class Trace(BaseModel):
    request_id: str
    ts: str
    mode: Literal["cloud", "onprem", "local"]
    embedding_b64: str
    classifier: ClassifierBlock
    decision: DecisionBlock
    execution: ExecutionBlock
    prompt: str | None = None
    response: str | None = None
    feedback: dict[str, Any] | None = None
    consent_at_capture: bool = False
    schema_version: int = 1
