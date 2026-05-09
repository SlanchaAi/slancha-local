"""Wire format for classifier requests/responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LocalModelDescriptor(BaseModel):
    backend: str
    id: str
    ctx_window: int
    capabilities: list[str] = Field(default_factory=list)
    est_throughput_tps: float | None = None
    free_vram_mb: int | None = None


class Preferences(BaseModel):
    cost_weight: float = 0.5
    quality_weight: float = 0.3
    latency_weight: float = 0.1
    privacy_weight: float = 0.1
    escalation_allowed: bool = True
    max_cost_per_1k: float | None = None
    max_latency_ms: int | None = None


class ClassifyRequest(BaseModel):
    embedding: list[float] = Field(min_length=512, max_length=512)
    prompt: str | None = None
    available_models: list[LocalModelDescriptor] = Field(default_factory=list)
    preferences: Preferences = Field(default_factory=Preferences)
    context_len: int = 0


class Decision(BaseModel):
    target: str
    fallbacks: list[str] = Field(default_factory=list)
    reason: str
    confidence: float


class ClassifyResponse(BaseModel):
    decision: Decision
    domain: str | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    language: str | None = None
    route: str | None = None
    jailbreak: bool = False
    pii: bool = False
    tool_calling: bool = False
    classifier_ms: float | None = None
