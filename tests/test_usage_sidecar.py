"""Tests for UsageSidecar — durable buffer + retry + DLQ."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from slancha_local.proxy.usage_sidecar import (
    OTEL_SEMCONV_VERSION,
    UsageSidecar,
    build_usage_event,
)


def _make_event(**overrides) -> dict:
    base = build_usage_event(
        request_id="req-abc",
        user_id="user-paul",
        specialist_id="demo-model",
        endpoint="/v1/chat/completions",
        tokens_in=230,
        tokens_out=1247,
        latency_ms=4830,
        ttft_ms=124,
        cost_cents=0,
        cloud_equivalent_cost_cents=7,
        status_code=200,
        route_target="mesh",
        pref_applied={"quality_weight": 0.6, "max_cost_cents": 5},
        decision_reason_structured={"winner": "demo-model", "alternatives_considered": []},
        model="demo-model",
    )
    base.update(overrides)
    return base


def test_build_usage_event_shape():
    """Payload matches slancha-api app/schemas/mesh_usage.py — counts only, no body."""
    event = _make_event()
    # Required fields per slancha-api MeshUsageEvent
    assert event["request_id"] == "req-abc"
    assert event["user_id"] == "user-paul"
    assert event["endpoint"] == "/v1/chat/completions"
    assert event["model"] == "demo-model"
    assert event["route"] == "mesh"
    assert event["tokens_in"] == 230
    assert event["tokens_out"] == 1247
    assert event["latency_ms"] == 4830
    assert event["status_code"] == 200
    # Optional fields
    assert event["specialist_id"] == "demo-model"
    assert event["ttft_ms"] == 124
    assert event["tokens_per_second"] == round(1247 / 4.83, 2)
    assert event["cost_cents"] == 0
    assert event["cloud_equivalent_cost_cents"] == 7
    assert event["fallback_fired"] is False
    # OTel GenAI semconv attrs (M10) — dotted aliases on the Pydantic side
    assert event["otel_semconv_version"] == OTEL_SEMCONV_VERSION
    assert event["gen_ai.request.model"] == "demo-model"
    assert event["gen_ai.usage.input_tokens"] == 230
    assert event["gen_ai.usage.output_tokens"] == 1247
    # Hard guarantee: no prompt/completion body anywhere
    serialized = json.dumps(event)
    assert "messages" not in serialized.lower()
    assert "choices" not in serialized.lower()
    assert "content" not in serialized.lower()


def test_tokens_per_second_zero_safe():
    """Avoid div-by-zero when latency or tokens are 0."""
    e = build_usage_event(
        request_id="r", user_id="u", specialist_id="s", endpoint="/v1/chat/completions",
        tokens_in=0, tokens_out=0, latency_ms=0, ttft_ms=None,
        status_code=200,
    )
    assert e["tokens_per_second"] is None


@pytest.mark.asyncio
async def test_enqueue_writes_durable_buffer(tmp_path: Path):
    """Durable buffer write happens BEFORE POST attempt (H16 ordering)."""
    transport = httpx.MockTransport(lambda req: httpx.Response(204))
    client = httpx.AsyncClient(transport=transport, timeout=2.0)
    sidecar = UsageSidecar(
        ingest_url="https://api.test", ingest_token="t", buffer_dir=tmp_path,
        http_client=client,
    )
    await sidecar.enqueue(_make_event())
    buf = (tmp_path / "usage-buffer.jsonl").read_text().strip().splitlines()
    assert len(buf) == 1
    parsed = json.loads(buf[0])
    assert parsed["request_id"] == "req-abc"
    assert "ts" in parsed
    await sidecar.aclose()


@pytest.mark.asyncio
async def test_post_success_no_dlq(tmp_path: Path):
    """200 response → buffer written, no DLQ entry."""
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"ack": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=2.0)
    sidecar = UsageSidecar(
        ingest_url="https://api.test", ingest_token="tk", buffer_dir=tmp_path,
        http_client=client,
    )
    await sidecar.enqueue(_make_event())
    assert len(captured) == 1
    assert captured[0].headers["authorization"] == "Bearer tk"
    assert not (tmp_path / "usage-dlq.jsonl").exists()
    await sidecar.aclose()


@pytest.mark.asyncio
async def test_post_client_error_dlq_immediately(tmp_path: Path):
    """4xx (not 429) is non-retryable → straight to DLQ."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad token"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=2.0)
    sidecar = UsageSidecar(
        ingest_url="https://api.test", ingest_token="tk", buffer_dir=tmp_path,
        http_client=client,
    )
    await sidecar.enqueue(_make_event())
    dlq_path = tmp_path / "usage-dlq.jsonl"
    assert dlq_path.exists()
    dlq_entries = dlq_path.read_text().strip().splitlines()
    assert len(dlq_entries) == 1
    entry = json.loads(dlq_entries[0])
    assert entry["reason"] == "http_401"
    assert entry["event"]["request_id"] == "req-abc"
    await sidecar.aclose()


@pytest.mark.asyncio
async def test_post_retries_on_5xx_then_dlq(tmp_path: Path, monkeypatch):
    """5xx repeats → exponential backoff → DLQ after exhaustion."""
    # Speed up by zeroing delays
    monkeypatch.setattr(
        "slancha_local.proxy.usage_sidecar._RETRY_DELAYS_S", (0.0, 0.0, 0.0)
    )
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": "unavail"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=2.0)
    sidecar = UsageSidecar(
        ingest_url="https://api.test", ingest_token="tk", buffer_dir=tmp_path,
        http_client=client,
    )
    await sidecar.enqueue(_make_event())
    assert calls["n"] == 4  # initial + 3 retries
    dlq_path = tmp_path / "usage-dlq.jsonl"
    entry = json.loads(dlq_path.read_text().strip())
    assert entry["reason"] == "retries_exhausted"
    await sidecar.aclose()


@pytest.mark.asyncio
async def test_post_recovers_on_transient_5xx(tmp_path: Path, monkeypatch):
    """5xx then 200 → POST succeeds, no DLQ."""
    monkeypatch.setattr(
        "slancha_local.proxy.usage_sidecar._RETRY_DELAYS_S", (0.0, 0.0, 0.0)
    )
    seq = iter([
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(200, json={"ack": True}),
    ])

    def handler(req: httpx.Request) -> httpx.Response:
        return next(seq)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=2.0)
    sidecar = UsageSidecar(
        ingest_url="https://api.test", ingest_token="tk", buffer_dir=tmp_path,
        http_client=client,
    )
    await sidecar.enqueue(_make_event())
    assert not (tmp_path / "usage-dlq.jsonl").exists()
    await sidecar.aclose()


@pytest.mark.asyncio
async def test_post_429_is_retryable(tmp_path: Path, monkeypatch):
    """429 (rate-limit) is retryable; not non-retryable 4xx."""
    monkeypatch.setattr(
        "slancha_local.proxy.usage_sidecar._RETRY_DELAYS_S", (0.0, 0.0, 0.0)
    )
    seq = iter([
        httpx.Response(429),
        httpx.Response(200),
    ])

    def handler(req: httpx.Request) -> httpx.Response:
        return next(seq)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=2.0)
    sidecar = UsageSidecar(
        ingest_url="https://api.test", ingest_token="tk", buffer_dir=tmp_path,
        http_client=client,
    )
    await sidecar.enqueue(_make_event())
    assert not (tmp_path / "usage-dlq.jsonl").exists()
    await sidecar.aclose()
