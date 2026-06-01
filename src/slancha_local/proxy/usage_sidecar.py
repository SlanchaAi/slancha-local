"""UsageSidecar — durable buffer + retry + DLQ for mesh telemetry POSTs.

Per Slancha-Mesh Protocol v0.1 §6 telemetry contract + H16 sidecar mid-stream
resilience pattern (mirrors Stripe webhook handling in slancha-api):

  1. After response close, build MeshUsageEvent matching slancha-api's
     /v1/admin/usage shape.
  2. Write to durable local jsonl (`~/.slancha-local/usage-buffer.jsonl`)
     BEFORE async POST. Local write must succeed — telemetry loss is
     acceptable, lost durable write is not.
  3. Async POST with Bearer mesh-ingest-token + idempotency via request_id.
  4. Retry on transient failures: 1s / 5s / 30s exponential.
  5. After N=3 failures append to DLQ (`~/.slancha-local/usage-dlq.jsonl`)
     and drop.
  6. Nightly reconcile script (out-of-band) replays DLQ → /v1/admin/usage.

Payload shape (counts-only, NO prompt/completion body, ever):

  {
    "request_id": str,                    # uuid (idempotency key)
    "user_id": str,
    "specialist_id": str,                 # e.g. "demo-model", "demo-model-v2"
    "endpoint": str,                      # "/v1/chat/completions"
    "tokens_in": int,
    "tokens_out": int,
    "latency_ms": int,
    "ttft_ms": int | None,
    "tokens_per_second": float | None,
    "cost_cents": int,                    # 0 for local mesh
    "cloud_equivalent_cost_cents_router_computed": int | None,
    "status_code": int,
    "route_target": str,                  # "mesh" | "direct" | "fallback"
    "fallback_fired": bool,
    "pref_applied": dict | None,
    "decision_reason_structured": dict | None,
    "otel_semconv_version": str,
    "gen_ai.request.model": str,
    "gen_ai.usage.input_tokens": int,
    "gen_ai.usage.output_tokens": int,
  }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OTEL_SEMCONV_VERSION = "1.36.0+dev"

_DEFAULT_BUFFER_DIR = Path("~/.slancha-local").expanduser()
_BUFFER_FILE = "usage-buffer.jsonl"
_DLQ_FILE = "usage-dlq.jsonl"

_RETRY_DELAYS_S: tuple[float, ...] = (1.0, 5.0, 30.0)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_usage_event(
    *,
    request_id: str,
    user_id: str,
    specialist_id: str,
    endpoint: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    ttft_ms: int | None,
    cost_cents: int = 0,
    cloud_equivalent_cost_cents: int | None = None,
    status_code: int,
    route_target: str = "mesh",
    fallback_fired: bool = False,
    pref_applied: dict[str, Any] | None = None,
    decision_reason_structured: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Construct the MeshUsageEvent payload — counts only, never body.

    Field names must match slancha-api app/schemas/mesh_usage.py exactly.
    Required: request_id, user_id, endpoint, model, route, tokens_in,
    tokens_out, latency_ms, status_code.
    """
    tokens_per_second: float | None = None
    if tokens_out > 0 and latency_ms > 0:
        tokens_per_second = round(tokens_out / (latency_ms / 1000.0), 2)
    model = model or specialist_id
    return {
        "request_id": request_id,
        "user_id": user_id,
        "endpoint": endpoint,
        "model": model,
        "route": route_target,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
        "status_code": status_code,
        "ttft_ms": ttft_ms,
        "tokens_per_second": tokens_per_second,
        "specialist_id": specialist_id,
        "cost_cents": cost_cents,
        "cloud_equivalent_cost_cents": cloud_equivalent_cost_cents,
        "fallback_fired": fallback_fired,
        "pref_applied": pref_applied,
        "decision_reason": decision_reason_structured,
        "otel_semconv_version": OTEL_SEMCONV_VERSION,
        "gen_ai.request.model": model,
        "gen_ai.usage.input_tokens": tokens_in,
        "gen_ai.usage.output_tokens": tokens_out,
    }


class UsageSidecar:
    """Durable buffer + retry + DLQ sender.

    Single shared instance per process; `enqueue(event)` is the public
    surface. Behavior:

      - Append to durable jsonl (always).
      - Spawn retry task to POST.
      - On exhaust → DLQ append.

    Construct with explicit ingest_url + token for tests; defaults read
    SLANCHA_API_BASE_URL + SLANCHA_MESH_INGEST_TOKEN env vars.
    """

    def __init__(
        self,
        *,
        ingest_url: str | None = None,
        ingest_token: str | None = None,
        buffer_dir: Path | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        api_base = (ingest_url or os.environ.get("SLANCHA_API_BASE_URL", "https://api.slancha.ai")).rstrip(
            "/"
        )
        self._ingest_url = f"{api_base}/v1/admin/usage"
        self._token = ingest_token or os.environ.get("SLANCHA_MESH_INGEST_TOKEN", "")
        self._buffer_dir = buffer_dir or _DEFAULT_BUFFER_DIR
        self._buffer_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_path = self._buffer_dir / _BUFFER_FILE
        self._dlq_path = self._buffer_dir / _DLQ_FILE
        self._http = http_client
        self._owns_http = http_client is None
        self._timeout_s = timeout_s
        if not self._token:
            logger.warning(
                "UsageSidecar: SLANCHA_MESH_INGEST_TOKEN unset — POSTs will "
                "fail auth and land in DLQ. Set the token once issued."
            )

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout_s)
        return self._http

    async def aclose(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()

    def _append(self, path: Path, event: dict[str, Any]) -> None:
        with path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    async def enqueue(self, event: dict[str, Any]) -> None:
        """Persist + fire-and-forget POST attempt. Safe for BackgroundTasks."""
        try:
            self._append(self._buffer_path, {"ts": _now_iso(), **event})
        except OSError as e:
            # Durability failure — log loudly; telemetry is lost but the
            # response itself proceeds. Cannot recover this in-band.
            logger.error("UsageSidecar: durable buffer write failed: %s", e)
            return
        await self._post_with_retries(event)

    async def _post_with_retries(self, event: dict[str, Any]) -> None:
        client = await self._client()
        for attempt, delay in enumerate([0.0, *_RETRY_DELAYS_S]):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                resp = await client.post(
                    self._ingest_url,
                    json=event,
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": "application/json",
                    },
                )
                if 200 <= resp.status_code < 300:
                    return  # Mac's endpoint is idempotent — duplicate posts no-op.
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    # Non-retryable client error (bad token, schema, etc.).
                    logger.warning(
                        "UsageSidecar: non-retryable %s on attempt %d body=%s",
                        resp.status_code,
                        attempt,
                        resp.text[:200],
                    )
                    self._append(
                        self._dlq_path,
                        {
                            "ts": _now_iso(),
                            "reason": f"http_{resp.status_code}",
                            "event": event,
                        },
                    )
                    return
                logger.info("UsageSidecar: transient %s on attempt %d", resp.status_code, attempt)
            except (TimeoutError, httpx.RequestError) as e:
                logger.info("UsageSidecar: network error on attempt %d: %s", attempt, e)
        # All retries exhausted → DLQ.
        logger.warning("UsageSidecar: retries exhausted, DLQ'ing request_id=%s", event.get("request_id"))
        self._append(self._dlq_path, {"ts": _now_iso(), "reason": "retries_exhausted", "event": event})


__all__ = ["UsageSidecar", "build_usage_event", "OTEL_SEMCONV_VERSION"]
