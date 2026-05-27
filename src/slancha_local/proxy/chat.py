"""POST /v1/chat/completions — orchestrates embed → classify → dispatch → trace.

Supports streaming (SSE passthrough) and non-streaming. Decision-trace header
is set on both paths.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import UTC, datetime

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from slancha_local.classifier_client.models import (
    ClassifyRequest,
    LocalModelDescriptor,
)
from slancha_local.classifier_client.rules_fallback import RulesFallbackClassifier
from slancha_local.embedder import embed_single
from slancha_local.proxy.mesh_fallback import resolve_local_fallback_target
from slancha_local.proxy.middleware import format_trace
from slancha_local.proxy.models import ChatCompletionRequest
from slancha_local.proxy.pref import resolve_preferences
from slancha_local.proxy.usage_sidecar import build_usage_event
from slancha_local.telemetry.schema import (
    ClassifierBlock,
    DecisionBlock,
    ExecutionBlock,
    Trace,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _flatten_messages(req: ChatCompletionRequest) -> str:
    parts: list[str] = []
    for m in req.messages:
        if isinstance(m.content, str):
            parts.append(m.content)
        elif isinstance(m.content, list):
            for chunk in m.content:
                if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                    parts.append(chunk["text"])
    return "\n".join(parts)


def _embedding_to_b64(vec: np.ndarray) -> str:
    return base64.b64encode(vec.astype(np.float32).tobytes()).decode("ascii")


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> dict:
    state = request.app.state
    settings = state.settings

    catalog = await state.probe.get()
    descriptors = [
        LocalModelDescriptor(
            backend=m.backend_id,
            id=m.model_id,
            ctx_window=m.ctx_window,
            capabilities=list(m.capabilities),
            est_throughput_tps=m.est_throughput_tps,
        )
        for m in catalog.all_models
    ]

    prompt_text = _flatten_messages(req)
    embed_t0 = time.perf_counter()
    embedding_vec = embed_single(prompt_text)
    embed_ms = (time.perf_counter() - embed_t0) * 1000.0

    # Accept slancha-api routing rules: X-Slancha-Pref header + `pref` body
    # (body wins). Bad input (e.g. unknown weights axis) → 422, same contract
    # as the gateway.
    try:
        preferences = resolve_preferences(
            header=request.headers.get("x-slancha-pref"),
            body_pref=req.pref,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"invalid X-Slancha-Pref: {e.errors()}") from e

    classify_req = ClassifyRequest(
        embedding=embedding_vec.tolist(),
        prompt=prompt_text if settings.share_prompts else None,
        available_models=descriptors,
        preferences=preferences,
        context_len=len(prompt_text),
    )

    # Explicit-model bypass: when user names a specific backend:model pair
    # (e.g. "vllm:demo-model", "generic-openai:demo-model-v2") OR a known
    # served model id (e.g. "demo-model", "demo-model-v2"), skip classifier
    # and dispatch directly. Falls through to classifier on "auto" or unknown.
    explicit_target = None
    explicit_match_reason = None
    if req.model and req.model != "auto":
        if state.registry.parse_target(req.model)[0] is not None:
            # User passed an already-namespaced target like vllm:demo-model.
            explicit_target = req.model
            explicit_match_reason = "explicit-model:namespaced"
        else:
            # Bare model id — look up across registered backends.
            for m in catalog.all_models:
                if m.model_id == req.model:
                    explicit_target = f"local:{m.backend_id}:{m.model_id}"
                    explicit_match_reason = f"explicit-model:bare={req.model}"
                    break

    # Mesh degrade (L1, steady state): an explicitly-requested specialist
    # whose backend is down has already dropped out of the probe catalog, so
    # the bare-model lookup above missed it. If we know a base model to
    # degrade to and it's currently served, retarget the base and flag the
    # degradation. Decided before the response starts → covers streaming too.
    if explicit_target is None and req.model and req.model != "auto":
        degrade_target = resolve_local_fallback_target(req.model, catalog)
        if degrade_target is not None:
            explicit_target = degrade_target
            explicit_match_reason = f"mesh-degrade:{req.model}->base"
            request.state.mesh_fallback = f"base-no-lora; specialist={req.model}"

    classifier_ms = 0.0
    classify_resp = None
    if explicit_target is None:
        classify_t0 = time.perf_counter()
        try:
            classify_resp = await state.classifier.classify(classify_req)
        except Exception as e:
            logger.warning("primary classifier failed; using rules fallback: %s", e)
            classify_resp = await RulesFallbackClassifier().classify(classify_req)
        classifier_ms = classify_resp.classifier_ms or (time.perf_counter() - classify_t0) * 1000.0
        target = classify_resp.decision.target
    else:
        # Synthesize a minimal classify_resp so downstream trace logging still
        # works without conditional checks. Confidence=1.0 reflects user-explicit.
        from slancha_local.classifier_client.models import (
            ClassifyResponse,
            Decision,
        )
        classify_resp = ClassifyResponse(
            domain=None, difficulty=None, language=None,
            jailbreak=False, pii=False, tool_calling=False,
            route=None,
            decision=Decision(
                target=explicit_target,
                fallbacks=[],
                reason=explicit_match_reason or "explicit-model",
                confidence=1.0,
            ),
            classifier_ms=0.0,
        )
        target = explicit_target

    # Build the decision-trace header BEFORE parse_target so even a malformed
    # classifier target still gets the header on its 502. Trace is the load-
    # bearing differentiator; every response needs it — success, streaming,
    # 502 (malformed target / backend-error), 503 (cloud-escalation), 400
    # (rejected), so gallery / brag / why CLI can introspect what went wrong.
    trace_str = format_trace(
        picked=target,
        reason=classify_resp.decision.reason,
        fallbacks=classify_resp.decision.fallbacks,
        domain=classify_resp.domain,
        difficulty=classify_resp.difficulty,
        jailbreak=classify_resp.jailbreak,
        pii=classify_resp.pii,
        tool_calling=classify_resp.tool_calling,
        confidence=classify_resp.decision.confidence,
        classifier_ms=classifier_ms,
        total_overhead_ms=embed_ms + classifier_ms,
    )
    request.state.decision_trace = trace_str

    scheme, backend_id, model_id = state.registry.parse_target(target)
    if scheme is None:
        raise HTTPException(status_code=502, detail=f"malformed classifier target: {target}")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    started = time.monotonic()
    response_body: dict | None = None
    tokens_in = tokens_out = 0
    status = "ok"

    def _enqueue_mesh_usage(*, tokens_in: int, tokens_out: int, status: str) -> None:
        """Fire-and-forget telemetry sidecar — per Slancha-Mesh Protocol v0.1 §6.

        No-op when state.usage_sidecar is absent (e.g. unit tests that build
        the proxy without the sidecar wired). Identity fields lifted off
        request.state stash that MeshAuthMiddleware set on verified
        requests; falls back to anonymous-mesh when middleware was dev-
        permissive or the route was unprotected.
        """
        sidecar = getattr(request.app.state, "usage_sidecar", None)
        if sidecar is None:
            return
        latency_ms = int((time.monotonic() - started) * 1000)
        status_code = 200 if status == "ok" else 502
        ttft = None  # streaming path tracks separately; non-streaming = full RT
        try:
            event = build_usage_event(
                request_id=request_id,
                user_id=getattr(request.state, "mesh_user_id", "anonymous"),
                specialist_id=target,
                endpoint="/v1/chat/completions",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                ttft_ms=ttft,
                cost_cents=0,
                status_code=status_code,
                route_target=getattr(request.state, "mesh_route_target", "mesh"),
                model=req.model,
            )
            asyncio.create_task(sidecar.enqueue(event))
        except Exception as e:  # noqa: BLE001
            # Telemetry must never break the user's response.
            logger.warning("mesh usage sidecar enqueue failed: %s", e)

    def _write_trace(*, tokens_in: int, tokens_out: int, status: str, response_text: str | None) -> None:
        latency_ms = int((time.monotonic() - started) * 1000)
        trace = Trace(
            request_id=request_id,
            ts=datetime.now(UTC).isoformat(),
            mode=settings.classifier_kind if settings.classifier_kind != "rules" else "local",
            embedding_b64=_embedding_to_b64(embedding_vec),
            classifier=ClassifierBlock(
                domain=classify_resp.domain,
                difficulty=classify_resp.difficulty,
                language=classify_resp.language,
                jailbreak=classify_resp.jailbreak,
                pii=classify_resp.pii,
                tool_calling=classify_resp.tool_calling,
                route=classify_resp.route,
                confidence=classify_resp.decision.confidence,
            ),
            decision=DecisionBlock(
                target=target,
                fallbacks=classify_resp.decision.fallbacks,
                reason=classify_resp.decision.reason,
            ),
            execution=ExecutionBlock(
                executed_target=target,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                status=status,
            ),
            prompt=prompt_text if settings.share_prompts else None,
            response=response_text if settings.share_traces else None,
            consent_at_capture=settings.share_traces,
        )
        state.trace_writer.write(trace)

    if scheme == "local":
        try:
            backend = state.registry.by_id(backend_id)
        except KeyError as e:
            raise HTTPException(status_code=502, detail=f"backend not registered: {backend_id}") from e

        # Streaming path
        if req.stream:
            from slancha_local.proxy.sse import StreamAccumulator

            # Default-on the usage chunk so streaming responses still produce
            # tokens_in/out for the telemetry sidecar. Without this, vLLM (and
            # llama.cpp's OpenAI-compat shim) emit deltas only — no usage —
            # and StreamAccumulator falls back to delta_count which is a
            # coarse estimate. Respect explicit client choice if they passed
            # stream_options themselves.
            if req.stream_options is None:
                req.stream_options = {"include_usage": True}
            elif "include_usage" not in req.stream_options:
                req.stream_options = {**req.stream_options, "include_usage": True}

            async def _gen():
                acc = StreamAccumulator()
                stream_status = "ok"
                try:
                    async for chunk in backend.chat_stream(model_id, req):
                        acc.feed(chunk)
                        yield bytes(chunk)
                except Exception as e:
                    stream_status = "error"
                    logger.exception("stream from backend failed: %s", e)
                    yield b"data: " + json.dumps({"error": {"message": str(e)}}).encode() + b"\n\n"
                finally:
                    _write_trace(
                        tokens_in=acc.usage_in,
                        tokens_out=acc.tokens_out_estimate,
                        status=stream_status,
                        response_text=acc.content if settings.share_traces else None,
                    )
                    _enqueue_mesh_usage(
                        tokens_in=acc.usage_in,
                        tokens_out=acc.tokens_out_estimate,
                        status=stream_status,
                    )

            return StreamingResponse(
                _gen(),
                media_type="text/event-stream",
                headers={
                    "slancha-decision-trace": trace_str,
                    "cache-control": "no-cache",
                    "connection": "keep-alive",
                    "x-accel-buffering": "no",
                },
            )

        # Non-streaming path
        try:
            response_body = await backend.chat(model_id, req)
            usage = response_body.get("usage", {}) if response_body else {}
            tokens_in = int(usage.get("prompt_tokens", 0) or 0)
            tokens_out = int(usage.get("completion_tokens", 0) or 0)
        except Exception as e:
            # Mesh degrade (L1, transient): the specialist backend was healthy
            # at probe but failed at dispatch (died inside the cache TTL). Retry
            # the mapped base once before surfacing 502. Skipped if we already
            # degraded pre-dispatch (mesh_fallback set) — that base just failed.
            degrade_target = (
                None
                if getattr(request.state, "mesh_fallback", None)
                else resolve_local_fallback_target(req.model, catalog)
            )
            if degrade_target is not None:
                fb_scheme, fb_backend_id, fb_model_id = state.registry.parse_target(degrade_target)
                try:
                    fb_backend = state.registry.by_id(fb_backend_id)
                    response_body = await fb_backend.chat(fb_model_id, req)
                    usage = response_body.get("usage", {}) if response_body else {}
                    tokens_in = int(usage.get("prompt_tokens", 0) or 0)
                    tokens_out = int(usage.get("completion_tokens", 0) or 0)
                    target = degrade_target  # trace reflects what served
                    request.state.mesh_fallback = f"base-no-lora; specialist={req.model}"
                    logger.warning(
                        "mesh specialist %s dispatch failed; degraded to %s", req.model, degrade_target
                    )
                except Exception as e2:
                    status = "error"
                    logger.exception("local backend + base fallback failed: %s", e2)
                    raise HTTPException(status_code=502, detail=f"local backend error: {e2}") from e2
            else:
                status = "error"
                logger.exception("local backend failed: %s", e)
                raise HTTPException(status_code=502, detail=f"local backend error: {e}") from e
    elif scheme == "cloud":
        # Cloud escalation lives in v0.2 — for now, surface a clear 503 unless
        # someone explicitly opts in via SLANCHA_API_KEY + classifier_kind=cloud.
        status = "rejected"
        if classify_resp.decision.target.startswith("cloud:reject:"):
            raise HTTPException(
                status_code=400,
                detail={"error": "request rejected by classifier", "reason": classify_resp.decision.reason},
            )
        raise HTTPException(
            status_code=503,
            detail=(
                "cloud escalation not enabled in this build. set "
                "SLANCHA_API_KEY + SLANCHA_CLASSIFIER_KIND=cloud, or change "
                "preferences to disallow escalation."
            ),
        )
    else:
        raise HTTPException(status_code=502, detail=f"unknown target scheme: {scheme}")

    # Non-streaming finalize (trace already on request.state)
    response_text = (
        response_body["choices"][0]["message"]["content"]
        if response_body and response_body.get("choices")
        else None
    )
    _write_trace(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        status=status,
        response_text=response_text,
    )
    _enqueue_mesh_usage(tokens_in=tokens_in, tokens_out=tokens_out, status=status)
    return response_body or {}
