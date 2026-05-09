"""POST /v1/chat/completions — orchestrates embed → classify → dispatch → trace.

Supports streaming (SSE passthrough) and non-streaming. Decision-trace header
is set on both paths.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from datetime import UTC, datetime

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from slancha_local.classifier_client.models import (
    ClassifyRequest,
    LocalModelDescriptor,
    Preferences,
)
from slancha_local.classifier_client.rules_fallback import RulesFallbackClassifier
from slancha_local.embedder import embed_single
from slancha_local.proxy.middleware import format_trace
from slancha_local.proxy.models import ChatCompletionRequest
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

    classify_req = ClassifyRequest(
        embedding=embedding_vec.tolist(),
        prompt=prompt_text if settings.share_prompts else None,
        available_models=descriptors,
        preferences=Preferences(),
        context_len=len(prompt_text),
    )

    classify_t0 = time.perf_counter()
    try:
        classify_resp = await state.classifier.classify(classify_req)
    except Exception as e:
        logger.warning("primary classifier failed; using rules fallback: %s", e)
        classify_resp = await RulesFallbackClassifier().classify(classify_req)
    classifier_ms = classify_resp.classifier_ms or (time.perf_counter() - classify_t0) * 1000.0

    target = classify_resp.decision.target
    scheme, backend_id, model_id = state.registry.parse_target(target)
    if scheme is None:
        raise HTTPException(status_code=502, detail=f"malformed classifier target: {target}")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    started = time.monotonic()
    response_body: dict | None = None
    tokens_in = tokens_out = 0
    status = "ok"

    # Build the decision-trace header now (we have everything we need)
    # so it can be set on both streaming and non-streaming responses before
    # the body bytes start flowing.
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

            async def _gen():
                accumulated = bytearray()
                tokens_out_streamed = 0
                stream_status = "ok"
                try:
                    async for chunk in backend.chat_stream(model_id, req):
                        accumulated.extend(chunk)
                        # crude token count estimate from SSE deltas: each chunk
                        # ≈ 1 token in OpenAI-compat streaming
                        tokens_out_streamed += chunk.count(b"data:")
                        yield bytes(chunk)
                except Exception as e:
                    stream_status = "error"
                    logger.exception("stream from backend failed: %s", e)
                    yield (b"data: " + json.dumps({"error": {"message": str(e)}}).encode() + b"\n\n")
                finally:
                    response_text = (
                        accumulated.decode("utf-8", errors="replace") if settings.share_traces else None
                    )
                    _write_trace(
                        tokens_in=0,
                        tokens_out=max(0, tokens_out_streamed - 1),  # subtract [DONE]
                        status=stream_status,
                        response_text=response_text,
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

    # Non-streaming finalize
    request.state.decision_trace = trace_str
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
    return response_body or {}
