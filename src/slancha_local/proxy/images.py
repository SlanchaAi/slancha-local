"""POST /v1/images/generations — OpenAI-compat image generation endpoint.

Routes to the configured image backend (ComfyUI v0.2; later: SwarmUI / diffusers).
Off by default — only attaches if `settings.comfy_enabled` is True or another
image backend is registered. Disabled = endpoint absent → 404, same privacy
red line as v0.1: opt-in for any non-loopback work.

Decision-trace header still emitted via DecisionTraceHeaderMiddleware; image
traces carry mode='image' so analytics can split modes.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from slancha_local.backends.comfy import ComfyBackend, ImageRequest

logger = logging.getLogger(__name__)

router = APIRouter()


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    n: int = Field(default=1, ge=1, le=10)
    size: str = Field(default="512x512")
    response_format: Literal["url", "b64_json"] = Field(default="url")
    model: str | None = Field(default=None)
    seed: int | None = Field(default=None)
    steps: int | None = Field(default=None, ge=1, le=200)
    workflow_path: str | None = Field(default=None)


@router.post("/v1/images/generations")
async def images_generations(request: Request, body: ImageGenerationRequest) -> dict[str, Any]:
    backend: ComfyBackend | None = getattr(request.app.state, "image_backend", None)
    if backend is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Image generation not enabled. Set SLANCHA_COMFY_ENABLED=true and ensure "
                "ComfyUI is reachable at SLANCHA_COMFY_BASE_URL."
            ),
        )

    request_id = str(uuid.uuid4())
    img_req = ImageRequest(
        prompt=body.prompt,
        n=body.n,
        size=body.size,
        response_format=body.response_format,
        model=body.model,
        seed=body.seed,
        steps=body.steps,
        workflow_path=body.workflow_path,
    )
    try:
        out = await backend.generate_image(img_req)
    except (TimeoutError, ConnectionError) as e:
        raise HTTPException(status_code=504, detail=f"image backend timeout: {e}") from e
    except Exception as e:
        logger.exception("image generation failed")
        raise HTTPException(status_code=500, detail=f"image generation failed: {e}") from e

    # Persist a minimal trace (mode='image') so the gallery + analytics can split modes.
    trace_writer = getattr(request.app.state, "trace_writer", None)
    if trace_writer is not None:
        try:
            trace = {
                "request_id": request_id,
                "ts": _now_iso(),
                "mode": "image",
                "classifier": {
                    "domain": "image_generation",
                    "difficulty": "n/a",
                    "language": "n/a",
                    "jailbreak": False,
                    "pii": False,
                    "tool_calling": False,
                    "route": "image_generation",
                    "confidence": 0.0,
                },
                "decision": {"target": "local:comfy", "fallbacks": [], "reason": "image_endpoint"},
                "execution": {
                    "executed_target": "local:comfy",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "latency_ms": 0,
                    "status": "ok",
                },
                "consent_at_capture": False,
                "schema_version": 1,
            }
            trace_writer.write(trace)
        except Exception as e:  # noqa: BLE001
            logger.debug("trace write skipped: %s", e)

    out["request_id"] = request_id
    return out


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
