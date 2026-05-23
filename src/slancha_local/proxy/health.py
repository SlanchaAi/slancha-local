"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
@router.get("/healthz")
async def health() -> dict:
    """/healthz alias added for AWS liveness-prober Lambda probing the
    mesh tunnel (Kubernetes-convention path)."""
    return {"status": "ok"}


@router.get("/health/detailed")
async def health_detailed(request: Request) -> dict:
    state = request.app.state
    catalog = await state.probe.get()
    backends = []
    for cap in catalog.capabilities:
        backends.append(
            {
                "id": cap.id,
                "healthy": cap.healthy,
                "base_url": cap.base_url,
                "models": [
                    {"id": m.model_id, "ctx": m.ctx_window, "capabilities": list(m.capabilities)}
                    for m in cap.models
                ],
            }
        )
    return {
        "status": "ok",
        "classifier_kind": state.settings.classifier_kind,
        "share_prompts": state.settings.share_prompts,
        "share_traces": state.settings.share_traces,
        "backends": backends,
    }
