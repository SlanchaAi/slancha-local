"""GET /v1/models — OpenAI-compat catalog of routable models.

Returns merged set of all healthy backends' models plus a synthetic "auto"
entry that selects via the classifier.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> dict:
    state = request.app.state
    catalog = await state.probe.get()
    now = int(time.time())

    data = [
        {
            "id": "auto",
            "object": "model",
            "created": now,
            "owned_by": "slancha-local",
            "permission": [],
            "context_length": max((m.ctx_window for m in catalog.all_models), default=8192),
            "metadata": {
                "description": ("auto-routed: slancha-local's classifier picks the right model per request."),
                "routing": "per-prompt",
            },
        }
    ]
    for m in catalog.all_models:
        data.append(
            {
                "id": f"{m.backend_id}:{m.model_id}",
                "object": "model",
                "created": now,
                "owned_by": m.backend_id,
                "permission": [],
                "context_length": m.ctx_window,
                "metadata": {
                    "backend": m.backend_id,
                    "capabilities": list(m.capabilities),
                    "est_throughput_tps": m.est_throughput_tps,
                },
            }
        )
    return {"object": "list", "data": data}
