"""In-mesh specialist degrade (L1).

When a self-hosted specialist backend is down but its base model is still
served by a healthy backend, the proxy serves the base instead of 502ing,
stamping X-Slancha-Fallback so callers can detect base-no-lora output.

Example: paul-voice-v8 (HF essay-DPO LoRA on :8004) is down; paul-voice
(vLLM v7d on :8003) is up — serve paul-voice and flag the degradation.

The probe-driven catalog drops a down backend within its TTL, so the
steady-state degrade is detectable pre-dispatch (the specialist is simply
absent from the catalog). The proxy also retries the base post-dispatch for
the transient window where the backend died inside the cache TTL.

See slancha-api docs/superpowers/specs/2026-05-24-mesh-failover-design.md.
"""

from __future__ import annotations

from slancha_local.capability.catalog import LocalCatalog

# Specialist model id → base model id to degrade to when the specialist
# backend is unavailable. The base must be a model id served by some
# registered backend (looked up in the live catalog at request time).
MESH_LOCAL_FALLBACK: dict[str, str] = {
    "paul-voice-v8": "paul-voice",
}


def resolve_local_fallback_target(specialist_model_id: str, catalog: LocalCatalog) -> str | None:
    """Return a `local:<backend_id>:<model_id>` target for the base model that
    `specialist_model_id` degrades to, or None when there is no mapping or the
    base model is not currently served by any healthy backend in `catalog`."""
    base_model_id = MESH_LOCAL_FALLBACK.get(specialist_model_id)
    if base_model_id is None:
        return None
    for m in catalog.all_models:
        if m.model_id == base_model_id:
            return f"local:{m.backend_id}:{m.model_id}"
    return None
