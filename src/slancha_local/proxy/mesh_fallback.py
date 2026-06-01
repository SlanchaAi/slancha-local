"""In-mesh specialist degrade (L1).

When a self-hosted specialist backend is down but its base model is still
served by a healthy backend, the proxy serves the base instead of 502ing,
stamping X-Slancha-Fallback so callers can detect base-no-lora output.

Example: a specialist LoRA (served on one backend) goes down while its base
model is still up on another backend — serve the base and flag the
degradation so the caller knows it got base-no-lora output.

The probe-driven catalog drops a down backend within its TTL, so the
steady-state degrade is detectable pre-dispatch (the specialist is simply
absent from the catalog). The proxy also retries the base post-dispatch for
the transient window where the backend died inside the cache TTL.

The specialist→base map is deployment-specific and **empty by default**;
configure it with the ``SLANCHA_MESH_FALLBACK_MAP`` env var — a JSON object of
``{"specialist_model_id": "base_model_id"}``. With no mapping configured this
layer is a no-op.
"""

from __future__ import annotations

import json
import logging
import os

from slancha_local.capability.catalog import LocalCatalog

logger = logging.getLogger(__name__)

FALLBACK_MAP_ENV = "SLANCHA_MESH_FALLBACK_MAP"


def _load_fallback_map() -> dict[str, str]:
    """Parse the specialist→base degrade map from env (JSON); default empty.

    Never raises: malformed JSON or a non-string map is logged and ignored so
    a bad env var can't break import or serving.
    """
    raw = os.environ.get(FALLBACK_MAP_ENV)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("%s is not valid JSON (%s); ignoring", FALLBACK_MAP_ENV, e)
        return {}
    if isinstance(data, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        return data
    logger.warning("%s must be a JSON object of string→string; ignoring", FALLBACK_MAP_ENV)
    return {}


# Specialist model id → base model id to degrade to when the specialist backend
# is unavailable. Deployment-specific, empty by default (nothing hardcoded); the
# base must be a model id served by some registered backend (looked up in the
# live catalog at request time). Configure via SLANCHA_MESH_FALLBACK_MAP.
MESH_LOCAL_FALLBACK: dict[str, str] = _load_fallback_map()


def resolve_local_fallback_target(
    specialist_model_id: str,
    catalog: LocalCatalog,
    *,
    fallback_map: dict[str, str] | None = None,
) -> str | None:
    """Return a `local:<backend_id>:<model_id>` target for the base model that
    `specialist_model_id` degrades to, or None when there is no mapping or the
    base model is not currently served by any healthy backend in `catalog`.

    `fallback_map` defaults to the module-level `MESH_LOCAL_FALLBACK`."""
    fb = MESH_LOCAL_FALLBACK if fallback_map is None else fallback_map
    base_model_id = fb.get(specialist_model_id)
    if base_model_id is None:
        return None
    for m in catalog.all_models:
        if m.model_id == base_model_id:
            return f"local:{m.backend_id}:{m.model_id}"
    return None
