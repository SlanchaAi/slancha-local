"""Lifespan integration — wire MeshHeartbeatLoop into the FastAPI app.

Lifespan handler that:
  - On startup: refresh CapabilityProbe ONCE so the catalog snapshot has
    data for the first heartbeat; build a MeshHeartbeatLoop pointed at
    SLANCHA_MESH_REGISTRY_URL; start the loop's background thread; attach
    to `app.state.mesh_heartbeat_loop` for visibility.
  - On shutdown: signal stop + join the thread within `stop_timeout_s`.

Honors `MeshHeartbeatLoop.enabled` — when SLANCHA_MESH_REGISTRY_URL is
unset or empty, the loop is constructed but never started. Booting the
proxy without the env set therefore has ZERO mesh side effects, matching
the existing slancha-local default-off posture.

The contract on the loop is "NEVER raises into the proxy" (per
heartbeat.py docstring). The lifespan respects that — startup never
fails because of a bogus registry URL.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from slancha_local.capability.probe import CapabilityProbe
from slancha_local.config import Settings
from slancha_local.mesh.heartbeat import LoadedSpecialist, MeshHeartbeatLoop

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Backends whose models should NOT be advertised to the mesh registry.
# Cloud backends serve via someone else's infra; the mesh router can't
# route to them as local nodes. OpenRouter is the canonical example.
_CLOUD_BACKEND_IDS = frozenset({"openrouter", "openai", "anthropic"})


def _is_cloud_backend(backend_id: str) -> bool:
    return backend_id.lower() in _CLOUD_BACKEND_IDS


def build_catalog_fn(probe: CapabilityProbe):
    """Return a sync `catalog_fn` closure suitable for MeshHeartbeatLoop.

    Reads the probe's cached snapshot (no await) and maps every healthy
    LOCAL backend's models to a LoadedSpecialist. Cloud-backed models
    are filtered out — they can't be mesh-routed.

    Returns an empty list if the probe hasn't refreshed yet. The mesh
    registry will see an empty `loaded_models` for those heartbeats;
    next refresh populates it.
    """

    def _fn() -> list[LoadedSpecialist]:
        snapshot = probe.snapshot()
        if snapshot is None:
            return []
        out: list[LoadedSpecialist] = []
        for cap in snapshot.capabilities:
            if not cap.healthy:
                continue
            if _is_cloud_backend(cap.id):
                continue
            for model in cap.models:
                out.append(
                    LoadedSpecialist(
                        specialist_id=model.model_id,
                        model_id=model.model_id,
                        # slancha-local doesn't currently classify per-model
                        # domain coverage at the registry layer. Default to
                        # "general" so the mesh router can still route to
                        # this node — slancha-mesh's allocator may downrank
                        # generic nodes but won't reject them.
                        domain="general",
                        estimated_tps=model.est_throughput_tps,
                    )
                )
        return out

    return _fn


def build_hardware_fn():
    """Return a sync `hardware_fn` closure populating the probe-shaped dict.

    Uses psutil + shutil + platform; returns the subset of fields
    slancha-mesh.NodeProbe accepts. arch is omitted here so
    `build_heartbeat_payload` falls through to `probe_arch()` which
    returns a valid `MeshArch` Literal (fixed in fe1c904).
    """
    import platform
    import shutil

    try:
        import psutil  # type: ignore[import]
    except ImportError:
        psutil = None  # type: ignore[assignment]

    def _fn() -> dict[str, object]:
        out: dict[str, object] = {
            "chip": platform.processor() or platform.machine() or "unknown",
        }
        if psutil is not None:
            vm = psutil.virtual_memory()
            out["ram_total_gb"] = round(vm.total / 1e9, 2)
            out["ram_available_gb"] = round(vm.available / 1e9, 2)
        # Disk free at the configured traces root is a useful signal
        # for whether this node can absorb training-checkpoint writes.
        try:
            usage = shutil.disk_usage("/")
            out["disk_free_gb"] = round(usage.free / 1e9, 2)
        except OSError:
            pass
        return out

    return _fn


@contextlib.asynccontextmanager
async def mesh_lifespan(app: "FastAPI") -> AsyncIterator[None]:
    """FastAPI lifespan that owns the MeshHeartbeatLoop.

    Reads `app.state.settings` + `app.state.probe` (populated earlier
    in `build_app`). Refreshes the probe once before constructing the
    loop so the first heartbeat carries real catalog data.
    """
    import os

    settings: Settings = app.state.settings
    probe: CapabilityProbe = app.state.probe

    # Refresh once so the first heartbeat has real loaded_models.
    # Tolerant: if no backends respond, snapshot() returns the empty
    # catalog and the heartbeat advertises zero specialists.
    try:
        await probe.refresh()
    except Exception:  # noqa: BLE001 — startup is non-fatal
        logger.exception("mesh-lifespan: probe.refresh() failed at startup")

    registry_url = os.environ.get("SLANCHA_MESH_REGISTRY_URL")
    loop = MeshHeartbeatLoop(
        registry_url=registry_url,
        node_url=f"http://{settings.bind_host}:{settings.bind_port}",
        friendly_name=getattr(settings, "node_friendly_name", "slancha-local"),
        catalog_fn=build_catalog_fn(probe),
        hardware_fn=build_hardware_fn(),
    )
    app.state.mesh_heartbeat_loop = loop

    if loop.enabled:
        try:
            loop.start()
            logger.info(
                "mesh-lifespan: heartbeat loop started → %s",
                loop.registry_url,
            )
        except Exception:  # noqa: BLE001 — NEVER fail proxy boot on mesh
            logger.exception(
                "mesh-lifespan: loop.start() failed; proxy continues without mesh"
            )
    else:
        logger.info(
            "mesh-lifespan: SLANCHA_MESH_REGISTRY_URL not set; mesh integration disabled"
        )

    try:
        yield
    finally:
        if loop.enabled:
            try:
                loop.stop(timeout=5.0)
            except Exception:  # noqa: BLE001 — best-effort shutdown
                logger.exception("mesh-lifespan: loop.stop() raised; ignoring")
