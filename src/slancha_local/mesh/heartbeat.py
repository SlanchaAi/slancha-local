"""Mesh heartbeat client — opt-in mesh-node registration.

Posts NodeHeartbeat shapes to a slancha-mesh registry every 5s. The
heartbeat declares this slancha-local instance + which models its
backends expose. Mesh router (in slancha-api or another orchestrator)
routes domain-matched requests to this instance's /v1/chat/completions.

Wire format mirrors slancha-mesh spec §5 verbatim. No dependency on
the slancha-mesh package; the heartbeat is a plain dict assembled
here. If slancha-mesh changes the wire format, this module's tests
break (golden-trace fixtures); intentional coupling break.

Usage from slancha-local proxy:
    from slancha_local.mesh import MeshHeartbeatLoop
    loop = MeshHeartbeatLoop(
        registry_url=os.environ.get("SLANCHA_MESH_REGISTRY_URL"),
        node_url=f"http://{settings.bind_host}:{settings.bind_port}",
        catalog_fn=lambda: list_loaded_specialists(),
    )
    if loop.enabled:
        loop.start()    # spawns daemon thread
    ...
    loop.stop()        # on shutdown
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL_S = 5.0
DEFAULT_POST_TIMEOUT_S = 3.0
NODE_ID_ENV = "SLANCHA_NODE_ID"
NODE_TOKEN_ENV = "SLANCHA_NODE_TOKEN"
REGISTRY_URL_ENV = "SLANCHA_MESH_REGISTRY_URL"


def _stable_node_id() -> str:
    """Stable node id across restarts on the same host.

    Priority:
      1. SLANCHA_NODE_ID env (operator-set)
      2. uuid5(host=hostname) — deterministic per-machine
    """
    env = os.environ.get(NODE_ID_ENV)
    if env:
        return env
    return uuid.uuid5(uuid.NAMESPACE_DNS, socket.gethostname()).hex


@dataclass
class LoadedSpecialist:
    """One specialist this slancha-local instance is serving.

    Caller's `catalog_fn` returns a list of these on each heartbeat tick.
    """

    specialist_id: str
    model_id: str
    domain: str
    difficulty_tiers: list[str] = field(default_factory=lambda: ["medium"])
    estimated_tps: float | None = None


def build_heartbeat_payload(
    *,
    node_id: str,
    node_url: str,
    friendly_name: str,
    loaded: list[LoadedSpecialist],
    health: str = "healthy",
    queue_depth: int = 0,
    chip: str = "unknown",
    arch: str = "unknown",
    ram_total_gb: float = 0.0,
    ram_available_gb: float = 0.0,
    available_backends: list[str] | None = None,
    disk_free_gb: float = 0.0,
) -> dict[str, Any]:
    """Build the POST /heartbeat body matching slancha-mesh spec §5.

    Pure function — caller-injected fields make this trivially testable.
    The slancha-mesh service expects exactly this JSON shape; any drift
    here breaks the contract.
    """
    now = datetime.now(timezone.utc).isoformat()
    return {
        "heartbeat": {
            "node_id": node_id,
            "ts": now,
            "hardware": {
                "node_id": node_id,
                "friendly_name": friendly_name,
                "chip": chip,
                "arch": arch,
                "ram_total_gb": ram_total_gb,
                "ram_available_gb": ram_available_gb,
                "unified_memory": False,
                "memory_bandwidth_gbs": None,
                "available_backends": available_backends or [],
                "disk_free_gb": disk_free_gb,
            },
            "loaded_models": [
                {
                    "specialist_id": s.specialist_id,
                    "model_id": s.model_id,
                    "loaded_at": now,
                    "estimated_tps": s.estimated_tps,
                }
                for s in loaded
            ],
            "util": {"queue_depth": queue_depth},
            "health": health,
        },
        "node_url": node_url,
    }


@dataclass
class MeshHeartbeatLoop:
    """Background daemon thread that posts heartbeats to a mesh registry.

    Construct with a `catalog_fn` closure that the proxy can update over
    its lifetime — when a new backend loads, the next heartbeat reflects
    it automatically (no need to recreate the loop).

    The loop NEVER raises into the proxy. HTTP failures + connection
    errors are logged at INFO and the next tick retries. If the
    registry is permanently gone, slancha-local just keeps serving
    local requests without mesh integration.
    """

    registry_url: str | None
    node_url: str
    friendly_name: str
    catalog_fn: Callable[[], list[LoadedSpecialist]]
    # Optional hardware-probe closure — returns {chip, arch, ram_*, ...}.
    # Default: empty dict (the heartbeat fields stay at defaults).
    hardware_fn: Callable[[], dict[str, Any]] = field(
        default_factory=lambda: (lambda: {})
    )
    node_id: str = field(default_factory=_stable_node_id)
    token: str | None = field(default_factory=lambda: os.environ.get(NODE_TOKEN_ENV))
    interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S
    post_timeout_s: float = DEFAULT_POST_TIMEOUT_S

    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _heartbeats_sent: int = field(default=0, init=False)
    _consecutive_failures: int = field(default=0, init=False)

    @property
    def enabled(self) -> bool:
        return bool(self.registry_url)

    @property
    def heartbeats_sent(self) -> int:
        return self._heartbeats_sent

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def post_once(self) -> bool:
        """Post a single heartbeat. Returns True on 2xx, False otherwise."""
        if not self.registry_url:
            return False
        loaded = self.catalog_fn()
        hardware = self.hardware_fn() or {}
        payload = build_heartbeat_payload(
            node_id=self.node_id,
            node_url=self.node_url,
            friendly_name=self.friendly_name,
            loaded=loaded,
            **hardware,
        )
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            resp = httpx.post(
                f"{self.registry_url.rstrip('/')}/heartbeat",
                json=payload,
                headers=headers,
                timeout=self.post_timeout_s,
            )
            if 200 <= resp.status_code < 300:
                self._heartbeats_sent += 1
                self._consecutive_failures = 0
                return True
            logger.info(
                "mesh heartbeat → %s returned %d", self.registry_url, resp.status_code
            )
        except (httpx.HTTPError, OSError) as exc:
            logger.info("mesh heartbeat to %s failed: %s", self.registry_url, exc)
        self._consecutive_failures += 1
        return False

    def start(self) -> None:
        if not self.enabled:
            logger.info("mesh heartbeat disabled (no registry_url)")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        logger.info(
            "starting mesh heartbeat loop → %s (interval=%.1fs node_id=%s)",
            self.registry_url, self.interval_s, self.node_id,
        )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="mesh-heartbeat"
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.post_once()
            self._stop.wait(self.interval_s)


__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL_S",
    "DEFAULT_POST_TIMEOUT_S",
    "LoadedSpecialist",
    "MeshHeartbeatLoop",
    "NODE_ID_ENV",
    "NODE_TOKEN_ENV",
    "REGISTRY_URL_ENV",
    "build_heartbeat_payload",
]
