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

import json
import logging
import os
import platform
import socket
import subprocess
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import httpx

# Mirrors slancha-mesh Arch literal (mesh/models.py:27). Spec §5
# requires arch ∈ {aarch64, x86_64, apple-silicon}. We probe at
# runtime; if our heuristic guesses wrong, the heartbeat 422s and
# slancha-local never registers — caught by paul-mac's M3 protocol
# golden traces (slancha-mesh-side). Default "x86_64" rather than
# "unknown" so a misdetect still validates against the strict schema
# even if it picks the wrong arch label.
MeshArch = Literal["aarch64", "x86_64", "apple-silicon"]

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL_S = 5.0
DEFAULT_POST_TIMEOUT_S = 3.0
NODE_ID_ENV = "SLANCHA_NODE_ID"
NODE_TOKEN_ENV = "SLANCHA_NODE_TOKEN"
REGISTRY_URL_ENV = "SLANCHA_MESH_REGISTRY_URL"


def probe_arch() -> MeshArch:
    """Detect the host's arch label matching slancha-mesh spec §5.

    Priority:
      1. Darwin + ARM → "apple-silicon" (M-series Macs)
      2. machine() in {aarch64, arm64} → "aarch64" (Linux ARM, Spark GB10)
      3. machine() in {x86_64, amd64} → "x86_64"
      4. fallback → "x86_64" (safer than "unknown" which fails validation)

    Why "apple-silicon" distinct from "aarch64": slancha-mesh's
    allocator routes MLX backends only to apple-silicon nodes; if a
    Mac mini labels itself aarch64, it would be considered for
    Linux-only vllm placement. The Darwin+arm64 check encodes that.
    """
    machine = platform.machine().lower()
    system = platform.system()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "apple-silicon"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    # Last resort — pick x86_64 (most common cloud host arch).
    # Misdetect-but-validates beats misdetect-and-422 for first-boot UX.
    logger.warning(
        "could not classify arch %r on %s; defaulting to x86_64",
        machine, system,
    )
    return "x86_64"


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


# ---------------------------------------------------------------------------
# Tailnet advertise-host resolution (producer side)
# ---------------------------------------------------------------------------
# slancha-local advertises ONE node_url the mesh gateway dials over a
# Tailscale/Headscale tailnet by MagicDNS. Bind (where we LISTEN, e.g.
# 0.0.0.0) and advertise (a routable MagicDNS name) are distinct — the old
# code conflated them and advertised loopback, unreachable from a cloud
# gateway. Mirrors slancha-mesh/mesh/tailnet.py but re-implemented here: this
# module carries no slancha-mesh dependency (see module docstring). The
# subprocess path follows probe_arch's never-raise posture.


def parse_magicdns_name(status: dict | str) -> str | None:
    """Pull `Self.DNSName` from a `tailscale status --json` payload.

    Returns the FQDN minus its trailing dot, or None if the payload is
    missing/empty/unparseable. Identical shape on Tailscale and Headscale.
    """
    if isinstance(status, str):
        try:
            status = json.loads(status)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(status, dict):
        return None
    self_obj = status.get("Self")
    if not isinstance(self_obj, dict):
        return None
    name = self_obj.get("DNSName")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.rstrip(".") or None


def resolve_magicdns_name(tailscale_bin: str = "tailscale") -> str | None:
    """Run `tailscale status --json` and return this node's MagicDNS name.

    Never raises: missing binary, non-zero exit, or unparseable output all
    yield None so the heartbeat loop survives.
    """
    try:
        out = subprocess.run(
            [tailscale_bin, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    return parse_magicdns_name(out.stdout)


def resolve_advertise_host(
    explicit: str | None,
    _resolver: Callable[[], str | None] = resolve_magicdns_name,
) -> str | None:
    """The host the registry should advertise for this node.

    Priority: explicit override > MagicDNS discovery > None. None means no
    tailnet name is available — the caller keeps the loopback bind host
    (dev mode). `_resolver` is injectable for tests.
    """
    if explicit:
        return explicit
    return _resolver()


def build_node_url(*, advertise_host: str | None, bind_host: str, bind_port: int) -> str:
    """Construct the advertised node_url.

    Uses `advertise_host` (a routable MagicDNS name) when present, else falls
    back to `bind_host` so non-tailnet dev is unchanged.
    """
    host = advertise_host or bind_host
    return f"http://{host}:{bind_port}"


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


def specialists_from_models(models: Iterable[Any]) -> list[LoadedSpecialist]:
    """Map healthy backend models → heartbeat loaded_models entries.

    slancha-local fronts generic OpenAI-compat backends with no per-model
    domain/specialist metadata, so specialist_id == model_id and domain is
    "general" (the mesh router does the matching; classification isn't this
    node's job). Duck-typed on `.model_id` / `.est_throughput_tps` to avoid
    a hard import of BackendModel.
    """
    return [
        LoadedSpecialist(
            specialist_id=m.model_id,
            model_id=m.model_id,
            domain="general",
            estimated_tps=getattr(m, "est_throughput_tps", None),
        )
        for m in models
    ]


def build_heartbeat_payload(
    *,
    node_id: str,
    node_url: str,
    friendly_name: str,
    loaded: list[LoadedSpecialist],
    health: str = "healthy",
    queue_depth: int = 0,
    chip: str = "unknown",
    arch: MeshArch | None = None,
    ram_total_gb: float = 0.0,
    ram_available_gb: float = 0.0,
    available_backends: list[str] | None = None,
    disk_free_gb: float = 0.0,
) -> dict[str, Any]:
    """Build the POST /heartbeat body matching slancha-mesh spec §5.

    Pure function — caller-injected fields make this trivially testable.
    The slancha-mesh service expects exactly this JSON shape; any drift
    here breaks the contract.

    `arch` defaults to probe_arch() so callers don't pass an invalid
    "unknown" string that fails slancha-mesh's NodeProbe.arch Literal
    validation (caught by mac M3 cross-repo verification 2026-05-16).
    """
    now = datetime.now(UTC).isoformat()
    if arch is None:
        arch = probe_arch()
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
    "MeshArch",
    "MeshHeartbeatLoop",
    "NODE_ID_ENV",
    "NODE_TOKEN_ENV",
    "REGISTRY_URL_ENV",
    "build_heartbeat_payload",
    "build_node_url",
    "parse_magicdns_name",
    "probe_arch",
    "resolve_advertise_host",
    "resolve_magicdns_name",
    "specialists_from_models",
]
