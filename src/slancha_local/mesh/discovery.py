"""Pull-discovery client — build a routing table by walking the tailnet.

A slancha-local router (a local home router, or a per-account cloud gateway
node) enumerates `tag:specialist` peers from `tailscale status --json` and
GETs each one's `/models?include=routing_meta` over the tailnet. This is the
*consume* side of slancha-mesh's pull discovery; the *produce* side is each
mesh node exposing `/models`.

Re-implemented here against the documented wire shape — **zero import of
slancha-mesh** (same cross-repo discipline as `mesh/heartbeat.py`: shapes
cross as plain JSON, never as shared Python types). Mirrors
slancha-mesh/mesh/discovery.py; drift surfaces as a contract-test failure.

Security: `node_url` is **host-pinned to the peer actually dialed**
(`pin_host`) — a node cannot advertise another node's address. `DNSName` is
control-plane-attested; `HostName` is self-reported and ignored. Never-raises:
a dead peer contributes nothing rather than aborting the pass.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

DEFAULT_SPECIALIST_TAG = "tag:specialist"
DEFAULT_NODE_INFO_PORT = 8088
FetchFn = Callable[[str, int], "dict | None"]


@dataclass(frozen=True)
class SpecialistPeer:
    host: str  # MagicDNS name, trailing dot stripped
    online: bool
    is_self: bool = False


@dataclass(frozen=True)
class DiscoveredSpecialist:
    specialist_id: str
    model_id: str | None = None
    domain: str | None = None
    capabilities: tuple[str, ...] = ()
    quality_router_observed: float | None = None
    node_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveryResult:
    specialists: dict[str, DiscoveredSpecialist] = field(default_factory=dict)
    reachable: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)


def _coerce(status: dict | str) -> dict | None:
    if isinstance(status, str):
        try:
            status = json.loads(status)
        except (json.JSONDecodeError, ValueError):
            return None
    return status if isinstance(status, dict) else None


def _host_of(node: dict) -> str | None:
    name = node.get("DNSName")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.rstrip(".") or None


def parse_specialist_peers(
    status: dict | str,
    specialist_tag: str = DEFAULT_SPECIALIST_TAG,
    include_self: bool = True,
) -> list[SpecialistPeer]:
    """Tagged, online specialist peers from a `tailscale status --json`."""
    data = _coerce(status)
    if data is None:
        return []
    peers: list[SpecialistPeer] = []
    if include_self:
        self_obj = data.get("Self")
        if isinstance(self_obj, dict) and specialist_tag in (self_obj.get("Tags") or []):
            host = _host_of(self_obj)
            if host:
                peers.append(SpecialistPeer(host=host, online=True, is_self=True))
    peer_map = data.get("Peer")
    if isinstance(peer_map, dict):
        for node in peer_map.values():
            if not isinstance(node, dict):
                continue
            if specialist_tag not in (node.get("Tags") or []):
                continue
            if not node.get("Online", False):
                continue
            host = _host_of(node)
            if host:
                peers.append(SpecialistPeer(host=host, online=True, is_self=False))
    return peers


def pin_host(node_url: str, peer_host: str) -> str:
    """Force `node_url`'s host to `peer_host`, keeping scheme/port/path.

    The node tells us which PORT; it does not get to tell us which HOST —
    that's the address we pulled from. This makes claim-hijack impossible.
    """
    parts = urlsplit(node_url)
    port = parts.port
    netloc = f"{peer_host}:{port}" if port is not None else peer_host
    return urlunsplit((parts.scheme or "http", netloc, parts.path, parts.query, parts.fragment))


def _merge(into: dict[str, DiscoveredSpecialist], spec: DiscoveredSpecialist) -> None:
    existing = into.get(spec.specialist_id)
    if existing is None:
        into[spec.specialist_id] = spec
        return
    merged_urls = tuple(dict.fromkeys((*existing.node_urls, *spec.node_urls)))
    into[spec.specialist_id] = DiscoveredSpecialist(
        specialist_id=existing.specialist_id,
        model_id=existing.model_id or spec.model_id,
        domain=existing.domain or spec.domain,
        capabilities=existing.capabilities or spec.capabilities,
        quality_router_observed=(
            existing.quality_router_observed
            if existing.quality_router_observed is not None
            else spec.quality_router_observed
        ),
        node_urls=merged_urls,
    )


def _specialists_from_models(payload: dict, peer_host: str) -> list[DiscoveredSpecialist]:
    out: list[DiscoveredSpecialist] = []
    for entry in payload.get("data", []) or []:
        if not isinstance(entry, dict):
            continue
        spec_id = entry.get("id")
        if not isinstance(spec_id, str) or not spec_id:
            continue
        meta = entry.get("routing_meta")
        if not isinstance(meta, dict):
            continue
        raw_urls = meta.get("node_urls") or []
        pinned = tuple(
            dict.fromkeys(pin_host(u, peer_host) for u in raw_urls if isinstance(u, str) and u)
        )
        if not pinned:
            continue
        quality = meta.get("quality") or {}
        out.append(
            DiscoveredSpecialist(
                specialist_id=spec_id,
                model_id=meta.get("model_id"),
                domain=meta.get("domain"),
                capabilities=tuple(meta.get("capabilities") or []),
                quality_router_observed=quality.get("router_observed"),
                node_urls=pinned,
            )
        )
    return out


def discover_specialists(
    status: dict | str,
    fetch: FetchFn,
    *,
    node_info_port: int = DEFAULT_NODE_INFO_PORT,
    specialist_tag: str = DEFAULT_SPECIALIST_TAG,
    include_self: bool = True,
) -> DiscoveryResult:
    """Walk the tailnet, pull each specialist node, aggregate into routes."""
    peers = parse_specialist_peers(status, specialist_tag=specialist_tag, include_self=include_self)
    specialists: dict[str, DiscoveredSpecialist] = {}
    reachable: list[str] = []
    unreachable: list[str] = []
    for peer in peers:
        payload = fetch(peer.host, node_info_port)
        if not isinstance(payload, dict):
            unreachable.append(peer.host)
            continue
        reachable.append(peer.host)
        for spec in _specialists_from_models(payload, peer.host):
            _merge(specialists, spec)
    return DiscoveryResult(specialists=specialists, reachable=reachable, unreachable=unreachable)


def make_http_fetch(*, token: str | None = None, timeout: float = 4.0) -> FetchFn:
    """Live `fetch(host, port)` → GET /models?include=routing_meta. Never raises."""
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def fetch(host: str, port: int) -> dict | None:
        try:
            resp = httpx.get(
                f"http://{host}:{port}/models",
                params={"include": "routing_meta"}, headers=headers, timeout=timeout,
            )
        except Exception:  # noqa: BLE001 — never-raise discovery contract
            return None
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            return None
        return data if isinstance(data, dict) else None

    return fetch


def tailnet_status(tailscale_bin: str = "tailscale") -> dict | None:
    """Parsed `tailscale status --json`, or None (never-raise)."""
    try:
        out = subprocess.run(
            [tailscale_bin, "status", "--json"],
            capture_output=True, text=True, timeout=4.0, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    try:
        data = json.loads(out.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def discover_live(
    *,
    tailscale_bin: str | None = None,
    node_info_port: int = DEFAULT_NODE_INFO_PORT,
    token: str | None = None,
) -> DiscoveryResult:
    """Convenience: read the live tailnet + pull. Empty result if not on a tailnet."""
    status = tailnet_status(tailscale_bin or os.environ.get("SLANCHA_TAILSCALE_BIN", "tailscale"))
    if status is None:
        return DiscoveryResult()
    return discover_specialists(
        status, fetch=make_http_fetch(token=token), node_info_port=node_info_port,
    )


__all__ = [
    "DEFAULT_NODE_INFO_PORT",
    "DEFAULT_SPECIALIST_TAG",
    "DiscoveredSpecialist",
    "DiscoveryResult",
    "FetchFn",
    "SpecialistPeer",
    "discover_live",
    "discover_specialists",
    "make_http_fetch",
    "parse_specialist_peers",
    "pin_host",
    "tailnet_status",
]
