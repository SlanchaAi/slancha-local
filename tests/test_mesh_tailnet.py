"""Tailnet advertise-host resolution for the mesh heartbeat producer.

slancha-local is the heartbeat PRODUCER: it advertises a single node_url
the mesh gateway dials over a Tailscale/Headscale tailnet by MagicDNS.
The old code advertised the *bind* address (loopback) — unreachable from
a cloud gateway. These tests lock the bind/advertise split:
`tailscale status --json` → `Self.DNSName`, with a never-raise contract so
a missing binary degrades to the loopback URL (dev mode) rather than
crashing the heartbeat loop.

Mirrors slancha-mesh/mesh/tests/test_tailnet.py (the consumer side) but
re-implemented here — heartbeat.py deliberately carries NO dependency on
the slancha-mesh package (see its module docstring).
"""

from __future__ import annotations

import json

from slancha_local.mesh.heartbeat import (
    LoadedSpecialist,
    build_node_url,
    parse_magicdns_name,
    resolve_advertise_host,
    resolve_magicdns_name,
    specialists_from_models,
)

# A captured `tailscale status --json` Self block. Tailscale and Headscale
# both populate Self.DNSName with a trailing-dot FQDN.
_STATUS_JSON = {
    "Self": {
        "HostName": "gb10-node",
        "DNSName": "gb10-node.tnet-example.ts.net.",
        "TailscaleIPs": ["100.64.0.1", "fd7a:115c:a1e0::1234"],
        "Online": True,
        "Tags": ["tag:specialist"],
    },
    "MagicDNSSuffix": "tnet-example.ts.net",
}


# ---------------------------------------------------------------------------
# parse_magicdns_name — pure parse of captured status JSON
# ---------------------------------------------------------------------------


def test_parse_magicdns_name_strips_trailing_dot():
    assert parse_magicdns_name(_STATUS_JSON) == "gb10-node.tnet-example.ts.net"


def test_parse_magicdns_name_accepts_json_string():
    assert parse_magicdns_name(json.dumps(_STATUS_JSON)) == "gb10-node.tnet-example.ts.net"


def test_parse_magicdns_name_none_on_missing_or_empty():
    assert parse_magicdns_name({}) is None
    assert parse_magicdns_name({"Self": {}}) is None
    assert parse_magicdns_name({"Self": {"DNSName": ""}}) is None
    assert parse_magicdns_name("not json at all") is None


# ---------------------------------------------------------------------------
# resolve_magicdns_name — subprocess, never-raise contract
# ---------------------------------------------------------------------------


def _fake_run(stdout="", returncode=0):
    class _R:
        pass

    r = _R()
    r.stdout = stdout
    r.returncode = returncode
    return r


def test_resolve_magicdns_name_parses_live_status(monkeypatch):
    monkeypatch.setattr(
        "slancha_local.mesh.heartbeat.subprocess.run",
        lambda *a, **kw: _fake_run(stdout=json.dumps(_STATUS_JSON), returncode=0),
    )
    assert resolve_magicdns_name() == "gb10-node.tnet-example.ts.net"


def test_resolve_magicdns_name_none_on_missing_binary(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("tailscale not installed")

    monkeypatch.setattr("slancha_local.mesh.heartbeat.subprocess.run", boom)
    assert resolve_magicdns_name() is None


def test_resolve_magicdns_name_none_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        "slancha_local.mesh.heartbeat.subprocess.run",
        lambda *a, **kw: _fake_run(stdout="", returncode=1),
    )
    assert resolve_magicdns_name() is None


def test_resolve_magicdns_name_none_on_unparseable(monkeypatch):
    monkeypatch.setattr(
        "slancha_local.mesh.heartbeat.subprocess.run",
        lambda *a, **kw: _fake_run(stdout="<<<garbage", returncode=0),
    )
    assert resolve_magicdns_name() is None


# ---------------------------------------------------------------------------
# resolve_advertise_host — priority: explicit > magicdns > None
# ---------------------------------------------------------------------------


def test_resolve_advertise_host_prefers_explicit_override():
    # Explicit wins even if the resolver would return something.
    got = resolve_advertise_host("myhost.example", _resolver=lambda: "other.ts.net")
    assert got == "myhost.example"


def test_resolve_advertise_host_falls_back_to_magicdns():
    got = resolve_advertise_host(None, _resolver=lambda: "gb10.taila.ts.net")
    assert got == "gb10.taila.ts.net"


def test_resolve_advertise_host_none_when_nothing_resolves():
    # No explicit + resolver returns None (no tailnet) → None. Caller then
    # keeps the loopback bind host (dev mode).
    assert resolve_advertise_host(None, _resolver=lambda: None) is None


# ---------------------------------------------------------------------------
# build_node_url — advertise host swaps in, port preserved, bind fallback
# ---------------------------------------------------------------------------


def test_build_node_url_uses_advertise_host_when_present():
    url = build_node_url(
        advertise_host="gb10.tnet-example.ts.net", bind_host="0.0.0.0", bind_port=8000
    )
    assert url == "http://gb10.tnet-example.ts.net:8000"


def test_build_node_url_falls_back_to_bind_host():
    # No advertise host (non-tailnet dev) → loopback URL unchanged.
    url = build_node_url(advertise_host=None, bind_host="127.0.0.1", bind_port=8000)
    assert url == "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# specialists_from_models — backend models → heartbeat loaded_models
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, model_id, tps=None):
        self.model_id = model_id
        self.est_throughput_tps = tps


def test_specialists_from_models_maps_id_and_tps():
    out = specialists_from_models([_FakeModel("qwen3:14b", 42.5)])
    assert len(out) == 1
    s = out[0]
    assert isinstance(s, LoadedSpecialist)
    assert s.specialist_id == "qwen3:14b"
    assert s.model_id == "qwen3:14b"
    assert s.estimated_tps == 42.5
    assert s.domain == "general"


def test_specialists_from_models_empty():
    assert specialists_from_models([]) == []


# ---------------------------------------------------------------------------
# build_heartbeat_loop — wires registry/node_url/catalog into the loop
# ---------------------------------------------------------------------------


class _FakeProbe:
    """Stand-in for CapabilityProbe exposing only the sync cached() read."""

    def __init__(self, catalog):
        self._catalog = catalog

    def cached(self):
        return self._catalog


def test_build_heartbeat_loop_disabled_without_registry(monkeypatch):
    monkeypatch.delenv("SLANCHA_MESH_REGISTRY_URL", raising=False)
    from slancha_local.config import Settings
    from slancha_local.proxy.main import build_heartbeat_loop

    loop = build_heartbeat_loop(Settings(), _FakeProbe(None))
    assert loop.enabled is False
    # None cache → empty loaded_models (no crash).
    assert loop.catalog_fn() == []


def test_build_heartbeat_loop_enabled_advertises_tailnet_host(monkeypatch):
    monkeypatch.setenv("SLANCHA_MESH_REGISTRY_URL", "http://reg.local:9000")
    monkeypatch.setenv("SLANCHA_MESH_ADVERTISE_HOST", "gb10.tnet-example.ts.net")
    from slancha_local.backends.base import BackendCapability, BackendModel
    from slancha_local.capability.catalog import LocalCatalog
    from slancha_local.config import Settings
    from slancha_local.proxy.main import build_heartbeat_loop

    cat = LocalCatalog(
        capabilities=(
            BackendCapability(
                id="ollama",
                healthy=True,
                base_url="http://127.0.0.1:11434",
                models=(
                    BackendModel(
                        backend_id="ollama",
                        model_id="qwen3:14b",
                        ctx_window=40960,
                        est_throughput_tps=12.0,
                    ),
                ),
            ),
        )
    )
    loop = build_heartbeat_loop(Settings(), _FakeProbe(cat))
    assert loop.enabled is True
    # Explicit advertise host wins; bind port preserved (default 8000).
    assert loop.node_url == "http://gb10.tnet-example.ts.net:8000"
    specs = loop.catalog_fn()
    assert [s.model_id for s in specs] == ["qwen3:14b"]
    assert specs[0].estimated_tps == 12.0
    assert specs[0].domain == "general"


def test_lifespan_runs_clean_without_mesh(monkeypatch):
    """Lifespan must not break startup when mesh is off (the default):
    heartbeat built but disabled, no thread, healthz still serves."""
    monkeypatch.delenv("SLANCHA_MESH_REGISTRY_URL", raising=False)
    from fastapi.testclient import TestClient

    from slancha_local.proxy.main import app

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
    assert app.state.mesh_heartbeat.enabled is False

