"""Startup wiring: mesh pull-discovery → BackendRegistry.

Opt-in (default off, zero behavior change). When enabled, discovered remote
specialists are registered as backends so the existing dispatch path routes to
them. discover_live is patched so tests never touch a real tailnet.
"""

from __future__ import annotations

from slancha_local.mesh.discovery import DiscoveredSpecialist, DiscoveryResult
from slancha_local.proxy.main import build_app


def test_discovery_off_by_default_does_not_call_tailnet(monkeypatch):
    monkeypatch.delenv("SLANCHA_MESH_DISCOVERY_ENABLED", raising=False)
    called = []
    monkeypatch.setattr("slancha_local.proxy.main.discover_live", lambda **k: called.append(1))
    build_app()
    assert called == []  # no tailnet walk unless explicitly enabled


def test_discovery_enabled_registers_remote_specialist_as_backend(monkeypatch):
    monkeypatch.setenv("SLANCHA_MESH_DISCOVERY_ENABLED", "1")
    result = DiscoveryResult(
        specialists={
            "demo-model": DiscoveredSpecialist(
                specialist_id="demo-model", model_id="vendor/demo-model",
                node_urls=("http://mac.ts.net:8004",),
            )
        },
        reachable=["mac.ts.net"], unreachable=[],
    )
    monkeypatch.setattr("slancha_local.proxy.main.discover_live", lambda **k: result)
    app = build_app()
    backend = app.state.registry.by_id("demo-model")  # dispatchable by specialist id
    assert backend._base_url == "http://mac.ts.net:8004"


def test_discovery_failure_does_not_break_startup(monkeypatch):
    monkeypatch.setenv("SLANCHA_MESH_DISCOVERY_ENABLED", "1")

    def boom(**k):
        raise RuntimeError("tailnet exploded")

    monkeypatch.setattr("slancha_local.proxy.main.discover_live", boom)
    app = build_app()  # must not raise — discovery is best-effort
    assert app is not None
