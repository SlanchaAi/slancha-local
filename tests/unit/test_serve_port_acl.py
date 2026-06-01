"""Serve-port resolution under mesh registration (slancha-mesh#8).

The slancha-mesh tailnet ACL opens `tag:gateway → tag:specialist:8003,8004`
ONLY. A node that binds + advertises the standalone default :8000 registers
but is un-routable from the gateway. These tests pin the cross-repo fix:

  - Standalone (no mesh registry): effective port stays :8000 — non-mesh
    users are untouched.
  - Mesh registration ON, no explicit port: effective port defaults to the
    ACL-permitted :8003 (vLLM convention), and the advertised node_url is
    therefore reachable by the gateway with NO manual --port override.
  - An explicit SLANCHA_BIND_PORT always wins, even under mesh registration —
    the operator's choice is never silently moved.

These assert the *port resolution* + *advertised node_url*, which is the
acceptance criterion for mesh#8 (registered AND routable).
"""

from __future__ import annotations

import pytest

from slancha_local.config import MESH_ACL_MODEL_PORT, MESH_ACL_MODEL_PORTS, Settings

REGISTRY_ENV = "SLANCHA_MESH_REGISTRY_URL"
BIND_PORT_ENV = "SLANCHA_BIND_PORT"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start each test from a known env — neither var set."""
    monkeypatch.delenv(REGISTRY_ENV, raising=False)
    monkeypatch.delenv(BIND_PORT_ENV, raising=False)


# ---------------------------------------------------------------------------
# effective_bind_port — the resolution rule
# ---------------------------------------------------------------------------


def test_standalone_default_port_unchanged():
    """No mesh registry → historical :8000 default, non-mesh users untouched."""
    settings = Settings()
    assert settings.mesh_registration_enabled() is False
    assert settings.effective_bind_port() == 8000


def test_mesh_registration_defaults_to_acl_model_port(monkeypatch):
    """Mesh registry set + no explicit port → ACL-permitted :8003, no override."""
    monkeypatch.setenv(REGISTRY_ENV, "http://registry.example.ts.net:9000")
    settings = Settings()
    assert settings.mesh_registration_enabled() is True
    assert settings.effective_bind_port() == MESH_ACL_MODEL_PORT == 8003
    # Acceptance: the resolved port is one the gateway→specialist ACL permits.
    assert settings.effective_bind_port() in MESH_ACL_MODEL_PORTS


def test_explicit_bind_port_wins_under_mesh(monkeypatch):
    """An explicit SLANCHA_BIND_PORT is never silently moved, even under mesh."""
    monkeypatch.setenv(REGISTRY_ENV, "http://registry.example.ts.net:9000")
    monkeypatch.setenv(BIND_PORT_ENV, "8004")
    settings = Settings()
    assert settings.effective_bind_port() == 8004


def test_explicit_bind_port_standalone_still_honored(monkeypatch):
    """Standalone explicit port keeps working — no mesh coupling."""
    monkeypatch.setenv(BIND_PORT_ENV, "9999")
    settings = Settings()
    assert settings.effective_bind_port() == 9999


# ---------------------------------------------------------------------------
# Advertised node_url — the end-to-end acceptance for mesh#8
# ---------------------------------------------------------------------------


def test_advertised_node_url_is_acl_routable_under_mesh(monkeypatch):
    """The documented mesh path advertises an ACL-permitted port with no --port.

    Mirrors mesh_lifespan: node_url is built from effective_bind_port(). With
    a tailnet advertise host and mesh registration on, the URL's port must be
    one the `tag:gateway → tag:specialist` ACL permits.
    """
    from slancha_local.mesh.heartbeat import build_node_url

    monkeypatch.setenv(REGISTRY_ENV, "http://registry.example.ts.net:9000")
    settings = Settings()
    node_url = build_node_url(
        advertise_host="gb10.tnet-example.ts.net",
        bind_host=settings.bind_host,
        bind_port=settings.effective_bind_port(),
    )
    assert node_url == "http://gb10.tnet-example.ts.net:8003"
    port = int(node_url.rsplit(":", 1)[1])
    assert port in MESH_ACL_MODEL_PORTS


def test_advertised_node_url_standalone_keeps_8000(monkeypatch):
    """Without mesh registration the advertised port stays the legacy :8000."""
    from slancha_local.mesh.heartbeat import build_node_url

    settings = Settings()
    node_url = build_node_url(
        advertise_host=None,
        bind_host=settings.bind_host,
        bind_port=settings.effective_bind_port(),
    )
    assert node_url == "http://127.0.0.1:8000"
