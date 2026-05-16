"""Mesh heartbeat client tests — opt-in mesh registration.

Validates the heartbeat wire format matches slancha-mesh spec §5
verbatim, and that the loop is genuinely opt-in (never starts threads
unless registry_url is set).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime
from unittest.mock import patch

import httpx
import pytest

from slancha_local.mesh.heartbeat import (
    DEFAULT_HEARTBEAT_INTERVAL_S,
    LoadedSpecialist,
    MeshHeartbeatLoop,
    _stable_node_id,
    build_heartbeat_payload,
)


# ---------------------------------------------------------------------------
# Node id
# ---------------------------------------------------------------------------


def test_stable_node_id_from_env(monkeypatch):
    monkeypatch.setenv("SLANCHA_NODE_ID", "test-node-42")
    assert _stable_node_id() == "test-node-42"


def test_stable_node_id_falls_back_to_hostname_hash(monkeypatch):
    monkeypatch.delenv("SLANCHA_NODE_ID", raising=False)
    nid = _stable_node_id()
    assert len(nid) == 32  # uuid5 hex
    # Re-invocation on same host produces same id (stable across restarts)
    assert _stable_node_id() == nid


# ---------------------------------------------------------------------------
# Payload shape (slancha-mesh spec §5)
# ---------------------------------------------------------------------------


def test_payload_top_level_keys():
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://127.0.0.1:8000",
        friendly_name="laptop",
        loaded=[],
    )
    assert set(p.keys()) == {"heartbeat", "node_url"}
    assert p["node_url"] == "http://127.0.0.1:8000"


def test_payload_heartbeat_required_fields_present():
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
    )
    hb = p["heartbeat"]
    # spec §5 required fields
    for k in ("node_id", "ts", "hardware", "loaded_models", "util", "health"):
        assert k in hb, f"required field {k!r} missing"
    # hardware shape
    for k in ("node_id", "friendly_name", "chip", "arch",
              "ram_total_gb", "ram_available_gb", "unified_memory",
              "memory_bandwidth_gbs", "available_backends", "disk_free_gb"):
        assert k in hb["hardware"], f"hardware.{k} missing"


def test_payload_loaded_models_shape():
    spec = LoadedSpecialist(
        specialist_id="qwen3-8b",
        model_id="Qwen/Qwen3-8B",
        domain="general",
        estimated_tps=42.5,
    )
    p = build_heartbeat_payload(
        node_id="n1", node_url="http://x", friendly_name="laptop",
        loaded=[spec],
    )
    lm = p["heartbeat"]["loaded_models"]
    assert len(lm) == 1
    assert lm[0]["specialist_id"] == "qwen3-8b"
    assert lm[0]["model_id"] == "Qwen/Qwen3-8B"
    assert lm[0]["estimated_tps"] == 42.5
    assert "loaded_at" in lm[0]


def test_payload_ts_iso8601_utc():
    p = build_heartbeat_payload(
        node_id="n1", node_url="http://x", friendly_name="laptop", loaded=[],
    )
    ts = p["heartbeat"]["ts"]
    # Parseable as ISO 8601
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None  # must be tz-aware


def test_payload_health_defaults_healthy():
    p = build_heartbeat_payload(
        node_id="n1", node_url="http://x", friendly_name="laptop", loaded=[],
    )
    assert p["heartbeat"]["health"] == "healthy"


def test_payload_health_can_be_overridden():
    p = build_heartbeat_payload(
        node_id="n1", node_url="http://x", friendly_name="laptop", loaded=[],
        health="degraded",
    )
    assert p["heartbeat"]["health"] == "degraded"


def test_payload_json_roundtrips_cleanly():
    """Payload must serialize via json.dumps without TypeError (no
    datetime objects leaking through)."""
    p = build_heartbeat_payload(
        node_id="n1", node_url="http://x", friendly_name="laptop",
        loaded=[LoadedSpecialist(
            specialist_id="x", model_id="y", domain="general",
            estimated_tps=10.0,
        )],
    )
    s = json.dumps(p)
    back = json.loads(s)
    assert back == p


# ---------------------------------------------------------------------------
# MeshHeartbeatLoop — enable/disable
# ---------------------------------------------------------------------------


def test_loop_disabled_when_no_registry_url():
    loop = MeshHeartbeatLoop(
        registry_url=None,
        node_url="http://x",
        friendly_name="laptop",
        catalog_fn=lambda: [],
    )
    assert loop.enabled is False
    loop.start()  # should NOT spawn a thread
    assert loop._thread is None
    loop.stop()


def test_loop_disabled_when_empty_registry_url():
    loop = MeshHeartbeatLoop(
        registry_url="",
        node_url="http://x",
        friendly_name="laptop",
        catalog_fn=lambda: [],
    )
    assert loop.enabled is False


def test_loop_enabled_when_registry_url_set():
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x",
        friendly_name="laptop",
        catalog_fn=lambda: [],
    )
    assert loop.enabled is True


def test_loop_post_once_returns_false_when_disabled():
    loop = MeshHeartbeatLoop(
        registry_url=None,
        node_url="http://x",
        friendly_name="laptop",
        catalog_fn=lambda: [],
    )
    assert loop.post_once() is False


# ---------------------------------------------------------------------------
# MeshHeartbeatLoop — post_once with mocked HTTP
# ---------------------------------------------------------------------------


def test_post_once_success(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        class R:
            status_code = 200
            def json(self_inner):
                return {"ack": True, "next_due_seconds": 5}
        return R()

    monkeypatch.setattr(
        "slancha_local.mesh.heartbeat.httpx.post", fake_post
    )
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://127.0.0.1:8000",
        friendly_name="laptop",
        catalog_fn=lambda: [LoadedSpecialist(
            specialist_id="qwen3-8b", model_id="Qwen/Qwen3-8B",
            domain="general",
        )],
    )
    assert loop.post_once() is True
    assert loop.heartbeats_sent == 1
    assert loop.consecutive_failures == 0
    # Verify wire shape sent
    assert captured["url"].endswith("/heartbeat")
    assert captured["json"]["node_url"] == "http://127.0.0.1:8000"
    assert len(captured["json"]["heartbeat"]["loaded_models"]) == 1


def test_post_once_failure_increments_counter(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(
        "slancha_local.mesh.heartbeat.httpx.post", fake_post
    )
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x", friendly_name="laptop",
        catalog_fn=lambda: [],
    )
    assert loop.post_once() is False
    assert loop.consecutive_failures == 1
    assert loop.heartbeats_sent == 0
    # Subsequent failures stack
    loop.post_once()
    assert loop.consecutive_failures == 2


def test_post_once_5xx_treated_as_failure(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        class R:
            status_code = 503
            def json(self_inner):
                return {}
        return R()

    monkeypatch.setattr(
        "slancha_local.mesh.heartbeat.httpx.post", fake_post
    )
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x", friendly_name="laptop",
        catalog_fn=lambda: [],
    )
    assert loop.post_once() is False
    assert loop.heartbeats_sent == 0


def test_bearer_token_sent_when_set(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["headers"] = headers
        class R:
            status_code = 200
            def json(self_inner):
                return {}
        return R()

    monkeypatch.setattr(
        "slancha_local.mesh.heartbeat.httpx.post", fake_post
    )
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x", friendly_name="laptop",
        catalog_fn=lambda: [],
        token="tok-xyz",
    )
    loop.post_once()
    assert captured["headers"] == {"Authorization": "Bearer tok-xyz"}


# ---------------------------------------------------------------------------
# Loop start/stop lifecycle
# ---------------------------------------------------------------------------


def test_start_then_stop_clean(monkeypatch):
    posts = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        posts["n"] += 1
        class R:
            status_code = 200
            def json(self_inner):
                return {}
        return R()

    monkeypatch.setattr(
        "slancha_local.mesh.heartbeat.httpx.post", fake_post
    )
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x", friendly_name="laptop",
        catalog_fn=lambda: [],
        interval_s=0.05,  # 50ms for fast test
    )
    loop.start()
    time.sleep(0.2)  # should fire ~4 times
    loop.stop()
    assert posts["n"] >= 2  # at least 2 posts in 200ms with 50ms interval


def test_start_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        "slancha_local.mesh.heartbeat.httpx.post",
        lambda *a, **kw: type("R", (), {"status_code": 200, "json": lambda s: {}})(),
    )
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x", friendly_name="laptop",
        catalog_fn=lambda: [],
        interval_s=10.0,
    )
    loop.start()
    first_thread = loop._thread
    loop.start()  # should NOT spawn a second thread
    assert loop._thread is first_thread
    loop.stop()
