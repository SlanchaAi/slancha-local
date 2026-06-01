"""Mesh heartbeat client tests — opt-in mesh registration.

Validates the heartbeat wire format matches slancha-mesh spec §5
verbatim, and that the loop is genuinely opt-in (never starts threads
unless registry_url is set).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import httpx

from slancha_local.mesh.heartbeat import (
    LoadedSpecialist,
    MeshHeartbeatLoop,
    _stable_node_id,
    build_heartbeat_payload,
    probe_arch,
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
# Arch probe — cross-repo schema contract
# ---------------------------------------------------------------------------


def test_probe_arch_darwin_arm_maps_to_apple_silicon(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    assert probe_arch() == "apple-silicon"


def test_probe_arch_linux_arm_maps_to_aarch64(monkeypatch):
    """Spark GB10 case — Linux aarch64."""
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    assert probe_arch() == "aarch64"


def test_probe_arch_intel_linux_maps_to_x86_64(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    assert probe_arch() == "x86_64"


def test_probe_arch_unknown_falls_back_to_x86_64(monkeypatch):
    """An unknown machine label must still satisfy the slancha-mesh
    Literal — return x86_64 rather than 'unknown' which would 422."""
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "weird-future-isa")
    assert probe_arch() == "x86_64"


def test_probe_arch_amd64_maps_to_x86_64(monkeypatch):
    """Windows / FreeBSD often report `AMD64` (uppercase too)."""
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.machine", lambda: "AMD64")
    assert probe_arch() == "x86_64"


def test_build_payload_default_arch_uses_probe(monkeypatch):
    """No `arch` kwarg → probe_arch() runs + caller gets a valid Literal."""
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
    )
    assert p["heartbeat"]["hardware"]["arch"] == "apple-silicon"


def test_build_payload_explicit_arch_overrides_probe(monkeypatch):
    """Caller's explicit arch wins (production probe in slancha-local
    will pass arch from its own platform probe path)."""
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
        arch="x86_64",  # explicit override
    )
    assert p["heartbeat"]["hardware"]["arch"] == "x86_64"


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
    for k in (
        "node_id",
        "friendly_name",
        "chip",
        "arch",
        "ram_total_gb",
        "ram_available_gb",
        "unified_memory",
        "memory_bandwidth_gbs",
        "available_backends",
        "disk_free_gb",
    ):
        assert k in hb["hardware"], f"hardware.{k} missing"


def test_payload_loaded_models_shape():
    spec = LoadedSpecialist(
        specialist_id="qwen3-8b",
        model_id="Qwen/Qwen3-8B",
        domain="general",
        estimated_tps=42.5,
    )
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
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
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
    )
    ts = p["heartbeat"]["ts"]
    # Parseable as ISO 8601
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None  # must be tz-aware


def test_payload_health_defaults_healthy():
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
    )
    assert p["heartbeat"]["health"] == "healthy"


def test_payload_health_can_be_overridden():
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
        health="degraded",
    )
    assert p["heartbeat"]["health"] == "degraded"


def test_payload_json_roundtrips_cleanly():
    """Payload must serialize via json.dumps without TypeError (no
    datetime objects leaking through)."""
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[
            LoadedSpecialist(
                specialist_id="x",
                model_id="y",
                domain="general",
                estimated_tps=10.0,
            )
        ],
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

            def json(self):
                return {"ack": True, "next_due_seconds": 5}

        return R()

    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", fake_post)
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://127.0.0.1:8000",
        friendly_name="laptop",
        catalog_fn=lambda: [
            LoadedSpecialist(
                specialist_id="qwen3-8b",
                model_id="Qwen/Qwen3-8B",
                domain="general",
            )
        ],
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

    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", fake_post)
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x",
        friendly_name="laptop",
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

            def json(self):
                return {}

        return R()

    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", fake_post)
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x",
        friendly_name="laptop",
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

            def json(self):
                return {}

        return R()

    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", fake_post)
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x",
        friendly_name="laptop",
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

            def json(self):
                return {}

        return R()

    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", fake_post)
    loop = MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x",
        friendly_name="laptop",
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
        node_url="http://x",
        friendly_name="laptop",
        catalog_fn=lambda: [],
        interval_s=10.0,
    )
    loop.start()
    first_thread = loop._thread
    loop.start()  # should NOT spawn a second thread
    assert loop._thread is first_thread
    loop.stop()


# ---------------------------------------------------------------------------
# Observability — silent heartbeat death (Windows dogfood finding #3, 2026-05-26)
# ---------------------------------------------------------------------------
# A tagged node fell off the mesh with ZERO local signal: failures were logged
# at INFO (suppressed under default WARNING) and `serve` kept reporting healthy.
# Fix: WARNING on the healthy→failing transition + recovery, INFO for the
# steady-state failure stream, and a `status()`/`last_success` surface.


def _ok_post(*a, **kw):
    return type("R", (), {"status_code": 200, "json": lambda s: {}})()


def _fail_post(*a, **kw):
    raise httpx.ConnectError("unreachable")


def _make_loop():
    return MeshHeartbeatLoop(
        registry_url="http://reg.local",
        node_url="http://x",
        friendly_name="laptop",
        catalog_fn=lambda: [],
    )


def test_first_failure_after_success_logs_warning(monkeypatch, caplog):
    """healthy→failing transition must surface at WARNING, not INFO."""
    loop = _make_loop()
    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", _ok_post)
    loop.post_once()  # one success first
    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", _fail_post)
    with caplog.at_level(logging.WARNING, logger="slancha_local.mesh.heartbeat"):
        loop.post_once()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("fallen off the mesh" in r.getMessage() for r in warnings)


def test_steady_state_failures_stay_below_warning(monkeypatch, caplog):
    """A long outage must NOT flood WARNING — only the transition is loud."""
    loop = _make_loop()
    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", _fail_post)
    with caplog.at_level(logging.INFO, logger="slancha_local.mesh.heartbeat"):
        loop.post_once()  # transition (WARNING) — failure #1
        caplog.clear()  # isolate the steady-state stream that follows
        loop.post_once()  # failure #2
        loop.post_once()  # failure #3
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    assert loop.consecutive_failures == 3


def test_recovery_logs_warning(monkeypatch, caplog):
    """Coming back onto the mesh leaves a trail at WARNING."""
    loop = _make_loop()
    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", _fail_post)
    loop.post_once()  # now failing
    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", _ok_post)
    with caplog.at_level(logging.WARNING, logger="slancha_local.mesh.heartbeat"):
        loop.post_once()  # recovery
    assert any("RECOVERED" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert loop.consecutive_failures == 0


def test_last_success_tracks_only_successful_posts(monkeypatch):
    loop = _make_loop()
    assert loop.last_success is None
    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", _fail_post)
    loop.post_once()
    assert loop.last_success is None  # failure must not stamp it
    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", _ok_post)
    loop.post_once()
    assert loop.last_success is not None
    datetime.fromisoformat(loop.last_success)  # parseable ISO-8601


def test_status_reflects_registration_health(monkeypatch):
    loop = _make_loop()
    s0 = loop.status()
    assert s0["enabled"] is True
    assert s0["registered"] is False  # nothing sent yet
    assert s0["heartbeats_sent"] == 0 and s0["last_success"] is None

    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", _ok_post)
    loop.post_once()
    s1 = loop.status()
    assert s1["registered"] is True and s1["heartbeats_sent"] == 1

    monkeypatch.setattr("slancha_local.mesh.heartbeat.httpx.post", _fail_post)
    loop.post_once()
    s2 = loop.status()
    # Still has a last_success, but no longer "registered" (in a failure streak)
    assert s2["registered"] is False
    assert s2["consecutive_failures"] == 1
    assert s2["last_success"] == s1["last_success"]


def test_status_when_disabled():
    loop = MeshHeartbeatLoop(
        registry_url=None,
        node_url="http://x",
        friendly_name="laptop",
        catalog_fn=lambda: [],
    )
    s = loop.status()
    assert s["enabled"] is False and s["registered"] is False
