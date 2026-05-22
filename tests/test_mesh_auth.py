"""Tests for MeshAuthMiddleware — HMAC verify + replay defense."""

from __future__ import annotations

import hmac
import secrets
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from slancha_local.proxy.mesh_auth import (
    MeshAuthMiddleware,
    _canonical_payload,
    _NonceCache,
)


KEY_HEX = secrets.token_hex(32)
KEY = bytes.fromhex(KEY_HEX)


def _now_iso(offset_s: int = 0) -> str:
    return (datetime.now(UTC) + timedelta(seconds=offset_s)).isoformat().replace("+00:00", "Z")


def _sign(*, user_id: str, ts: str, nonce: str, route_target: str, origin_id: str, key: bytes = KEY) -> str:
    payload = _canonical_payload(
        user_id=user_id, timestamp=ts, nonce=nonce, route_target=route_target, origin_id=origin_id
    )
    return hmac.new(key, payload, sha256).hexdigest()


def _build_app(*, enforce: bool, key_hex: str = KEY_HEX) -> TestClient:
    app = FastAPI()
    app.add_middleware(MeshAuthMiddleware, hmac_key_hex=key_hex, enforce=enforce)

    @app.post("/v1/chat/completions")
    def chat():
        return JSONResponse({"ok": True})

    @app.get("/v1/models")
    def models():
        return JSONResponse({"data": []})

    @app.get("/health")
    def health():
        return JSONResponse({"ok": True})

    return TestClient(app)


def _good_headers(**override) -> dict[str, str]:
    user_id = override.get("user_id", "user-paul")
    ts = override.get("ts", _now_iso())
    nonce = override.get("nonce", secrets.token_hex(16))
    route_target = override.get("route_target", "mesh")
    origin_id = override.get("origin_id", "paul-mesh-spark")
    sig = override.get("sig") or _sign(
        user_id=user_id, ts=ts, nonce=nonce, route_target=route_target, origin_id=origin_id
    )
    return {
        "X-Slancha-User-Id": user_id,
        "X-Slancha-Timestamp": ts,
        "X-Slancha-Nonce": nonce,
        "X-Slancha-Route-Target": route_target,
        "X-Slancha-Origin-Id": origin_id,
        "X-Slancha-Forward-Sig": sig,
    }


def test_dev_mode_no_key_pass_through():
    """When key absent, middleware is no-op even on protected routes."""
    client = _build_app(enforce=True, key_hex="")
    r = client.post("/v1/chat/completions", json={})
    assert r.status_code == 200


def test_unprotected_routes_skip_auth():
    """/health + /v1/models bypass mesh auth even when enforce=True."""
    client = _build_app(enforce=True)
    for path in ("/health", "/v1/models"):
        r = client.get(path)
        assert r.status_code == 200, path


def test_enforce_missing_headers_401():
    client = _build_app(enforce=True)
    r = client.post("/v1/chat/completions", json={})
    assert r.status_code == 401
    body = r.json()
    assert body["error"] == "mesh_auth_missing"
    assert "X-Slancha-User-Id" in body["missing"]


def test_enforce_valid_hmac_200():
    client = _build_app(enforce=True)
    r = client.post("/v1/chat/completions", json={}, headers=_good_headers())
    assert r.status_code == 200


def test_enforce_bad_hmac_401():
    client = _build_app(enforce=True)
    r = client.post(
        "/v1/chat/completions", json={},
        headers=_good_headers(sig="0" * 64),
    )
    assert r.status_code == 401
    assert r.json()["error"] == "mesh_auth_hmac_mismatch"


def test_enforce_skewed_timestamp_401():
    """Timestamps outside ±300s window rejected."""
    client = _build_app(enforce=True)
    skewed_ts = _now_iso(offset_s=-3600)  # 1hr in past
    r = client.post(
        "/v1/chat/completions", json={},
        headers=_good_headers(ts=skewed_ts),
    )
    assert r.status_code == 401
    assert r.json()["error"] == "mesh_auth_skew"


def test_enforce_malformed_timestamp_401():
    client = _build_app(enforce=True)
    r = client.post(
        "/v1/chat/completions", json={},
        headers=_good_headers(ts="not-a-timestamp"),
    )
    assert r.status_code == 401
    assert r.json()["error"] == "mesh_auth_bad_timestamp"


def test_enforce_nonce_replay_401():
    """Second request with same nonce rejected."""
    client = _build_app(enforce=True)
    headers = _good_headers(nonce="deadbeef" * 4)
    r1 = client.post("/v1/chat/completions", json={}, headers=headers)
    assert r1.status_code == 200
    r2 = client.post("/v1/chat/completions", json={}, headers=headers)
    assert r2.status_code == 401
    assert r2.json()["error"] == "mesh_auth_nonce_replay"


def test_dev_mode_logs_but_passes_on_bad_sig(caplog):
    """When enforce=False, bad sig logs warning but request proceeds."""
    client = _build_app(enforce=False)
    with caplog.at_level("WARNING"):
        r = client.post(
            "/v1/chat/completions", json={},
            headers=_good_headers(sig="0" * 64),
        )
    assert r.status_code == 200
    assert any("HMAC mismatch" in rec.message for rec in caplog.records)


def test_nonce_cache_basic():
    cache = _NonceCache(max_size=10, ttl_s=600)
    assert not cache.seen_before("a")
    assert cache.seen_before("a")
    assert not cache.seen_before("b")


def test_nonce_cache_size_eviction():
    """LRU eviction when max_size exceeded."""
    cache = _NonceCache(max_size=2, ttl_s=600)
    cache.seen_before("a")
    cache.seen_before("b")
    cache.seen_before("c")  # evicts "a"
    assert not cache.seen_before("a")  # a is gone, treated as new


def test_canonical_payload_no_body_hash():
    """B1 invariant: payload contains identity claims only, no body."""
    payload = _canonical_payload(
        user_id="u1", timestamp="t1", nonce="n1", route_target="mesh", origin_id="o1",
    )
    s = payload.decode()
    assert s == "u1|t1|n1|mesh|o1"
    # No body_hash field anywhere — L@E body inspection cap (B1) honored.
