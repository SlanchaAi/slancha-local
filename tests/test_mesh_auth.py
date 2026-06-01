"""Tests for MeshAuthMiddleware — paul-mac sig scheme + replay defense.

Sig header: `v1:<kid>:<timestamp_ms>:<nonce>:<hex_mac>`
Identity: X-Slancha-User-Id, X-Slancha-Route-Target, X-Slancha-Mesh-Origin-Id
Payload: f"{user_id}|{ts_ms}|{nonce}|{route_target}|{mesh_origin_id}"
"""

from __future__ import annotations

import hmac
import secrets
import time
from hashlib import sha256

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
KID = "v1"


def _ts_ms(offset_s: int = 0) -> int:
    return int((time.time() + offset_s) * 1000)


def _sign(
    *,
    user_id: str,
    ts_ms: int,
    nonce: str,
    route_target: str,
    mesh_origin_id: str,
    key: bytes = KEY,
) -> str:
    payload = _canonical_payload(
        user_id=user_id,
        timestamp_ms=ts_ms,
        nonce=nonce,
        route_target=route_target,
        mesh_origin_id=mesh_origin_id,
    )
    mac = hmac.new(key, payload, sha256).hexdigest()
    return f"v1:{KID}:{ts_ms}:{nonce}:{mac}"


def _build_app(*, enforce: bool, key_present: bool = True, monkeypatch=None) -> TestClient:
    if monkeypatch and key_present:
        monkeypatch.setenv(f"SLANCHA_MESH_HMAC_KEY_{KID.upper()}", KEY_HEX)
    elif monkeypatch:
        monkeypatch.delenv(f"SLANCHA_MESH_HMAC_KEY_{KID.upper()}", raising=False)
    app = FastAPI()
    app.add_middleware(MeshAuthMiddleware, enforce=enforce)

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
    ts_ms = override.get("ts_ms", _ts_ms())
    nonce = override.get("nonce", secrets.token_hex(16))
    route_target = override.get("route_target", "mesh")
    mesh_origin_id = override.get("mesh_origin_id", "paul-mesh-spark")
    sig = override.get("sig") or _sign(
        user_id=user_id,
        ts_ms=ts_ms,
        nonce=nonce,
        route_target=route_target,
        mesh_origin_id=mesh_origin_id,
    )
    return {
        "X-Slancha-User-Id": user_id,
        "X-Slancha-Route-Target": route_target,
        "X-Slancha-Mesh-Origin-Id": mesh_origin_id,
        "X-Slancha-Forward-Sig": sig,
    }


def test_dev_mode_no_key_pass_through(monkeypatch):
    """When key absent, middleware is no-op even on protected routes."""
    client = _build_app(enforce=True, key_present=False, monkeypatch=monkeypatch)
    r = client.post("/v1/chat/completions", json={})
    assert r.status_code == 200


def test_unprotected_routes_skip_auth(monkeypatch):
    client = _build_app(enforce=True, monkeypatch=monkeypatch)
    for path in ("/health", "/v1/models"):
        r = client.get(path)
        assert r.status_code == 200, path


def test_enforce_missing_headers_401(monkeypatch):
    client = _build_app(enforce=True, monkeypatch=monkeypatch)
    r = client.post("/v1/chat/completions", json={})
    assert r.status_code == 401
    body = r.json()
    assert body["error"] == "mesh_auth_missing"


def test_enforce_valid_hmac_200(monkeypatch):
    client = _build_app(enforce=True, monkeypatch=monkeypatch)
    r = client.post("/v1/chat/completions", json={}, headers=_good_headers())
    assert r.status_code == 200


def test_enforce_bad_hmac_401(monkeypatch):
    client = _build_app(enforce=True, monkeypatch=monkeypatch)
    headers = _good_headers()
    # Replace the hex_mac portion of the sig
    parts = headers["X-Slancha-Forward-Sig"].split(":")
    parts[-1] = "0" * 64
    headers["X-Slancha-Forward-Sig"] = ":".join(parts)
    r = client.post("/v1/chat/completions", json={}, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"] == "mesh_auth_hmac_mismatch"


def test_enforce_unknown_version_401(monkeypatch):
    client = _build_app(enforce=True, monkeypatch=monkeypatch)
    headers = _good_headers()
    parts = headers["X-Slancha-Forward-Sig"].split(":")
    parts[0] = "v9"
    headers["X-Slancha-Forward-Sig"] = ":".join(parts)
    r = client.post("/v1/chat/completions", json={}, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"] == "mesh_auth_unknown_version"


def test_enforce_unknown_kid_401(monkeypatch):
    client = _build_app(enforce=True, monkeypatch=monkeypatch)
    headers = _good_headers()
    parts = headers["X-Slancha-Forward-Sig"].split(":")
    parts[1] = "v99"
    headers["X-Slancha-Forward-Sig"] = ":".join(parts)
    r = client.post("/v1/chat/completions", json={}, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"] == "mesh_auth_unknown_kid"


def test_enforce_bad_sig_shape_401(monkeypatch):
    """Forward-sig with wrong number of colon-delimited fields rejected."""
    client = _build_app(enforce=True, monkeypatch=monkeypatch)
    headers = _good_headers(sig="not:enough:fields")
    r = client.post("/v1/chat/completions", json={}, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"] == "mesh_auth_bad_sig_shape"


def test_enforce_skewed_timestamp_401(monkeypatch):
    client = _build_app(enforce=True, monkeypatch=monkeypatch)
    r = client.post(
        "/v1/chat/completions",
        json={},
        headers=_good_headers(ts_ms=_ts_ms(offset_s=-3600)),
    )
    assert r.status_code == 401
    assert r.json()["error"] == "mesh_auth_skew"


def test_enforce_malformed_timestamp_401(monkeypatch):
    client = _build_app(enforce=True, monkeypatch=monkeypatch)
    headers = _good_headers()
    parts = headers["X-Slancha-Forward-Sig"].split(":")
    parts[2] = "not-a-timestamp"
    headers["X-Slancha-Forward-Sig"] = ":".join(parts)
    r = client.post("/v1/chat/completions", json={}, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"] == "mesh_auth_bad_timestamp"


def test_enforce_nonce_replay_401(monkeypatch):
    client = _build_app(enforce=True, monkeypatch=monkeypatch)
    headers = _good_headers(nonce="deadbeef" * 4)
    r1 = client.post("/v1/chat/completions", json={}, headers=headers)
    assert r1.status_code == 200
    r2 = client.post("/v1/chat/completions", json={}, headers=headers)
    assert r2.status_code == 401
    assert r2.json()["error"] == "mesh_auth_nonce_replay"


def test_dev_mode_logs_but_passes_on_bad_sig(monkeypatch, caplog):
    client = _build_app(enforce=False, monkeypatch=monkeypatch)
    headers = _good_headers()
    parts = headers["X-Slancha-Forward-Sig"].split(":")
    parts[-1] = "0" * 64
    headers["X-Slancha-Forward-Sig"] = ":".join(parts)
    with caplog.at_level("WARNING"):
        r = client.post("/v1/chat/completions", json={}, headers=headers)
    assert r.status_code == 200
    assert any("HMAC mismatch" in rec.message for rec in caplog.records)


def test_nonce_cache_basic():
    cache = _NonceCache(max_size=10, ttl_s=600)
    assert not cache.seen_before("a")
    assert cache.seen_before("a")
    assert not cache.seen_before("b")


def test_nonce_cache_size_eviction():
    cache = _NonceCache(max_size=2, ttl_s=600)
    cache.seen_before("a")
    cache.seen_before("b")
    cache.seen_before("c")
    assert not cache.seen_before("a")


def test_canonical_payload_no_body_hash():
    """B1 invariant: payload contains identity claims only, no body."""
    payload = _canonical_payload(
        user_id="u1",
        timestamp_ms=1234567890,
        nonce="n1",
        route_target="mesh",
        mesh_origin_id="o1",
    )
    assert payload == b"u1|1234567890|n1|mesh|o1"
