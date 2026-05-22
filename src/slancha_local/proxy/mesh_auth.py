"""MeshAuthMiddleware — verify X-Slancha-Forward-Sig HMAC + replay defense.

Per Slancha-Mesh Protocol v0.1 §4 + §4.1 auth direction matrix:

  HMAC scope: SaaS → Mesh.
  Payload:    HMAC-SHA256(key, user_id || timestamp || nonce || route_target || origin_id)
              (NO body_hash — B1 fix; L@E never reads body so request size is unbounded)
  Replay:     ±300s timestamp tolerance + nonce LRU dedup 600s window

Headers expected on inbound chat-completion requests (set by L@E in prod;
test-time call sites use a helper to attach them):

    X-Slancha-User-Id        ULID/UUID of the slancha user
    X-Slancha-Timestamp      ISO-8601 UTC (e.g. 2026-05-22T19:55:00Z)
    X-Slancha-Nonce          ≥16 hex chars, unique per request
    X-Slancha-Route-Target   "mesh" | "direct" | "fallback"
    X-Slancha-Origin-Id      registered mesh origin id (allowlist entry)
    X-Slancha-Forward-Sig    hex SHA-256 HMAC over the canonical payload

Enable enforcement by setting SLANCHA_MESH_AUTH_ENFORCE=true. When OFF
(default during substrate development), missing/invalid signatures are
logged but pass through — useful while wiring up L@E. Flip to true once
L@E origin-request reliably signs every request.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from collections import OrderedDict
from datetime import UTC, datetime
from hashlib import sha256

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Routes that REQUIRE mesh auth. Health + models discovery + /docs stay open
# so probes work and clients can introspect.
_PROTECTED_PREFIXES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/images/generations",
    "/v1/embeddings",
    "/v1/audio",
)

# Replay defense.
_TIMESTAMP_TOLERANCE_S = 300
_NONCE_TTL_S = 600
_NONCE_MAX = 50_000  # bound memory; LRU evict beyond this


class _NonceCache:
    """Tiny in-process LRU + TTL cache. Single-worker uvicorn → no lock needed."""

    def __init__(self, *, max_size: int = _NONCE_MAX, ttl_s: int = _NONCE_TTL_S) -> None:
        self._max = max_size
        self._ttl = ttl_s
        self._seen: OrderedDict[str, float] = OrderedDict()

    def seen_before(self, nonce: str) -> bool:
        now = time.monotonic()
        # Evict expired (cheap because OrderedDict keeps insertion order).
        while self._seen:
            oldest_nonce, oldest_ts = next(iter(self._seen.items()))
            if now - oldest_ts > self._ttl:
                self._seen.popitem(last=False)
            else:
                break
        if nonce in self._seen:
            return True
        self._seen[nonce] = now
        # Bound memory.
        while len(self._seen) > self._max:
            self._seen.popitem(last=False)
        return False


def _canonical_payload(
    *,
    user_id: str,
    timestamp: str,
    nonce: str,
    route_target: str,
    origin_id: str,
) -> bytes:
    """Identity-only payload — NO body_hash so L@E never reads body (B1)."""
    return "|".join((user_id, timestamp, nonce, route_target, origin_id)).encode("utf-8")


def _verify_hmac(
    *,
    key: bytes,
    sig_hex: str,
    payload: bytes,
) -> bool:
    expected = hmac.new(key, payload, sha256).hexdigest()
    return hmac.compare_digest(expected, sig_hex)


def _parse_timestamp(raw: str) -> float | None:
    """Parse ISO-8601 UTC → epoch seconds. Returns None on bad input."""
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


class MeshAuthMiddleware(BaseHTTPMiddleware):
    """HMAC-verified mesh auth for protected routes.

    Init reads SLANCHA_MESH_HMAC_KEY (32+ random bytes hex-encoded) from
    env. If absent, middleware logs once and is a no-op — useful for
    local dev. Production deploys set both SLANCHA_MESH_HMAC_KEY and
    SLANCHA_MESH_AUTH_ENFORCE=true.
    """

    def __init__(
        self,
        app,
        *,
        hmac_key_hex: str | None = None,
        enforce: bool | None = None,
        nonce_cache: _NonceCache | None = None,
    ) -> None:
        super().__init__(app)
        self._key_hex = hmac_key_hex or os.environ.get("SLANCHA_MESH_HMAC_KEY", "")
        self._enforce = enforce if enforce is not None else (
            os.environ.get("SLANCHA_MESH_AUTH_ENFORCE", "false").lower() == "true"
        )
        self._nonce = nonce_cache or _NonceCache()
        if not self._key_hex:
            logger.warning(
                "MeshAuthMiddleware: SLANCHA_MESH_HMAC_KEY unset — middleware "
                "is a NO-OP. Production must set the key and "
                "SLANCHA_MESH_AUTH_ENFORCE=true."
            )

    @property
    def _key(self) -> bytes:
        return bytes.fromhex(self._key_hex) if self._key_hex else b""

    async def dispatch(self, request: Request, call_next):
        # Only gate the protected surface.
        if not any(request.url.path.startswith(p) for p in _PROTECTED_PREFIXES):
            return await call_next(request)
        # Dev-mode pass-through when key absent.
        if not self._key_hex:
            return await call_next(request)

        user_id = request.headers.get("x-slancha-user-id", "")
        timestamp = request.headers.get("x-slancha-timestamp", "")
        nonce = request.headers.get("x-slancha-nonce", "")
        route_target = request.headers.get("x-slancha-route-target", "")
        origin_id = request.headers.get("x-slancha-origin-id", "")
        sig_hex = request.headers.get("x-slancha-forward-sig", "")

        # Missing-field check — strict reject if enforcing, log if not.
        missing = [
            name for name, val in (
                ("X-Slancha-User-Id", user_id),
                ("X-Slancha-Timestamp", timestamp),
                ("X-Slancha-Nonce", nonce),
                ("X-Slancha-Route-Target", route_target),
                ("X-Slancha-Origin-Id", origin_id),
                ("X-Slancha-Forward-Sig", sig_hex),
            )
            if not val
        ]
        if missing:
            logger.warning("mesh auth: missing headers %s on %s", missing, request.url.path)
            if self._enforce:
                return JSONResponse(
                    status_code=401,
                    content={"error": "mesh_auth_missing", "missing": missing},
                )
            return await call_next(request)

        # Timestamp window.
        ts_epoch = _parse_timestamp(timestamp)
        if ts_epoch is None:
            logger.warning("mesh auth: malformed X-Slancha-Timestamp=%r", timestamp)
            if self._enforce:
                return JSONResponse(
                    status_code=401, content={"error": "mesh_auth_bad_timestamp"}
                )
            return await call_next(request)
        skew = abs(time.time() - ts_epoch)
        if skew > _TIMESTAMP_TOLERANCE_S:
            logger.warning(
                "mesh auth: timestamp skew %.0fs > %ds tolerance", skew, _TIMESTAMP_TOLERANCE_S
            )
            if self._enforce:
                return JSONResponse(
                    status_code=401,
                    content={"error": "mesh_auth_skew", "skew_s": int(skew)},
                )

        # Nonce replay.
        if self._nonce.seen_before(nonce):
            logger.warning("mesh auth: nonce replay %s", nonce[:12])
            if self._enforce:
                return JSONResponse(
                    status_code=401, content={"error": "mesh_auth_nonce_replay"}
                )

        # HMAC verify.
        payload = _canonical_payload(
            user_id=user_id,
            timestamp=timestamp,
            nonce=nonce,
            route_target=route_target,
            origin_id=origin_id,
        )
        if not _verify_hmac(key=self._key, sig_hex=sig_hex, payload=payload):
            logger.warning("mesh auth: HMAC mismatch user=%s origin=%s", user_id, origin_id)
            if self._enforce:
                return JSONResponse(
                    status_code=401, content={"error": "mesh_auth_hmac_mismatch"}
                )

        # Stash identity for downstream sidecar telemetry.
        request.state.mesh_user_id = user_id
        request.state.mesh_origin_id = origin_id
        request.state.mesh_route_target = route_target

        return await call_next(request)


__all__ = ["MeshAuthMiddleware"]
