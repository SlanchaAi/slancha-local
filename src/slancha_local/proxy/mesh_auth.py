"""MeshAuthMiddleware — verify X-Slancha-Forward-Sig HMAC + replay defense.

Per Slancha-Mesh Protocol v0.1 §4 + §4.1 auth direction matrix.
Signature scheme per paul-mac issuance 2026-05-22:

  Header: X-Slancha-Forward-Sig
    Shape: `v1:<kid>:<timestamp_ms>:<nonce>:<hex_mac>` (5 colon-delimited fields)
    Example: `v1:v1:1779489000000:a3b9...:fa5d...`

  Identity headers (separately):
    X-Slancha-User-Id          ULID/UUID of the slancha user
    X-Slancha-Route-Target     "mesh" | "direct" | "fallback"
    X-Slancha-Mesh-Origin-Id   registered mesh origin id (allowlist entry)

  HMAC payload (canonical):
    f"{user_id}|{timestamp_ms}|{nonce}|{route_target}|{mesh_origin_id}"
    (NO body_hash — B1 fix; L@E never reads body, request size unbounded)

  Replay defense: ±300s timestamp window + nonce LRU dedup 600s.
  KID-aware key lookup: env SLANCHA_MESH_HMAC_KEY_<KID> (hex-encoded, 32 bytes).

Enable enforcement: SLANCHA_MESH_AUTH_ENFORCE=true. When OFF (dev default),
missing/invalid sigs log WARNING and pass through. Flip true once L@E signs
every request OR once you've smoke-tested manual signing locally.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from collections import OrderedDict
from hashlib import sha256

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

_PROTECTED_PREFIXES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/images/generations",
    "/v1/embeddings",
    "/v1/audio",
)

_SIG_VERSION = "v1"
_TIMESTAMP_TOLERANCE_MS = 300_000  # ±300s
_NONCE_TTL_S = 600
_NONCE_MAX = 50_000


class _NonceCache:
    """Tiny in-process LRU + TTL cache. Single-worker uvicorn → no lock needed."""

    def __init__(self, *, max_size: int = _NONCE_MAX, ttl_s: int = _NONCE_TTL_S) -> None:
        self._max = max_size
        self._ttl = ttl_s
        self._seen: OrderedDict[str, float] = OrderedDict()

    def seen_before(self, nonce: str) -> bool:
        now = time.monotonic()
        while self._seen:
            oldest_nonce, oldest_ts = next(iter(self._seen.items()))
            if now - oldest_ts > self._ttl:
                self._seen.popitem(last=False)
            else:
                break
        if nonce in self._seen:
            return True
        self._seen[nonce] = now
        while len(self._seen) > self._max:
            self._seen.popitem(last=False)
        return False


def _canonical_payload(
    *,
    user_id: str,
    timestamp_ms: int,
    nonce: str,
    route_target: str,
    mesh_origin_id: str,
) -> bytes:
    """Identity-only payload — NO body_hash (B1 fix)."""
    return f"{user_id}|{timestamp_ms}|{nonce}|{route_target}|{mesh_origin_id}".encode("utf-8")


def _key_for_kid(kid: str) -> bytes | None:
    """Look up HMAC key bytes for a given KID via env SLANCHA_MESH_HMAC_KEY_<KID>.

    KID is case-folded to UPPER for env var name. Returns None if absent.
    """
    env_name = f"SLANCHA_MESH_HMAC_KEY_{kid.upper()}"
    hex_str = os.environ.get(env_name, "").strip()
    if not hex_str:
        return None
    try:
        return bytes.fromhex(hex_str)
    except ValueError:
        logger.error("mesh auth: %s not valid hex; rejecting all sigs for KID=%s", env_name, kid)
        return None


class MeshAuthMiddleware(BaseHTTPMiddleware):
    """HMAC-verified mesh auth for protected routes.

    Construction reads env at init. SLANCHA_MESH_HMAC_KEY_V1 (and any
    additional KIDs) provides keys; SLANCHA_MESH_AUTH_ENFORCE=true flips
    strict mode. Absent keys → middleware is a no-op (dev mode).
    """

    def __init__(
        self,
        app,
        *,
        enforce: bool | None = None,
        nonce_cache: _NonceCache | None = None,
        active_kids: tuple[str, ...] = ("v1",),
    ) -> None:
        super().__init__(app)
        self._enforce = enforce if enforce is not None else (
            os.environ.get("SLANCHA_MESH_AUTH_ENFORCE", "false").lower() == "true"
        )
        self._nonce = nonce_cache or _NonceCache()
        self._active_kids = active_kids
        self._keys: dict[str, bytes] = {}
        for kid in active_kids:
            k = _key_for_kid(kid)
            if k is not None:
                self._keys[kid] = k
        if not self._keys:
            logger.warning(
                "MeshAuthMiddleware: no HMAC keys present (checked KIDs=%s via "
                "SLANCHA_MESH_HMAC_KEY_<KID> env). Middleware is NO-OP. "
                "Production must set key(s) + SLANCHA_MESH_AUTH_ENFORCE=true.",
                active_kids,
            )

    def _reject(self, code: str, http_status: int = 401, extra: dict | None = None):
        body: dict = {"error": code}
        if extra:
            body.update(extra)
        return JSONResponse(status_code=http_status, content=body)

    async def dispatch(self, request: Request, call_next):
        if not any(request.url.path.startswith(p) for p in _PROTECTED_PREFIXES):
            return await call_next(request)
        if not self._keys:
            return await call_next(request)  # dev pass-through

        sig_hdr = request.headers.get("x-slancha-forward-sig", "")
        user_id = request.headers.get("x-slancha-user-id", "")
        route_target = request.headers.get("x-slancha-route-target", "")
        mesh_origin_id = request.headers.get("x-slancha-mesh-origin-id", "")

        missing = [
            n for n, v in (
                ("X-Slancha-User-Id", user_id),
                ("X-Slancha-Route-Target", route_target),
                ("X-Slancha-Mesh-Origin-Id", mesh_origin_id),
                ("X-Slancha-Forward-Sig", sig_hdr),
            )
            if not v
        ]
        if missing:
            logger.warning("mesh auth: missing headers %s on %s", missing, request.url.path)
            if self._enforce:
                return self._reject("mesh_auth_missing", extra={"missing": missing})
            return await call_next(request)

        # Parse compound sig: v1:<kid>:<timestamp_ms>:<nonce>:<hex_mac>
        parts = sig_hdr.split(":")
        if len(parts) != 5:
            logger.warning("mesh auth: forward-sig has %d parts (expect 5)", len(parts))
            if self._enforce:
                return self._reject("mesh_auth_bad_sig_shape")
            return await call_next(request)
        version, kid, ts_str, nonce, hex_mac = parts

        if version != _SIG_VERSION:
            logger.warning("mesh auth: unknown sig version %r", version)
            if self._enforce:
                return self._reject("mesh_auth_unknown_version")
            return await call_next(request)

        key = self._keys.get(kid)
        if key is None:
            logger.warning("mesh auth: unknown KID %r (active=%s)", kid, list(self._keys.keys()))
            if self._enforce:
                return self._reject("mesh_auth_unknown_kid")
            return await call_next(request)

        try:
            ts_ms = int(ts_str)
        except ValueError:
            logger.warning("mesh auth: malformed timestamp_ms %r", ts_str)
            if self._enforce:
                return self._reject("mesh_auth_bad_timestamp")
            return await call_next(request)

        now_ms = int(time.time() * 1000)
        skew_ms = abs(now_ms - ts_ms)
        if skew_ms > _TIMESTAMP_TOLERANCE_MS:
            logger.warning("mesh auth: timestamp skew %dms > tolerance %dms", skew_ms, _TIMESTAMP_TOLERANCE_MS)
            if self._enforce:
                return self._reject("mesh_auth_skew", extra={"skew_ms": skew_ms})

        if self._nonce.seen_before(nonce):
            logger.warning("mesh auth: nonce replay %s", nonce[:12])
            if self._enforce:
                return self._reject("mesh_auth_nonce_replay")

        payload = _canonical_payload(
            user_id=user_id,
            timestamp_ms=ts_ms,
            nonce=nonce,
            route_target=route_target,
            mesh_origin_id=mesh_origin_id,
        )
        expected = hmac.new(key, payload, sha256).hexdigest()
        if not hmac.compare_digest(expected, hex_mac):
            logger.warning(
                "mesh auth: HMAC mismatch user=%s origin=%s kid=%s",
                user_id, mesh_origin_id, kid,
            )
            if self._enforce:
                return self._reject("mesh_auth_hmac_mismatch")

        request.state.mesh_user_id = user_id
        request.state.mesh_origin_id = mesh_origin_id
        request.state.mesh_route_target = route_target
        return await call_next(request)


__all__ = ["MeshAuthMiddleware"]
