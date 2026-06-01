"""Accept slancha-api's X-Slancha-Pref routing rules on the local proxy.

slancha-api lets agents steer routing via a price/accuracy/latency weight
simplex plus flat levers, expressed as an `X-Slancha-Pref` header (RFC 8941
Structured Field Dictionary) or a JSON `pref` body. This module accepts the
SAME inputs at slancha-local and maps them onto the proxy's existing
`Preferences` (which the classifier/selector already honor).

Scope vs slancha-api (`app/mesh/pref.py`):
  - We re-implement the field names + weights validation (no cross-repo
    import — slancha-local has no slancha-api dependency).
  - The header parser is a deliberate RFC 8941 dictionary SUBSET: flat
    `key=value` scalars + booleans (`?1`/`?0`). Structured values (the
    `weights` dict, require/exclude lists) come via the JSON body, where
    Pydantic gives full fidelity. This avoids pulling in `http_sfv` for a
    lean proxy that otherwise makes zero non-loopback calls (ADR-002).
  - Gateway-only concerns (admin ceiling, provider-shape translation,
    service-tier presets) are NOT ported — they belong at the gateway, not
    a single-node local proxy.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

from slancha_local.classifier_client.models import Preferences

logger = logging.getLogger(__name__)

# Header (hyphen) → body (underscore) field names. Subset of slancha-api's
# map covering the levers slancha-local maps onto Preferences.
_HEADER_TO_BODY_FIELD_MAP: dict[str, str] = {
    "max-cost-per-1m-usd": "max_cost_per_1m_usd",
    "max-cost-cents": "max_cost_per_1m_usd",  # legacy alias → new field
    "max-latency-ms-p95": "max_latency_ms_p95",
    "quality-weight": "quality_weight",
    "allow-fallbacks": "allow_fallbacks",
}

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_ALLOWED_AXES = {"price", "accuracy", "latency"}


def _coerce_scalar(raw: str) -> Any:
    """Coerce one RFC 8941 dictionary value (subset: bool/int/float/string)."""
    if raw in ("?1", "?0"):
        return raw == "?1"
    if _INT_RE.match(raw):
        return int(raw)
    if _FLOAT_RE.match(raw):
        return float(raw)
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    return raw


def parse_pref_header(header: str | None) -> dict[str, Any]:
    """Parse an `X-Slancha-Pref` header into body-style (underscored) keys.

    Subset of RFC 8941: comma-separated `key=value` members with bare scalar
    values. Unknown keys and members without `=` are dropped. Never raises —
    malformed input yields `{}` so a bad header degrades to "no preference".
    """
    if not header:
        return {}
    out: dict[str, Any] = {}
    for member in header.split(","):
        member = member.strip()
        if "=" not in member:
            continue
        key, _, val = member.partition("=")
        body_key = _HEADER_TO_BODY_FIELD_MAP.get(key.strip().lower())
        if body_key is None:
            continue
        out[body_key] = _coerce_scalar(val.strip())
    return out


class SlanchaPrefInput(BaseModel):
    """Routing-relevant subset of slancha-api's SlanchaPref body.

    `extra="ignore"` so the full slancha-api shape (zdr, region, require,
    service_tier, …) is ACCEPTED, not rejected — slancha-local just maps the
    fields it can act on.
    """

    weights: dict[str, float] | None = None
    quality_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    max_latency_ms_p95: int | None = Field(default=None, ge=0)
    max_cost_per_1m_usd: float | None = Field(default=None, ge=0)
    allow_fallbacks: bool | None = None

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _validate_weights(self) -> SlanchaPrefInput:
        if self.weights is not None:
            bad = set(self.weights) - _ALLOWED_AXES
            if bad:
                raise ValueError(
                    f"weights has unknown axis/axes {sorted(bad)}; allowed: {sorted(_ALLOWED_AXES)}"
                )
            for k, v in self.weights.items():
                # `not (v >= 0)` would let NaN through (nan >= 0 is False);
                # require finite ≥ 0 so the normalizer stays deterministic.
                if not math.isfinite(v) or v < 0:
                    raise ValueError(f"weights[{k!r}]={v} must be a finite value ≥ 0")
        return self


def resolve_preferences(
    *,
    header: str | None = None,
    body_pref: dict[str, Any] | None = None,
) -> Preferences:
    """Merge header + body pref and map onto slancha-local `Preferences`.

    Precedence: body overrides header (matching slancha-api). Returns
    `Preferences()` defaults for anything unset. Raises pydantic
    ValidationError on invalid input (e.g. a bad weights axis) so the caller
    can surface a 422 — same contract as the gateway.
    """
    merged: dict[str, Any] = parse_pref_header(header)
    if body_pref:
        merged.update(body_pref)
    if not merged:
        return Preferences()

    sp = SlanchaPrefInput.model_validate(merged)
    updates: dict[str, Any] = {}

    # 3-axis simplex supersedes the standalone quality_weight lever (matches
    # slancha-api). All-zero / absent → leave default weights untouched.
    if sp.weights and sum(sp.weights.values()) > 0:
        total = sum(sp.weights.values())
        updates["cost_weight"] = sp.weights.get("price", 0.0) / total
        updates["quality_weight"] = sp.weights.get("accuracy", 0.0) / total
        updates["latency_weight"] = sp.weights.get("latency", 0.0) / total
        updates["privacy_weight"] = 0.0
    elif sp.quality_weight is not None:
        updates["quality_weight"] = sp.quality_weight

    if sp.max_latency_ms_p95 is not None:
        updates["max_latency_ms"] = sp.max_latency_ms_p95
    if sp.max_cost_per_1m_usd is not None:
        # Preferences.max_cost_per_1k is per-1k tokens; the wire field is
        # per-1M USD. 1M = 1000 × 1k.
        updates["max_cost_per_1k"] = sp.max_cost_per_1m_usd / 1000.0
    if sp.allow_fallbacks is not None:
        updates["escalation_allowed"] = sp.allow_fallbacks

    return Preferences().model_copy(update=updates)


__all__ = [
    "SlanchaPrefInput",
    "parse_pref_header",
    "resolve_preferences",
]
