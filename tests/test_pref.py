"""X-Slancha-Pref acceptance on the slancha-local proxy.

slancha-api lets agents express routing rules (the price/accuracy/latency
weight simplex + flat levers) via an `X-Slancha-Pref` header or a `pref`
JSON body. This locks slancha-local accepting the SAME inputs and mapping
them onto its existing `Preferences` (fed to the classifier/selector).

Header parsing is a documented RFC 8941 dictionary SUBSET (flat scalar
levers); the full shape including the `weights` simplex comes via the JSON
body. Body wins over header, matching slancha-api precedence.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slancha_local.classifier_client.models import Preferences
from slancha_local.proxy.pref import (
    SlanchaPrefInput,
    parse_pref_header,
    resolve_preferences,
)

# ---------------------------------------------------------------------------
# parse_pref_header — RFC 8941 dictionary subset
# ---------------------------------------------------------------------------


def test_parse_header_scalars_and_bool():
    out = parse_pref_header("max-latency-ms-p95=3000, quality-weight=0.6, allow-fallbacks=?1")
    assert out["max_latency_ms_p95"] == 3000
    assert out["quality_weight"] == 0.6
    assert out["allow_fallbacks"] is True


def test_parse_header_false_bool_and_legacy_cost_alias():
    out = parse_pref_header("allow-fallbacks=?0, max-cost-cents=5")
    assert out["allow_fallbacks"] is False
    # legacy max-cost-cents maps to the new field name
    assert out["max_cost_per_1m_usd"] == 5


def test_parse_header_drops_unknown_keys():
    out = parse_pref_header("quality-weight=0.5, totally-made-up=9")
    assert "quality_weight" in out
    assert "totally-made-up" not in out
    assert "totally_made_up" not in out


def test_parse_header_empty_and_malformed():
    assert parse_pref_header(None) == {}
    assert parse_pref_header("") == {}
    # a segment with no '=' is skipped, not crashed
    assert parse_pref_header("garbage-no-equals") == {}


# ---------------------------------------------------------------------------
# SlanchaPrefInput — weights validation mirrors slancha-api
# ---------------------------------------------------------------------------


def test_weights_rejects_unknown_axis():
    with pytest.raises(ValidationError):
        SlanchaPrefInput(weights={"speed": 1.0})  # only price/accuracy/latency


def test_weights_rejects_negative_or_nan():
    with pytest.raises(ValidationError):
        SlanchaPrefInput(weights={"price": -1.0})
    with pytest.raises(ValidationError):
        SlanchaPrefInput(weights={"latency": float("nan")})


def test_unknown_fields_ignored_not_rejected():
    # "accept those as well": the full slancha-api shape must not 422 here.
    sp = SlanchaPrefInput.model_validate(
        {"weights": {"price": 1.0}, "service_tier": "fast_premium", "zdr": True}
    )
    assert sp.weights == {"price": 1.0}


# ---------------------------------------------------------------------------
# resolve_preferences — map onto slancha-local Preferences
# ---------------------------------------------------------------------------


def test_resolve_weights_simplex_normalizes_to_local_weights():
    prefs = resolve_preferences(header=None, body_pref={"weights": {"price": 3, "accuracy": 1}})
    assert isinstance(prefs, Preferences)
    # price=3, accuracy=1, latency=0 → normalized 0.75 / 0.25 / 0.0
    assert prefs.cost_weight == pytest.approx(0.75)
    assert prefs.quality_weight == pytest.approx(0.25)
    assert prefs.latency_weight == pytest.approx(0.0)
    assert prefs.privacy_weight == pytest.approx(0.0)


def test_resolve_all_zero_weights_keeps_defaults():
    default = Preferences()
    prefs = resolve_preferences(header=None, body_pref={"weights": {"price": 0, "accuracy": 0}})
    assert prefs.cost_weight == default.cost_weight
    assert prefs.quality_weight == default.quality_weight


def test_resolve_flat_levers_map_and_unit_convert():
    prefs = resolve_preferences(
        header=None,
        body_pref={
            "max_latency_ms_p95": 2500,
            "max_cost_per_1m_usd": 4000.0,  # USD per 1M tok
            "allow_fallbacks": False,
        },
    )
    assert prefs.max_latency_ms == 2500
    # per-1M-usd → per-1k: 4000 / 1000 = 4.0
    assert prefs.max_cost_per_1k == pytest.approx(4.0)
    assert prefs.escalation_allowed is False


def test_resolve_body_wins_over_header():
    prefs = resolve_preferences(
        header="quality-weight=0.2",
        body_pref={"quality_weight": 0.9},
    )
    assert prefs.quality_weight == pytest.approx(0.9)


def test_resolve_header_only_when_no_body():
    prefs = resolve_preferences(header="allow-fallbacks=?0", body_pref=None)
    assert prefs.escalation_allowed is False


def test_resolve_empty_returns_defaults():
    prefs = resolve_preferences(header=None, body_pref=None)
    assert prefs == Preferences()


def test_resolve_bad_weights_raises():
    with pytest.raises(ValidationError):
        resolve_preferences(header=None, body_pref={"weights": {"speed": 1.0}})


# ---------------------------------------------------------------------------
# Request model accepts pref
# ---------------------------------------------------------------------------


def test_chat_request_accepts_pref_field():
    from slancha_local.proxy.models import ChatCompletionRequest

    req = ChatCompletionRequest.model_validate(
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "hi"}],
            "pref": {"weights": {"latency": 1.0}},
        }
    )
    assert req.pref == {"weights": {"latency": 1.0}}
