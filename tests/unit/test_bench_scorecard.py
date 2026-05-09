"""Bench scorecard formatting."""

from __future__ import annotations

from slancha_local.bench.scorecard import BenchResult, render_scorecard


def _example() -> BenchResult:
    return BenchResult(
        name="adversarial-self-bench",
        classifier_version="v1-local",
        samples=17,
        accuracy=0.706,
        p50_classifier_ms=1.2,
        p95_classifier_ms=3.2,
        p99_classifier_ms=3.2,
        head_breakdown={
            "domain": 1.0,
            "jailbreak": 0.5,
            "pii": 0.8,
            "tool_calling": 0.333,
            "language": 1.0,
        },
        hardware="macOS-15.7-arm64",
        backend="local-classifier-only",
    )


def test_scorecard_renders_non_empty():
    out = render_scorecard(_example())
    assert "slancha bench" in out
    assert "70.6%" in out
    assert "100.0%" in out


def test_scorecard_no_template_holes():
    out = render_scorecard(_example())
    assert "{" not in out and "}" not in out


def test_scorecard_to_dict_round_trip():
    r = _example()
    d = r.to_dict()
    assert d["accuracy"] == 0.706
    assert d["head_breakdown"]["domain"] == 1.0
