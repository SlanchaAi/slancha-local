"""Unit tests for :mod:`slancha_local.train.eval_row` — the aggregator
that turns per-sample scores into an EvalPass-shaped row for
``slancha_local.train.gate.decide``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slancha_local.train.eval_row import (
    ACCEPTABLE_SCORE_THRESHOLD,
    FAILURE_SCORE_THRESHOLD,
    EvalSample,
    aggregate_eval_pass,
    append_eval_row,
    read_eval_row,
)
from slancha_local.train.gate import EVAL_ROW_FIELDS, GateThresholds, decide


def _sample(domain="general", model="gpt-x", score=4.0, failure_kind=None) -> EvalSample:
    return EvalSample(domain=domain, served_model=model, score=score, failure_kind=failure_kind)


def test_aggregate_emits_full_eval_row_schema():
    row = aggregate_eval_pass(
        [_sample()],
        router_version="v1",
        judge_model="gpt-judge",
        holdout_version=7,
        ts="2026-05-28T05:00:00Z",
    )
    assert tuple(row.keys()) == tuple(EVAL_ROW_FIELDS)
    assert row["ts"] == "2026-05-28T05:00:00Z"
    assert row["router_version"] == "v1"
    assert row["holdout_version"] == 7
    assert row["n_eval"] == 1
    assert row["judge_model"] == "gpt-judge"


def test_aggregate_empty_row_has_zero_stats():
    row = aggregate_eval_pass([], router_version="v1", judge_model="j", holdout_version=1, ts="t")
    assert row["n_eval"] == 0
    assert row["mean_score"] == 0.0
    assert row["median_score"] == 0.0
    assert row["pct_acceptable"] == 0.0
    assert row["pct_failure"] == 0.0
    assert row["per_domain_mean"] == {}
    assert row["per_model_mean"] == {}


def test_aggregate_means_and_pcts():
    samples = [
        _sample(domain="general", score=5.0),
        _sample(domain="general", score=4.0),
        _sample(domain="general", score=0.0),  # below FAILURE
        _sample(domain="code", score=3.0),  # exactly acceptable
        _sample(domain="code", score=2.0),  # neither acceptable nor failure
    ]
    row = aggregate_eval_pass(samples, router_version="v1", judge_model="j", holdout_version=1, ts="t")
    assert row["n_eval"] == 5
    assert row["mean_score"] == pytest.approx((5 + 4 + 0 + 3 + 2) / 5, abs=1e-4)
    assert row["median_score"] == pytest.approx(3.0, abs=1e-4)
    # Acceptable (>=3.0): 5, 4, 3 → 3/5
    assert row["pct_acceptable"] == pytest.approx(0.6, abs=1e-4)
    # Failure (<1.0): only the 0.0 → 1/5
    assert row["pct_failure"] == pytest.approx(0.2, abs=1e-4)
    assert row["per_domain_mean"]["general"] == pytest.approx((5 + 4 + 0) / 3, abs=1e-4)
    assert row["per_domain_mean"]["code"] == pytest.approx(2.5, abs=1e-4)


def test_failure_kinds_increment_counters():
    samples = [
        _sample(failure_kind="dispatch", score=0.0),
        _sample(failure_kind="dispatch", score=0.0),
        _sample(failure_kind="scorer", score=0.0),
        _sample(score=4.0),
    ]
    row = aggregate_eval_pass(samples, router_version="v1", judge_model="j", holdout_version=1, ts="t")
    assert row["n_dispatch_failures"] == 2
    assert row["n_scorer_failures"] == 1


def test_thresholds_match_mesh_constants():
    """Drift here means the slancha-local aggregator silently disagrees
    with mesh's runner on what 'acceptable' / 'failure' means."""
    assert ACCEPTABLE_SCORE_THRESHOLD == 3.0
    assert FAILURE_SCORE_THRESHOLD == 1.0


def test_per_model_mean_partitions_by_served_model():
    samples = [
        _sample(model="m1", score=4.0),
        _sample(model="m1", score=2.0),
        _sample(model="m2", score=5.0),
    ]
    row = aggregate_eval_pass(samples, router_version="v1", judge_model="j", holdout_version=1, ts="t")
    assert row["per_model_mean"] == {"m1": 3.0, "m2": 5.0}


def test_aggregate_row_feeds_decide_round_trip(tmp_path):
    """End-to-end: build two rows, persist them, gate.decide() promotes
    the challenger over the champion."""
    champ_samples = [_sample(domain="general", model="m1", score=3.0) for _ in range(60)] + [
        _sample(domain="code", model="m1", score=3.0) for _ in range(60)
    ]
    chall_samples = [_sample(domain="general", model="m2", score=4.0) for _ in range(60)] + [
        _sample(domain="code", model="m2", score=4.0) for _ in range(60)
    ]
    champ = aggregate_eval_pass(
        champ_samples, router_version="v1", judge_model="j", holdout_version=1, ts="t1"
    )
    chall = aggregate_eval_pass(
        chall_samples, router_version="v2", judge_model="j", holdout_version=1, ts="t2"
    )
    v = decide(champ, chall, GateThresholds(min_n_eval=100))
    assert v.accept is True
    assert v.mean_delta == pytest.approx(1.0, abs=1e-4)

    out = tmp_path / "eval_results.jsonl"
    append_eval_row(out, champ)
    append_eval_row(out, chall)
    again = read_eval_row(out)  # last row
    assert again == chall


def test_append_creates_parent_dir(tmp_path):
    out = tmp_path / "deep" / "nested" / "eval.jsonl"
    row = aggregate_eval_pass([_sample()], router_version="v", judge_model="j", holdout_version=1, ts="t")
    append_eval_row(out, row)
    assert out.is_file()


def test_read_eval_row_accepts_pretty_json(tmp_path):
    row = aggregate_eval_pass([_sample()], router_version="v", judge_model="j", holdout_version=1, ts="t")
    path = tmp_path / "row.json"
    path.write_text(json.dumps(row, indent=2))
    assert read_eval_row(path) == row


def test_read_eval_row_raises_on_empty(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_eval_row(p)


def test_read_eval_row_raises_on_garbage(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{not json")
    with pytest.raises(ValueError):
        read_eval_row(p)


def test_ts_defaults_to_now_when_omitted():
    row = aggregate_eval_pass([_sample()], router_version="v", judge_model="j", holdout_version=1)
    # ISO UTC with trailing Z, length 20
    assert row["ts"].endswith("Z")
    assert len(row["ts"]) == 20


def test_provenance_fields_passed_through():
    row = aggregate_eval_pass(
        [_sample()],
        router_version="v",
        judge_model="j",
        holdout_version=1,
        fast_head_version=3,
        overrides_version=17,
        elapsed_seconds=12.345,
        ts="t",
    )
    assert row["fast_head_version"] == 3
    assert row["overrides_version"] == 17
    assert row["elapsed_seconds"] == 12.345


def test_per_domain_uses_fmean_skipping_empty_buckets():
    """Mesh's runner drops domains whose score-list is empty from
    per_domain_mean. Verify we do the same."""
    # No way to create an empty per-domain bucket via the public
    # API since EvalSample always carries a (domain, score), but we
    # can verify the post-aggregation dict has no empty values by
    # inspection — every key has a float.
    row = aggregate_eval_pass(
        [_sample(domain="a", score=3.0)],
        router_version="v",
        judge_model="j",
        holdout_version=1,
        ts="t",
    )
    for v in row["per_domain_mean"].values():
        assert isinstance(v, float)


def test_round_trip_via_path_object(tmp_path: Path):
    """read_eval_row uses Path.read_text — verify pathlib API works."""
    out = tmp_path / "x.json"
    row = aggregate_eval_pass([_sample()], router_version="v", judge_model="j", holdout_version=1, ts="t")
    out.write_text(json.dumps(row))
    back = read_eval_row(out)
    assert back == row
