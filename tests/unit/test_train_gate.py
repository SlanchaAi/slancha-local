"""Unit tests for slancha_local.train.gate — the local mirror of
``mesh.eval.gate``. Behavior parity with mesh is asserted separately by
``tests/test_gate_cross_repo_compat.py`` (skips when slancha-mesh isn't
on disk).
"""

from __future__ import annotations

from slancha_local.train.gate import GateThresholds, PromotionVerdict, decide


def pytest_approx(value: float, tol: float = 1e-9) -> float:
    """Inline approx so we don't drag ``pytest.approx`` into a non-fixture
    context — also lets us use it inside dict equality checks."""

    class _Approx(float):
        def __eq__(self, other):
            return abs(float(other) - value) < tol

        def __ne__(self, other):
            return not self.__eq__(other)

        def __hash__(self):
            return hash(value)

    return _Approx(value)


def _row(
    *,
    mean: float,
    n: int = 200,
    judge: str = "gpt-judge",
    version: str = "v1",
    per_domain: dict[str, float] | None = None,
) -> dict:
    return {
        "router_version": version,
        "n_eval": n,
        "judge_model": judge,
        "mean_score": mean,
        "per_domain_mean": per_domain or {},
    }


def test_accept_when_lift_meets_threshold_and_no_regression():
    champ = _row(mean=0.60, per_domain={"general": 0.6, "code": 0.6})
    chall = _row(mean=0.70, version="v2", per_domain={"general": 0.7, "code": 0.65})
    v = decide(champ, chall)
    assert v.accept is True
    assert v.reject_reasons == ()
    assert v.mean_delta == pytest_approx(0.10)
    assert v.per_domain_deltas == {"general": pytest_approx(0.1), "code": pytest_approx(0.05)}
    assert v.champion_version == "v1"
    assert v.challenger_version == "v2"


def test_reject_when_mean_lift_below_threshold():
    champ = _row(mean=0.60)
    chall = _row(mean=0.62, version="v2")  # +0.02 < default 0.05
    v = decide(champ, chall)
    assert v.accept is False
    assert any("mean_delta" in r for r in v.reject_reasons)


def test_reject_on_per_domain_cliff_even_when_mean_lifts():
    """The whole reason this gate exists: a challenger that lifts the
    headline mean while tanking one domain must NOT promote."""
    champ = _row(mean=0.60, per_domain={"general": 0.5, "code": 0.7})
    chall = _row(
        mean=0.65,  # +0.05 mean — passes mean check
        version="v2",
        per_domain={"general": 0.9, "code": 0.40},  # code: -0.30 → cliff
    )
    v = decide(champ, chall)
    assert v.accept is False
    assert any("'code'" in r and "regression" in r for r in v.reject_reasons)


def test_reject_when_under_min_n_eval():
    champ = _row(mean=0.6, n=50)
    chall = _row(mean=0.7, n=200, version="v2")
    v = decide(champ, chall)
    assert v.accept is False
    assert any("champion n_eval 50" in r for r in v.reject_reasons)


def test_reject_on_judge_mismatch_by_default():
    champ = _row(mean=0.6, judge="gpt-judge-a")
    chall = _row(mean=0.7, version="v2", judge="gpt-judge-b")
    v = decide(champ, chall)
    assert v.accept is False
    assert any("judge_model mismatch" in r for r in v.reject_reasons)


def test_judge_mismatch_allowed_when_opted_in():
    champ = _row(mean=0.6, judge="gpt-judge-a", per_domain={"general": 0.6})
    chall = _row(mean=0.7, version="v2", judge="gpt-judge-b", per_domain={"general": 0.7})
    v = decide(champ, chall, GateThresholds(require_judge_match=False))
    assert v.accept is True
    # But the mismatch is still recorded so an operator can see it later.
    assert v.judge_model_champion == "gpt-judge-a"
    assert v.judge_model_challenger == "gpt-judge-b"


def test_per_domain_check_skips_domains_only_one_side_saw():
    """A domain present only on the challenger isn't a regression — the
    champion never saw it."""
    champ = _row(mean=0.6, per_domain={"general": 0.6})
    chall = _row(mean=0.7, version="v2", per_domain={"general": 0.7, "newly_observed_domain": 0.3})
    v = decide(champ, chall)
    assert v.accept is True
    assert v.per_domain_deltas == {"general": pytest_approx(0.1)}


def test_to_row_round_trips_reject_reasons_as_list():
    champ = _row(mean=0.6)
    chall = _row(mean=0.62, version="v2")
    v = decide(champ, chall)
    row = v.to_row()
    assert isinstance(row["reject_reasons"], list)
    assert row["accept"] is False


def test_verdict_carries_thresholds_for_audit():
    champ = _row(mean=0.6, per_domain={"general": 0.6})
    chall = _row(mean=0.7, version="v2", per_domain={"general": 0.7})
    thr = GateThresholds(mean_score_delta=0.02, per_domain_max_regression=0.20, min_n_eval=50)
    v = decide(champ, chall, thr)
    assert v.thresholds == {
        "mean_score_delta": 0.02,
        "per_domain_max_regression": 0.20,
        "min_n_eval": 50,
        "require_judge_match": True,
    }


def test_decided_at_is_iso_utc():
    champ = _row(mean=0.6, per_domain={"general": 0.6})
    chall = _row(mean=0.7, version="v2", per_domain={"general": 0.7})
    v = decide(champ, chall)
    assert v.decided_at.endswith("Z")
    assert len(v.decided_at) == 20  # 2026-05-28T04:50:53Z


def test_promotion_verdict_is_frozen():
    v = PromotionVerdict(accept=True, reject_reasons=(), mean_delta=0.0)
    try:
        v.accept = False  # type: ignore[misc]
    except Exception:  # FrozenInstanceError
        return
    raise AssertionError("PromotionVerdict must be frozen")
