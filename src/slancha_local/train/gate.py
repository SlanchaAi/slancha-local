"""Local mirror of slancha-mesh's promotion gate (mesh.eval.gate).

Why mirrored, not imported
--------------------------
slancha-local ships **public** with no slancha-mesh runtime dependency
(the package isn't on PyPI; the design discipline is documented in
``src/slancha_local/mesh/__init__.py``). The gate is the contract by
which any slancha-local-side training pass decides whether a freshly
retrained head should replace the incumbent: we mirror the shape +
behavior here so a deployment can run ``slancha promote-heads`` on a
machine that has never seen slancha-mesh on disk.

The mirror is held honest by ``tests/test_gate_cross_repo_compat.py``,
which AST-parses slancha-mesh's gate.py + runner.py when that checkout
is present and asserts:
    • GateThresholds field names + defaults match,
    • PromotionVerdict field names match,
    • EvalPass.to_row() schema matches our :data:`EVAL_ROW_FIELDS`,
    • decide() produces identical verdicts on a curated fixture pair.
When slancha-mesh is *not* on disk the guard skips cleanly — alignment
is only enforced where it can be checked, same discipline as
``test_pref_cross_repo_compat.py``.

Eval row schema
---------------
:data:`EVAL_ROW_FIELDS` is the documented union of fields any row that
the gate consumes (or any downstream tooling reads) is expected to
carry. It's a flat tuple — easy to diff against ``EvalPass.to_row()``
upstream — rather than a class hierarchy, because the wire format IS
the contract; representations on either side are implementation details.

The gate itself only reads ``mean_score``, ``per_domain_mean``,
``n_eval``, ``judge_model``, and ``router_version`` — everything else
is carried through for audit. Verdicts include the thresholds dict so a
later operator can ask "why did we promote this?" months later.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_MEAN_SCORE_DELTA = 0.05
DEFAULT_PER_DOMAIN_MAX_REGRESSION = 0.15
DEFAULT_MIN_N_EVAL = 100

EVAL_ROW_FIELDS: tuple[str, ...] = (
    "ts",
    "router_version",
    "fast_head_version",
    "overrides_version",
    "holdout_version",
    "n_eval",
    "judge_model",
    "mean_score",
    "median_score",
    "pct_acceptable",
    "pct_failure",
    "per_domain_mean",
    "per_model_mean",
    "elapsed_seconds",
    "n_dispatch_failures",
    "n_scorer_failures",
)
"""Fields any row written by a slancha-local eval pass must carry to be
consumable by the gate. Mirrors ``mesh.eval.runner.EvalPass.to_row()``."""


@dataclass(frozen=True)
class GateThresholds:
    """Mirrors :class:`mesh.eval.gate.GateThresholds`.

    ``mean_score_delta`` — headline lift required to even consider a
    promotion. Smaller values trade noise tolerance for sensitivity.

    ``per_domain_max_regression`` — how much any single domain may slip
    in absolute judge-score points before the gate refuses. Set larger
    than the typical inter-pass noise on the held-out mean.

    ``min_n_eval`` — refuse if either side ran fewer prompts than this;
    avoids being fooled by a tiny pass.

    ``require_judge_match`` — when True, refuse cross-judge comparisons.
    When False, the gate still records the mismatch in the verdict.
    """

    mean_score_delta: float = DEFAULT_MEAN_SCORE_DELTA
    per_domain_max_regression: float = DEFAULT_PER_DOMAIN_MAX_REGRESSION
    min_n_eval: int = DEFAULT_MIN_N_EVAL
    require_judge_match: bool = True


@dataclass(frozen=True)
class PromotionVerdict:
    """Mirrors :class:`mesh.eval.gate.PromotionVerdict`.

    The decision plus enough audit detail to explain itself later — the
    SRE persona's "every promotion is an event" requirement.
    """

    accept: bool
    reject_reasons: tuple[str, ...]
    mean_delta: float
    per_domain_deltas: dict[str, float] = field(default_factory=dict)
    champion_version: str = ""
    challenger_version: str = ""
    n_eval_champion: int = 0
    n_eval_challenger: int = 0
    judge_model_champion: str = ""
    judge_model_challenger: str = ""
    decided_at: str = ""
    thresholds: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "reject_reasons": list(self.reject_reasons),
        }


def _row_field(row: dict[str, Any], key: str, default: Any) -> Any:
    v = row.get(key)
    return default if v is None else v


def decide(
    champion: dict[str, Any],
    challenger: dict[str, Any],
    thresholds: GateThresholds | None = None,
) -> PromotionVerdict:
    """Return a :class:`PromotionVerdict` for ``(champion, challenger)``.

    Inputs are eval-row dicts matching :data:`EVAL_ROW_FIELDS`. Domains
    not present in **both** rows are skipped from the per-domain check —
    we cannot say a domain regressed if the champion never saw it.

    Behavior must remain bit-identical to ``mesh.eval.gate.decide``;
    drift is caught by the cross-repo guard tests.
    """
    if thresholds is None:
        thresholds = GateThresholds()
    champion_mean = float(_row_field(champion, "mean_score", 0.0))
    challenger_mean = float(_row_field(challenger, "mean_score", 0.0))
    mean_delta = challenger_mean - champion_mean

    champion_per_dom = _row_field(champion, "per_domain_mean", {}) or {}
    challenger_per_dom = _row_field(challenger, "per_domain_mean", {}) or {}
    shared_domains = sorted(set(champion_per_dom) & set(challenger_per_dom))
    per_dom_deltas: dict[str, float] = {
        d: float(challenger_per_dom[d]) - float(champion_per_dom[d]) for d in shared_domains
    }

    judge_a = str(_row_field(champion, "judge_model", "unknown"))
    judge_b = str(_row_field(challenger, "judge_model", "unknown"))
    n_a = int(_row_field(champion, "n_eval", 0))
    n_b = int(_row_field(challenger, "n_eval", 0))

    reasons: list[str] = []

    if thresholds.require_judge_match and judge_a != judge_b:
        reasons.append(f"judge_model mismatch: champion={judge_a!r} challenger={judge_b!r}")
    if n_a < thresholds.min_n_eval:
        reasons.append(f"champion n_eval {n_a} below min_n_eval {thresholds.min_n_eval}")
    if n_b < thresholds.min_n_eval:
        reasons.append(f"challenger n_eval {n_b} below min_n_eval {thresholds.min_n_eval}")
    if mean_delta < thresholds.mean_score_delta:
        reasons.append(f"mean_delta {mean_delta:+.3f} below required {thresholds.mean_score_delta:+.3f}")
    for d in shared_domains:
        delta = per_dom_deltas[d]
        if delta < -thresholds.per_domain_max_regression:
            reasons.append(
                f"per-domain regression on {d!r}: {delta:+.3f} exceeds "
                f"-{thresholds.per_domain_max_regression:.3f}"
            )

    return PromotionVerdict(
        accept=not reasons,
        reject_reasons=tuple(reasons),
        mean_delta=mean_delta,
        per_domain_deltas=per_dom_deltas,
        champion_version=str(_row_field(champion, "router_version", "")),
        challenger_version=str(_row_field(challenger, "router_version", "")),
        n_eval_champion=n_a,
        n_eval_challenger=n_b,
        judge_model_champion=judge_a,
        judge_model_challenger=judge_b,
        decided_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        thresholds=asdict(thresholds),
    )
