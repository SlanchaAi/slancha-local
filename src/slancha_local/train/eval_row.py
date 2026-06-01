"""Build :data:`EVAL_ROW_FIELDS`-shaped eval-pass rows from per-sample
scores so :func:`slancha_local.train.gate.decide` can be invoked on
champion/challenger router pairs.

Mirrors the aggregation in ``mesh.eval.runner.run_eval_pass`` but takes
already-scored samples as input rather than dispatching prompts itself.
slancha-local's existing scoring loop lives in
``slancha_local.train.eval`` (pairwise base-vs-FT) and the embedded
LLM-judge tier; this module is the shape-converter that turns either
into a single EvalPass-shaped row the promotion gate can consume.

Why a separate module: the mesh ``EvalPass`` lives in slancha-mesh's
private repo and pulls in seed verification + endpoint dispatch deps
slancha-local doesn't ship. This module re-emits the same row shape
without those dependencies, audited against
``mesh.eval.runner.EvalPass.to_row`` by
``tests/test_gate_cross_repo_compat.py``.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slancha_local.train.gate import EVAL_ROW_FIELDS

# Match mesh.eval.runner constants so a slancha-local-built row aggregates
# the same way a mesh-built row would on the same input data.
ACCEPTABLE_SCORE_THRESHOLD = 3.0
FAILURE_SCORE_THRESHOLD = 1.0


@dataclass(frozen=True)
class EvalSample:
    """One scored sample contributing to an aggregate eval pass.

    ``score`` is on the same 0..5 ladder as
    ``mesh.quality_probe.LocalJudgeScorer`` so the
    ACCEPTABLE/FAILURE thresholds carry the same semantics across repos.
    Callers using a {win, tie, loss}-style local judge should map
    pairwise verdicts onto the same ladder (e.g. ft-win → 5, tie → 3,
    base-win → 1) before calling :func:`aggregate_eval_pass`.

    ``failure_kind`` lets the aggregator carry over the
    n_dispatch_failures / n_scorer_failures counters mesh exposes. Set
    to ``None`` for normal samples.
    """

    domain: str
    served_model: str
    score: float
    failure_kind: str | None = None  # "dispatch" | "scorer" | None


def aggregate_eval_pass(
    samples: list[EvalSample],
    *,
    router_version: str,
    judge_model: str,
    holdout_version: int,
    fast_head_version: int | None = None,
    overrides_version: int | None = None,
    elapsed_seconds: float = 0.0,
    ts: str | None = None,
    artifact_sha256: str | None = None,
    holdout_manifest_sha256: str | None = None,
    training_corpus_hash: str | None = None,
    base_model_fingerprint: str | None = None,
    router_config_hash: str | None = None,
    code_sha: str | None = None,
) -> dict[str, Any]:
    """Aggregate ``samples`` into a JSON-serializable dict matching
    :data:`EVAL_ROW_FIELDS` (i.e. the output of
    ``mesh.eval.runner.EvalPass.to_row``).

    Domains / served-models with zero samples are omitted from the
    per-domain / per-model dicts — same behavior as mesh's runner.

    ``ts`` defaults to ``time.gmtime`` ISO-UTC ``"%Y-%m-%dT%H:%M:%SZ"``
    for parity with mesh; pass an explicit value for deterministic tests
    or to stamp the row with the time the underlying eval started.
    """
    scores = [s.score for s in samples]
    n_eval = len(scores)

    per_domain_scores: dict[str, list[float]] = defaultdict(list)
    per_model_scores: dict[str, list[float]] = defaultdict(list)
    n_dispatch_failures = 0
    n_scorer_failures = 0
    for s in samples:
        per_domain_scores[s.domain].append(s.score)
        per_model_scores[s.served_model].append(s.score)
        if s.failure_kind == "dispatch":
            n_dispatch_failures += 1
        elif s.failure_kind == "scorer":
            n_scorer_failures += 1

    mean = statistics.fmean(scores) if scores else 0.0
    median = statistics.median(scores) if scores else 0.0
    pct_acc = sum(1 for x in scores if x >= ACCEPTABLE_SCORE_THRESHOLD) / n_eval if n_eval else 0.0
    pct_fail = sum(1 for x in scores if x < FAILURE_SCORE_THRESHOLD) / n_eval if n_eval else 0.0

    row = {
        "ts": ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "router_version": router_version,
        "fast_head_version": fast_head_version,
        "overrides_version": overrides_version,
        "holdout_version": holdout_version,
        "n_eval": n_eval,
        "judge_model": judge_model,
        "mean_score": round(mean, 4),
        "median_score": round(median, 4),
        "pct_acceptable": round(pct_acc, 4),
        "pct_failure": round(pct_fail, 4),
        "per_domain_mean": {d: round(statistics.fmean(s), 4) for d, s in per_domain_scores.items() if s},
        "per_model_mean": {m: round(statistics.fmean(s), 4) for m, s in per_model_scores.items() if s},
        "elapsed_seconds": round(elapsed_seconds, 3),
        "n_dispatch_failures": n_dispatch_failures,
        "n_scorer_failures": n_scorer_failures,
        "artifact_sha256": artifact_sha256,
        "holdout_manifest_sha256": holdout_manifest_sha256,
        "training_corpus_hash": training_corpus_hash,
        "base_model_fingerprint": base_model_fingerprint,
        "router_config_hash": router_config_hash,
        "code_sha": code_sha,
    }
    # Key-set parity is a hard invariant; the cross-repo guard
    # asserts the same in CI when slancha-mesh is on disk.
    assert tuple(row) == tuple(EVAL_ROW_FIELDS), (
        f"eval_row schema drift: built={tuple(row)} expected={tuple(EVAL_ROW_FIELDS)}"
    )
    return row


def append_eval_row(output: Path, row: dict[str, Any]) -> None:
    """Append one eval-pass row to ``output`` as a JSONL line.

    Same on-disk format as ``mesh.eval.runner.append_pass`` so the
    resulting file can be read by mesh's dashboard / gate tooling
    interchangeably with mesh-written rows.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_eval_row(path: Path) -> dict[str, Any]:
    """Read a single eval-pass row from a JSON file (one object) or a
    JSONL file (uses the last line — typical for an appended log).

    Raises :class:`ValueError` on empty input or unparseable JSON.
    """
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"eval-row file is empty: {path}")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Single-line: try as JSON (handles pretty-printed or compact one-object).
    # Multi-line: try whole-text first (pretty-printed object), then fall
    # back to JSONL last-line.
    if len(lines) == 1:
        try:
            return json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not parse eval-row JSON at {path}: {exc}") from exc
    try:
        # Whole-file parse handles a pretty-printed single object spanning
        # multiple lines. JSONL would fail here (Extra data after line 1).
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse last JSONL row at {path}: {exc}") from exc
