"""Post-FT eval harness — TensorZero-pattern two-tier (heuristic + judge).

Compares FT'd model against base on the val set. Heuristics filter the
obviously-broken (empty / non-utf8); LLM judge does pairwise win-rate.
Promotion gated on win_rate ≥ threshold (default 0.55).

This module is data-only — no inference; the runner script wires it to
the actual model providers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HeuristicResult:
    sample_idx: int
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class PairwiseEval:
    sample_idx: int
    base_response: str
    ft_response: str
    judge_pick: str  # "base" | "ft" | "tie"
    judge_reason: str = ""


@dataclass
class EvalReport:
    total_samples: int
    heuristic_passed: int
    judge_evaluated: int
    base_wins: int
    ft_wins: int
    ties: int
    win_rate_ft: float
    promote: bool
    threshold: float


def heuristic_check(response: str, names: list[str]) -> HeuristicResult:
    """Apply enabled heuristics; return pass/fail + reasons."""
    failures: list[str] = []
    if "non_empty_response" in names and not (response or "").strip():
        failures.append("empty_response")
    if "valid_utf8" in names:
        try:
            response.encode("utf-8")
        except UnicodeError:
            failures.append("invalid_utf8")
    return HeuristicResult(sample_idx=-1, passed=not failures, reasons=failures)


def judge_pairwise_pick(prompt: str, base: str, ft: str, judge_id: str) -> str:
    """Stub: returns 'tie' until wired to a real judge model.

    Real impl: send (prompt, base, ft) to the judge model with a rubric prompt
    asking which response is better; parse "BASE" | "FT" | "TIE" out of the
    judge's reply. Defer to v0.1.x.
    """
    # NOTE: Phase 1 stub. Returns 'tie' so promotion is conservatively blocked.
    return "tie"


def evaluate(
    val_jsonl: Path,
    *,
    base_responses: list[str],
    ft_responses: list[str],
    heuristics: list[str],
    judge_id: str,
    threshold: float = 0.55,
) -> EvalReport:
    """Score base vs FT on the val set. Returns promotion decision."""
    samples = [json.loads(line) for line in val_jsonl.read_text().splitlines() if line.strip()]
    n = min(len(samples), len(base_responses), len(ft_responses))
    pairs: list[PairwiseEval] = []
    h_pass = 0
    for i in range(n):
        ft_h = heuristic_check(ft_responses[i], heuristics)
        if not ft_h.passed:
            continue
        h_pass += 1
        prompt = samples[i].get("messages", [{}])[0].get("content", "")
        pick = judge_pairwise_pick(prompt, base_responses[i], ft_responses[i], judge_id)
        pairs.append(
            PairwiseEval(
                sample_idx=i,
                base_response=base_responses[i],
                ft_response=ft_responses[i],
                judge_pick=pick,
            )
        )
    base_wins = sum(1 for p in pairs if p.judge_pick == "base")
    ft_wins = sum(1 for p in pairs if p.judge_pick == "ft")
    ties = sum(1 for p in pairs if p.judge_pick == "tie")
    decided = base_wins + ft_wins
    win_rate = (ft_wins / decided) if decided else 0.0
    return EvalReport(
        total_samples=n,
        heuristic_passed=h_pass,
        judge_evaluated=len(pairs),
        base_wins=base_wins,
        ft_wins=ft_wins,
        ties=ties,
        win_rate_ft=win_rate,
        promote=win_rate >= threshold,
        threshold=threshold,
    )
