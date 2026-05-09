"""Post-FT eval harness — TensorZero-pattern two-tier (heuristic + judge).

Compares FT'd model against base on the val set. Heuristics filter the
obviously-broken (empty / non-utf8); LLM judge does pairwise win-rate.
Promotion gated on win_rate ≥ threshold (default 0.55).

Judge resolution order:
1. SLANCHA_TRAIN_DRY_RUN=1 → 'tie' (test path, zero network)
2. SLANCHA_JUDGE_URL set → POST OpenAI-compat /v1/chat/completions there
3. else SLANCHA_API_KEY set → slancha cloud /v1/chat/completions
4. else OPENAI_API_KEY set → OpenAI /v1/chat/completions
5. else 'tie' (no judge available; promotion blocked)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


JUDGE_SYSTEM = """You are a strict pairwise judge. Compare two assistant responses to the same user query.
Pick which is better on:
- Correctness and factual accuracy
- Faithfulness to the user's request
- Clarity and helpfulness

Output exactly one token on the first line: BASE, FT, or TIE.
Then one short sentence of justification on the next line. Nothing else.
"""

JUDGE_TEMPLATE = """User query:
{prompt}

Response A (BASE):
{base}

Response B (FT):
{ft}
"""


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


def _resolve_judge_endpoint() -> tuple[str, str, str] | None:
    """Return (base_url, api_key, model) or None if no real judge available."""
    if os.environ.get("SLANCHA_TRAIN_DRY_RUN") == "1":
        return None
    if url := os.environ.get("SLANCHA_JUDGE_URL"):
        key = os.environ.get("SLANCHA_JUDGE_API_KEY") or os.environ.get("SLANCHA_API_KEY") or ""
        model = os.environ.get("SLANCHA_JUDGE_MODEL", "gpt-4o-mini")
        return url.rstrip("/"), key, model
    if key := os.environ.get("SLANCHA_API_KEY"):
        return "https://api.slancha.ai", key, os.environ.get("SLANCHA_JUDGE_MODEL", "claude-haiku")
    if key := os.environ.get("OPENAI_API_KEY"):
        return "https://api.openai.com", key, os.environ.get("SLANCHA_JUDGE_MODEL", "gpt-4o-mini")
    return None


def _parse_judge_reply(text: str) -> tuple[str, str]:
    """Parse 'BASE|FT|TIE' from the first non-empty line. Lower-case the verdict."""
    if not text:
        return "tie", "empty judge reply"
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "tie", "empty judge reply"
    first = lines[0].upper()
    m = re.search(r"\b(BASE|FT|TIE)\b", first)
    if not m:
        # fallback: scan whole text
        m = re.search(r"\b(BASE|FT|TIE)\b", text.upper())
    verdict = m.group(1).lower() if m else "tie"
    reason = lines[1] if len(lines) > 1 else ""
    return verdict, reason


def judge_pairwise_pick(
    prompt: str,
    base: str,
    ft: str,
    judge_id: str,
    *,
    timeout_s: float = 30.0,
) -> tuple[str, str]:
    """Call the judge model. Returns (verdict, reason).

    verdict ∈ {"base", "ft", "tie"}. Falls back to ("tie", "<why>") if
    no judge configured or call fails — promotion stays conservatively blocked.

    `judge_id` is informational (logged, threaded into config). The actual
    endpoint comes from env via _resolve_judge_endpoint().
    """
    endpoint = _resolve_judge_endpoint()
    if endpoint is None:
        return "tie", "no judge endpoint configured (dry-run or missing API key)"

    base_url, api_key, model_id = endpoint
    if judge_id and ":" in judge_id:
        # Allow config to override model: e.g. "openai:gpt-5.4-mini" → use gpt-5.4-mini
        _, _, model_override = judge_id.partition(":")
        if model_override:
            model_id = model_override

    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_TEMPLATE.format(prompt=prompt, base=base, ft=ft)},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(base_url=base_url, headers=headers, timeout=timeout_s) as c:
            r = c.post("/v1/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return _parse_judge_reply(text)
    except (httpx.HTTPError, KeyError, ValueError) as e:
        logger.warning("judge call failed: %s", e)
        return "tie", f"judge error: {type(e).__name__}"


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
        pick, reason = judge_pairwise_pick(prompt, base_responses[i], ft_responses[i], judge_id)
        pairs.append(
            PairwiseEval(
                sample_idx=i,
                base_response=base_responses[i],
                ft_response=ft_responses[i],
                judge_pick=pick,
                judge_reason=reason,
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
