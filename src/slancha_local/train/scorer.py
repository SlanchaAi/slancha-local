"""Holdout-scoring primitive for the cluster-head promotion loop.

Used by :mod:`slancha_local.train.promote_head` to turn a
``(prompt, response)`` pair into a 0..5 ladder score that the gate
aggregates. The 0..5 ladder + ACCEPTABLE / FAILURE thresholds match
``mesh.eval.runner`` so a slancha-local-built row gates identically
to a mesh-built one (the same discipline as
:mod:`slancha_local.train.eval_row`).

PURE STANDALONE: no slancha-mesh runtime dependency. The
:class:`Scorer` Protocol lets callers plug in whatever judge they
like — mesh's ``LocalJudgeScorer``, a recorded-fixture replay, an
in-memory fake for tests. :class:`HttpxLocalJudgeScorer` is a thin
OpenAI-chat-completions-compatible LLM-judge default for deployments
that don't need anything fancier.

JUDGE-MATCH IS THE COMPARABILITY GUARANTEE: every scorer call MUST
record the ``judge_model`` that produced the score. The gate refuses
cross-judge comparisons by default (``require_judge_match=True``),
so a single :class:`Scorer` instance is used for BOTH incumbent and
candidate in the same promotion run — never two different scorers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class ScoreError(RuntimeError):
    """Raised when a scorer transport call fails terminally.

    The orchestrator catches this, records the failure on the
    sample's ``failure_kind="scorer"``, and continues with the next
    prompt — one bad judge call must not abort the eval pass.
    """


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of one ``score(prompt, response)`` call.

    ``score`` is on the 0..5 ladder shared with
    ``mesh.quality_probe.LocalJudgeScorer`` (5 = excellent, 3 =
    acceptable, 1 = failure, 0 = no answer). ``judge_model`` is the
    name of the LLM (or rule-based judge) that produced the score —
    the gate uses this to enforce ``require_judge_match``.
    """

    score: float
    judge_model: str


class Scorer(Protocol):
    """Judge ``response`` to ``prompt``, return a 0..5 ladder score.

    Implementations should raise :class:`ScoreError` on any transport
    failure they consider terminal. The orchestrator catches that
    error and records the sample as a scorer failure rather than
    aborting the whole eval pass.
    """

    def score(self, prompt: str, response: str) -> ScoreResult: ...


# Default judge prompt — explicit 0..5 ladder, asks for ONLY a digit.
# Kept minimal so a smaller judge model still produces a parseable
# answer; callers wanting richer rubrics can ship their own Scorer.
_JUDGE_SYSTEM_PROMPT = """You are an impartial grader. Score the assistant's response on this 0..5 ladder:

5 = excellent, fully correct, well-explained
4 = correct with minor issues
3 = acceptable, mostly correct
2 = partially correct, important gaps
1 = failure: incorrect, unhelpful, or off-topic
0 = no answer / refused / empty

Reply with ONLY a single digit 0-5. No explanation, no punctuation."""

_JUDGE_USER_TEMPLATE = """PROMPT:
{prompt}

ASSISTANT RESPONSE:
{response}

Your score (single digit 0-5):"""

_SCORE_RE = re.compile(r"\b([0-5])\b")


def parse_judge_score(text: str) -> float:
    """Extract a 0..5 score from raw judge output.

    Tolerant of whitespace + minor preamble: takes the FIRST digit in
    [0,5] found in the response. Raises :class:`ScoreError` if no
    valid digit is present — better to count this as a scorer failure
    than silently coerce to 0 and bias the mean down.
    """
    stripped = text.strip()
    match = _SCORE_RE.search(stripped)
    if not match:
        raise ScoreError(f"judge response did not contain a 0..5 digit: {text!r}")
    return float(match.group(1))


class HttpxLocalJudgeScorer:
    """OpenAI-chat-completions-compatible LLM-judge scorer.

    Thin by design — meant as a "your-judge-isn't-deployed-yet" stub.
    Most production callers will plug in mesh's ``LocalJudgeScorer``
    (or a fixture replay for offline CI) via the :class:`Scorer`
    Protocol.

    Constructor takes the endpoint URL (e.g.
    ``http://localhost:8000/v1``), the judge model name, and an
    optional API key. ``temperature=0`` and a seed are passed so
    repeated runs on the same prompt+response produce the same score
    (the deterministic-comparability guarantee).
    """

    def __init__(
        self,
        endpoint_url: str,
        judge_model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        seed: int = 0,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.judge_model = judge_model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.seed = seed

    def score(self, prompt: str, response: str) -> ScoreResult:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ScoreError(
                "HttpxLocalJudgeScorer requires the 'httpx' extra; install "
                "slancha-local[promote] or pass a custom Scorer implementation"
            ) from e

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.judge_model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _JUDGE_USER_TEMPLATE.format(
                        prompt=prompt, response=response
                    ),
                },
            ],
            "temperature": 0.0,
            "seed": self.seed,
        }
        url = f"{self.endpoint_url}/chat/completions"
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
            r.raise_for_status()
            data = r.json()
        except Exception as e:  # noqa: BLE001
            raise ScoreError(f"judge call to {self.judge_model!r} failed: {e}") from e

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ScoreError(
                f"judge {self.judge_model!r} returned malformed response: {data!r}"
            ) from e

        return ScoreResult(score=parse_judge_score(str(text)), judge_model=self.judge_model)
