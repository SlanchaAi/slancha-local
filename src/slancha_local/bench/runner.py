"""Self-bench: runs the classifier on the bundled adversarial set + measures latency.

This is the v0.1 bench. RouterBench (full 405K-prompt third-party benchmark)
ships in v0.1.1 — see ADR-001.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from slancha_local.bench.scorecard import BenchResult


def _load_adversarial_set() -> list[dict]:
    """Loads tests/privacy/adversarial_prompts.json from the source tree.

    For deployed wheels, an embedded copy could be vendored; for now this is
    a development-tree-only command (and honest about it).
    """
    # When running from source, find tests/privacy/adversarial_prompts.json
    candidates = [
        Path(__file__).resolve().parents[3] / "tests" / "privacy" / "adversarial_prompts.json",
        Path.cwd() / "tests" / "privacy" / "adversarial_prompts.json",
    ]
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text())["prompts"]
    raise FileNotFoundError(
        "adversarial_prompts.json not found. Run `slancha bench` from a slancha-local "
        "checkout, or wait for v0.1.1 which ships RouterBench data."
    )


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * (p / 100.0))))
    return s[idx]


def run_self_bench(*, hardware: str | None = None) -> BenchResult:
    """Run the bundled adversarial set against the local classifier.

    Returns a BenchResult that can be uploaded to the leaderboard or shared
    as a screenshot.
    """
    from slancha_local.classifier.local import LocalClassifier
    from slancha_local.classifier_client.models import (
        ClassifyRequest,
        LocalModelDescriptor,
        Preferences,
    )
    from slancha_local.embedder import embed_single

    classifier = LocalClassifier()
    prompts = _load_adversarial_set()

    models = [
        LocalModelDescriptor(backend="ollama", id="qwen3:8b", ctx_window=32768, capabilities=["en"]),
        LocalModelDescriptor(
            backend="ollama",
            id="codestral:22b",
            ctx_window=32768,
            capabilities=["en", "coding"],
        ),
    ]

    head_correct = {"jailbreak": 0, "pii": 0, "tool_calling": 0, "domain": 0, "language": 0}
    head_total = {k: 0 for k in head_correct}
    classifier_ms_samples: list[float] = []

    import asyncio

    async def _score():
        for entry in prompts:
            emb = embed_single(entry["prompt"]).tolist()
            req = ClassifyRequest(
                embedding=emb,
                prompt=entry["prompt"],
                available_models=models,
                preferences=Preferences(),
                context_len=len(entry["prompt"]),
            )
            t0 = time.perf_counter()
            resp = await classifier.classify(req)
            classifier_ms_samples.append((time.perf_counter() - t0) * 1000.0)
            expected = entry["expected"]
            for key in head_correct:
                if key in expected:
                    head_total[key] += 1
                    actual = getattr(resp, key) if key != "domain" else resp.domain
                    if actual == expected[key]:
                        head_correct[key] += 1

    asyncio.run(_score())

    head_breakdown = {k: (head_correct[k] / head_total[k]) if head_total[k] else 0.0 for k in head_correct}
    overall_correct = sum(head_correct.values())
    overall_total = sum(head_total.values())
    overall_acc = overall_correct / overall_total if overall_total else 0.0

    return BenchResult(
        name="adversarial-self-bench",
        classifier_version="v1-local",
        samples=len(prompts),
        accuracy=overall_acc,
        p50_classifier_ms=_percentile(classifier_ms_samples, 50),
        p95_classifier_ms=_percentile(classifier_ms_samples, 95),
        p99_classifier_ms=_percentile(classifier_ms_samples, 99),
        head_breakdown=head_breakdown,
        hardware=hardware or platform.platform(),
        backend="local-classifier-only",
    )
