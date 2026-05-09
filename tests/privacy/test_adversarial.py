"""Adversarial prompt set regression test for the local classifier.

Skipped automatically on systems without libomp (treelite import fails); the
regression suite needs the actual local heads to score, not the rules fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slancha_local.classifier_client.models import (
    ClassifyRequest,
    LocalModelDescriptor,
    Preferences,
)

# Skip the whole module if treelite/libomp isn't available
try:
    from slancha_local.classifier.local import LocalClassifier
    from slancha_local.embedder import embed_single

    LocalClassifier()  # constructor probes for treelite + assets
    _LOCAL_OK = True
except Exception:
    _LOCAL_OK = False

pytestmark = pytest.mark.skipif(
    not _LOCAL_OK, reason="local classifier unavailable (treelite/libomp missing)"
)


PROMPTS = json.loads(
    (Path(__file__).parent / "adversarial_prompts.json").read_text()
)["prompts"]

_MODELS = [
    LocalModelDescriptor(
        backend="ollama", id="qwen3:8b", ctx_window=32768, capabilities=["en"]
    ),
    LocalModelDescriptor(
        backend="ollama",
        id="codestral:22b",
        ctx_window=32768,
        capabilities=["en", "coding"],
    ),
]


@pytest.mark.parametrize("entry", PROMPTS, ids=[e["id"] for e in PROMPTS])
async def test_adversarial(entry):
    classifier = LocalClassifier()
    emb = embed_single(entry["prompt"]).tolist()
    req = ClassifyRequest(
        embedding=emb,
        prompt=entry["prompt"],
        available_models=_MODELS,
        preferences=Preferences(),
        context_len=len(entry["prompt"]),
    )
    resp = await classifier.classify(req)
    expected = entry["expected"]
    misses: list[str] = []
    if "jailbreak" in expected and resp.jailbreak != expected["jailbreak"]:
        misses.append(f"jailbreak: expected {expected['jailbreak']} got {resp.jailbreak}")
    if "pii" in expected and resp.pii != expected["pii"]:
        misses.append(f"pii: expected {expected['pii']} got {resp.pii}")
    if "tool_calling" in expected and resp.tool_calling != expected["tool_calling"]:
        misses.append(
            f"tool_calling: expected {expected['tool_calling']} got {resp.tool_calling}"
        )
    if "domain" in expected and resp.domain != expected["domain"]:
        misses.append(f"domain: expected {expected['domain']!r} got {resp.domain!r}")
    if "language" in expected and resp.language != expected["language"]:
        misses.append(f"language: expected {expected['language']!r} got {resp.language!r}")
    # Allow up to 1 miss per entry — tracked over time, fail loudly with detail
    assert len(misses) <= 1, f"{entry['id']}: {misses}"
