"""Unit tests for LocalClassifier._select_target — the pre-fix selector
collapsed every non-coding non-hard prompt onto available[0]. These tests
pin per-prompt diversity: biology/history/etc must NOT route to a coder
when a non-coder is available."""

from __future__ import annotations

import pytest

from slancha_local.classifier.local import LocalClassifier
from slancha_local.classifier_client.models import LocalModelDescriptor, Preferences


def _coder():
    return LocalModelDescriptor(
        backend="ollama",
        id="codestral:22b",
        ctx_window=32768,
        capabilities=["en", "coding"],
        est_throughput_tps=20.0,
    )


def _general():
    return LocalModelDescriptor(
        backend="ollama",
        id="qwen3:8b",
        ctx_window=32768,
        capabilities=["en"],
        est_throughput_tps=40.0,
    )


def _reasoner():
    return LocalModelDescriptor(
        backend="ollama",
        id="deepseek-r1:14b",
        ctx_window=16384,
        capabilities=["en", "hard"],
        est_throughput_tps=15.0,
    )


def _tool_caller():
    return LocalModelDescriptor(
        backend="ollama",
        id="qwen3-tool:8b",
        ctx_window=32768,
        capabilities=["en", "tool_use"],
        est_throughput_tps=30.0,
    )


def _select(
    *,
    domain,
    difficulty="medium",
    jailbreak=False,
    pii=False,
    needs_tools=False,
    available,
    escalation=True,
    ctx=100,
):
    prefs = Preferences(escalation_allowed=escalation)
    return LocalClassifier._select_target(
        domain=domain,
        difficulty=difficulty,
        jailbreak=jailbreak,
        pii=pii,
        needs_tools=needs_tools,
        available=available,
        preferences=prefs,
        context_len=ctx,
    )


class TestPerPromptDiversity:
    def test_biology_picks_generalist_not_coder_when_coder_first(self):
        target, _, reason, _ = _select(
            domain="biology",
            available=[_coder(), _general()],
        )
        assert target == "local:ollama:qwen3:8b", reason

    def test_history_picks_generalist_when_coder_first(self):
        target, _, _, _ = _select(domain="history", available=[_coder(), _general()])
        assert target == "local:ollama:qwen3:8b"

    def test_only_coders_present_falls_back(self):
        target, _, reason, _ = _select(domain="biology", available=[_coder()])
        assert target == "local:ollama:codestral:22b"
        assert "only coders" in reason


class TestDomainSpecific:
    def test_cs_domain_prefers_coder(self):
        target, _, _, _ = _select(
            domain="computer science",
            available=[_general(), _coder()],
        )
        assert target == "local:ollama:codestral:22b"

    def test_math_prefers_reasoner_over_generalist(self):
        target, _, reason, _ = _select(
            domain="math",
            available=[_general(), _reasoner()],
        )
        assert target == "local:ollama:deepseek-r1:14b"
        assert "STEM" in reason

    def test_math_falls_back_to_generalist_when_no_reasoner(self):
        target, _, _, _ = _select(domain="math", available=[_coder(), _general()])
        assert target == "local:ollama:qwen3:8b"

    def test_hard_difficulty_picks_reasoner(self):
        target, _, reason, _ = _select(
            domain="biology",
            difficulty="hard",
            available=[_general(), _reasoner()],
        )
        assert target == "local:ollama:deepseek-r1:14b"
        assert "hard-capable" in reason


class TestToolCalling:
    def test_needs_tools_picks_tool_capable(self):
        target, _, reason, _ = _select(
            domain="other",
            needs_tools=True,
            available=[_general(), _tool_caller()],
        )
        assert target == "local:ollama:qwen3-tool:8b"
        assert "tool-capable" in reason

    def test_math_needs_tools_overrides_to_reasoning_not_tools(self):
        # The slancha-api domain-precedence fix: dy/dx = 3x^2 must NOT route
        # to tool_use just because the tool-head fired.
        target, _, _, _ = _select(
            domain="math",
            needs_tools=True,
            available=[_tool_caller(), _reasoner()],
        )
        assert target == "local:ollama:deepseek-r1:14b"

    def test_cs_needs_tools_overrides_to_coder_not_tools(self):
        target, _, _, _ = _select(
            domain="computer science",
            needs_tools=True,
            available=[_tool_caller(), _coder()],
        )
        assert target == "local:ollama:codestral:22b"


class TestEscalation:
    def test_no_local_with_escalation_goes_cloud(self):
        target, _, _, _ = _select(domain="other", available=[], escalation=True)
        assert target.startswith("cloud:")

    def test_no_local_without_escalation_rejects(self):
        target, _, _, _ = _select(domain="other", available=[], escalation=False)
        assert target.startswith("cloud:reject:")

    def test_context_overflow_with_escalation(self):
        target, _, _, _ = _select(
            domain="other",
            available=[_general()],
            ctx=1_000_000,
        )
        assert target.startswith("cloud:")


def test_classify_signature_passes_needs_tools(monkeypatch):
    """Regression: chat.py was previously calling _select_target without
    needs_tools — adding it as a kwarg without backfilling all call sites
    would break classify(). This test makes sure classify() forwards it."""
    captured = {}

    def fake_select(**kw):
        captured.update(kw)
        return ("local:ollama:qwen3:8b", [], "stub", 1.0)

    monkeypatch.setattr(LocalClassifier, "_select_target", staticmethod(fake_select))

    import asyncio

    import numpy as np

    from slancha_local.classifier_client.models import ClassifyRequest

    cls = LocalClassifier.__new__(LocalClassifier)
    cls._labels = {
        "domain": {"labels": ["other"]},
        "difficulty": {"labels": ["medium"]},
        "language": {"labels": ["en"]},
    }

    class _StubModel:
        pass

    _head_names = ("domain", "difficulty", "language", "jailbreak", "pii", "tool_calling")
    cls._heads = {n: _StubModel() for n in _head_names}

    def fake_multi(model, x, labels):
        return labels[0], 0.9

    def fake_binary(model, x):
        return 0.9 if id(model) == id(cls._heads["tool_calling"]) else 0.1

    monkeypatch.setattr(LocalClassifier, "_predict_multiclass", staticmethod(fake_multi))
    monkeypatch.setattr(LocalClassifier, "_predict_binary", staticmethod(fake_binary))

    req = ClassifyRequest(
        embedding=np.zeros(512, dtype=np.float32).tolist(),
        prompt="x",
        available_models=[_general()],
        preferences=Preferences(),
        context_len=10,
    )
    asyncio.run(cls.classify(req))
    assert "needs_tools" in captured
    assert captured["needs_tools"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
