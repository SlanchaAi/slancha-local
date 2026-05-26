"""LocalClassifier: 6 treelite heads + route selector — runs in-process, zero network."""

from __future__ import annotations

import json
import logging
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from slancha_local.classifier_client.base import ClassifierClient
from slancha_local.classifier_client.models import ClassifyRequest, ClassifyResponse, Decision

logger = logging.getLogger(__name__)

_ASSET_ROOT = Path(str(files("slancha_local.assets") / "classifier_v1"))

# Map slancha-api domain labels → routing capability tags
# (slancha-api uses MMLU-Pro categories: "computer science", "math", etc.)
_DOMAIN_TO_CAP = {
    "computer science": "coding",
    "engineering": "coding",
    "math": "math",
    "physics": "math",
    "chemistry": "math",
    "biology": "general",
    "economics": "general",
    "business": "general",
    "health": "general",
    "history": "general",
    "law": "general",
    "philosophy": "general",
    "psychology": "general",
    "other": "general",
}

_DIFFICULTY_TO_DOMAIN_PREF = {
    "hard": "hard",  # prefer "hard"-capable model
    "medium": None,
    "easy": None,
}

# Mirrors slancha-api app/classifier/router.py: math/physics/CS reasoning is
# the primary signal; tool_calling head false-positives on `dy/dx = 3x^2` etc.
# These domains skip the tool_use branch even when needs_tools=True.
_DOMAIN_PRECEDENCE_OVER_TOOLS = {"math", "physics", "computer science", "engineering"}


class LocalClassifier(ClassifierClient):
    """Runs the 6 classifier heads + selector locally. No network calls."""

    def __init__(self, asset_root: Path | None = None) -> None:
        root = asset_root or _ASSET_ROOT
        with open(root / "labels.json") as f:
            self._labels = json.load(f)
        self._heads = self._load_heads(root)

    def _load_heads(self, root: Path) -> dict[str, Any]:
        try:
            import treelite
        except ImportError as e:
            raise RuntimeError(
                "treelite not installed. Install slancha-local[classifier] or [dev], "
                "or set SLANCHA_CLASSIFIER_KIND=rules to use the rule-based fallback."
            ) from e

        heads: dict[str, Any] = {}
        for name in ["domain", "jailbreak", "pii", "difficulty", "tool_calling", "language"]:
            path = root / f"mmbert_tl_{name}.bin"
            if path.exists():
                heads[name] = treelite.Model.deserialize(str(path))
            else:
                logger.warning("classifier head missing: %s", path)
        return heads

    @staticmethod
    def _predict_multiclass(model: Any, x: np.ndarray, labels: list[str]) -> tuple[str, float]:
        from treelite import gtil

        raw = gtil.predict(model, x).squeeze().flatten()
        if raw.ndim == 0:
            return labels[0], float(raw)
        probs = raw
        if probs.min() < 0 or probs.sum() < 0.5:
            exp = np.exp(probs - probs.max())
            probs = exp / exp.sum()
        idx = int(np.argmax(probs))
        return labels[idx], float(probs[idx])

    @staticmethod
    def _predict_binary(model: Any, x: np.ndarray) -> float:
        from treelite import gtil

        raw = gtil.predict(model, x).squeeze()
        return float(raw.flat[0]) if raw.size == 1 else float(raw.flat[-1])

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        import time

        x = np.array(request.embedding, dtype=np.float32).reshape(1, -1)
        t0 = time.perf_counter()

        domain_label, domain_conf = self._predict_multiclass(
            self._heads["domain"], x, self._labels["domain"]["labels"]
        )
        diff_label, diff_conf = self._predict_multiclass(
            self._heads["difficulty"], x, self._labels["difficulty"]["labels"]
        )
        lang_label, lang_conf = self._predict_multiclass(
            self._heads["language"], x, self._labels["language"]["labels"]
        )
        jb_prob = self._predict_binary(self._heads["jailbreak"], x)
        pii_prob = self._predict_binary(self._heads["pii"], x)
        tool_prob = self._predict_binary(self._heads["tool_calling"], x)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        is_jailbreak = jb_prob >= 0.5
        has_pii = pii_prob >= 0.5
        needs_tools = tool_prob >= 0.5

        target, fallbacks, reason, confidence = self._select_target(
            domain=domain_label,
            difficulty=diff_label,
            jailbreak=is_jailbreak,
            pii=has_pii,
            needs_tools=needs_tools,
            available=request.available_models,
            preferences=request.preferences,
            context_len=request.context_len,
        )

        return ClassifyResponse(
            decision=Decision(target=target, fallbacks=fallbacks, reason=reason, confidence=confidence),
            domain=domain_label,
            difficulty=diff_label if diff_label in ("easy", "medium", "hard") else None,
            language=lang_label,
            route=f"{domain_label}_{diff_label}".replace(" ", "_"),
            jailbreak=is_jailbreak,
            pii=has_pii,
            tool_calling=needs_tools,
            classifier_ms=elapsed_ms,
        )

    @staticmethod
    def _select_target(
        *,
        domain: str,
        difficulty: str,
        jailbreak: bool,
        pii: bool,
        needs_tools: bool,
        available: list,
        preferences,
        context_len: int,
    ) -> tuple[str, list[str], str, float]:
        """Rule-based selector. First match wins.

        Policy on jailbreak: the v1 classifier head has known false-positive
        weaknesses (e.g., flags "tell me a joke about cats" at 0.999, flags
        Spanish prompts at 0.96). We surface the signal in the trace header
        so users can act on it, but we DO NOT auto-reject. The user's
        downstream model has its own safety alignment; routing-time policy
        belongs to the user.

        To opt back into local rejection, run with SLANCHA_REJECT_JAILBREAK=1
        (TODO Phase 1.1) or override _select_target in a fork.
        """
        # Note: jailbreak flag is reported via the trace header; not used as
        # a hard gate here. See module docstring.

        # No local options → escalate (or reject)
        if not available:
            if preferences.escalation_allowed:
                return (
                    "cloud:openai:gpt-5.4-mini",
                    [],
                    "no local models available — escalating per preferences",
                    0.5,
                )
            return (
                "cloud:reject:no-local",
                [],
                "no local models available + escalation_allowed=false",
                0.4,
            )

        # Context overflow
        max_local_ctx = max(m.ctx_window for m in available)
        if context_len > max_local_ctx:
            if preferences.escalation_allowed:
                return (
                    "cloud:openai:gpt-5.4-mini",
                    [f"local:{m.backend}:{m.id}" for m in available],
                    f"context {context_len} > local max {max_local_ctx}",
                    0.7,
                )
            m = available[0]
            return (
                f"local:{m.backend}:{m.id}",
                [],
                f"context {context_len} > local max {max_local_ctx} but escalation disabled — clamping",
                0.4,
            )

        cap = _DOMAIN_TO_CAP.get(domain, "general")

        def _fallback_list(picked) -> list[str]:
            return [f"local:{a.backend}:{a.id}" for a in available if a is not picked]

        # Tool-calling: route to a tool_use-capable model unless the domain
        # itself overrides (math/CS reasoning beats the tool-head false
        # positives on `dy/dx = 3x^2` and `def f():`). Mirrors slancha-api.
        if needs_tools and domain not in _DOMAIN_PRECEDENCE_OVER_TOOLS:
            tool_models = [m for m in available if "tool_use" in m.capabilities]
            if tool_models:
                m = tool_models[0]
                return (
                    f"local:{m.backend}:{m.id}",
                    _fallback_list(m),
                    f"needs_tools=True, domain={domain} — tool-capable model preferred",
                    0.8,
                )

        # Coding domain → coder if available
        if cap == "coding":
            coders = [m for m in available if "coding" in m.capabilities]
            if coders:
                m = coders[0]
                return (
                    f"local:{m.backend}:{m.id}",
                    _fallback_list(m),
                    f"domain={domain} (coding) — coding-capable model preferred",
                    0.85,
                )

        # Math / physics / chemistry domain → reasoning-capable preferred
        # (these are STEM-reasoning, not coding). Falls through to non-coder
        # generalist if no hard-capable model exists.
        if cap == "math":
            hard = [m for m in available if "hard" in m.capabilities]
            if hard:
                m = hard[0]
                return (
                    f"local:{m.backend}:{m.id}",
                    _fallback_list(m),
                    f"domain={domain} (STEM reasoning) — hard-capable model preferred",
                    0.8,
                )

        # Hard difficulty (any non-coding domain not already routed) → hard
        if difficulty == "hard":
            hard = [m for m in available if "hard" in m.capabilities]
            if hard:
                m = hard[0]
                return (
                    f"local:{m.backend}:{m.id}",
                    _fallback_list(m),
                    "difficulty=hard — hard-capable model preferred",
                    0.8,
                )

        # General domains (biology, history, law, ...) — prefer a non-coding
        # generalist over a coder if any exists, so non-CS prompts don't
        # collapse to the coding model just because it happens to be
        # available[0]. This is the per-prompt diversity fix; without it
        # every general prompt routes to whichever model probe yielded first.
        non_coders = [m for m in available if "coding" not in m.capabilities]
        if non_coders:
            m = non_coders[0]
            return (
                f"local:{m.backend}:{m.id}",
                _fallback_list(m),
                f"domain={domain}, difficulty={difficulty} — non-coding generalist preferred",
                0.7,
            )

        # Last resort: only coders are available
        m = available[0]
        return (
            f"local:{m.backend}:{m.id}",
            _fallback_list(m),
            f"domain={domain}, difficulty={difficulty} — first available (only coders present)",
            0.55,
        )
