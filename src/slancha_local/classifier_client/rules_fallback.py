"""Rule-based fallback classifier for when other classifiers are unavailable."""

from __future__ import annotations

import re

from slancha_local.classifier_client.base import ClassifierClient
from slancha_local.classifier_client.models import ClassifyRequest, ClassifyResponse, Decision

_CODE_KEYWORDS = re.compile(
    r"\b(python|javascript|typescript|rust|golang|c\+\+|function|class|"
    r"refactor|debug|stacktrace|implement|algorithm|fibonacci|recursion|"
    r"compile|bug|exception|traceback)\b",
    re.IGNORECASE,
)
_JB_KEYWORDS = re.compile(
    r"\b(ignore (?:all )?(?:previous|prior) instructions|"
    r"jailbreak|DAN |pretend you are|forget you are an AI|"
    r"act as (?:a )?linux shell|dump (?:your )?system prompt)\b",
    re.IGNORECASE,
)
_DEFAULT_CLOUD_TARGET = "cloud:openai:gpt-5.4-mini"


class RulesFallbackClassifier(ClassifierClient):
    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        prompt = request.prompt or ""
        models = request.available_models

        if _JB_KEYWORDS.search(prompt):
            return ClassifyResponse(
                decision=Decision(
                    target=_DEFAULT_CLOUD_TARGET,
                    fallbacks=[],
                    reason="rules-fallback: jailbreak keyword detected — escalating",
                    confidence=0.7,
                ),
                jailbreak=True,
                route="jailbreak_rejected",
            )

        if not models:
            return ClassifyResponse(
                decision=Decision(
                    target=_DEFAULT_CLOUD_TARGET,
                    fallbacks=[],
                    reason="rules-fallback: no local models available",
                    confidence=0.5,
                ),
                route="cloud_overflow",
            )

        max_local_ctx = max(m.ctx_window for m in models)
        if request.context_len > max_local_ctx:
            return ClassifyResponse(
                decision=Decision(
                    target=_DEFAULT_CLOUD_TARGET,
                    fallbacks=[f"local:{m.backend}:{m.id}" for m in models],
                    reason=(
                        f"rules-fallback: context {request.context_len} > "
                        f"local max {max_local_ctx}"
                    ),
                    confidence=0.6,
                ),
                route="cloud_overflow",
            )

        if _CODE_KEYWORDS.search(prompt):
            coders = [m for m in models if "coding" in m.capabilities]
            if coders:
                m = coders[0]
                return ClassifyResponse(
                    decision=Decision(
                        target=f"local:{m.backend}:{m.id}",
                        fallbacks=[f"local:{models[0].backend}:{models[0].id}"],
                        reason="rules-fallback: coding keyword + coding-capable local",
                        confidence=0.7,
                    ),
                    route="code_general",
                    domain="coding",
                )

        m = models[0]
        return ClassifyResponse(
            decision=Decision(
                target=f"local:{m.backend}:{m.id}",
                fallbacks=[_DEFAULT_CLOUD_TARGET],
                reason="rules-fallback: default to first local",
                confidence=0.55,
            ),
            route="general_qa",
        )
