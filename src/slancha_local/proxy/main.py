"""FastAPI app factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from slancha_local.backends.ollama import OllamaBackend
from slancha_local.backends.registry import BackendRegistry
from slancha_local.capability.probe import CapabilityProbe
from slancha_local.classifier_client.base import ClassifierClient
from slancha_local.classifier_client.cloud import CloudClassifierClient
from slancha_local.classifier_client.rules_fallback import RulesFallbackClassifier
from slancha_local.config import Settings
from slancha_local.proxy import chat, decisions, health
from slancha_local.proxy.middleware import DecisionTraceHeaderMiddleware
from slancha_local.telemetry.local_writer import LocalTraceWriter

logger = logging.getLogger(__name__)


def _build_classifier(settings: Settings) -> ClassifierClient:
    if settings.classifier_kind == "local":
        try:
            from slancha_local.classifier.local import LocalClassifier

            return LocalClassifier()
        except (ImportError, OSError, FileNotFoundError, RuntimeError) as e:
            # treelite/libomp/asset issues — degrade to rules fallback so the
            # proxy stays usable. doctor surfaces this state.
            logger.warning(
                "local classifier unavailable (%s) — falling back to rules. "
                "On macOS, run `brew install libomp` to enable the local "
                "classifier, then restart slancha.",
                e,
            )
            return RulesFallbackClassifier()
    if settings.classifier_kind == "cloud":
        return CloudClassifierClient(
            base_url=settings.api_base_url,
            api_key=settings.api_key,
            timeout_s=settings.classifier_timeout_s,
        )
    return RulesFallbackClassifier()


def build_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="slancha-local", version="0.0.1")
    app.add_middleware(DecisionTraceHeaderMiddleware)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(decisions.router)

    ollama = OllamaBackend(base_url=settings.ollama_base_url)
    registry = BackendRegistry([ollama])
    probe = CapabilityProbe([ollama], ttl_s=settings.capability_ttl_s)
    classifier = _build_classifier(settings)
    trace_writer = LocalTraceWriter(root=settings.traces_root)

    app.state.settings = settings
    app.state.registry = registry
    app.state.probe = probe
    app.state.classifier = classifier
    app.state.trace_writer = trace_writer
    return app


app = build_app()
