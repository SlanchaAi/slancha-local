"""FastAPI app factory."""

from __future__ import annotations

import logging
import os
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI

from slancha_local.backends.comfy import ComfyBackend
from slancha_local.backends.llamacpp import LlamaCppBackend
from slancha_local.backends.ollama import OllamaBackend
from slancha_local.backends.openai_compat import (
    GenericOpenAIBackend,
    LMStudioBackend,
    MLXBackend,
    OpenRouterBackend,
    VLLMBackend,
)
from slancha_local.backends.registry import BackendRegistry
from slancha_local.capability.probe import CapabilityProbe
from slancha_local.classifier_client.base import ClassifierClient
from slancha_local.classifier_client.cloud import CloudClassifierClient
from slancha_local.classifier_client.rules_fallback import RulesFallbackClassifier
from slancha_local.config import Settings
from slancha_local.mesh.heartbeat import (
    REGISTRY_URL_ENV,
    LoadedSpecialist,
    MeshHeartbeatLoop,
    build_node_url,
    resolve_advertise_host,
    specialists_from_models,
)
from slancha_local.proxy import chat, decisions, health, images, models_endpoint
from slancha_local.proxy.mesh_auth import MeshAuthMiddleware
from slancha_local.proxy.middleware import DecisionTraceHeaderMiddleware
from slancha_local.proxy.usage_sidecar import UsageSidecar
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


def build_heartbeat_loop(settings: Settings, probe: CapabilityProbe) -> MeshHeartbeatLoop:
    """Construct the mesh heartbeat loop for this node.

    Opt-in: with no SLANCHA_MESH_REGISTRY_URL the loop is disabled (never
    starts a thread) and the tailnet/MagicDNS resolution is skipped — no
    `tailscale` subprocess on a default boot. When enabled, the advertised
    node_url is the tailnet MagicDNS name (or SLANCHA_MESH_ADVERTISE_HOST),
    falling back to the bind host for non-tailnet dev. `catalog_fn` reads the
    probe's cached snapshot synchronously (it runs in a daemon thread and
    cannot await).
    """
    registry_url = os.environ.get(REGISTRY_URL_ENV)
    advertise_host = (
        resolve_advertise_host(settings.mesh_advertise_host) if registry_url else None
    )
    node_url = build_node_url(
        advertise_host=advertise_host,
        bind_host=settings.bind_host,
        bind_port=settings.bind_port,
    )

    def _catalog_fn() -> list[LoadedSpecialist]:
        catalog = probe.cached()
        if catalog is None:
            return []
        models = [m for cap in catalog.healthy_backends for m in cap.models]
        return specialists_from_models(models)

    return MeshHeartbeatLoop(
        registry_url=registry_url,
        node_url=node_url,
        friendly_name=socket.gethostname(),
        catalog_fn=_catalog_fn,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start/stop the mesh heartbeat alongside the proxy.

    No-op unless SLANCHA_MESH_REGISTRY_URL is set. When enabled, warm the
    capability cache once so the first heartbeat carries real loaded_models,
    then run the daemon thread for the app's lifetime.
    """
    loop = build_heartbeat_loop(app.state.settings, app.state.probe)
    if loop.enabled:
        await app.state.probe.refresh()
        loop.start()
    app.state.mesh_heartbeat = loop
    try:
        yield
    finally:
        loop.stop()


def build_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="slancha-local", version="0.0.1", lifespan=_lifespan)
    # Order matters: last-added runs OUTERMOST. MeshAuthMiddleware must
    # gate inbound BEFORE DecisionTraceHeaderMiddleware records trace.
    app.add_middleware(DecisionTraceHeaderMiddleware)
    app.add_middleware(MeshAuthMiddleware)
    # UsageSidecar: stash on app.state so chat handlers schedule BackgroundTasks
    # against the single shared instance (durable buffer + retry + DLQ).
    app.state.usage_sidecar = UsageSidecar()
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(decisions.router)
    app.include_router(models_endpoint.router)
    app.include_router(images.router)

    backends_list: list = []
    if settings.ollama_enabled:
        backends_list.append(OllamaBackend(base_url=settings.ollama_base_url))
    if settings.llamacpp_enabled:
        backends_list.append(LlamaCppBackend(base_url=settings.llamacpp_base_url))
    if settings.vllm_enabled:
        backends_list.append(VLLMBackend(base_url=settings.vllm_base_url))
    if settings.mlx_enabled:
        backends_list.append(MLXBackend(base_url=settings.mlx_base_url))
    if settings.lmstudio_enabled:
        backends_list.append(LMStudioBackend(base_url=settings.lmstudio_base_url))
    if settings.generic_openai_base_url:
        backends_list.append(GenericOpenAIBackend(base_url=settings.generic_openai_base_url))
    if settings.openrouter_enabled and settings.openrouter_api_key:
        backends_list.append(
            OpenRouterBackend(
                base_url=settings.openrouter_base_url,
                api_key=settings.openrouter_api_key,
                extra_headers={
                    "HTTP-Referer": settings.openrouter_referer,
                    "X-Title": settings.openrouter_app_title,
                },
            )
        )
    registry = BackendRegistry(backends_list)
    probe = CapabilityProbe(backends_list, ttl_s=settings.capability_ttl_s)
    classifier = _build_classifier(settings)
    trace_writer = LocalTraceWriter(root=settings.traces_root)

    app.state.settings = settings
    app.state.registry = registry
    app.state.probe = probe
    app.state.classifier = classifier
    app.state.trace_writer = trace_writer

    if settings.comfy_enabled:
        app.state.image_backend = ComfyBackend(
            base_url=settings.comfy_base_url,
            default_workflow=settings.comfy_default_workflow,
            poll_interval_s=settings.comfy_poll_interval_s,
            timeout_s=settings.comfy_timeout_s,
        )
    return app


app = build_app()
