"""Settings — pydantic-settings, env-var-driven."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# vLLM-convention model-serving port. The slancha-mesh tailnet ACL opens
# `tag:gateway → tag:specialist:8003,8004` ONLY — a node that binds/advertises
# the standalone default :8000 registers but is un-routable from the gateway
# (slancha-mesh#8). When mesh registration is enabled and the operator has not
# pinned a port, slancha-local serves + advertises this ACL-permitted port.
MESH_ACL_MODEL_PORT = 8003

# The set of ports the slancha-mesh `tag:gateway → tag:specialist` ACL permits.
# A node whose advertised port is outside this set is registered-but-unroutable.
MESH_ACL_MODEL_PORTS = frozenset({8003, 8004})

# Env var that pins the bind port. Presence (not value) is the signal that the
# operator chose a port explicitly, so the mesh default-to-8003 must not override
# it. Mirrors pydantic-settings' env_prefix + field-name convention.
BIND_PORT_ENV = "SLANCHA_BIND_PORT"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SLANCHA_", env_file=".env", extra="ignore")

    # Classifier
    classifier_kind: Literal["local", "cloud", "rules"] = Field(default="local")
    api_base_url: str = Field(default="https://api.slancha.ai")
    api_key: str | None = Field(default=None)
    classifier_timeout_s: float = Field(default=2.0)

    # Backends
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    ollama_enabled: bool = Field(default=True)
    llamacpp_base_url: str = Field(default="http://127.0.0.1:8080")
    llamacpp_enabled: bool = Field(default=True)
    vllm_base_url: str = Field(default="http://127.0.0.1:8000")
    vllm_enabled: bool = Field(default=False)
    mlx_base_url: str = Field(default="http://127.0.0.1:8081")
    mlx_enabled: bool = Field(default=False)
    lmstudio_base_url: str = Field(default="http://127.0.0.1:1234")
    lmstudio_enabled: bool = Field(default=False)
    generic_openai_base_url: str | None = Field(default=None)  # opt-in via env
    capability_ttl_s: int = Field(default=30)

    # Cloud router — OpenRouter (NON-LOCAL, opt-in)
    # Defaults disabled to preserve ADR-002 (zero non-loopback calls on default install).
    openrouter_base_url: str = Field(default="https://openrouter.ai/api")
    openrouter_enabled: bool = Field(default=False)
    openrouter_api_key: str | None = Field(default=None)
    # Optional but encouraged by OpenRouter for analytics + rate limit headroom:
    openrouter_referer: str = Field(default="https://slancha.ai")
    openrouter_app_title: str = Field(default="slancha-local")

    # Multimodal — image generation (ComfyUI)
    comfy_base_url: str = Field(default="http://127.0.0.1:8188")
    comfy_enabled: bool = Field(default=False)  # opt-in; off by default
    comfy_default_workflow: str | None = Field(default=None)  # path to JSON workflow template
    comfy_poll_interval_s: float = Field(default=1.0)
    comfy_timeout_s: float = Field(default=300.0)

    # Sharing / opt-in
    share_prompts: bool = Field(default=False)
    share_traces: bool = Field(default=False)

    # Storage
    traces_root: Path = Field(default=Path.home() / ".slancha" / "traces")

    # Server
    bind_host: str = Field(default="127.0.0.1")
    bind_port: int = Field(default=8000)

    # Mesh — what the heartbeat advertises to the gateway over a tailnet.
    # None → auto-discover this node's MagicDNS name via `tailscale status
    # --json` (Self.DNSName). Set explicitly to override. Distinct from
    # bind_host: bind broadly (0.0.0.0 on a tailnet), advertise a routable
    # name. Env: SLANCHA_MESH_ADVERTISE_HOST.
    mesh_advertise_host: str | None = Field(default=None)

    # Mesh pull-discovery — when enabled, the router builds (part of) its
    # routing table by walking the tailnet for tag:specialist peers and
    # pulling each node's /models, registering them as remote backends. The
    # consume side of slancha-mesh pull discovery; default off (zero behavior
    # change). Env: SLANCHA_MESH_DISCOVERY_ENABLED / SLANCHA_MESH_DISCOVERY_PORT.
    mesh_discovery_enabled: bool = Field(default=False)
    mesh_discovery_port: int = Field(default=8088)

    @staticmethod
    def mesh_registration_enabled() -> bool:
        """True when this node is opted in to mesh-registry heartbeats.

        Registration is gated entirely by SLANCHA_MESH_REGISTRY_URL being set
        and non-empty — the same signal `mesh_lifespan` / `MeshHeartbeatLoop`
        use. Kept here (not a Settings field) because the registry URL is read
        straight from the environment at lifespan time, never via Settings.
        """
        return bool(os.environ.get("SLANCHA_MESH_REGISTRY_URL"))

    def effective_bind_port(self) -> int:
        """Port the proxy should bind AND advertise to the mesh.

        Standalone (no mesh registration): the historical default :8000,
        unchanged. Under mesh registration with no explicit SLANCHA_BIND_PORT:
        default to the ACL-permitted model port :8003 so the advertised
        node_url is reachable by `tag:gateway → tag:specialist` (slancha-mesh#8).
        An explicit SLANCHA_BIND_PORT always wins — the operator's override is
        never silently moved.
        """
        if self.mesh_registration_enabled() and BIND_PORT_ENV not in os.environ:
            return MESH_ACL_MODEL_PORT
        return self.bind_port
