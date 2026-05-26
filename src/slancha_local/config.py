"""Settings — pydantic-settings, env-var-driven."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
