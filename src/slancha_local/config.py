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
    capability_ttl_s: int = Field(default=30)

    # Sharing / opt-in
    share_prompts: bool = Field(default=False)
    share_traces: bool = Field(default=False)

    # Storage
    traces_root: Path = Field(default=Path.home() / ".slancha" / "traces")

    # Server
    bind_host: str = Field(default="127.0.0.1")
    bind_port: int = Field(default=8000)
