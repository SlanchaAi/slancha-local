"""BackendRegistry parser + lookup."""

from __future__ import annotations

import pytest

from slancha_local.backends.base import BackendCapability
from slancha_local.backends.ollama import OllamaBackend
from slancha_local.backends.registry import BackendRegistry


def test_parse_target_local_three_parts():
    s, b, m = BackendRegistry.parse_target("local:ollama:qwen3:8b")
    assert s == "local"
    assert b == "ollama"
    assert m == "qwen3:8b"


def test_parse_target_cloud():
    s, b, m = BackendRegistry.parse_target("cloud:openai:gpt-5.4-mini")
    assert s == "cloud"
    assert b == "openai"
    assert m == "gpt-5.4-mini"


def test_parse_target_malformed_returns_none():
    s, b, m = BackendRegistry.parse_target("garbage")
    assert s is None and b is None and m is None


def test_registry_by_id():
    ollama = OllamaBackend(base_url="http://x")
    reg = BackendRegistry([ollama])
    assert reg.by_id("ollama") is ollama


def test_registry_unknown_raises():
    reg = BackendRegistry([])
    with pytest.raises(KeyError):
        reg.by_id("missing")
