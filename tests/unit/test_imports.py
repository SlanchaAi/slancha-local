"""Smoke test: every public module imports cleanly."""

from __future__ import annotations


def test_top_level_import():
    import slancha_local

    assert slancha_local.__version__ == "0.0.1"


def test_proxy_imports():
    from slancha_local.proxy.main import build_app
    from slancha_local.proxy.middleware import format_trace
    from slancha_local.proxy.models import ChatCompletionRequest

    assert callable(build_app)
    assert callable(format_trace)
    assert ChatCompletionRequest is not None


def test_classifier_clients_import():
    from slancha_local.classifier_client import (
        CloudClassifierClient,
        RulesFallbackClassifier,
    )

    assert CloudClassifierClient is not None
    assert RulesFallbackClassifier is not None


def test_backends_import():
    from slancha_local.backends import BackendRegistry, OllamaBackend

    assert OllamaBackend is not None
    assert BackendRegistry is not None


def test_telemetry_import():
    from slancha_local.telemetry import LocalTraceWriter, Trace

    assert LocalTraceWriter is not None
    assert Trace is not None


def test_cli_import():
    from slancha_local.cli import app

    assert app is not None
