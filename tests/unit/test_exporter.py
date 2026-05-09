"""Trace bundle exporter: PII redaction + tarball structure."""

from __future__ import annotations

import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from slancha_local.telemetry.exporter import _redact, export_bundle


def test_redact_email():
    assert _redact("contact me at john@example.com please") == "contact me at <EMAIL> please"


def test_redact_ssn():
    assert _redact("SSN 123-45-6789") == "SSN <SSN>"


def test_redact_credit_card():
    out = _redact("card 4111-1111-1111-1111 expires soon")
    assert "<CARD>" in out
    assert "4111" not in out


def test_redact_openai_key():
    out = _redact("my key is sk-proj-aBc123dEfGhIjKlMnOpQrStUvWxYz")
    assert "<API_KEY>" in out
    assert "sk-proj" not in out


def test_redact_phone():
    out = _redact("call me at +1-555-123-4567")
    assert "<PHONE>" in out


def test_redact_none_safe():
    assert _redact(None) is None


def test_redact_clean_text_unchanged():
    text = "This is a normal sentence about cats."
    assert _redact(text) == text


def _write_trace(root: Path, ts: str, prompt: str, response: str) -> None:
    (root / "2026-05-09.jsonl").parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "request_id": "req-1",
            "ts": ts,
            "mode": "local",
            "embedding_b64": "AAAA",
            "classifier": {
                "domain": "general",
                "difficulty": "easy",
                "language": "en",
                "jailbreak": False,
                "pii": False,
                "tool_calling": False,
                "route": "general_qa",
                "confidence": 0.7,
            },
            "decision": {"target": "local:ollama:qwen3:8b", "fallbacks": [], "reason": "ok"},
            "execution": {
                "executed_target": "local:ollama:qwen3:8b",
                "tokens_in": 1,
                "tokens_out": 1,
                "latency_ms": 100,
                "status": "ok",
            },
            "prompt": prompt,
            "response": response,
            "feedback": None,
            "consent_at_capture": True,
            "schema_version": 1,
        }
    )
    target = root / f"{ts[:10]}.jsonl"
    with open(target, "a") as f:
        f.write(line + "\n")


def test_export_bundle_creates_tarball(tmp_path: Path):
    traces_root = tmp_path / "traces"
    traces_root.mkdir()
    _write_trace(traces_root, "2026-05-09T12:00:00+00:00", "hi there", "hello back")
    _write_trace(traces_root, "2026-05-09T12:01:00+00:00", "another", "reply")

    out = tmp_path / "bundle.tar.gz"
    n, sz = export_bundle(traces_root=traces_root, out_path=out, since=None)
    assert n == 2
    assert sz > 0
    assert out.exists()

    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
        assert any(n.endswith("manifest.json") for n in names)
        assert any(n.endswith("traces.jsonl") for n in names)


def test_export_bundle_redacts_pii(tmp_path: Path):
    traces_root = tmp_path / "traces"
    traces_root.mkdir()
    _write_trace(
        traces_root,
        "2026-05-09T12:00:00+00:00",
        "email me at user@example.com about SSN 123-45-6789",
        "noted, contact at user@example.com confirmed",
    )

    out = tmp_path / "bundle.tar.gz"
    export_bundle(traces_root=traces_root, out_path=out, since=None)
    with tarfile.open(out, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith("traces.jsonl")]
        assert len(members) == 1
        f = tar.extractfile(members[0])
        assert f is not None
        content = f.read().decode()

    assert "<EMAIL>" in content
    assert "<SSN>" in content
    assert "user@example.com" not in content
    assert "123-45-6789" not in content


def test_export_bundle_since_filter(tmp_path: Path):
    traces_root = tmp_path / "traces"
    traces_root.mkdir()
    _write_trace(traces_root, "2026-05-08T10:00:00+00:00", "old", "old reply")
    _write_trace(traces_root, "2026-05-09T12:00:00+00:00", "new", "new reply")

    out = tmp_path / "bundle.tar.gz"
    n, _ = export_bundle(
        traces_root=traces_root,
        out_path=out,
        since=datetime(2026, 5, 9, tzinfo=UTC),
    )
    assert n == 1


def test_export_bundle_empty_dir(tmp_path: Path):
    traces_root = tmp_path / "traces"
    traces_root.mkdir()
    out = tmp_path / "bundle.tar.gz"
    n, _ = export_bundle(traces_root=traces_root, out_path=out, since=None)
    assert n == 0
    assert out.exists()
