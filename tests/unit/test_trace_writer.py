"""LocalTraceWriter consent gate + JSONL append."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from slancha_local.telemetry.local_writer import LocalTraceWriter
from slancha_local.telemetry.schema import (
    ClassifierBlock,
    DecisionBlock,
    ExecutionBlock,
    Trace,
)


def make_trace(consent: bool = False) -> Trace:
    return Trace(
        request_id=str(uuid.uuid4()),
        ts="2026-05-09T14:23:11.482Z",
        mode="local",
        embedding_b64="AAAA",
        classifier=ClassifierBlock(
            domain="general",
            difficulty="medium",
            language="en",
            jailbreak=False,
            pii=False,
            tool_calling=False,
            route="general_qa",
            confidence=0.87,
        ),
        decision=DecisionBlock(target="local:ollama:qwen3:8b", fallbacks=[], reason="rules"),
        execution=ExecutionBlock(
            executed_target="local:ollama:qwen3:8b",
            tokens_in=10,
            tokens_out=5,
            latency_ms=300,
            status="ok",
        ),
        consent_at_capture=consent,
    )


def test_writer_appends_jsonl(tmp_path: Path):
    writer = LocalTraceWriter(root=tmp_path)
    writer.write(make_trace())
    writer.write(make_trace())
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)


def test_writer_consent_false_strips_prompt_and_response(tmp_path: Path):
    t = make_trace(consent=False)
    t.prompt = "secret prompt"
    t.response = "secret reply"
    writer = LocalTraceWriter(root=tmp_path)
    writer.write(t)
    line = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[0])
    assert line["prompt"] is None
    assert line["response"] is None


def test_writer_consent_true_keeps_prompt_and_response(tmp_path: Path):
    t = make_trace(consent=True)
    t.prompt = "shared prompt"
    t.response = "shared reply"
    writer = LocalTraceWriter(root=tmp_path)
    writer.write(t)
    line = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[0])
    assert line["prompt"] == "shared prompt"
    assert line["response"] == "shared reply"


def test_writer_rejects_system_paths():
    with pytest.raises(ValueError):
        LocalTraceWriter(root="/etc/slancha")
