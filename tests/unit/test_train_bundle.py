"""Smoke test: bundle pipeline emits train.jsonl + val.jsonl with expected counts."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from slancha_local.train.bundle import build_train_bundle


def _trace(rid: str, route: str, prompt: str, response: str, *, consent: bool = True) -> dict:
    emb = np.random.default_rng(hash(rid) & 0xFFFF).standard_normal(512).astype(np.float32)
    return {
        "request_id": rid,
        "ts": "2026-05-09T10:00:00.000Z",
        "mode": "local",
        "embedding_b64": base64.b64encode(emb.tobytes()).decode(),
        "classifier": {
            "domain": "general",
            "difficulty": "easy",
            "language": "en",
            "jailbreak": False,
            "pii": False,
            "tool_calling": False,
            "route": route,
            "confidence": 0.7,
        },
        "decision": {"target": "local:ollama:qwen3:8b", "fallbacks": [], "reason": "r"},
        "execution": {
            "executed_target": "local:ollama:qwen3:8b",
            "tokens_in": 5,
            "tokens_out": 5,
            "latency_ms": 100,
            "status": "ok",
        },
        "prompt": prompt,
        "response": response,
        "feedback": None,
        "consent_at_capture": consent,
        "schema_version": 1,
    }


def test_bundle_emits_train_and_val(tmp_path: Path):
    traces = [_trace(f"r{i}", "general_qa", f"prompt {i}", f"reply {i}") for i in range(20)]
    stats = build_train_bundle(traces, out_dir=tmp_path / "out", val_fraction=0.1, cluster=False)
    assert stats.train_count + stats.val_count == 20
    assert stats.val_count >= 1
    assert (tmp_path / "out" / "train.jsonl").exists()
    assert (tmp_path / "out" / "val.jsonl").exists()


def test_bundle_skips_no_consent(tmp_path: Path):
    traces = [_trace(f"r{i}", "x", "p", "r", consent=False) for i in range(5)]
    stats = build_train_bundle(traces, out_dir=tmp_path / "out", cluster=False)
    assert stats.train_count == 0
    assert stats.skipped_no_consent == 5


def test_bundle_skips_missing_response(tmp_path: Path):
    traces = [_trace("r0", "x", "p", "")]
    stats = build_train_bundle(traces, out_dir=tmp_path / "out", cluster=False)
    assert stats.train_count == 0
    assert stats.skipped_no_response == 1


def test_bundle_jsonl_records_chat_format(tmp_path: Path):
    traces = [_trace(f"r{i}", "x", f"hi {i}", f"hello {i}") for i in range(8)]
    build_train_bundle(traces, out_dir=tmp_path / "out", cluster=False, val_fraction=0.0)
    line = json.loads((tmp_path / "out" / "train.jsonl").read_text().splitlines()[0])
    assert "messages" in line
    assert line["messages"][0]["role"] == "user"
    assert line["messages"][1]["role"] == "assistant"
    assert "metadata" in line
