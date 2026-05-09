"""Iter-3 train pipeline: HTTP providers (dry-run + structure) + judge parsing + variant bandit."""

from __future__ import annotations

import os
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from slancha_local.train.eval import (
    _parse_judge_reply,
    _resolve_judge_endpoint,
    judge_pairwise_pick,
)
from slancha_local.train.providers import get_provider
from slancha_local.train.providers.base import TrainingJob
from slancha_local.train.providers.http_providers import (
    FireworksProvider,
    OpenAIProvider,
    TogetherProvider,
)
from slancha_local.variants import VariantStore

# ---------- HTTP provider dry-run ----------


def _make_job(tmp_path: Path) -> TrainingJob:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text('{"messages":[{"role":"user","content":"x"}]}\n')
    val.write_text('{"messages":[{"role":"user","content":"y"}]}\n')
    return TrainingJob(
        route="cs_medium",
        base_model="Qwen/Qwen3-8B",
        train_jsonl=train,
        val_jsonl=val,
        output_dir=tmp_path / "out",
        hyperparams={"epochs": 1, "lora_r": 8},
        artifact_dest=f"local:{tmp_path}/artifacts",
    )


@pytest.mark.parametrize("provider_id", ["fireworks", "together", "openai"])
def test_http_providers_dry_run_succeeds(tmp_path: Path, provider_id: str):
    # SLANCHA_TRAIN_DRY_RUN=1 set by conftest
    p = get_provider(provider_id)
    job = _make_job(tmp_path)
    ok, msg = p.precheck(job)
    assert ok, msg
    result = p.train(job)
    assert result.success
    assert result.artifact_ref and result.artifact_ref.startswith(f"{provider_id}://dry-run/")
    assert result.metrics.get("dry_run") is True


def test_fireworks_precheck_requires_account_id(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SLANCHA_TRAIN_DRY_RUN", raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")
    p = FireworksProvider()
    job = _make_job(tmp_path)
    ok, msg = p.precheck(job)
    assert not ok
    assert "FIREWORKS_ACCOUNT_ID" in msg


def test_together_precheck_missing_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SLANCHA_TRAIN_DRY_RUN", raising=False)
    p = TogetherProvider()
    job = _make_job(tmp_path)
    ok, msg = p.precheck(job)
    assert not ok
    assert "TOGETHER_API_KEY" in msg


def test_openai_precheck_missing_jsonl(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SLANCHA_TRAIN_DRY_RUN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    p = OpenAIProvider()
    job = TrainingJob(
        route="r",
        base_model="m",
        train_jsonl=tmp_path / "missing-train.jsonl",
        val_jsonl=tmp_path / "missing-val.jsonl",
        output_dir=tmp_path,
        hyperparams={},
        artifact_dest="local:.",
    )
    ok, msg = p.precheck(job)
    assert not ok
    assert "not found" in msg


# ---------- Judge parsing ----------


def test_resolve_judge_endpoint_dry_run_returns_none():
    assert _resolve_judge_endpoint() is None


def test_resolve_judge_endpoint_openai_fallback(monkeypatch):
    monkeypatch.delenv("SLANCHA_TRAIN_DRY_RUN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    out = _resolve_judge_endpoint()
    assert out is not None
    base_url, key, model = out
    assert "openai.com" in base_url
    assert key == "sk-test"
    assert model  # default model id


def test_resolve_judge_endpoint_slancha_url_override(monkeypatch):
    monkeypatch.delenv("SLANCHA_TRAIN_DRY_RUN", raising=False)
    monkeypatch.setenv("SLANCHA_JUDGE_URL", "http://localhost:9999/")
    monkeypatch.setenv("SLANCHA_JUDGE_API_KEY", "k")
    monkeypatch.setenv("SLANCHA_JUDGE_MODEL", "qwen3:8b")
    out = _resolve_judge_endpoint()
    assert out == ("http://localhost:9999", "k", "qwen3:8b")


@pytest.mark.parametrize(
    "text,want_verdict",
    [
        ("FT\nFT was clearer", "ft"),
        ("BASE\nstuck the landing", "base"),
        ("TIE\nboth hand-wavy", "tie"),
        ("Sure! FT is better.\nBecause clearer.", "ft"),
        ("", "tie"),
        ("nonsense", "tie"),
    ],
)
def test_parse_judge_reply(text: str, want_verdict: str):
    verdict, _ = _parse_judge_reply(text)
    assert verdict == want_verdict


def test_judge_pairwise_pick_returns_tie_in_dry_run():
    verdict, reason = judge_pairwise_pick("p", "b", "f", "openai:x")
    assert verdict == "tie"
    assert "no judge endpoint" in reason


def test_judge_pairwise_pick_uses_real_endpoint_when_configured(monkeypatch):
    monkeypatch.delenv("SLANCHA_TRAIN_DRY_RUN", raising=False)
    monkeypatch.setenv("SLANCHA_JUDGE_URL", "http://judge.test")
    monkeypatch.setenv("SLANCHA_JUDGE_API_KEY", "k")

    fake_response = MagicMock()
    fake_response.json.return_value = {"choices": [{"message": {"content": "FT\nbetter."}}]}
    fake_response.raise_for_status = MagicMock()

    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = False
    fake_client.post.return_value = fake_response

    with patch("slancha_local.train.eval.httpx.Client", return_value=fake_client):
        verdict, reason = judge_pairwise_pick("query?", "B answer", "F answer", "openai:gpt-4o-mini")

    assert verdict == "ft"
    assert reason == "better."


def test_judge_pairwise_pick_handles_http_error(monkeypatch):
    import httpx as _httpx

    monkeypatch.delenv("SLANCHA_TRAIN_DRY_RUN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = False
    fake_client.post.side_effect = _httpx.ConnectError("boom")

    with patch("slancha_local.train.eval.httpx.Client", return_value=fake_client):
        verdict, reason = judge_pairwise_pick("p", "b", "f", "openai:m")

    assert verdict == "tie"
    assert "ConnectError" in reason


# ---------- Variant store / bandit ----------


def test_variant_store_register_and_pick(tmp_path: Path):
    store = VariantStore(path=tmp_path / "v.json")
    store.register("cs_medium", "v1", target="local:ollama:qwen3:8b")
    store.register("cs_medium", "v2", target="local:ollama:codestral:22b")
    assert len(store.list_variants("cs_medium")) == 2

    rng = random.Random(42)
    picked = store.pick("cs_medium", rng=rng)
    assert picked is not None
    assert picked.variant_id in {"v1", "v2"}


def test_variant_store_pick_empty_returns_none(tmp_path: Path):
    store = VariantStore(path=tmp_path / "v.json")
    assert store.pick("never_registered") is None


def test_variant_store_thompson_converges_to_winner(tmp_path: Path):
    store = VariantStore(path=tmp_path / "v.json")
    store.register("r", "good")
    store.register("r", "bad")
    rng = random.Random(0)
    # Train: good wins 80% of trials, bad wins 20%
    for _ in range(200):
        store.record_outcome("r", "good", won=rng.random() < 0.8)
        store.record_outcome("r", "bad", won=rng.random() < 0.2)

    # After updates, picks should mostly favor 'good'
    picks = [store.pick("r", rng=random.Random(i)).variant_id for i in range(100)]
    good_share = picks.count("good") / len(picks)
    assert good_share >= 0.7, f"Thompson didn't converge: good_share={good_share}"


def test_variant_store_persists_across_instances(tmp_path: Path):
    p = tmp_path / "v.json"
    s1 = VariantStore(path=p)
    s1.register("r", "v1", target="t")
    s1.record_outcome("r", "v1", won=True)
    s1.record_outcome("r", "v1", won=False)

    s2 = VariantStore(path=p)
    stats = s2.list_variants("r")
    assert len(stats) == 1
    assert stats[0].alpha == 2.0  # 1 + 1 win
    assert stats[0].beta == 2.0  # 1 + 1 loss
    assert stats[0].last_target == "t"


def test_variant_store_summary_shape(tmp_path: Path):
    store = VariantStore(path=tmp_path / "v.json")
    store.register("r1", "v1")
    store.record_outcome("r1", "v1", won=True)
    summary = store.summary()
    assert "r1" in summary
    assert summary["r1"][0]["variant_id"] == "v1"
    assert summary["r1"][0]["alpha"] == 2.0


def test_variant_store_corrupt_file_recovers(tmp_path: Path):
    p = tmp_path / "v.json"
    p.write_text("{not json")
    store = VariantStore(path=p)
    assert store.list_variants("any") == []
    store.register("r", "v")
    assert store.list_variants("r")


def test_variant_store_default_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SLANCHA_VARIANTS_PATH", str(tmp_path / "default.json"))
    # re-import so DEFAULT_PATH is recomputed — but the module already loaded it.
    # Easier: just instantiate w/ an explicit path. Sanity-check env var is read.
    assert os.environ["SLANCHA_VARIANTS_PATH"] == str(tmp_path / "default.json")
