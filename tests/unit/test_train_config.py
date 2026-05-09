"""TrainConfig: TOML round-trip + provider resolution + eval scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from slancha_local.train.config import example_config, load_config
from slancha_local.train.eval import (
    EvalReport,
    HeuristicResult,
    evaluate,
    heuristic_check,
)
from slancha_local.train.providers import get_provider
from slancha_local.train.providers.axolotl_provider import AxolotlProvider
from slancha_local.train.providers.http_providers import (
    FireworksProvider,
    OpenAIProvider,
    TogetherProvider,
)


def test_example_config_loads(tmp_path: Path):
    cfg_path = tmp_path / "slancha-train.toml"
    cfg_path.write_text(example_config())
    cfg = load_config(cfg_path)
    assert cfg.cadence == "weekly"
    assert len(cfg.routes) == 2
    assert cfg.routes[0].route == "computer_science_medium"
    assert cfg.routes[0].provider == "axolotl"
    assert cfg.routes[1].hyperparams.lora_r == 32  # creative override


def test_get_provider_resolves_known_ids():
    assert isinstance(get_provider("axolotl"), AxolotlProvider)
    assert isinstance(get_provider("fireworks"), FireworksProvider)
    assert isinstance(get_provider("together"), TogetherProvider)
    assert isinstance(get_provider("openai"), OpenAIProvider)


def test_get_provider_rejects_unknown():
    with pytest.raises(ValueError):
        get_provider("bogus")


def test_heuristic_empty_response_fails():
    r = heuristic_check("", ["non_empty_response"])
    assert not r.passed
    assert "empty_response" in r.reasons


def test_heuristic_valid_string_passes():
    r = heuristic_check("Hello world", ["non_empty_response", "valid_utf8"])
    assert r.passed


def test_evaluate_promotion_threshold(tmp_path: Path):
    val = tmp_path / "val.jsonl"
    import json as _json

    val.write_text(
        "\n".join(
            _json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"prompt {i}"},
                        {"role": "assistant", "content": "x"},
                    ]
                }
            )
            for i in range(5)
        )
    )
    base_responses = ["base"] * 5
    ft_responses = ["ft response"] * 5
    report = evaluate(
        val,
        base_responses=base_responses,
        ft_responses=ft_responses,
        heuristics=["non_empty_response"],
        judge_id="openai:gpt-5.4-mini",
        threshold=0.55,
    )
    assert isinstance(report, EvalReport)
    assert report.total_samples == 5
    assert report.heuristic_passed == 5
    # Stub judge returns 'tie' → win_rate is 0.0 → promote=False (correct guard)
    assert not report.promote
    assert report.ties == 5


def test_evaluate_filters_failed_heuristics(tmp_path: Path):
    val = tmp_path / "val.jsonl"
    import json as _json

    val.write_text(
        _json.dumps({"messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "x"}]})
        + "\n"
    )
    report = evaluate(
        val,
        base_responses=["b"],
        ft_responses=[""],  # fails non_empty
        heuristics=["non_empty_response"],
        judge_id="x",
    )
    assert report.heuristic_passed == 0
    assert report.judge_evaluated == 0


def test_heuristic_result_dataclass():
    h = HeuristicResult(sample_idx=3, passed=False, reasons=["empty_response"])
    assert h.sample_idx == 3
    assert not h.passed
