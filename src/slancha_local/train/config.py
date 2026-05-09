"""Declarative training config — patterns cribbed from TensorZero's tensorzero.toml.

`slancha-train.toml` declares:
- which routes get fine-tuned
- with what base model + hyperparameters
- via what provider (axolotl/fireworks/together/openai)
- on what cadence (manual or quarterly)
- what eval to run post-FT

Single source of truth for the training pipeline. Server-side (Spark) reads it
on each FT run kickoff.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class HyperParams(BaseModel):
    """LoRA SFT hyperparams. Defaults sized for Spark GB10 121GB ceiling."""

    max_seq_len: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 3
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4


class EvalSpec(BaseModel):
    """Post-FT eval. Mirrors TensorZero's two-tier (boolean + rubric) eval pattern."""

    enabled: bool = True
    # judge model — defaults to the cheapest cloud judge; can be swapped for local
    judge: str = "openai:gpt-5.4-mini"
    # how many val samples to compare base-vs-finetuned on
    sample_n: int = 100
    # win-rate threshold to promote the FT'd LoRA to the route's catalog
    promote_if_win_rate_gte: float = 0.55
    # heuristic checks (boolean: any failure = drop the run)
    heuristics: list[str] = Field(default_factory=lambda: ["non_empty_response", "valid_utf8"])


class RouteSpec(BaseModel):
    """One route → one FT job. Multiple routes can share base + provider."""

    route: str  # e.g. "computer_science_medium" — matches classifier.route
    base_model: str  # e.g. "Qwen/Qwen3-8B"
    provider: Literal["axolotl", "fireworks", "together", "openai", "torchtune", "unsloth"] = "axolotl"
    min_train_examples: int = 200  # don't FT if cluster has fewer
    hyperparams: HyperParams = Field(default_factory=HyperParams)
    eval: EvalSpec = Field(default_factory=EvalSpec)
    # promotion target: where to put a successful LoRA (HF hub, local dir, slancha cloud)
    artifact_dest: str = "local:.slancha-train/artifacts"


class TrainConfig(BaseModel):
    """Top-level config. Roughly mirrors tensorzero.toml's structure."""

    storage_root: Path = Path("~/.slancha-train").expanduser()
    cadence: Literal["manual", "hourly", "daily", "weekly", "quarterly"] = "manual"
    routes: list[RouteSpec] = Field(default_factory=list)
    # Global GPU pre-check threshold (MB free required to launch any FT)
    gpu_min_free_mb: int = 40_000


def load_config(path: Path) -> TrainConfig:
    """Load + validate slancha-train.toml."""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = tomllib.loads(path.read_text())
    return TrainConfig.model_validate(data)


def example_config() -> str:
    """Return a documented example slancha-train.toml as a string."""
    return """\
# slancha-train.toml — declarative FT config, patterns cribbed from TensorZero
storage_root = "~/.slancha-train"
cadence = "weekly"
gpu_min_free_mb = 40000

[[routes]]
route = "computer_science_medium"
base_model = "Qwen/Qwen3-8B"
provider = "axolotl"
min_train_examples = 200
artifact_dest = "local:.slancha-train/artifacts"

[routes.hyperparams]
max_seq_len = 2048
lora_r = 16
learning_rate = 2e-4
epochs = 3

[routes.eval]
enabled = true
judge = "openai:gpt-5.4-mini"
sample_n = 100
promote_if_win_rate_gte = 0.55
heuristics = ["non_empty_response", "valid_utf8"]

[[routes]]
route = "creative_writing_medium"
base_model = "Qwen/Qwen3-8B"
provider = "axolotl"
min_train_examples = 500

[routes.hyperparams]
# creative writing benefits from longer context + more epochs
max_seq_len = 4096
lora_r = 32
epochs = 5
"""
