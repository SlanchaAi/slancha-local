"""TrainingProvider ABC — pluggable backends for SFT.

Patterns cribbed from TensorZero's optimization-providers abstraction:
they support OpenAI, GCP Vertex, Fireworks, Together, plus self-host
(axolotl, torchtune, unsloth). Same idea here.

Concrete impls: axolotl (local CUDA), fireworks (HTTP), together (HTTP),
openai (HTTP), torchtune, unsloth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingJob:
    """What we hand to a provider. Same shape regardless of impl."""

    route: str
    base_model: str
    train_jsonl: Path
    val_jsonl: Path
    output_dir: Path
    hyperparams: dict  # route_spec.hyperparams.model_dump()
    artifact_dest: str  # "local:.../artifacts" or "hf:repo/name"


@dataclass
class TrainingResult:
    """What providers return. Path to LoRA OR a remote artifact ref."""

    success: bool
    artifact_path: Path | None  # local LoRA dir
    artifact_ref: str | None  # remote ref (HF repo, fireworks model id, etc)
    metrics: dict  # {"train_loss": ..., "eval_loss": ..., "epochs_completed": ...}
    error: str | None = None


class TrainingProvider(ABC):
    """Abstract interface every FT backend implements."""

    id: str

    @abstractmethod
    def precheck(self, job: TrainingJob) -> tuple[bool, str]:
        """Return (ok, message). E.g. axolotl checks GPU memory; fireworks checks API key."""

    @abstractmethod
    def train(self, job: TrainingJob) -> TrainingResult:
        """Run the FT. Blocking. Caller is responsible for any async wrapping."""
