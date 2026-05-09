"""Axolotl provider — local CUDA SFT on Spark."""

from __future__ import annotations

import logging

from slancha_local.train.providers.base import TrainingJob, TrainingProvider, TrainingResult
from slancha_local.train.spark_runner import emit_axolotl_config, launch_axolotl, precheck_gpu

logger = logging.getLogger(__name__)


class AxolotlProvider(TrainingProvider):
    id = "axolotl"

    def precheck(self, job: TrainingJob) -> tuple[bool, str]:
        return precheck_gpu(min_free_mb=40_000)

    def train(self, job: TrainingJob) -> TrainingResult:
        cfg_path = emit_axolotl_config(
            base_model=job.base_model,
            train_jsonl=job.train_jsonl,
            val_jsonl=job.val_jsonl,
            output_dir=job.output_dir,
            max_seq_len=job.hyperparams.get("max_seq_len", 2048),
            lora_r=job.hyperparams.get("lora_r", 16),
            lora_alpha=job.hyperparams.get("lora_alpha", 32),
            learning_rate=job.hyperparams.get("learning_rate", 2e-4),
            epochs=job.hyperparams.get("epochs", 3),
            batch_size=job.hyperparams.get("micro_batch_size", 4),
            gradient_accumulation_steps=job.hyperparams.get("gradient_accumulation_steps", 4),
        )
        rc = launch_axolotl(cfg_path)
        if rc != 0:
            return TrainingResult(
                success=False,
                artifact_path=None,
                artifact_ref=None,
                metrics={},
                error=f"axolotl exited with {rc}",
            )
        out_dir = job.output_dir / "out"
        return TrainingResult(
            success=True,
            artifact_path=out_dir if out_dir.exists() else None,
            artifact_ref=None,
            metrics={"epochs_completed": job.hyperparams.get("epochs", 3)},
        )
