"""HTTP-based fine-tuning providers — Fireworks / Together / OpenAI.

Each follows the same upload-jsonl → create-job → poll → retrieve-artifact
pattern. Real network calls when SLANCHA_TRAIN_DRY_RUN is unset and the
provider's API key is present. Dry-run mode (default in tests) returns a
synthetic success without hitting the network.

Patterns cribbed from TensorZero's optimization-providers abstraction.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

from slancha_local.train.providers.base import TrainingJob, TrainingProvider, TrainingResult

logger = logging.getLogger(__name__)


def _is_dry_run() -> bool:
    return os.environ.get("SLANCHA_TRAIN_DRY_RUN", "0") == "1"


class _RemoteHTTPProvider(TrainingProvider):
    """Common upload→create→poll→retrieve scaffolding.

    Subclasses fill in: api_base, api_key_env, docs, and the four hook
    methods (_upload_dataset, _create_job, _poll_job, _extract_artifact_ref).
    """

    id = "_remote-http"
    api_base: str = ""
    api_key_env: str = ""
    docs: str = ""
    poll_interval_s: float = 30.0
    poll_timeout_s: float = 6 * 60 * 60  # 6h cap

    def precheck(self, job: TrainingJob) -> tuple[bool, str]:
        if _is_dry_run():
            return True, f"{self.id} (dry-run)"
        if self.api_key_env and not os.environ.get(self.api_key_env):
            return False, f"{self.api_key_env} not set in env"
        if not job.train_jsonl.exists():
            return False, f"train_jsonl not found: {job.train_jsonl}"
        if not job.val_jsonl.exists():
            return False, f"val_jsonl not found: {job.val_jsonl}"
        return True, f"{self.id} OK"

    def train(self, job: TrainingJob) -> TrainingResult:
        ok, msg = self.precheck(job)
        if not ok:
            return TrainingResult(False, None, None, {}, error=msg)
        if _is_dry_run():
            return self._dry_run_result(job)
        try:
            with self._client() as client:
                logger.info("[%s] uploading train.jsonl", self.id)
                dataset_ref = self._upload_dataset(client, job)
                logger.info("[%s] dataset ref: %s", self.id, dataset_ref)
                job_id = self._create_job(client, job, dataset_ref)
                logger.info("[%s] job created: %s", self.id, job_id)
                final = self._poll_job(client, job_id)
                logger.info("[%s] terminal status: %s", self.id, final.get("status"))
                if not self._is_success(final):
                    return TrainingResult(
                        False, None, None, final, error=f"job ended in {final.get('status')}"
                    )
                ref = self._extract_artifact_ref(final)
                return TrainingResult(
                    success=True,
                    artifact_path=None,
                    artifact_ref=ref,
                    metrics=final,
                )
        except httpx.HTTPError as e:
            return TrainingResult(False, None, None, {}, error=f"HTTP error: {e}")
        except Exception as e:
            return TrainingResult(False, None, None, {}, error=f"{type(e).__name__}: {e}")

    def _client(self) -> httpx.Client:
        api_key = os.environ.get(self.api_key_env, "")
        return httpx.Client(
            base_url=self.api_base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def _dry_run_result(self, job: TrainingJob) -> TrainingResult:
        return TrainingResult(
            success=True,
            artifact_path=None,
            artifact_ref=f"{self.id}://dry-run/{job.route}",
            metrics={"dry_run": True, "route": job.route},
        )

    def _is_success(self, job_state: dict) -> bool:
        status = (job_state.get("status") or "").upper()
        return status in {"COMPLETED", "SUCCEEDED", "SUCCESS"}

    # --- subclass hooks ---

    def _upload_dataset(self, client: httpx.Client, job: TrainingJob) -> str:
        raise NotImplementedError

    def _create_job(self, client: httpx.Client, job: TrainingJob, dataset_ref: str) -> str:
        raise NotImplementedError

    def _poll_job(self, client: httpx.Client, job_id: str) -> dict:
        deadline = time.time() + self.poll_timeout_s
        while time.time() < deadline:
            state = self._get_job(client, job_id)
            status = (state.get("status") or "").upper()
            if status in {
                "COMPLETED",
                "SUCCEEDED",
                "SUCCESS",
                "FAILED",
                "CANCELLED",
                "CANCELED",
                "ERROR",
            }:
                return state
            time.sleep(self.poll_interval_s)
        return {"status": "TIMEOUT"}

    def _get_job(self, client: httpx.Client, job_id: str) -> dict:
        raise NotImplementedError

    def _extract_artifact_ref(self, job_state: dict) -> str | None:
        raise NotImplementedError


# ---------- Fireworks ----------


class FireworksProvider(_RemoteHTTPProvider):
    """Fireworks fine-tunes API.

    Account-scoped endpoints: /v1/accounts/{account_id}/datasets and
    /v1/accounts/{account_id}/fineTuningJobs. Account id from
    FIREWORKS_ACCOUNT_ID env (set by user during onboarding).
    """

    id = "fireworks"
    api_base = "https://api.fireworks.ai"
    api_key_env = "FIREWORKS_API_KEY"
    docs = "https://docs.fireworks.ai/fine-tuning/fine-tuning-models"

    def _account_id(self) -> str:
        return os.environ.get("FIREWORKS_ACCOUNT_ID", "")

    def precheck(self, job: TrainingJob) -> tuple[bool, str]:
        ok, msg = super().precheck(job)
        if not ok or _is_dry_run():
            return ok, msg
        if not self._account_id():
            return False, "FIREWORKS_ACCOUNT_ID not set"
        return True, "fireworks OK"

    def _upload_dataset(self, client: httpx.Client, job: TrainingJob) -> str:
        acc = self._account_id()
        ds_id = f"slancha-{job.route}-{int(time.time())}"
        client.post(
            f"/v1/accounts/{acc}/datasets",
            json={"datasetId": ds_id, "dataset": {"format": "CHAT"}},
        ).raise_for_status()
        with open(job.train_jsonl, "rb") as fh:
            client.post(
                f"/v1/accounts/{acc}/datasets/{ds_id}:upload",
                files={"file": ("train.jsonl", fh, "application/jsonl")},
            ).raise_for_status()
        return f"accounts/{acc}/datasets/{ds_id}"

    def _create_job(self, client: httpx.Client, job: TrainingJob, dataset_ref: str) -> str:
        acc = self._account_id()
        hp = job.hyperparams
        body = {
            "baseModel": job.base_model,
            "dataset": dataset_ref,
            "loraRank": hp.get("lora_r", 16),
            "epochs": hp.get("epochs", 3),
            "learningRate": hp.get("learning_rate", 2e-4),
            "batchSize": hp.get("micro_batch_size", 4),
        }
        r = client.post(f"/v1/accounts/{acc}/fineTuningJobs", json=body)
        r.raise_for_status()
        return r.json()["name"]  # full resource name

    def _get_job(self, client: httpx.Client, job_id: str) -> dict:
        r = client.get(f"/v1/{job_id}")
        r.raise_for_status()
        return r.json()

    def _extract_artifact_ref(self, job_state: dict) -> str | None:
        return job_state.get("outputModel") or job_state.get("name")


# ---------- Together ----------


class TogetherProvider(_RemoteHTTPProvider):
    """Together fine-tunes API.

    Together's API uploads the file first (POST /v1/files), then creates
    a job referencing that file id (POST /v1/fine-tunes).
    """

    id = "together"
    api_base = "https://api.together.xyz"
    api_key_env = "TOGETHER_API_KEY"
    docs = "https://docs.together.ai/docs/fine-tuning-overview"

    def _upload_dataset(self, client: httpx.Client, job: TrainingJob) -> str:
        with open(job.train_jsonl, "rb") as fh:
            r = client.post(
                "/v1/files",
                files={"file": ("train.jsonl", fh, "application/jsonl")},
                data={"purpose": "fine-tune"},
            )
            r.raise_for_status()
            return r.json()["id"]

    def _create_job(self, client: httpx.Client, job: TrainingJob, dataset_ref: str) -> str:
        hp = job.hyperparams
        body = {
            "training_file": dataset_ref,
            "model": job.base_model,
            "n_epochs": hp.get("epochs", 3),
            "learning_rate": hp.get("learning_rate", 2e-4),
            "batch_size": hp.get("micro_batch_size", 4),
            "lora": True,
            "lora_r": hp.get("lora_r", 16),
            "lora_alpha": hp.get("lora_alpha", 32),
            "lora_dropout": hp.get("lora_dropout", 0.05),
        }
        r = client.post("/v1/fine-tunes", json=body)
        r.raise_for_status()
        return r.json()["id"]

    def _get_job(self, client: httpx.Client, job_id: str) -> dict:
        r = client.get(f"/v1/fine-tunes/{job_id}")
        r.raise_for_status()
        return r.json()

    def _extract_artifact_ref(self, job_state: dict) -> str | None:
        return job_state.get("output_name") or job_state.get("model_output_name")


# ---------- OpenAI ----------


class OpenAIProvider(_RemoteHTTPProvider):
    """OpenAI fine-tunes API.

    Files endpoint then fine_tuning/jobs. OpenAI's offering is full SFT
    on their hosted models; LoRA hyperparams from the spec are mapped
    where they have analogues.
    """

    id = "openai"
    api_base = "https://api.openai.com"
    api_key_env = "OPENAI_API_KEY"
    docs = "https://platform.openai.com/docs/guides/fine-tuning"

    def _upload_dataset(self, client: httpx.Client, job: TrainingJob) -> str:
        with open(job.train_jsonl, "rb") as fh:
            r = client.post(
                "/v1/files",
                files={"file": ("train.jsonl", fh, "application/jsonl")},
                data={"purpose": "fine-tune"},
            )
            r.raise_for_status()
            return r.json()["id"]

    def _create_job(self, client: httpx.Client, job: TrainingJob, dataset_ref: str) -> str:
        hp = job.hyperparams
        body: dict[str, Any] = {
            "training_file": dataset_ref,
            "model": job.base_model,
            "hyperparameters": {
                "n_epochs": hp.get("epochs", 3),
                "batch_size": hp.get("micro_batch_size", 4),
                "learning_rate_multiplier": 1.0,
            },
        }
        r = client.post("/v1/fine_tuning/jobs", json=body)
        r.raise_for_status()
        return r.json()["id"]

    def _get_job(self, client: httpx.Client, job_id: str) -> dict:
        r = client.get(f"/v1/fine_tuning/jobs/{job_id}")
        r.raise_for_status()
        return r.json()

    def _is_success(self, job_state: dict) -> bool:
        return (job_state.get("status") or "").lower() == "succeeded"

    def _extract_artifact_ref(self, job_state: dict) -> str | None:
        return job_state.get("fine_tuned_model")


def write_dry_run_summary(provider_id: str, job: TrainingJob, dest: str) -> dict:
    """Helper: drop a JSON summary of a dry-run for audit/debug."""
    out = {
        "provider": provider_id,
        "route": job.route,
        "base_model": job.base_model,
        "train_jsonl": str(job.train_jsonl),
        "val_jsonl": str(job.val_jsonl),
        "hyperparams": job.hyperparams,
        "artifact_dest": job.artifact_dest,
        "ts": int(time.time()),
    }
    with open(dest, "w") as f:
        f.write(json.dumps(out, indent=2))
    return out
