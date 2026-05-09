"""HTTP-based FT providers — stubs for v0.1.

Each follows the same upload-jsonl-then-poll-job pattern. Real implementations
(Fireworks fine-tunes API, Together fine-tunes API, OpenAI FT API) are
deferred until we have user volume to justify the API spend. Stubs return
not-implemented so the abstraction is wired without faking the work.
"""

from __future__ import annotations

import logging
import os

from slancha_local.train.providers.base import TrainingJob, TrainingProvider, TrainingResult

logger = logging.getLogger(__name__)


class _RemoteHTTPProvider(TrainingProvider):
    id = "_remote-http"
    api_key_env: str = ""
    docs: str = ""

    def precheck(self, job: TrainingJob) -> tuple[bool, str]:
        if self.api_key_env and not os.environ.get(self.api_key_env):
            return False, f"{self.api_key_env} not set in env"
        return True, f"{self.id} OK (note: not yet wired)"

    def train(self, job: TrainingJob) -> TrainingResult:
        return TrainingResult(
            success=False,
            artifact_path=None,
            artifact_ref=None,
            metrics={},
            error=(
                f"{self.id} provider not yet wired. See {self.docs} for the upload + "
                "poll + retrieve pattern. Until then, use provider=axolotl on Spark."
            ),
        )


class FireworksProvider(_RemoteHTTPProvider):
    id = "fireworks"
    api_key_env = "FIREWORKS_API_KEY"
    docs = "https://docs.fireworks.ai/fine-tuning/fine-tuning-models"


class TogetherProvider(_RemoteHTTPProvider):
    id = "together"
    api_key_env = "TOGETHER_API_KEY"
    docs = "https://docs.together.ai/docs/fine-tuning-overview"


class OpenAIProvider(_RemoteHTTPProvider):
    id = "openai"
    api_key_env = "OPENAI_API_KEY"
    docs = "https://platform.openai.com/docs/guides/fine-tuning"
