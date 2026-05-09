"""Pluggable training providers — patterns cribbed from TensorZero."""

from slancha_local.train.providers.axolotl_provider import AxolotlProvider
from slancha_local.train.providers.base import TrainingJob, TrainingProvider, TrainingResult
from slancha_local.train.providers.http_providers import (
    FireworksProvider,
    OpenAIProvider,
    TogetherProvider,
)


def get_provider(provider_id: str) -> TrainingProvider:
    """Resolve a provider id to an instance. Constructor is parameterless."""
    table: dict[str, type[TrainingProvider]] = {
        "axolotl": AxolotlProvider,
        "fireworks": FireworksProvider,
        "together": TogetherProvider,
        "openai": OpenAIProvider,
    }
    if provider_id not in table:
        raise ValueError(f"unknown provider: {provider_id}; have {list(table)}")
    return table[provider_id]()


__all__ = [
    "AxolotlProvider",
    "FireworksProvider",
    "OpenAIProvider",
    "TogetherProvider",
    "TrainingJob",
    "TrainingProvider",
    "TrainingResult",
    "get_provider",
]
