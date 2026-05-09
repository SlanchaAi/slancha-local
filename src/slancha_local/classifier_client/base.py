"""ABC for classifier transports."""

from __future__ import annotations

from abc import ABC, abstractmethod

from slancha_local.classifier_client.models import ClassifyRequest, ClassifyResponse


class ClassifierClient(ABC):
    @abstractmethod
    async def classify(self, request: ClassifyRequest) -> ClassifyResponse: ...
