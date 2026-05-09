"""HTTP client for slancha cloud /v1/classify-routed endpoint (opt-in)."""

from __future__ import annotations

import logging

import httpx

from slancha_local.classifier_client.base import ClassifierClient
from slancha_local.classifier_client.models import ClassifyRequest, ClassifyResponse

logger = logging.getLogger(__name__)


class CloudClassifierClient(ClassifierClient):
    def __init__(self, *, base_url: str, api_key: str | None, timeout_s: float = 2.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        url = f"{self._base_url}/v1/classify-routed"
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        body = request.model_dump(exclude_none=True)
        resp = await self._client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return ClassifyResponse.model_validate(resp.json())

    async def aclose(self) -> None:
        await self._client.aclose()
