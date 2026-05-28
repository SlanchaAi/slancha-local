"""Holdout-dispatch primitive for the cluster-head promotion loop.

Used by :mod:`slancha_local.train.promote_head` to send a holdout
prompt to a chosen served_model and get back the response that the
scorer will judge. Kept narrow on purpose: the dispatcher does ONE
thing — turn ``(prompt, served_model)`` into ``(response_text,
elapsed_ms)`` — so the orchestrator can wire incumbent vs candidate
routers freely without the dispatcher caring which head decided the
model.

PURE STANDALONE: no slancha-mesh runtime dependency. The
:class:`Dispatcher` Protocol lets callers plug in whatever transport
they like (mesh's runner, a recorded-fixture replay, an in-memory
fake for tests). :class:`HttpxDispatcher` is a thin
OpenAI-chat-completions-compatible default for deployments that don't
need anything fancier.

Failure model: every transport error becomes a :class:`DispatchError`
the orchestrator can catch + count into ``n_dispatch_failures`` on
the eval row, mirroring ``mesh.eval.runner``'s behavior.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class DispatchError(RuntimeError):
    """Raised when a dispatch transport call fails terminally.

    The orchestrator catches this, records the failure on the
    sample's ``failure_kind="dispatch"``, and continues with the next
    prompt — one bad dispatch must not abort the eval pass.
    """


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of one ``dispatch(prompt, served_model)`` call.

    ``response_text`` is the raw model response (no scoring yet).
    ``elapsed_ms`` is wall time including any retries the transport
    chose to do. ``served_model`` is the model that actually
    answered — the dispatcher MUST echo the value it was asked to use
    so the orchestrator can attribute the sample to the right row in
    ``per_model_mean``.
    """

    response_text: str
    served_model: str
    elapsed_ms: float


class Dispatcher(Protocol):
    """Send ``prompt`` to ``served_model``, return the response.

    Implementations should raise :class:`DispatchError` on any
    transport failure they consider terminal (timeout, 5xx after
    retries, malformed JSON, etc.). The orchestrator catches that
    error and records the sample as a dispatch failure rather than
    aborting the whole eval pass.
    """

    def dispatch(self, prompt: str, served_model: str) -> DispatchResult: ...


class HttpxDispatcher:
    """OpenAI-chat-completions-compatible dispatcher built on httpx.

    Thin by design — meant as a "your-mesh-is-not-deployed-yet" stub.
    Most production callers will plug in mesh's runner (or their own
    multiplexed transport) via the :class:`Dispatcher` Protocol.

    Constructor takes the endpoint URL (e.g. ``http://localhost:8000/v1``)
    and an optional API key. Per-call ``served_model`` becomes the
    ``model`` field in the request body.
    """

    def __init__(
        self,
        endpoint_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def dispatch(self, prompt: str, served_model: str) -> DispatchResult:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover - exercised only in light deps
            raise DispatchError(
                "HttpxDispatcher requires the 'httpx' extra; install slancha-local[promote] "
                "or pass a custom Dispatcher implementation"
            ) from e

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": served_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        url = f"{self.endpoint_url}/chat/completions"
        started = time.perf_counter()
        try:
            response = httpx.post(
                url, json=payload, headers=headers, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:  # noqa: BLE001 - transport errors all collapse to DispatchError
            raise DispatchError(f"dispatch to {served_model!r} failed: {e}") from e
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise DispatchError(
                f"dispatch to {served_model!r} returned malformed response: {data!r}"
            ) from e

        return DispatchResult(
            response_text=str(text),
            served_model=served_model,
            elapsed_ms=elapsed_ms,
        )
