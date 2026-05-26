"""RemoteMeshBackend — a pull-discovered mesh specialist as a Backend.

A discovered specialist is an OpenAI-compat endpoint at its host-pinned
`node_url`. Wrapping it as a `Backend` with `id == specialist_id` lets it drop
straight into the existing `BackendRegistry`, so the proxy's dispatch path
(chat.py) routes to remote mesh specialists with no change — the registry
already keys backends by `id` and speaks the OpenAI-compat surface.

`probe()` is overridden to return the capability we already learned during
discovery, so the CapabilityProbe doesn't re-hit every remote node each TTL.

Replica note: when a specialist is served by several nodes, this first cut
uses `node_urls[0]` (one backend per specialist, to match the id-keyed
registry). Multi-node load-balancing is a follow-up.
"""

from __future__ import annotations

from slancha_local.backends.base import BackendCapability, BackendModel
from slancha_local.backends.openai_compat import (
    OpenAICompatBackend,
    _generic_capabilities,
    _generic_ctx,
)
from slancha_local.mesh.discovery import DiscoveryResult


class RemoteMeshBackend(OpenAICompatBackend):
    """An OpenAI-compat backend pointing at a remote mesh node over the tailnet."""

    infer_capabilities = staticmethod(_generic_capabilities)
    ctx_default = staticmethod(_generic_ctx)

    def __init__(
        self,
        *,
        specialist_id: str,
        base_url: str,
        model_id: str,
        capabilities: tuple[str, ...] = (),
        domain: str | None = None,
        token: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(base_url=base_url, api_key=token, timeout_s=timeout_s)
        self.id = specialist_id  # per-instance id — BackendRegistry keys on this
        self._model_id = model_id
        self._capabilities = tuple(capabilities)
        self._domain = domain

    async def probe(self) -> BackendCapability:
        """Report the capability learned at discovery — no per-TTL network hit.

        The node was reachable when discovered; routing freshness comes from
        re-running discovery, not from re-probing each remote node here.
        """
        caps = self._capabilities or type(self).infer_capabilities(self._model_id)
        model = BackendModel(
            backend_id=self.id,
            model_id=self._model_id,
            ctx_window=type(self).ctx_default(self._model_id),
            capabilities=caps,
        )
        return BackendCapability(
            id=self.id, healthy=True, base_url=self._base_url, models=(model,),
        )


def backends_from_discovery(
    result: DiscoveryResult, *, token: str | None = None,
) -> list[RemoteMeshBackend]:
    """One RemoteMeshBackend per discovered specialist that has a node_url."""
    out: list[RemoteMeshBackend] = []
    for spec in result.specialists.values():
        if not spec.node_urls:
            continue
        out.append(
            RemoteMeshBackend(
                specialist_id=spec.specialist_id,
                base_url=spec.node_urls[0],
                model_id=spec.model_id or spec.specialist_id,
                capabilities=spec.capabilities,
                domain=spec.domain,
                token=token,
            )
        )
    return out


__all__ = ["RemoteMeshBackend", "backends_from_discovery"]
