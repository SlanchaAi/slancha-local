"""CapabilityProbe: TTL-cached aggregate of backend.probe() results."""

from __future__ import annotations

import asyncio
import time

from slancha_local.backends.base import Backend, BackendCapability
from slancha_local.capability.catalog import LocalCatalog


class CapabilityProbe:
    def __init__(self, backends: list[Backend], *, ttl_s: int = 30) -> None:
        self._backends = backends
        self._ttl_s = ttl_s
        self._cache: LocalCatalog | None = None
        self._cached_at: float = 0.0
        self._lock = asyncio.Lock()

    async def refresh(self) -> LocalCatalog:
        results = await asyncio.gather(*(b.probe() for b in self._backends), return_exceptions=True)
        capabilities = tuple(r for r in results if isinstance(r, BackendCapability) and r.healthy)
        catalog = LocalCatalog(capabilities=capabilities)
        self._cache = catalog
        self._cached_at = time.monotonic()
        return catalog

    async def get(self) -> LocalCatalog:
        async with self._lock:
            if self._cache is None or (time.monotonic() - self._cached_at) > self._ttl_s:
                return await self.refresh()
            return self._cache

    def cached(self) -> LocalCatalog | None:
        """Last cached catalog without triggering a probe (None if never run).

        Sync, non-blocking: the mesh heartbeat runs in a daemon thread and
        cannot await `get()`. It reads this snapshot; the async request path
        keeps the cache warm.
        """
        return self._cache
