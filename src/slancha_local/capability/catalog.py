"""LocalCatalog: aggregate view across all healthy backends."""

from __future__ import annotations

from dataclasses import dataclass, field

from slancha_local.backends.base import BackendCapability, BackendModel


@dataclass(frozen=True)
class LocalCatalog:
    capabilities: tuple[BackendCapability, ...] = field(default_factory=tuple)

    @property
    def all_models(self) -> tuple[BackendModel, ...]:
        return tuple(m for cap in self.capabilities for m in cap.models)

    @property
    def healthy_backends(self) -> tuple[BackendCapability, ...]:
        return tuple(c for c in self.capabilities if c.healthy)
