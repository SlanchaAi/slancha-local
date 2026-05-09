"""BackendRegistry: lookup by id + parse target strings."""

from __future__ import annotations

from slancha_local.backends.base import Backend


class BackendRegistry:
    def __init__(self, backends: list[Backend]) -> None:
        self._backends: dict[str, Backend] = {b.id: b for b in backends}

    def by_id(self, backend_id: str) -> Backend:
        if backend_id not in self._backends:
            raise KeyError(f"backend not registered: {backend_id}")
        return self._backends[backend_id]

    def all(self) -> list[Backend]:
        return list(self._backends.values())

    @staticmethod
    def parse_target(target: str) -> tuple[str | None, str | None, str | None]:
        """`local:ollama:qwen3:8b` → ('local', 'ollama', 'qwen3:8b')."""
        parts = target.split(":", 2)
        if len(parts) != 3:
            return None, None, None
        return parts[0], parts[1], parts[2]
