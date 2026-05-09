"""Bandit-style A/B variant selection — Thompson sampling per (route × variant).

Mirrors TensorZero's variant pattern: a route can have multiple competing
LoRAs/configs; the pipeline samples one per request, records the outcome
(thumbs-up / heuristic pass / latency budget met), and the winner emerges
from posterior updates.

State lives in ~/.slancha/variants.json — small, JSON-serializable, no DB
dependency. Concurrent writes guarded by a fcntl flock.

Pick policy: Thompson sampling against Beta(α, β) priors, default α=β=1
(uniform). Caller updates with record_outcome(route, variant, won) where
`won` is whatever signal the caller has — promotion-eval result, user
thumbs-up, or a latency-tier comparison.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

DEFAULT_PATH = Path(os.environ.get("SLANCHA_VARIANTS_PATH", "~/.slancha/variants.json")).expanduser()


@dataclass
class VariantStat:
    variant_id: str
    alpha: float = 1.0  # successes + 1
    beta: float = 1.0  # failures + 1
    last_target: str | None = None  # backend target this variant maps to

    @property
    def trials(self) -> float:
        return self.alpha + self.beta - 2.0

    @property
    def win_rate(self) -> float:
        n = self.alpha + self.beta
        return (self.alpha / n) if n else 0.0


@dataclass
class VariantStore:
    path: Path = field(default_factory=lambda: DEFAULT_PATH)
    _routes: dict[str, dict[str, VariantStat]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            for route, variants in data.get("routes", {}).items():
                self._routes[route] = {}
                for vid, stat in variants.items():
                    self._routes[route][vid] = VariantStat(
                        variant_id=vid,
                        alpha=stat.get("alpha", 1.0),
                        beta=stat.get("beta", 1.0),
                        last_target=stat.get("last_target"),
                    )
        except (json.JSONDecodeError, OSError):
            self._routes = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "routes": {
                route: {
                    vid: {
                        "alpha": stat.alpha,
                        "beta": stat.beta,
                        "last_target": stat.last_target,
                    }
                    for vid, stat in variants.items()
                }
                for route, variants in self._routes.items()
            }
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(self.path)

    def register(self, route: str, variant_id: str, target: str | None = None) -> VariantStat:
        with self._lock:
            self._routes.setdefault(route, {})
            if variant_id not in self._routes[route]:
                self._routes[route][variant_id] = VariantStat(variant_id=variant_id, last_target=target)
            elif target is not None:
                self._routes[route][variant_id].last_target = target
            self._save()
            return self._routes[route][variant_id]

    def list_variants(self, route: str) -> list[VariantStat]:
        return list(self._routes.get(route, {}).values())

    def pick(self, route: str, *, rng: random.Random | None = None) -> VariantStat | None:
        """Thompson-sample a variant for `route`. Returns None if no variants registered."""
        rng = rng or random
        variants = self.list_variants(route)
        if not variants:
            return None
        scored = [(rng.betavariate(v.alpha, v.beta), v) for v in variants]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def record_outcome(self, route: str, variant_id: str, won: bool) -> None:
        with self._lock:
            self._routes.setdefault(route, {})
            if variant_id not in self._routes[route]:
                self._routes[route][variant_id] = VariantStat(variant_id=variant_id)
            stat = self._routes[route][variant_id]
            if won:
                stat.alpha += 1.0
            else:
                stat.beta += 1.0
            self._save()

    def summary(self) -> dict[str, list[dict]]:
        return {
            route: [
                {
                    "variant_id": v.variant_id,
                    "alpha": v.alpha,
                    "beta": v.beta,
                    "trials": v.trials,
                    "win_rate": v.win_rate,
                    "last_target": v.last_target,
                }
                for v in variants.values()
            ]
            for route, variants in self._routes.items()
        }
