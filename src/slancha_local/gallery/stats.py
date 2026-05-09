"""Trace-derived per-model + global stats. Pure-data, easy to test."""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass
class ModelStats:
    target: str  # e.g. "local:ollama:codestral:22b"
    backend: str
    model_id: str
    use_count: int = 0
    total_latency_ms: int = 0
    top_combos: list[tuple[str, str, int]] = field(default_factory=list)
    # combos = (domain, difficulty, count) sorted by count desc

    @property
    def avg_latency_ms(self) -> int:
        return int(self.total_latency_ms / self.use_count) if self.use_count else 0


@dataclass
class GalleryView:
    total_routed: int
    total_local: int
    total_cloud: int
    distinct_models: int
    models: list[ModelStats]
    window_days: int

    @property
    def pct_local(self) -> float:
        return (self.total_local / self.total_routed * 100) if self.total_routed else 0.0


def _read_traces(root: Path, days: int) -> list[dict]:
    if not root.exists():
        return []
    cutoff = datetime.now(UTC) - timedelta(days=days)
    out: list[dict] = []
    for f in sorted(root.glob("*.jsonl")):
        try:
            text = f.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                ts = datetime.fromisoformat(t["ts"].replace("Z", "+00:00"))
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
            if ts >= cutoff:
                out.append(t)
    return out


def compute_view(traces_root: Path, *, days: int = 30) -> GalleryView:
    traces = _read_traces(traces_root, days)
    by_target: dict[str, list[dict]] = collections.defaultdict(list)
    for t in traces:
        by_target[t["decision"]["target"]].append(t)

    models: list[ModelStats] = []
    for target, ts in by_target.items():
        parts = target.split(":", 2)
        backend = parts[1] if len(parts) >= 3 else "?"
        model_id = parts[2] if len(parts) >= 3 else target
        combos: collections.Counter = collections.Counter()
        total_latency = 0
        for t in ts:
            cls = t.get("classifier", {})
            combos[(cls.get("domain", "?"), cls.get("difficulty", "?"))] += 1
            total_latency += t.get("execution", {}).get("latency_ms", 0) or 0
        top = sorted(((d, di, c) for (d, di), c in combos.items()), key=lambda x: -x[2])[:3]
        models.append(
            ModelStats(
                target=target,
                backend=backend,
                model_id=model_id,
                use_count=len(ts),
                total_latency_ms=total_latency,
                top_combos=top,
            )
        )
    models.sort(key=lambda m: -m.use_count)

    total = len(traces)
    local = sum(1 for t in traces if t["decision"]["target"].startswith("local:"))
    return GalleryView(
        total_routed=total,
        total_local=local,
        total_cloud=total - local,
        distinct_models=len(models),
        models=models,
        window_days=days,
    )
