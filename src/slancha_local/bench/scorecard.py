"""Bench scorecard: ASCII renderer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchResult:
    name: str
    classifier_version: str
    samples: int
    accuracy: float  # 0.0–1.0
    p50_classifier_ms: float
    p95_classifier_ms: float
    p99_classifier_ms: float
    head_breakdown: dict[str, float]  # head_name → accuracy
    hardware: str = "unknown"
    backend: str = "n/a"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "classifier_version": self.classifier_version,
            "samples": self.samples,
            "accuracy": self.accuracy,
            "p50_classifier_ms": self.p50_classifier_ms,
            "p95_classifier_ms": self.p95_classifier_ms,
            "p99_classifier_ms": self.p99_classifier_ms,
            "head_breakdown": self.head_breakdown,
            "hardware": self.hardware,
            "backend": self.backend,
        }


def _row(text: str, width: int) -> str:
    return f"║{text:<{width - 2}}║"


def render_scorecard(r: BenchResult) -> str:
    w = 72
    bar = "═" * (w - 2)
    lat = f"{r.p50_classifier_ms:.1f} / {r.p95_classifier_ms:.1f} / {r.p99_classifier_ms:.1f}"
    lines = [
        "╔" + bar + "╗",
        _row(f"  slancha bench · {r.name}", w),
        "╠" + bar + "╣",
        _row(f"  Classifier:      {r.classifier_version}", w),
        _row(f"  Samples:         {r.samples}", w),
        _row(f"  Hardware:        {r.hardware}", w),
        _row(f"  Backend:         {r.backend}", w),
        _row("", w),
        _row(f"  Overall accuracy: {r.accuracy:>6.1%}", w),
        _row(f"  Latency p50/p95/p99: {lat} ms", w),
        _row("", w),
        _row("  Per-head accuracy", w),
        _row("  -----------------", w),
    ]
    for head, acc in sorted(r.head_breakdown.items()):
        lines.append(_row(f"  {head:<22} {acc:>6.1%}", w))
    lines.append("╚" + bar + "╝")
    lines.append("")
    lines.append("Reproduce: `slancha bench`. Submit to leaderboard: `slancha bench --upload` (opt-in).")
    return "\n".join(lines)
