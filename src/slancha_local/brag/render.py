"""ASCII brag screen — shareable routing summary."""

from __future__ import annotations

import collections
import json
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path

from slancha_local import __version__


def _read_traces_within(root: Path, days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out: list[dict] = []
    if not root.exists():
        return out
    for f in sorted(root.glob("*.jsonl")):
        try:
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    ts = datetime.fromisoformat(t["ts"].replace("Z", "+00:00"))
                    if ts >= cutoff:
                        out.append(t)
                except (ValueError, json.JSONDecodeError, KeyError):
                    continue
        except OSError:
            continue
    return out


def render_brag(traces_root: Path, *, days: int = 7) -> str:
    traces = _read_traces_within(traces_root, days)
    total = len(traces)
    local = sum(1 for t in traces if t["decision"]["target"].startswith("local:"))
    cloud = total - local
    by_combo: collections.Counter = collections.Counter()
    for t in traces:
        cls = t.get("classifier", {})
        by_combo[(cls.get("domain", "?"), cls.get("difficulty", "?"), t["decision"]["target"])] += 1
    top = by_combo.most_common(3)

    host = platform.node() or "host"
    pct_local = (local / total * 100) if total else 0
    pct_cloud = (cloud / total * 100) if total else 0
    line_w = 70

    lines = [
        "╔" + "═" * (line_w - 2) + "╗",
        f"║  slancha brag · {host[:line_w-22]:<{line_w-22}}║",
        "╠" + "═" * (line_w - 2) + "╣",
        f"║  Version  slancha-local {__version__:<{line_w-30}}║",
        "║" + " " * (line_w - 2) + "║",
        f"║  Last {days} days" + " " * (line_w - 16) + "║",
        f"║  Routed:  {total:>5} prompts" + " " * (line_w - 28) + "║",
        f"║  Local:   {local:>5} ({pct_local:>5.1f}%)" + " " * (line_w - 31) + "║",
        f"║  Cloud:   {cloud:>5} ({pct_cloud:>5.1f}%)" + " " * (line_w - 31) + "║",
        "║" + " " * (line_w - 2) + "║",
        "║  Top routing combos" + " " * (line_w - 22) + "║",
    ]
    for (domain, diff, target), count in top:
        text = f"  {domain[:14]:14}+{diff[:8]:8} → {target[:24]:24} ({count:>3})"
        lines.append(f"║{text:<{line_w-2}}║")
    if not top:
        lines.append("║  (no traces yet — run a few requests then re-run slancha brag)" + " " * (line_w - 64) + "║")
    lines.append("╚" + "═" * (line_w - 2) + "╝")
    lines.append("")
    lines.append("(share the screenshot — slancha.ai/local/bench coming soon)")
    return "\n".join(lines)
