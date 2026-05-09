"""htop-style TUI for slancha-local. The screenshot is the launch artifact."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static


def _read_recent_traces(root: Path, n: int = 20) -> list[dict]:
    out: list[dict] = []
    if not root.exists():
        return out
    for f in sorted(root.glob("*.jsonl"), reverse=True):
        try:
            text = f.read_text()
        except OSError:
            continue
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= n:
                return out
    return out


class SlanchaTUI(App):
    """Live routing dashboard. Press q to quit."""

    CSS = """
    Screen { layout: vertical; }
    #top { height: 9; }
    #middle { height: 1fr; }
    #backends-panel, #counts-panel { width: 1fr; }
    Static.title { text-style: bold; padding: 0 1; background: $primary 30%; }
    DataTable { height: 1fr; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh_now", "Refresh", show=True),
    ]

    def __init__(
        self,
        *,
        proxy_url: str = "http://127.0.0.1:8000",
        traces_root: Path | None = None,
    ) -> None:
        super().__init__()
        self._proxy_url = proxy_url
        self._traces_root = traces_root or (Path.home() / ".slancha" / "traces")
        self._client = httpx.Client(timeout=2.0)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, name="slancha-local")
        with Horizontal(id="top"):
            with Vertical(id="backends-panel"):
                yield Static("Backends", classes="title")
                yield DataTable(id="backends")
            with Vertical(id="counts-panel"):
                yield Static("This session", classes="title")
                yield DataTable(id="counts")
        with Vertical(id="middle"):
            yield Static("Recent decisions  (q quit · r refresh)", classes="title")
            yield DataTable(id="decisions", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        bt = self.query_one("#backends", DataTable)
        bt.add_columns("backend", "status", "models")

        ct = self.query_one("#counts", DataTable)
        ct.add_columns("metric", "value")

        dt = self.query_one("#decisions", DataTable)
        dt.add_columns("time", "domain", "diff", "picked", "ms")

        self.set_interval(2.0, self.refresh_data)
        self.refresh_data()

    def action_refresh_now(self) -> None:
        self.refresh_data()

    def refresh_data(self) -> None:
        # backends from /health/detailed
        try:
            r = self._client.get(f"{self._proxy_url}/health/detailed")
            data = r.json() if r.status_code == 200 else {"backends": []}
        except Exception:
            data = {"backends": []}

        bt = self.query_one("#backends", DataTable)
        bt.clear()
        for b in data.get("backends", []):
            bt.add_row(
                b["id"],
                "[green]healthy[/green]" if b["healthy"] else "[red]down[/red]",
                str(len(b["models"])),
            )

        # decisions from local traces
        traces = _read_recent_traces(self._traces_root, n=20)

        # session counters
        ct = self.query_one("#counts", DataTable)
        ct.clear()
        total = len(traces)
        local_count = sum(1 for t in traces if t["decision"]["target"].startswith("local:"))
        cloud_count = total - local_count
        latencies = [t["execution"]["latency_ms"] for t in traces if t["execution"]["latency_ms"]]
        p95 = _percentile(latencies, 95) if latencies else 0
        ct.add_row("routed", str(total))
        ct.add_row("local", str(local_count))
        ct.add_row("cloud", str(cloud_count))
        ct.add_row("p95 latency ms", str(p95))

        dt = self.query_one("#decisions", DataTable)
        dt.clear()
        for t in traces:
            cls = t.get("classifier", {})
            dt.add_row(
                _short_time(t["ts"]),
                cls.get("domain", "?")[:18],
                cls.get("difficulty", "?")[:6],
                t["decision"]["target"][:32],
                str(t["execution"]["latency_ms"]),
            )


def _percentile(values: list[int], p: int) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * (p / 100.0))))
    return s[idx]


def _short_time(ts: str) -> str:
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return d.astimezone().strftime("%H:%M:%S")
    except (ValueError, KeyError):
        return ts[-12:-4] if len(ts) >= 12 else ts
