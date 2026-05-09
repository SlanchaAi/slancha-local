"""FastAPI + inline HTMX UI for slancha gallery.

Localhost-only by default. Renders a card per routed model with stats.
"""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from slancha_local.gallery.stats import GalleryView, compute_view

_INLINE_CSS = """
:root {
  --bg: #0e0f12; --fg: #e6e6e6; --muted: #8a8f99; --accent: #7ad9d4;
  --card: #181a20; --card-border: #2a2d36;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 14px; line-height: 1.5; }
header { padding: 1.5rem; border-bottom: 1px solid var(--card-border); }
header h1 { margin: 0 0 0.4rem 0; font-size: 1.2rem; letter-spacing: -0.01em; }
header .meta { color: var(--muted); font-size: 0.85rem; }
header .pill { background: #283038; padding: 2px 8px; border-radius: 4px;
  margin-right: 6px; color: var(--accent); }
main { padding: 1.25rem; max-width: 1200px; margin: 0 auto; }
.summary { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1.25rem;
  color: var(--muted); }
.summary span strong { color: var(--fg); font-size: 1.1em; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 14px; }
.card { background: var(--card); border: 1px solid var(--card-border);
  border-radius: 8px; padding: 14px 16px; }
.card h2 { font-size: 0.95rem; margin: 0 0 0.3rem 0; color: var(--accent);
  word-break: break-all; }
.card .backend { font-size: 0.78rem; color: var(--muted); margin-bottom: 0.6rem;
  letter-spacing: 0.04em; text-transform: uppercase; }
.card .stat { display: flex; justify-content: space-between; padding: 4px 0;
  border-top: 1px dashed var(--card-border); }
.card .stat:first-of-type { border-top: none; }
.card .stat .lbl { color: var(--muted); }
.card .combos { margin-top: 0.4rem; padding-top: 0.5rem;
  border-top: 1px solid var(--card-border); font-size: 0.85rem; }
.card .combos .combo { display: flex; justify-content: space-between;
  color: var(--muted); padding: 2px 0; }
.card .combos .combo strong { color: var(--fg); }
.empty { color: var(--muted); padding: 2rem 0; text-align: center; }
footer { padding: 1rem; color: var(--muted); font-size: 0.8rem; text-align: center;
  border-top: 1px solid var(--card-border); }
.refresh { float: right; padding: 4px 10px; background: var(--card);
  border: 1px solid var(--card-border); color: var(--muted); border-radius: 4px;
  cursor: pointer; font-family: inherit; font-size: 0.8rem; }
.refresh:hover { color: var(--fg); }
"""


def _render_card(m) -> str:
    combos_html = (
        "".join(
            f'<div class="combo"><span><strong>{d}</strong>+{di}</span><span>{c}</span></div>'
            for d, di, c in m.top_combos
        )
        or '<div class="combo"><em>no traces yet</em></div>'
    )
    used_row = (
        f'<div class="stat"><span class="lbl">used</span>'
        f"<span><strong>{m.use_count}</strong> times</span></div>"
    )
    lat_row = (
        f'<div class="stat"><span class="lbl">avg latency</span><span>{m.avg_latency_ms} ms</span></div>'
    )
    return (
        '<div class="card">'
        f"<h2>{m.model_id}</h2>"
        f'<div class="backend">{m.backend}</div>'
        f"{used_row}{lat_row}"
        f'<div class="combos">{combos_html}</div>'
        "</div>"
    )


def _render(view: GalleryView, host: str) -> str:
    cards = [_render_card(m) for m in view.models]
    grid_html = (
        "\n".join(cards)
        if cards
        else (
            '<div class="empty">No routed traces in the last '
            f"{view.window_days} days. Run some prompts through `slancha serve` first.</div>"
        )
    )
    pills = (
        f'<span class="pill">{view.distinct_models} models</span>'
        f'<span class="pill">{view.total_routed} routed</span>'
    )
    meta_line = (
        f"Last {view.window_days} days · {view.total_local} local "
        f"({view.pct_local:.1f}%) · {view.total_cloud} cloud · host {host}"
    )
    footer = (
        '<a href="https://github.com/SlanchaAi/slancha-local" '
        'style="color:var(--accent)">slancha-local</a> · '
        "Apache 2.0 · zero phone-home in default install"
    )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>slancha gallery — {host}</title>\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<style>{_INLINE_CSS}</style>\n</head>\n<body>\n"
        "<header>\n"
        '  <button class="refresh" onclick="location.reload()">refresh</button>\n'
        f"  <h1>slancha gallery {pills}</h1>\n"
        f'  <div class="meta">{meta_line}</div>\n'
        "</header>\n<main>\n"
        f'  <div class="grid">{grid_html}</div>\n'
        "</main>\n"
        f"<footer>{footer}</footer>\n"
        "</body>\n</html>"
    )


def build_gallery_app(*, traces_root: Path, window_days: int = 30) -> FastAPI:
    app = FastAPI(title="slancha gallery", version="0.0.1")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        view = compute_view(traces_root, days=window_days)
        host = request.headers.get("host", "localhost")
        return HTMLResponse(_render(view, host=host))

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok")

    return app


def run_gallery(
    *,
    traces_root: Path,
    host: str = "127.0.0.1",
    port: int = 8001,
    window_days: int = 30,
) -> None:
    """Run the gallery server in the foreground."""
    app = build_gallery_app(traces_root=traces_root, window_days=window_days)
    uvicorn.run(app, host=host, port=port, log_level="warning")
