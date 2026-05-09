"""slancha-local CLI: serve, doctor, version, trace, why, brag."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from slancha_local import __version__

app = typer.Typer(help="slancha-local — local LLM router. Apache 2.0.")
console = Console()


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind host (default: 127.0.0.1)"),
    port: int | None = typer.Option(None, help="Bind port (default: 8000)"),
) -> None:
    """Start the proxy on host:port."""
    from slancha_local.config import Settings

    settings = Settings()
    uvicorn.run(
        "slancha_local.proxy.main:app",
        host=host or settings.bind_host,
        port=port or settings.bind_port,
        reload=False,
    )


@app.command()
def doctor(
    capture: bool = typer.Option(False, "--capture", help="Print bytes the next request would egress"),
) -> None:
    """Probe backends + classifier; print status."""
    from slancha_local.config import Settings

    settings = Settings()

    async def _probe() -> None:
        from slancha_local.backends.llamacpp import LlamaCppBackend
        from slancha_local.backends.ollama import OllamaBackend
        from slancha_local.backends.openai_compat import (
            GenericOpenAIBackend,
            LMStudioBackend,
            MLXBackend,
            VLLMBackend,
        )

        backend_specs: list[tuple[str, object]] = []
        if settings.ollama_enabled:
            backend_specs.append(("ollama", OllamaBackend(base_url=settings.ollama_base_url)))
        if settings.llamacpp_enabled:
            backend_specs.append(("llamacpp", LlamaCppBackend(base_url=settings.llamacpp_base_url)))
        if settings.vllm_enabled:
            backend_specs.append(("vllm", VLLMBackend(base_url=settings.vllm_base_url)))
        if settings.mlx_enabled:
            backend_specs.append(("mlx", MLXBackend(base_url=settings.mlx_base_url)))
        if settings.lmstudio_enabled:
            backend_specs.append(("lmstudio", LMStudioBackend(base_url=settings.lmstudio_base_url)))
        if settings.generic_openai_base_url:
            backend_specs.append(
                ("generic-openai", GenericOpenAIBackend(base_url=settings.generic_openai_base_url))
            )

        caps = []
        for label, b in backend_specs:
            cap = await b.probe()
            caps.append((label, cap))
            await b.aclose()  # type: ignore[attr-defined]

        table = Table(title="slancha-local doctor")
        table.add_column("component")
        table.add_column("status")
        table.add_column("detail")
        for label, cap in caps:
            table.add_row(
                label,
                "[green]healthy[/green]" if cap.healthy else "[red]down[/red]",
                f"{cap.base_url} ({len(cap.models)} models)",
            )
            for m in cap.models:
                table.add_row(
                    f"  └─ {m.model_id}",
                    "loaded",
                    f"ctx={m.ctx_window}, caps={','.join(m.capabilities)}",
                )
        table.add_row(
            "classifier",
            f"[cyan]{settings.classifier_kind}[/cyan]",
            f"api_key={'set' if settings.api_key else 'unset'}",
        )
        table.add_row("traces_root", "configured", str(settings.traces_root))
        table.add_row(
            "share_prompts",
            "[yellow]yes[/yellow]" if settings.share_prompts else "no",
            "raw prompt sent to classifier?",
        )
        table.add_row(
            "share_traces",
            "[yellow]yes[/yellow]" if settings.share_traces else "no",
            "full prompt+response captured?",
        )
        console.print(table)

        if capture:
            console.print()
            console.print("[bold]What the next /v1/chat/completions request would egress:[/bold]")
            console.print()
            if settings.classifier_kind == "local":
                console.print("  → [green]127.0.0.1:11434[/green] (Ollama probe + chat)")
                console.print()
                console.print("[bold green]Default config: zero outbound non-loopback packets.[/bold green]")
            elif settings.classifier_kind == "cloud":
                console.print("  → [green]127.0.0.1:11434[/green] (Ollama probe + chat)")
                console.print(
                    f"  → [yellow]{settings.api_base_url}/v1/classify-routed[/yellow] "
                    "(cloud classifier — opt-in)"
                )
                if settings.share_prompts:
                    console.print("    [red]NOTE: --share-prompts is ON; raw prompt text is included.[/red]")
            else:
                console.print("  → [green]127.0.0.1:11434[/green] (Ollama probe + chat)")
                console.print("  rules-based classifier; no classifier network calls.")
            console.print()
            console.print("Verify externally:")
            console.print("  [dim]sudo tcpdump -i any -n 'not (host 127.0.0.1) and not (port 53)'[/dim]")

    asyncio.run(_probe())


@app.command()
def trace(last: int = typer.Option(10, help="Number of recent decisions")) -> None:
    """Print the last N routing decisions in a Rich table."""
    from slancha_local.config import Settings

    settings = Settings()
    decisions = _read_recent_decisions(settings.traces_root, last)
    if not decisions:
        typer.echo(f"no traces found at {settings.traces_root}")
        return
    table = Table(title=f"slancha trace — last {len(decisions)} decisions")
    table.add_column("time")
    table.add_column("picked")
    table.add_column("reason", overflow="fold", max_width=40)
    table.add_column("ms", justify="right")
    for d in decisions:
        table.add_row(
            d["ts"][11:19],
            d["decision"]["target"],
            d["decision"]["reason"][:60],
            str(d["execution"]["latency_ms"]),
        )
    console.print(table)


@app.command()
def why(request_id: str) -> None:
    """Explain a routing decision in plain English."""
    from slancha_local.config import Settings

    settings = Settings()
    t = _find_decision(settings.traces_root, request_id)
    if not t:
        typer.echo(f"no trace found for {request_id}")
        raise typer.Exit(1)
    cls = t["classifier"]
    dec = t["decision"]
    typer.echo(
        f"Your prompt was classified as {cls.get('domain') or '?'}-domain, "
        f"{cls.get('difficulty') or '?'}-difficulty, in {cls.get('language') or '?'}.\n"
        f"slancha picked {dec['target']} because {dec['reason']}.\n"
        f"Confidence: {(cls.get('confidence') or 0):.2f}.\n"
        f"Fallbacks: {', '.join(dec.get('fallbacks') or []) or 'none'}.\n"
        f"Latency: {t['execution']['latency_ms']}ms."
    )


@app.command()
def brag(days: int = typer.Option(7, help="Look-back window in days")) -> None:
    """Print a shareable ASCII routing summary."""
    from slancha_local.brag.render import render_brag
    from slancha_local.config import Settings

    typer.echo(render_brag(Settings().traces_root, days=days))


@app.command(name="train-bundle")
def train_bundle(
    out: str = typer.Option("./train-bundle", help="Output dir for train.jsonl + val.jsonl"),
    val_fraction: float = typer.Option(0.1, help="Fraction of each cluster reserved for val"),
    no_cluster: bool = typer.Option(False, help="Skip KMeans; group by route only"),
) -> None:
    """Cluster traces + split train/val into axolotl-compat JSONL."""
    import json as _json

    from slancha_local.config import Settings
    from slancha_local.train.bundle import build_train_bundle

    settings = Settings()
    traces: list[dict] = []
    if settings.traces_root.exists():
        for f in sorted(settings.traces_root.glob("*.jsonl")):
            try:
                for line in f.read_text().splitlines():
                    if line.strip():
                        traces.append(_json.loads(line))
            except (OSError, _json.JSONDecodeError):
                continue
    if not traces:
        typer.echo(f"no traces at {settings.traces_root}; need consent_at_capture=true entries")
        raise typer.Exit(1)
    stats = build_train_bundle(traces, out_dir=Path(out), val_fraction=val_fraction, cluster=not no_cluster)
    typer.echo(
        f"emitted train={stats.train_count} val={stats.val_count} "
        f"clusters={stats.clusters} routes={len(stats.routes)} → {out}"
    )
    if stats.skipped_no_response or stats.skipped_no_consent:
        typer.echo(f"skipped: no_response={stats.skipped_no_response} no_consent={stats.skipped_no_consent}")


@app.command()
def tui(
    proxy_url: str = typer.Option("http://127.0.0.1:8000", help="URL of the running slancha-local proxy"),
) -> None:
    """htop-style live routing dashboard. Press q to quit."""
    from slancha_local.config import Settings
    from slancha_local.tui.app import SlanchaTUI

    SlanchaTUI(proxy_url=proxy_url, traces_root=Settings().traces_root).run()


@app.command()
def bench(
    upload: bool = typer.Option(False, "--upload", help="Upload to slancha.ai/local/bench"),
) -> None:
    """Run the local classifier against the adversarial set and print a scorecard."""
    from slancha_local.bench.runner import run_self_bench
    from slancha_local.bench.scorecard import render_scorecard

    typer.echo("Running adversarial self-bench against the local classifier...")
    result = run_self_bench()
    typer.echo(render_scorecard(result))
    if upload:
        typer.echo("[upload not yet wired — coming in v0.1.1; copy the scorecard above for now]")


@app.command()
def demo(
    proxy_url: str = typer.Option("http://127.0.0.1:8000", help="URL of the running slancha-local proxy"),
) -> None:
    """Send 5 representative prompts through the proxy and print decision-trace headers."""
    from slancha_local.bench.demo import run_demo

    raise typer.Exit(code=run_demo(proxy_url=proxy_url))


@app.command()
def gallery(
    host: str = typer.Option("127.0.0.1", help="Bind host (default: 127.0.0.1)"),
    port: int = typer.Option(8001, help="Bind port (default: 8001)"),
    days: int = typer.Option(30, help="Look-back window in days for stats"),
) -> None:
    """Open a localhost web UI showing your model collection + routing stats."""
    from slancha_local.config import Settings
    from slancha_local.gallery.web import run_gallery

    typer.echo(f"slancha gallery → http://{host}:{port}")
    run_gallery(traces_root=Settings().traces_root, host=host, port=port, window_days=days)


@app.command()
def export(
    out: str = typer.Option("slancha-traces.tar.gz", help="Output path for the bundle"),
    since: str | None = typer.Option(
        None, help="Only include traces with ts >= this ISO date (e.g. 2026-05-01)"
    ),
    upload: bool = typer.Option(False, "--upload", help="POST bundle to slancha cloud"),
) -> None:
    """Bundle redacted traces into a .tar.gz for inspection (and opt-in upload)."""
    from datetime import datetime

    from slancha_local.config import Settings
    from slancha_local.telemetry.exporter import export_bundle

    settings = Settings()
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since).replace(tzinfo=UTC)
        except ValueError as e:
            typer.echo(f"invalid --since: {e}")
            raise typer.Exit(2) from e
    out_path = Path(out).resolve()
    n, sz = export_bundle(traces_root=settings.traces_root, out_path=out_path, since=since_dt)
    typer.echo(f"wrote {n} traces ({sz / 1024:.1f} KB) → {out_path}")
    typer.echo(f"audit:  tar tvf {out_path}")
    typer.echo(f"inspect: tar -xOf {out_path} '*/traces.jsonl' | head -3")
    if upload:
        typer.echo(
            "[upload not yet wired — coming with v0.1.1 cloud receiver. "
            "until then, the bundle is a deliverable you can share manually.]"
        )


@app.command()
def catalog() -> None:
    """Print the merged backend catalog as a Rich table."""
    import asyncio

    from slancha_local.backends.llamacpp import LlamaCppBackend
    from slancha_local.backends.ollama import OllamaBackend
    from slancha_local.capability.probe import CapabilityProbe
    from slancha_local.config import Settings

    settings = Settings()

    async def _probe_all():
        from slancha_local.backends.openai_compat import (
            GenericOpenAIBackend,
            LMStudioBackend,
            MLXBackend,
            VLLMBackend,
        )

        backends: list = []
        if settings.ollama_enabled:
            backends.append(OllamaBackend(base_url=settings.ollama_base_url))
        if settings.llamacpp_enabled:
            backends.append(LlamaCppBackend(base_url=settings.llamacpp_base_url))
        if settings.vllm_enabled:
            backends.append(VLLMBackend(base_url=settings.vllm_base_url))
        if settings.mlx_enabled:
            backends.append(MLXBackend(base_url=settings.mlx_base_url))
        if settings.lmstudio_enabled:
            backends.append(LMStudioBackend(base_url=settings.lmstudio_base_url))
        if settings.generic_openai_base_url:
            backends.append(GenericOpenAIBackend(base_url=settings.generic_openai_base_url))
        probe = CapabilityProbe(backends, ttl_s=1)
        return await probe.refresh()

    cat = asyncio.run(_probe_all())
    table = Table(title="slancha catalog — merged routable models")
    table.add_column("backend")
    table.add_column("model")
    table.add_column("ctx", justify="right")
    table.add_column("capabilities")
    if not cat.all_models:
        typer.echo(
            "No models found. Is your backend running?\n"
            f"  ollama:   {settings.ollama_base_url}\n"
            f"  llamacpp: {settings.llamacpp_base_url} (enabled: {settings.llamacpp_enabled})"
        )
        return
    for m in cat.all_models:
        table.add_row(
            m.backend_id,
            m.model_id,
            str(m.ctx_window),
            ", ".join(m.capabilities) or "-",
        )
    console.print(table)
    console.print(
        f"\n[dim]{len(cat.all_models)} models across {len(cat.healthy_backends)} healthy backend(s).[/dim]"
    )


def _read_recent_decisions(root: Path, n: int) -> list[dict]:
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


def _find_decision(root: Path, request_id: str) -> dict | None:
    if not root.exists():
        return None
    for f in root.glob("*.jsonl"):
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
            except json.JSONDecodeError:
                continue
            if t.get("request_id") == request_id:
                return t
    return None


if __name__ == "__main__":
    app()
