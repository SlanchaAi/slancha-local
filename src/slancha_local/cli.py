"""slancha-local CLI: serve, doctor, version, trace, why, brag."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from slancha_local import __version__


def _force_utf8_streams() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows (no-op elsewhere).

    Windows consoles default to cp1252, which can't encode Rich's box-drawing
    glyphs (│ ┌ ─ └) → UnicodeEncodeError on every Rich-output command
    (doctor/trace/catalog/brag/tui). Found on a real Win10 box, 2026-05-26;
    `PYTHONIOENCODING=utf-8` was the manual workaround — this bakes it in.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # detached/!TextIOWrapper — best effort
                pass


_force_utf8_streams()

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
    snapshot_in: str = typer.Option(
        "",
        "--snapshot-in",
        help="Path to a prior cluster snapshot (carries cluster ids forward). "
        "Defaults to {out}/cluster_snapshot.npz if that file exists.",
    ),
    no_snapshot_out: bool = typer.Option(
        False,
        "--no-snapshot-out",
        help="Skip writing the post-fit cluster snapshot.",
    ),
) -> None:
    """Cluster traces + split train/val into axolotl-compat JSONL."""
    import json as _json

    from slancha_local.config import Settings
    from slancha_local.train.bundle import SNAPSHOT_FILENAME, build_train_bundle

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
    # Default snapshot_in to the bundle-local snapshot if both pair files exist.
    snapshot_in_path: Path | None
    if snapshot_in:
        snapshot_in_path = Path(snapshot_in)
    else:
        candidate_npz = Path(out) / SNAPSHOT_FILENAME
        candidate_json = candidate_npz.with_suffix(".json")
        snapshot_in_path = candidate_npz if candidate_npz.exists() and candidate_json.exists() else None
    stats = build_train_bundle(
        traces,
        out_dir=Path(out),
        val_fraction=val_fraction,
        cluster=not no_cluster,
        snapshot_in=snapshot_in_path,
        snapshot_out=False if no_snapshot_out else True,
    )
    typer.echo(
        f"emitted train={stats.train_count} val={stats.val_count} "
        f"clusters={stats.clusters} routes={len(stats.routes)} → {out}"
    )
    if stats.snapshot_path is not None:
        typer.echo(
            f"snapshot: {stats.snapshot_path} "
            f"(revived_ids={stats.snapshot_revived_ids} retired_routes={stats.snapshot_retired_routes})"
        )
    if stats.skipped_no_response or stats.skipped_no_consent:
        typer.echo(f"skipped: no_response={stats.skipped_no_response} no_consent={stats.skipped_no_consent}")


@app.command(name="gate-decide")
def gate_decide(
    champion: Path = typer.Option(  # noqa: B008  (typer-required sentinel)
        ..., "--champion", help="Path to the champion eval-row JSON (or JSONL — last row wins)."
    ),
    challenger: Path = typer.Option(  # noqa: B008
        ..., "--challenger", help="Path to the challenger eval-row JSON / JSONL."
    ),
    mean_score_delta: float = typer.Option(
        0.05, "--mean-delta", help="Minimum mean_score lift required to accept."
    ),
    per_domain_max_regression: float = typer.Option(
        0.15,
        "--per-domain-max-regression",
        help="Max per-domain mean drop tolerated. Domains seen by only one side are skipped.",
    ),
    min_n_eval: int = typer.Option(
        100,
        "--min-n-eval",
        help="Both rows must have at least this many evaluated samples.",
    ),
    require_judge_match: bool = typer.Option(
        True,
        "--require-judge-match/--no-require-judge-match",
        help="Reject if champion.judge_model != challenger.judge_model.",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit the verdict as a JSON object on stdout (script-friendly)."
    ),
    promotions_log: Path | None = typer.Option(  # noqa: B008  (typer-required sentinel)
        None,
        "--promotions-log",
        help="Append the verdict as a JSONL row to this path (parents created if missing). "
        "Mirrors mesh.eval.gate's `--promotions-log` (default upstream: dashboard/promotions.jsonl). "
        "Omit to skip event-sourcing — useful for dry-run / CI gating without log pollution.",
    ),
) -> None:
    """Decide whether to promote a challenger router over a champion.

    Reads two eval-pass rows produced by ``slancha_local.train.eval_row``
    (or the equivalent shape from ``mesh.eval.runner.EvalPass.to_row``)
    and prints the verdict. Exit status: 0 on accept, 2 on reject —
    matches the convention CI gates use for promote/no-promote.
    """
    from slancha_local.train.eval_row import read_eval_row
    from slancha_local.train.gate import GateThresholds, append_verdict, decide

    try:
        champ_row = read_eval_row(champion)
        chall_row = read_eval_row(challenger)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    verdict = decide(
        champ_row,
        chall_row,
        GateThresholds(
            mean_score_delta=mean_score_delta,
            per_domain_max_regression=per_domain_max_regression,
            min_n_eval=min_n_eval,
            require_judge_match=require_judge_match,
        ),
    )

    if promotions_log is not None:
        append_verdict(promotions_log, verdict)

    if json_out:
        typer.echo(json.dumps(verdict.to_row(), ensure_ascii=False))
    else:
        status = "ACCEPT" if verdict.accept else "REJECT"
        color = typer.colors.GREEN if verdict.accept else typer.colors.YELLOW
        typer.secho(f"{status} — {verdict.champion_version} → {verdict.challenger_version}", fg=color)
        typer.echo(f"  mean_delta:       {verdict.mean_delta:+.4f}")
        if verdict.per_domain_deltas:
            worst = min(verdict.per_domain_deltas.items(), key=lambda kv: kv[1])
            typer.echo(
                f"  per_domain:       {len(verdict.per_domain_deltas)} compared, "
                f"worst {worst[0]}={worst[1]:+.4f}"
            )
        typer.echo(
            f"  n_eval:           champion={verdict.n_eval_champion} challenger={verdict.n_eval_challenger}"
        )
        typer.echo(
            f"  judge_model:      champion={verdict.judge_model_champion} "
            f"challenger={verdict.judge_model_challenger}"
        )
        if verdict.reject_reasons:
            typer.echo("  reject_reasons:")
            for r in verdict.reject_reasons:
                typer.echo(f"    - {r}")

    if not verdict.accept:
        raise typer.Exit(code=2)


@app.command(name="promote-head")
def promote_head_cmd(
    store_root: Path = typer.Option(  # noqa: B008
        ...,
        "--store-root",
        help="Root directory for the pointer-store (e.g. assets/heads).",
    ),
    holdout: Path = typer.Option(  # noqa: B008
        ...,
        "--holdout",
        help="JSONL of HoldoutPrompt rows: {prompt: str, domain: str}.",
    ),
    head_bytes: Path = typer.Option(  # noqa: B008
        ...,
        "--head-bytes",
        help="Path to the freshly-retrained head .bin (treelite-serialized).",
    ),
    label_table: Path = typer.Option(  # noqa: B008
        ...,
        "--label-table",
        help="Path to the label_table.json from head_retrain.HeadRetrainResult.",
    ),
    holdout_version: int = typer.Option(
        ..., "--holdout-version", help="Integer version pin for the holdout."
    ),
    mean_score_delta: float = typer.Option(0.05, "--mean-delta", help="Minimum mean_score lift to accept."),
    per_domain_max_regression: float = typer.Option(
        0.15, "--per-domain-max-regression", help="Max per-domain regression tolerated."
    ),
    min_n_eval: int = typer.Option(100, "--min-n-eval", help="Min holdout samples per side."),
    promotions_log: Path | None = typer.Option(  # noqa: B008
        None,
        "--promotions-log",
        help="Append the verdict as JSONL to this path. Omit to skip event-sourcing.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Run the full pipeline (stage + verify + eval + gate) but NEVER "
            "write to the pointer store or promotions log. Useful for "
            "what-would-happen checks."
        ),
    ),
) -> None:
    """Stage + verify + evaluate-pair + gate a freshly retrained cluster head.

    The orchestrator runs the full P2b.3 promotion pipeline:

    \b
      1. Stage the candidate in a tempdir (head + label_table + sidecar).
      2. ``verify_load`` the head bytes (treelite-deserialize smoke test).
      3. Evaluate BOTH incumbent and candidate on the same holdout in the
         SAME run (same scorer instance → judge-match guaranteed).
      4. ``gate.decide(champion=incumbent, challenger=candidate)``.
      5. ACCEPT → commit into the store + flip ACTIVE. REJECT → rmtree
         staging, ACTIVE untouched. Either way → append verdict.

    Exit status: 0 on accept, 2 on reject — matches gate-decide.

    This CLI is intentionally lightweight: it can only run with a stub
    in-memory dispatcher + scorer because production users almost always
    wire a real mesh runner / quality probe. For programmatic use, import
    :func:`slancha_local.train.promote_head.promote_head` directly.
    """
    from slancha_local.train.head_retrain import HeadRetrainResult
    from slancha_local.train.pointer_store import PointerStore
    from slancha_local.train.promote_head import (
        HeadRouter,
        HoldoutPrompt,
        PromoteHeadError,
        promote_head,
    )

    try:
        head_payload = head_bytes.read_bytes()
        label_table_data = json.loads(label_table.read_text(encoding="utf-8"))
        holdout_rows: list[HoldoutPrompt] = []
        for line in holdout.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            holdout_rows.append(HoldoutPrompt(prompt=row["prompt"], domain=row["domain"]))
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    head_result = HeadRetrainResult(
        head_bytes=head_payload,
        label_table=label_table_data,
        n_classes=len(label_table_data),
        n_samples=0,
        embedding_dim=0,
    )

    # Stub routers + dispatcher + scorer: ALL prompts → "stub-model",
    # every response scored 0. The CLI's job is to surface the pipeline
    # mechanics + sidecar/gate behavior for ops smoke testing; real
    # users plug in a real Dispatcher/Scorer programmatically.
    typer.secho(
        "warning: built-in routers/dispatcher/scorer are stubs — "
        "promote_head is meant to be driven programmatically with real "
        "mesh runners. Use --dry-run for a smoke pass.",
        fg=typer.colors.YELLOW,
        err=True,
    )

    from slancha_local.train.dispatcher import DispatchResult
    from slancha_local.train.gate import GateThresholds
    from slancha_local.train.scorer import ScoreResult

    class _StubDispatcher:
        def dispatch(self, prompt: str, served_model: str) -> DispatchResult:
            return DispatchResult(response_text="", served_model=served_model, elapsed_ms=0.0)

    class _StubScorer:
        judge_model = "stub-judge"

        def score(self, prompt: str, response: str) -> ScoreResult:
            return ScoreResult(score=0.0, judge_model=self.judge_model)

    inc_router = HeadRouter(pick=lambda _p: "stub-model-incumbent")
    cand_router_factory = lambda _staging: HeadRouter(pick=lambda _p: "stub-model-candidate")  # noqa: E731

    store = PointerStore(root=store_root)
    try:
        verdict = promote_head(
            store,
            head_result=head_result,
            holdout=holdout_rows,
            incumbent_router=inc_router,
            candidate_router_factory=cand_router_factory,
            dispatcher=_StubDispatcher(),
            scorer=_StubScorer(),
            holdout_version=holdout_version,
            thresholds=GateThresholds(
                mean_score_delta=mean_score_delta,
                per_domain_max_regression=per_domain_max_regression,
                min_n_eval=min_n_eval,
            ),
            promotions_log=promotions_log,
            dry_run=dry_run,
        )
    except PromoteHeadError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    status = "ACCEPT" if verdict.accept else "REJECT"
    color = typer.colors.GREEN if verdict.accept else typer.colors.YELLOW
    typer.secho(f"{status} — {verdict.champion_version} → {verdict.challenger_version}", fg=color)
    typer.echo(f"  mean_delta: {verdict.mean_delta:+.4f}")
    if verdict.reject_reasons:
        typer.echo("  reject_reasons:")
        for r in verdict.reject_reasons:
            typer.echo(f"    - {r}")
    if dry_run:
        typer.secho("  (dry-run: no store / log writes)", fg=typer.colors.BLUE)

    if not verdict.accept:
        raise typer.Exit(code=2)


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
