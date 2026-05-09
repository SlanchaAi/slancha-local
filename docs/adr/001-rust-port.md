# ADR-001 — Rust port from day 1, Python canonical

**Status:** ACCEPTED
**Date:** 2026-05-09
**Decision-makers:** Paul, James

---

## Context

slancha-local needs to ship to r/LocalLLaMA. That audience prefers single-binary CLIs (like `ollama`, `koboldcpp`, `llamafile`) and tolerates Python with mild groans. We have to choose: Python-only, Rust-only, or both.

Existing slancha-api is Python+FastAPI+ONNX. Vendoring its embedder + classifier is fastest in Python.

## Decision

Ship **Python v0.1 in week 14** (canonical, fully featured) and **Rust v0.2 in week 18** (single-binary, perf-first).

Both target the same OpenAI-compat surface and the same trace-header contract. Both speak to the same cloud endpoints. Different installation, identical UX.

## Rationale

### Why not Python-only

- 250MB wheel + Python deps + ONNX runtime install path = 30s to first-routed-request on a clean machine (vs 3s for a Rust binary).
- r/LocalLLaMA reads "pip install" as "30 minutes of debugging CUDA + onnxruntime build" even when it's not. Reputation damage is real.
- PyInstaller binaries work but feel heavy (~150MB, slower startup). Acceptable but not delightful.

### Why not Rust-only

- Phase 1 has lots of glue work (rule engine, classifier wrapper, telemetry, bench harness, TUI). Iteration speed matters.
- Vendoring slancha-api's existing Python embedder + treelite is ~1 hour. Rewriting in Rust is ~2 weeks.
- Bench harness wants pandas/parquet/HF datasets — Rust ecosystem here is brutal.
- Solo developer (Paul) is faster in Python.

### Why both

- **Python = canonical.** Full feature set ships first. Bench, TUI, gallery stay Python.
- **Rust = perf binary.** `slancha` Rust binary handles the hot path (proxy + embedder + classifier + dispatch). Python helpers (`slancha-tui`, `slancha bench`, `slancha gallery`) installed alongside, invoked by the Rust binary via subprocess when needed.
- This gives us the brew-installable single-binary feel without sacrificing iteration speed.
- ~30MB Rust binary vs ~250MB Python wheel.
- Rust binary as v0.2 release lets us announce "Python today, Rust in 4 weeks" — the *roadmap itself* is launch-day content.

## Architecture

### Rust binary scope (`slancha`)

Statically linked. Embedded ONNX + treelite weights. ~30MB.

Components:
- HTTP server (axum) — OpenAI-compat surface at `/v1/chat/completions`
- ONNX runtime (`ort` crate) — mmBERT-small INT8 embedder
- Treelite C API + Rust bindings — 6 classifier heads
- Route selector — Rust port of the rule engine
- Backend dispatch — `reqwest` async clients per backend
- Trace writer — `serde_json` + `tokio::fs`
- Metrics + structured logs — `tracing` crate

### Python helper scope (`slancha-helpers`)

Stays Python. Optional install. Invoked by Rust binary via subprocess.

Components:
- TUI (`slancha tui`) — textual-based
- Bench (`slancha bench`) — pandas / RouterBench harness
- Gallery (`slancha gallery`) — FastAPI + HTMX
- Brag (`slancha brag`) — ASCII generator
- Trace export (`slancha export`) — bundle creator
- Doctor (`slancha doctor`) — diagnostics

The Rust binary detects whether helpers are installed. `slancha tui` invokes Python if installed, otherwise prints install instructions. `slancha serve`, `slancha why`, `slancha doctor` are in Rust core.

### Cross-language contract

Single source of truth = the trace JSON schema (`schema.py` in Python, `Trace` struct in Rust, both pinned to `schema_version=1`). CI test asserts both produce byte-identical JSON for a fixed input.

Same for the classifier wire models (`ClassifyRequest`, `ClassifyResponse`).

### Distribution

- Python: PyPI (`pip install slancha-local`), Docker image, source tarball.
- Rust: Homebrew tap (`brew install SlanchaAi/tap/slancha`), GitHub release binaries (linux x86_64 / linux aarch64 / darwin x86_64 / darwin aarch64), eventually `cargo install slancha`.

## Consequences

### Good

- Best of both worlds: Python iteration speed + Rust binary deliciousness.
- Launch post can credibly say "Python today, Rust in 4 weeks" — the roadmap is content.
- Rust build forces us to lock the cross-language schema contracts early, which prevents drift.
- 30MB binary changes the install conversation entirely.

### Bad

- Maintaining two implementations of the hot path is real work. Mitigation: hot path is small (~1500 LOC each); auxiliary tools are only Python; classifier weights are shared.

### Neutral

- Rust binary won't have the bench/TUI/gallery; users running TUI from Rust binary need pip-installed helper.

## Alternatives considered

1. **Python + PyInstaller binary** (no Rust). Rejected: PyInstaller binaries on macOS hit codesigning / notarization friction; Linux ARM PyInstaller is iffy; binary size is ~150MB even after stripping.

2. **Go instead of Rust.** Rejected:
   - ONNX runtime Go bindings are less mature than Rust's `ort`.
   - r/LocalLLaMA aesthetic prefers Rust over Go for systems-y inference tools.

3. **C++.** Rejected for engineering speed reasons.

4. **Wasm + JS runtime.** Way out of scope.

## Migration plan

Phase 1 (weeks 1–14): Python only. v0.1 release at week 14.

Phase 1.5 (weeks 14–18): Rust port. Mirror the Phase 1 architecture line-by-line in Rust. v0.2 release at week 18.

Cross-language CI: from week 14 onward, every PR runs both Python tests and Rust tests; trace-schema round-trip test asserts byte-identical output.
