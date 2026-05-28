# Changelog

All notable changes to slancha-local. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), [SemVer](https://semver.org/).

## [Unreleased]

### Added — self-organizing cluster-head training loop (P1–P2b.3)

The full end-to-end pipeline that lets `slancha-local` discover its own routing specializations from observed traffic, without any mesh runtime dependency. Reversible, event-sourced, runs on commodity hardware.

#### P1 — stable cluster identity

- `slancha_local.train.cluster.cluster_by_route` — group traces by `classifier.route`, KMeans on embeddings, capacity-bounded `k` (#11)
- P1.5 — retained-centroid stickiness with bounded LRU (#12). Cluster IDs survive incremental retraining so downstream artifacts can reference them.

#### P2a — snapshot persistence

- `ClusterSnapshot` persisted to `.npz` + `.json` sidecar (#13). Torn-write detection via per-save consistency token.

#### P2b.1–P2b.2 — gate decision substrate

- `slancha_local.train.gate.decide(champion, challenger, thresholds)` — local mirror of `mesh.eval.gate.decide()` with cross-repo guard (#14).
- `EvalPass` row aggregator + `slancha gate-decide` CLI (#15).
- `gate.append_verdict()` + `--promotions-log` (#16). Every promote/reject decision is event-sourced to disk.

#### P2b.3 — closed cluster-head training loop

- **Phase 2a — pointer store** (#17): versioned artifact store at `<root>/<component>/<version>/...` with atomic `ACTIVE` pointer flip. `rollback()` walks back chronologically.
- **Phase 2b — head retrain** (#18): snapshot-driven `(embedding, cluster_id)` supervised set; LightGBM multiclass → treelite convert → serialize → `write_candidate` into the pointer store. Optional `slancha-local[promote]` extra for the lightgbm/treelite deps. `n_classes >= 2` guard.
- **Phase 2d — cluster-head selector READ path** (#19): 7th head loads via `active_path()` best-effort; cluster_id→cap mapping from JSON sidecar in the same versioned dir; branch fires only when ACTIVE + loaded + confidence > 0.7 (env-tunable). Safe-by-default (no ACTIVE = identical to today; every failure is inert, never raises).
- **Phase 2c — promote_head WRITE path** (#20, #21): orchestrator wires `head_retrain → run_eval_pair (incumbent + candidate on the same holdout in the same run with a shared judge) → gate.decide(champion=incumbent, challenger=candidate) → on ACCEPT: atomic `write_candidate` + `promote` + `append_verdict` / on REJECT: `discard_staged` + `append_verdict`. `slancha promote-head --dry-run` CLI subcommand for side-effect-free probing. Dispatcher + Scorer Protocols + thin httpx defaults.
- **Cap-vocabulary contract enforcement** (#21): `KNOWN_CAPS = frozenset({"coding","math","general"})` shared between writer (`train.promote_head`) and reader (`classifier.cluster_head._CLUSTER_CAP_TO_MODEL_CAP`). The writer parses the upstream compound `<domain>_<difficulty>` `classifier.route` form (e.g. `code_easy` → `coding`, `math_hard` → `math`, everything else → `general`) and defensively raises `PromoteHeadError` at promote-time if any cap somehow ends up out-of-vocab — never silent serve-time no-op. Expanding the vocabulary requires updating both reader and writer in the same change.

The result: traffic → stable clusters → retrained head → eval against incumbent → reversible promote → cluster-aware routing for coding/math vs generalist, all event-sourced and operator-reversible via `gate --promotions-log` rollback.

## [0.0.1] — 2026-05-09

Initial private alpha. **Not yet on PyPI.**

### Added
- FastAPI proxy at `/v1/chat/completions` (OpenAI-compatible, streaming + non-streaming)
- `/v1/models` endpoint — clients can list available routable models including the synthetic `auto`
- `/v1/decisions/last` and `/v1/decisions/{request_id}` — read-only routing-decision history
- `/health` and `/health/detailed` endpoints
- `LocalClassifier` — runs in-process via 6 treelite XGBoost heads + mmBERT-small ONNX embedder. ~4ms classifier latency on M2 Max CPU
- `CloudClassifierClient` — opt-in upgrade to cloud-v-next via `SLANCHA_CLASSIFIER_KIND=cloud`
- `RulesFallbackClassifier` — keyword-based fallback (kicks in if treelite unavailable)
- `OllamaBackend` — speaks `/api/tags` for probe and `/v1/chat/completions` for dispatch
- `LlamaCppBackend` — speaks `/v1/models` for probe and `/v1/chat/completions` for dispatch
- `BackendRegistry` + `CapabilityProbe` (TTL-cached merged catalog)
- `LocalTraceWriter` — JSONL traces with consent gate; macOS-symlink-safe path validation
- `DecisionTraceHeaderMiddleware` — emits `slancha-decision-trace` HTTP response header on every chat response
- CLI: `slancha serve | doctor [--capture] | trace [--last N] | why <id> | brag | tui | bench [--upload] | demo | version`
- `slancha bench` — adversarial-self-bench (17 prompts) with per-head accuracy + latency p50/p95/p99 scorecard
- `slancha demo` — sends 5 representative prompts through the proxy and prints the decision-trace header for each
- `slancha tui` — htop-style live routing dashboard (textual)
- `slancha brag` — shareable ASCII summary of routing activity
- ADR-001 (`docs/adr/001-rust-port.md`) — Python v0.1, Rust v0.2 port plan
- ADR-002 (`docs/adr/002-privacy-red-lines.md`) — five committed-in-writing red lines
- Adversarial regression set (`tests/privacy/adversarial_prompts.json`) — 17 entries covering jailbreak / PII / tool / domain / language
- Stub Ollama (`scripts/stub_ollama.py`) — for live testing without installing Ollama

### Test posture
- 74 unit+integration tests passing in 0.7s
- 17 adversarial regression tests passing
- Live e2e verified: streaming + non-streaming round-trip, decision-trace header, zero egress in default mode

### Known issues (v0.1)
- Bench accuracy: **70.6%** overall — domain + language perfect, jailbreak / PII / tool heads have weaknesses (tracked in README)
- Spanish/non-English prompts sometimes false-positive on jailbreak head — surfaced in trace, never auto-rejected
- Streaming token count is crude (counts SSE deltas)
- Wheel size: ~20MB (28MB unpacked) — bundles classifier weights so install is one step

### Deferred to v0.1.1
- RouterBench full benchmark integration
- Retrained binary heads (multilingual + r/LocalLLaMA-style prompts)
- `slancha export` for trace-bundle upload
- HF Spaces demo page

### Deferred to v0.2 (per ADR-001)
- Rust single-binary port
- vLLM, MLX, LM Studio backends
- Cloud escalation when local can't handle (BYOK + slancha cloud)
- Brew tap publication
