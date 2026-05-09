# Changelog

All notable changes to slancha-local. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), [SemVer](https://semver.org/).

## [Unreleased]

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
