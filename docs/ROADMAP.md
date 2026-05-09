# slancha-local roadmap

> **Status as of 2026-05-09:** v0.0.1 base build is feature-complete and 176 tests green.
> The launch path is the blocker, not more features. This document freezes everything
> built-but-unverified or designed-but-unbuilt as a plan, so we ship v0.0.1 first
> and let real user feedback dictate iter 4+.

## Current state

| Layer | Status |
|---|---|
| FastAPI proxy (`/v1/chat/completions` + `/v1/models` + `/v1/decisions/*` + `/health*`) | ✅ shipped, 176 tests green |
| 6 chat backends (Ollama, llama.cpp, vLLM, MLX, LM Studio, generic OpenAI-compat) | ✅ shipped |
| ComfyUI image backend + `/v1/images/generations` | ✅ shipped, opt-in |
| mmBERT classifier (6 treelite heads) | ✅ shipped |
| Decision-trace HTTP header (load-bearing differentiator) | ✅ shipped |
| 12 CLI commands (serve doctor trace why brag tui bench demo catalog gallery export train-bundle) | ✅ shipped |
| Train pipeline (cluster + bundle + axolotl + Spark runner + receiver + storage) | ✅ shipped |
| TensorZero patterns (declarative TOML config + pluggable providers + 2-tier eval) | ✅ shipped |
| Real Fireworks/Together/OpenAI HTTP fine-tunes API (dry-run gated) | ✅ shipped |
| Real OpenAI-compat judge call | ✅ shipped |
| Thompson-sampling variant bandit (`variants.py`) | ✅ shipped, NOT WIRED into proxy/chat.py yet |
| Pluggable Storage (JSONL default + ClickHouse fan-out) | ✅ shipped |
| Spark HANDOFF runbook + smoke script | ✅ shipped |
| **Spark smoke verification** | ⏳ **NEXT — gates everything below** |
| Public repo + PyPI + asciinema + launch post | ❌ blocking, not started |
| James Apache 2.0 sign-off | ❌ pinged via Discord earlier; not closed |

## Why we're freezing iter 4+ as plan

iter 1–3 produced 22 commits worth of plumbing (variant bandit, ClickHouse fan-out, 3 cloud
fine-tune providers, judge call, ComfyUI image backend) without a single real prompt flowing
through any of it. That's YAGNI at scale — the load-bearing question is "does the base build
run on Linux/CUDA the same as on macOS dev?" and "do r/LocalLLaMA people actually use it?",
neither of which is answered by more code.

The next concrete blocker is **Spark smoke** (`scripts/spark_smoke.sh`). After that, **launch ops**
(public repo, PyPI, asciinema, post). After *that*, plan items below get re-evaluated against
actual user feedback rather than speculation.

## Phase 0 — Spark smoke (NEXT)

Run `bash scripts/spark_smoke.sh` on `promaxgb10-d325.local`. See `docs/spark/HANDOFF.md` for
the full Spark-side runbook. Smoke verifies:

- clone + venv + install on linux/aarch64 + Python 3.12
- libgomp present (treelite native dep on Linux)
- 176 tests pass (Mac dev parity)
- proxy starts, `/health/detailed` responds 200
- decision-trace header emits on a real chat completion (assuming Ollama installed)

Report by editing `docs/spark/STATUS.md` and committing. Mac side reads it on next session.

## Phase 1 — Launch ops (BLOCKED on Phase 0 green)

| Step | Status |
|---|---|
| Flip github.com/SlanchaAi/slancha-local public | ❌ |
| `python -m hatchling build` + PyPI publish | ❌ |
| Brew tap publish | ❌ |
| Record asciinema via `docs/launch/asciinema-script.sh` | ❌ |
| James Apache 2.0 sign-off (already pinged on Discord; close the loop) | ❌ |
| Run RouterBench, paste numbers in launch post | ❌ |
| Post to r/LocalLLaMA (draft in slancha-business/strategy/slancha-local-plans/V2-LAUNCH-POST.md) | ❌ |

## Phase 2 — Variant bandit wiring (DESIGNED, NOT WIRED)

The Thompson-sampling `VariantStore` in `src/slancha_local/variants.py` is built and tested
in isolation but not yet wired into the proxy decision pipeline. Full design:

- Before backend dispatch in `src/slancha_local/proxy/chat.py`, query
  `VariantStore.list_variants(classifier.route)`.
- If non-empty, call `pick(route)` to Thompson-sample a variant and override `decision.target`
  to the variant's `last_target`.
- Add `variant_id: str | None` to the trace JSON schema; bump `schema_version` to 2.
- `slancha variants` CLI: `list <route>`, `register <route> <variant_id> <target>`, `summary`.
- `record_outcome()` hook fires on (a) explicit user thumbs-up/down via `/v1/decisions/{id}/feedback`,
  (b) heuristic pass (non-empty + status=ok + latency_ms < SLA), or (c) judge eval at FT promotion time.

**Why not yet:** zero users → zero variants registered → bandit is a no-op. Wire the day a
second LoRA exists for any route.

## Phase 3 — Audio backends (DESIGNED, NOT BUILT)

Per `docs/train/MULTIMODAL.md` v0.3 / v0.4:

- **piper TTS** + `POST /v1/audio/speech` (OpenAI-compat: model + input + voice → audio/wav).
  Probe via piper CLI presence; opt-in via `SLANCHA_PIPER_ENABLED=true`.
- **whisper.cpp ASR** + `POST /v1/audio/transcriptions`. Probe via whisper.cpp HTTP server
  `/inference`; opt-in via `SLANCHA_WHISPER_ENABLED=true`.

**Why not yet:** ComfyUI image was the prosumer-popular hook; audio is incremental. Adds
surface area before there's a demonstrated demand. Defer until a user asks.

## Phase 4 — Spark productionization (DESIGNED)

Per `docs/spark/HANDOFF.md` Phases A/B/C and `docs/train/SPARK-RUNBOOK.md`:

- Receiver as systemd unit on Spark, exposed via Cloudflare tunnel `train.laulpogan.com`
- Cron-driven hourly bundle ingestion → KMeans cluster → axolotl SFT
- Auto-promotion: per-quarter best LoRA per route gets pushed back as a downloadable artifact
- Multi-LoRA hot-swap via vLLM for routes with >1 promoted variant

**Why not yet:** zero traces in production → nothing to cluster, nothing to FT.

## Phase 5 — Mode classifier (DESIGNED)

Per `docs/train/MULTIMODAL.md` §C, option 2: train a 4-class head (text / image / audio_in / audio_out)
on prompt embeddings so `POST /v1/chat/completions` auto-routes "draw me a cat" prompts to ComfyUI.

**Why not yet:** needs ~5K mode-tagged traces from production traffic. Defer to v0.5.

## Phase 6 — Cloud judge productionization (DESIGNED)

`src/slancha_local/train/eval.py` currently resolves the judge endpoint via env-var fallback
chain (SLANCHA_JUDGE_URL → SLANCHA_API_KEY → OPENAI_API_KEY → tie). For real auto-promotion,
we want:

- A single hosted judge endpoint at api.slancha.ai/v1/judge that wraps a strong model + a
  rubric prompt + caching by (prompt-hash, base-resp-hash, ft-resp-hash)
- Scoring multiple FT runs per prompt with bias-mitigation (response order randomization)
- A small public eval set so users can verify the judge is sane

**Why not yet:** chicken-and-egg with slancha cloud existing as a billable service.

## Phase 7 — ClickHouse productionization (DESIGNED)

Storage abstraction is in place (`src/slancha_local/train/storage.py`). To turn it on:

- Spin up ClickHouse on Spark or external (~$10/mo on a t3.small)
- Set SLANCHA_TRAIN_STORAGE=clickhouse + the four CLICKHOUSE_* env vars
- JSONL stays as durable backing; CH is the queryable analytics fan-out

**Why not yet:** with ten traces total in production, JSONL plus jq is enough.

## Anti-roadmap (will NOT build)

- Video gen (compute too heavy for prosumer hardware; v1+ if at all)
- Cross-modal routing loops (text → image → text); too speculative; user can chain manually
- Embedding-only RAG mode; different abstraction; fork into a sibling project if needed
- Multi-tenant cloud-hosted slancha-local (defeats the privacy positioning)
- Re-implementing the LiteLLM / OpenRouter feature set (we route to *local* backends; that's
  the entire point)

## Decision log

- **2026-05-09**: froze iter 4+ as roadmap; Phase 0 (Spark smoke) gates everything below.
  Reason: 22 commits of plumbing without a single real prompt = YAGNI at scale. Ship base
  build first.
