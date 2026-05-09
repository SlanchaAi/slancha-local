# Spark router test plan — concurrent models, no OOM

> Goal: prove slancha-local routes correctly across diverse local models on Spark
> without crashing the box. This is the integration test that gates the launch
> claim "we route between *your* local models intelligently". Today the router is
> only smoke-tested with zero models loaded.

## Hardware budget

DGX Spark GB10 — **128GB unified memory** (per nvidia-smi); per memory `finetuning_corpus.md`
the practical ceiling for FT use is **121GB** (the rest goes to OS + kernel + Python + Ollama
runtime).

**Reservation breakdown:**

| Reserved for | Size |
|---|---|
| OS + kernel + DGX baseline | ~4 GB |
| Ollama runtime + Python + slancha-local proxy | ~4 GB |
| KV cache headroom (5 concurrent inferences × ~6K ctx avg) | ~25 GB |
| Buffer for memory pressure / fragmentation | ~10 GB |
| **Available for loaded model weights** | **~85 GB** |

Concurrent-model plan must stay under 85GB resident. Budget below uses ~57GB → 28GB headroom.

## Model selection — Tier 1 (required, ~38GB)

Covers all four primary classifier routes: easy/general, medium/general, code,
creative-writing. Plus a reasoning specialist.

| Model | Ollama tag | Quant | Approx VRAM | Route(s) it covers |
|---|---|---|---|---|
| Qwen3-4B | `qwen3:4b` | q4_K_M | ~2.6 GB | general_easy |
| Qwen3-8B | `qwen3:8b` | q4_K_M | ~4.7 GB | general_medium, fallback for most routes |
| Codestral-22B | `codestral:22b` | q4_K_M | ~13.0 GB | computer_science_* |
| Gemma2-9B-it | `gemma2:9b` | q4_K_M | ~5.5 GB | creative_writing_* |
| Phi-4 (14B) | `phi4:14b` | q4_K_M | ~8.5 GB | reasoning_medium / general_hard |
| DeepSeek-R1-distill-Qwen-14B | `deepseek-r1:14b` | q4_K_M | ~8.5 GB | reasoning_hard |
| **Subtotal Tier 1** | | | **~42.8 GB** | |

Numbers are **approximate** — verify each pull with `ollama show <tag> | grep parameter` and
`du -sh ~/.ollama/models/blobs/*<tag>*` before assuming. Quant defaults to q4_K_M unless overridden.

## Model selection — Tier 2 (optional, +12GB; only if testing image route)

| Model | Vehicle | Approx VRAM | Route |
|---|---|---|---|
| Flux-schnell | ComfyUI + diffusers | ~12 GB | image_generation |

Comfy holds it in shared memory; LRU evicts when chat models need RAM under pressure. Skip
this tier if we're not exercising `/v1/images/generations` in this round.

Tier 1 + Tier 2 = ~55GB resident. Still 30GB headroom.

## DO NOT load these (would push past budget)

- Llama-3.3-70B q4 (~42 GB) — alone fits, but contention with the rest will OOM on KV under load
- Qwen3-30B q4 (~18 GB) — fine standalone; combined with Tier 1 hits 60+, too tight
- xtts-v2 + whisper-large together (~6 GB) — fine in isolation but adds another concurrent runtime
  to manage; defer to dedicated audio test

## Pre-flight

On Spark via SSH:

```bash
# 1. Install Ollama (first run only)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Verify the daemon is up
systemctl --user status ollama || ollama serve >/tmp/ollama.log 2>&1 &
sleep 2
curl -sf http://127.0.0.1:11434/api/tags

# 3. Baseline GPU snapshot
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader

# 4. Pull Tier 1 (in parallel for speed; total ~25GB download)
for m in qwen3:4b qwen3:8b codestral:22b gemma2:9b phi4:14b deepseek-r1:14b; do
  ollama pull "$m" &
done
wait

# 5. Pre-load each model once so first-prompt latency doesn't skew the test
for m in qwen3:4b qwen3:8b codestral:22b gemma2:9b phi4:14b deepseek-r1:14b; do
  ollama run "$m" 'hi' --keepalive 30m
done

# 6. Snapshot resident set
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
# expected: memory.used ≈ 42-45 GB, memory.free ≈ 80-85 GB

# 7. Start the proxy on a free port (8765 is taken by mcp_agent_mail per Spark STATUS findings)
cd ~/Source/slancha-local && source .venv/bin/activate
slancha serve --host 127.0.0.1 --port 8766 >/tmp/slancha-serve.log 2>&1 &

# 8. Sanity
curl -sf http://127.0.0.1:8766/health
curl -sf http://127.0.0.1:8766/v1/models | python3 -m json.tool   # should list 6 models
```

## Test matrix

### A. Routing diversity (24 prompts, 4 per route × 6 routes)

Inputs span domains so the classifier picks different `route` values, then the registry resolves
to different `target` Ollama tags. Validates the "decision-trace shows different models for
different prompts" claim.

| Category | Prompt examples |
|---|---|
| general_easy | "What time is it in Tokyo?", "Spell potato", "Hi", "Give me a joke" |
| general_medium | "Explain the difference between TCP and UDP", "Write a haiku about trees", "Why is the sky blue?", "Summarize WW2 in one paragraph" |
| computer_science_medium | "Implement binary search in Python", "What's the diff between mutex and semaphore?", "Explain Big-O of quicksort", "Refactor this for-loop into a list comprehension: ..." |
| creative_writing_medium | "Write a 4-line poem about loneliness", "Continue this story: ...", "Draft a short pitch for a sci-fi novel about ...", "Write dialogue between a barista and a wizard" |
| reasoning_medium | "If A is taller than B and B is taller than C, who is shortest?", "5 sweaters take 5 hours to dry, how long for 30?", "What's the next number: 2, 6, 12, 20, ?", "Solve: a+b=10, a-b=2" |
| reasoning_hard | "Prove √2 is irrational", "Explain the halting problem with an example", "What's Bayes' theorem applied to spam filters?", "Why does Cantor's diagonal argument prove ℝ is uncountable?" |

**Pass criteria:**
- Every route hit by ≥1 successful 200 response.
- `picked=` field in the decision-trace header maps to ≥4 distinct models across the 24 prompts.
- No 5xx responses from the router itself (backend errors okay if we log them; OOM crash NOT okay).

### B. Concurrency (10 simultaneous requests)

Send 10 prompts in parallel via `asyncio.gather`. Validates that the proxy + classifier + Ollama
handle concurrent dispatch without deadlock or OOM.

```python
import asyncio, httpx, time
PROMPTS = [...24 prompts from matrix A, take first 10...]

async def hit(client, prompt):
    t0 = time.perf_counter()
    r = await client.post("http://127.0.0.1:8766/v1/chat/completions",
        json={"model":"auto","messages":[{"role":"user","content":prompt}],"stream":False},
        timeout=60.0)
    return r.status_code, r.headers.get("slancha-decision-trace",""), (time.perf_counter()-t0)*1000

async def main():
    async with httpx.AsyncClient() as c:
        results = await asyncio.gather(*(hit(c,p) for p in PROMPTS))
        for sc, trace, ms in results:
            print(f"{sc} {ms:.0f}ms picked={trace.split('|')[0] if trace else 'none'}")

asyncio.run(main())
```

**Pass criteria:**
- All 10 return HTTP 200 (or 502 with header — not crash).
- p95 latency under 30s on Spark (large model + cold-ish cache; not a perf benchmark, just liveness).
- No process killed by OOM (check `dmesg | tail -50` for OOM-killer hits after run).

### C. Memory floor watchdog (continuous during A + B)

Run nvidia-smi sampling at 1Hz in background, logging `memory.free`. After test, scan log for
the minimum value.

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -l 1 \
    > /tmp/gpu_floor.log 2>&1 &
SMI_PID=$!
# ... run tests A + B ...
kill $SMI_PID
sort -n /tmp/gpu_floor.log | head -1
# expected: ≥30000 (i.e. ≥30 GB always free)
```

**Pass criteria:** memory.free never dropped below **20 GB** (gives us margin for one extra
concurrent request landing while a big model is generating).

### D. Fault-tolerance (regression of decision-trace header on errors)

Per the iter-3 fix shipped in `0337be2` — verify the header is on the 502 path under real conditions.

```bash
# 1. Kill ollama mid-test
pkill -STOP ollama  # SIGSTOP — process freezes, slancha sees timeouts

# 2. Hit the router
curl -s -D - -X POST http://127.0.0.1:8766/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"stream":false}' \
  | head -25

# Expected: HTTP 502, body contains "local backend error",
# header slancha-decision-trace: picked=local:ollama:... | reason=... | ...

# 3. Resume ollama for cleanup
pkill -CONT ollama
```

**Pass criteria:** 502 response carries a parseable `slancha-decision-trace` header.

### E. Trace audit

After A–D run, read `~/.slancha/traces/*.jsonl`. Verify:

```bash
for f in ~/.slancha/traces/*.jsonl; do
  python3 -c "
import json, sys
for i, line in enumerate(open('$f')):
    t = json.loads(line)
    assert t['schema_version'] == 1
    assert t['mode'] in ('local','cloud','image')
    assert 'classifier' in t and 'decision' in t and 'execution' in t
    assert t['classifier']['route']
    assert t['decision']['target'] == t['execution']['executed_target']
print(f'$f: $(wc -l < $f) traces, all schema-valid')
"
done
```

**Pass criteria:** every trace passes schema check; total trace count ≥ 30 (24 from A + 10 from B).

## Tear-down

```bash
# 1. Stop the proxy
pkill -f "slancha serve" || true

# 2. Stop ollama (frees the resident models)
systemctl --user stop ollama 2>/dev/null || pkill ollama || true

# 3. Verify GPU clean
sleep 2 && nvidia-smi --query-gpu=memory.used --format=csv,noheader
# expected: < 2000 MB (just baseline)

# 4. (Optional) Remove the pulled models if disk pressure
# du -sh ~/.ollama/models  # check first
# ollama rm qwen3:4b qwen3:8b codestral:22b gemma2:9b phi4:14b deepseek-r1:14b
```

Disk: 6 models × ~6GB avg = ~35GB. Spark almost certainly has room; don't auto-rm unless we see disk
pressure.

## Failure handling

| Symptom | Likely cause | Action |
|---|---|---|
| OOM in `dmesg` after concurrent test | KV cache outgrew budget | Drop one Tier 1 model (probably codestral:22b → codestral:7b), re-run |
| All requests pick same model | classifier rules-fallback active | Verify treelite imported; `slancha doctor` should show classifier=local |
| Decision-trace header absent on 502 | Regression of iter-3 fix | Reopen `tests/integration/test_decision_trace_on_errors.py` |
| Ollama unresponsive partway through | Single-process bottleneck | Run with `OLLAMA_NUM_PARALLEL=4`; defer; not a router bug |
| HF 401 on a pull | Gated model; need HF_TOKEN | Set `HF_TOKEN` per `docs/launch/API-KEYS.md` Tier 2 |
| `slancha train-bundle` exits clean but no LoRA produced | dry-run still on | `unset SLANCHA_TRAIN_DRY_RUN` (only for FT runs, not router test) |

## Output artifacts

After a green run, commit these to `docs/spark/`:

- `STATUS.md` — append a new dated line: `2026-MM-DDTHH:MM:SSZ [router-test] pass · 6 models · 30 prompts · p95=Xms · floor=YGB`
- `ROUTER-TEST-RESULTS.md` (new) — full Markdown table of (prompt → picked-model → latency_ms → status)
  for the 24+10 = 34 prompts. This is screenshot-bait for the launch post.
- `~/.slancha/traces/*.jsonl` — copy a redacted sample (consent_at_capture=false anyway, but
  double-check) to `docs/spark/sample-traces.jsonl` for the launch post asciinema.

## What this plan does NOT cover

- Streaming SSE under concurrency (separate test; harder to assert correctness)
- Cross-process FT runs (covered by separate FT smoke; gated on Tier-1 API keys)
- Long-context routing (8K+ token prompts; KV-cache budget would need a redo)
- Backends other than Ollama (llama.cpp, vLLM, MLX, LM Studio — out of scope; covered by unit tests)

These get added once the basic router test passes and we have user signal on which to prioritize.
