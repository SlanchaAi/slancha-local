# slancha-local

Local LLM router. OpenAI-compatible. Apache 2.0. The classifier ships in the wheel — your prompts never leave the box by default.

```
brew install SlanchaAi/tap/slancha-local         # not yet available; pip for now
pip install slancha-local
slancha serve
# point any OpenAI-compatible client at http://127.0.0.1:8000
```

**Fastest path (zero host setup):** use [Docker compose](#docker-compose) below — it bundles Ollama + the proxy, no Python or PATH fuss.

**Windows:** use Python 3.12 (3.10 is too old), and the module form — the `slancha` console script often isn't on PATH:

```powershell
winget install -e --id Python.Python.3.12
py -3.12 -m pip install slancha-local
py -3.12 -m slancha_local serve
```

slancha-local sits in front of Ollama, llama.cpp, vLLM, MLX, LM Studio, or any OpenAI-compat endpoint and picks the right model per prompt with a small classifier (mmBERT-small embedder + 6 treelite heads, ~150MB, runs on CPU in ~10ms). Every routed request comes back with a `slancha-decision-trace` HTTP response header naming domain, difficulty, picked model, fallbacks, and reason in plain English.

> **Classifier runtime note:** the local classifier needs the `treelite` runtime (and `libomp` on macOS — `brew install libomp`). If it can't load on your platform, slancha-local **automatically falls back to rules-based routing** rather than failing — routing still works, just without the learned heads. `slancha doctor` shows which classifier is active and how to enable the learned one.

## Docker compose

```bash
git clone https://github.com/SlanchaAi/slancha-local.git
cd slancha-local
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen3:8b
curl localhost:8000/v1/chat/completions -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
```

## Supported backends

| backend | env var to enable | default url | default state |
|---|---|---|---|
| Ollama | `SLANCHA_OLLAMA_ENABLED` | http://127.0.0.1:11434 | **on** |
| llama.cpp server | `SLANCHA_LLAMACPP_ENABLED` | http://127.0.0.1:8080 | **on** |
| vLLM | `SLANCHA_VLLM_ENABLED=true` | http://127.0.0.1:8000 | off |
| MLX (mlx_lm.server) | `SLANCHA_MLX_ENABLED=true` | http://127.0.0.1:8081 | off |
| LM Studio | `SLANCHA_LMSTUDIO_ENABLED=true` | http://127.0.0.1:1234 | off |
| any OpenAI-compat | `SLANCHA_GENERIC_OPENAI_BASE_URL=...` | (none) | off until URL set |

`slancha-local catalog` shows the merged routable model list across all enabled backends.

## Default install: zero phone-home

The classifier runs locally in-process. The default install makes zero outbound network calls except to your local backends (Ollama on `127.0.0.1:11434`).

Verify:

```bash
slancha doctor --capture
# prints exactly what the next request would egress (default: nothing)

# external verification:
sudo tcpdump -i any -n 'not (host 127.0.0.1) and not (port 53)' &
curl localhost:8000/v1/chat/completions \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
# should capture zero packets
```

## Privacy red lines

See [`docs/adr/002-privacy-red-lines.md`](docs/adr/002-privacy-red-lines.md). Five committed-in-writing limits we won't cross.

## Opt-in tiers (unlock cloud + FT)

```bash
# Opt in to share embeddings with the latest cloud classifier ($9/mo, "experimental v-next"):
SLANCHA_CLASSIFIER_KIND=cloud SLANCHA_API_KEY=... slancha serve

# Opt in to capture full prompt+response pairs locally (for FT corpus export):
SLANCHA_SHARE_TRACES=true slancha serve

# Opt in to send raw prompts to the cloud classifier (instead of just embeddings):
SLANCHA_SHARE_PROMPTS=true slancha serve
```

All three opt-ins are independently togglable. None are required for the local install.

## CLI

| Command | What it does |
|---|---|
| `slancha serve` | Start the proxy on `127.0.0.1:8000` |
| `slancha doctor` | Probe backends + classifier config; print status |
| `slancha doctor --capture` | Print every byte the next request would egress |
| `slancha trace --last 10` | Show the last 10 routing decisions in a Rich table |
| `slancha why <request_id>` | Explain a routing decision in plain English |
| `slancha brag` | ASCII summary of your routing activity (shareable) |
| `slancha bench` | Run RouterBench on your local stack (v0.1.1) |
| `slancha tui` | htop-style live-routing TUI (v0.1.1) |
| `slancha gallery` | Localhost web UI of your model collection (v0.1.1) |
| `slancha version` | Print version |

## Roadmap

- **v0.1** (this release): Ollama + llama.cpp + vLLM + MLX + LM Studio + any-OpenAI-compat backends (see the table above — non-Ollama are off by default). Local classifier. CLI + decisions endpoints.
- **v0.1.1** (next week): TUI + brag mode + gallery + bench harness + RouterBench numbers.
- **v0.2** (4 weeks): Rust port (single binary, ~30MB).
- **v0.3+**: opt-in trace export → FT credits, community classifier registry.

## Architecture

See [`docs/architecture.md`](docs/architecture.md). Short version:

```
Client → POST /v1/chat/completions
    │
    ▼
slancha-local proxy (FastAPI)
    │  1. mmBERT-small embed (~5ms, CPU)
    │  2. 6 treelite heads classify (domain/difficulty/lang/jailbreak/pii/tool)
    │  3. Route selector picks target from local capabilities
    │  4. Dispatch to local backend
    │
    ▼
Ollama (or llama.cpp / vLLM / MLX / LM Studio)
```

## Known issues (v0.1)

`slancha bench` self-bench scores **70.6%** on the bundled adversarial set. Honest breakdown:

| head | accuracy | note |
|---|---|---|
| domain | 100% | strong on MMLU-Pro categories |
| language | 100% | en/zh/es/fr/de/ja correctly identified |
| pii | 80% | misses some api-key formats; flags some legit decoys |
| jailbreak | 50% | over-fires on benign English ("tell me a joke about cats" → 0.999) and misses some real attempts |
| tool_calling | 33% | head essentially under-trained |

**Practical implication:** the proxy does NOT auto-reject on jailbreak/PII flags by default — the signal goes into the `slancha-decision-trace` header for downstream policy. Your local model has its own safety alignment; we don't second-guess it.

**v0.1.1 plan:** retrain the binary heads on a broader corpus (multilingual + r/LocalLLaMA-style prompts). Track via GitHub issues.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The classifier weights themselves are also Apache 2.0. They ship as a snapshot from when this version was built; a newer version may be available via the cloud classifier upgrade path (`SLANCHA_CLASSIFIER_KIND=cloud`).

## Contributing

Issues + PRs welcome. Adversarial prompts that break the jailbreak/PII detector go in `tests/privacy/adversarial_prompts.json` — send a PR with the prompt + expected flag.

## Related

- Slancha cloud router (paid): https://slancha.ai
