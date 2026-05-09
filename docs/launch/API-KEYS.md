# API keys — what we need for end-to-end testing

> Audience: Paul. Each entry says **what** the key gates, **why** it matters, **how** to get it,
> and **how it gets to the test environment** (env var name + which `.env` file or shell export).

slancha-local enforces **opt-in** for every external network call. None of these keys are
required for the default install / smoke / launch. They unlock specific test paths.

## Tier 1 — minimum viable for end-to-end FT + judge tests

These three are enough to verify the train pipeline against real APIs without burning much money.

### 1. `OPENAI_API_KEY` (required for judge + OpenAI fine-tunes path)

- **Gates:** real judge call in `src/slancha_local/train/eval.py::judge_pairwise_pick`. Also exercises
  the `OpenAIProvider` fine-tunes path (less interesting; OpenAI doesn't do LoRA anyway).
- **Why:** the judge is the load-bearing decision in promotion. A real call (gpt-4o-mini, $0.15/1M
  input) verifies the rubric prompt parses correctly and we get verdicts back, not just `tie`.
- **How to obtain:** https://platform.openai.com/api-keys → create key with `model.read` +
  `chat.completion` scopes. Set spend limit to $5/mo for safety.
- **Cost ceiling for tests:** ~50 judge calls × 200 tokens = $0.01. Negligible.
- **Where it goes:** `~/Source/slancha-local/.env` + `OPENAI_API_KEY=sk-...`

### 2. `FIREWORKS_API_KEY` + `FIREWORKS_ACCOUNT_ID` (required for Fireworks FT real path)

- **Gates:** `src/slancha_local/train/providers/http_providers.py::FireworksProvider` upload→create→poll
  →retrieve flow against the real Fireworks API. This is the user-volume off-Spark FT path; without
  this we never validate anything beyond dry-run.
- **Why:** Fireworks is the "user runs `slancha train` and wants to FT in the cloud" target. Their LoRA
  pricing is good (~$0.40/1M training tokens). Most-likely real-world cloud FT path for users.
- **How to obtain:** https://app.fireworks.ai/settings/users/api-keys → create key. Account ID is
  visible in the URL after login (e.g. `https://app.fireworks.ai/dashboard/<ACCOUNT_ID>`).
- **Cost ceiling for tests:** ~$2 for one FT run on Qwen3-1.5B with 200 examples × 3 epochs. Stay
  small.
- **Where it goes:** `.env` + `FIREWORKS_API_KEY=fw_...` + `FIREWORKS_ACCOUNT_ID=...`

### 3. `TOGETHER_API_KEY` (alternate FT path; redundant with Fireworks but covers a different shape)

- **Gates:** `TogetherProvider` real flow. Together's API is a different shape (single `/v1/files`
  upload, `/v1/fine-tunes` create) — testing both surfaces both providers' edge cases.
- **Why:** Lots of users prefer Together (broader model catalog, sometimes cheaper). Want at least one
  real FT run through their API to verify our adapter.
- **How to obtain:** https://api.together.xyz/settings/api-keys
- **Cost ceiling:** similar to Fireworks, ~$2 per run.
- **Where it goes:** `.env` + `TOGETHER_API_KEY=...`

## Tier 2 — useful but not blocking

### 4. `HF_TOKEN` (HuggingFace — for downloading gated models on Spark)

- **Gates:** Spark can pull Llama-3.3, Qwen3 (gated), Mistral models from HF. Also lets us push
  trained LoRAs back as HF artifacts when `RouteSpec.artifact_dest = "hf:slanchaai/<route>"`.
- **Why:** Some Tier 1 router-test models (notably any Llama variant, sometimes Qwen3-30B) require
  HF auth. Avoids "401 Unauthorized" surprises on `ollama pull`.
- **How to obtain:** https://huggingface.co/settings/tokens → "read" scope is enough; "write" if we
  intend to push LoRAs back.
- **Cost:** free.
- **Where it goes:** Spark side, `~/.huggingface/token` via `huggingface-cli login` OR
  `HF_TOKEN=hf_...` in shell.

### 5. `SLANCHA_API_KEY` (slancha cloud classifier — when slancha-api is up)

- **Gates:** the cloud classifier path (`SLANCHA_CLASSIFIER_KIND=cloud`). Tests that the proxy can fall
  through to slancha cloud when `local` classifier degrades.
- **Why:** symmetry with what users do who set `--classifier=cloud`. Currently slancha-api may not be
  publicly serving classify requests — verify before bothering.
- **How to obtain:** internal — provision via slancha-api / slancha-website admin, or generate via
  `slancha-business` tooling.
- **Where it goes:** `.env` + `SLANCHA_API_KEY=...` + `SLANCHA_CLASSIFIER_KIND=cloud`

## Tier 3 — optional / Phase 4

### 6. Cloudflare API token (`CF_API_TOKEN`)

- **Gates:** programmatic provisioning of `train.laulpogan.com` tunnel for the Spark receiver. Per
  global `CLAUDE.md` "Public uplink — laulpogan.com via Cloudflare" pattern.
- **Why:** lets the Mac trigger Spark FT runs over a public HTTPS endpoint instead of going through
  the local network. Phase 4 production substrate; not blocking.
- **How to obtain:** Cloudflare dash → My Profile → API Tokens → "Edit zone DNS" + "Cloudflare Tunnel"
  scopes for the laulpogan.com zone.
- **Where it goes:** `~/.cloudflared/credentials` per the cloudflared CLI flow; not an env var.

## Don't get

### Anthropic API key — INTENTIONALLY NOT NEEDED

Per global `CLAUDE.md` "Harness Claude Code, not the API": when slancha-local needs to call Claude
(e.g. as a judge model), we wrap the Claude Code CLI subprocess and inherit its OAuth. Setting
`ANTHROPIC_API_KEY` would silently flip from cheap subscription to pay-per-token billing.
**Strip `ANTHROPIC_API_KEY` from any subprocess env we spawn.**

## How to wire all of these into the test environment

```bash
# ~/Source/slancha-local/.env (NEVER commit; .gitignore already excludes it)
OPENAI_API_KEY=sk-...
FIREWORKS_API_KEY=fw_...
FIREWORKS_ACCOUNT_ID=...
TOGETHER_API_KEY=...
HF_TOKEN=hf_...     # only needed on Spark side, can live in ~/.huggingface/token instead
SLANCHA_API_KEY=...  # if testing cloud classifier
```

`pydantic-settings` loads `.env` automatically via `Settings(env_file=".env")`. Tests use
`tests/conftest.py` autouse fixture which strips all of these before each test, so .env never leaks
into pytest runs.

## Verification commands (run after setting each)

```bash
cd ~/Source/slancha-local && source .venv/bin/activate

# OpenAI judge
python -c "
import os
os.environ.pop('SLANCHA_TRAIN_DRY_RUN', None)
from slancha_local.train.eval import judge_pairwise_pick
v, r = judge_pairwise_pick('What is 2+2?', '4', 'four', 'openai:gpt-4o-mini')
print(f'verdict={v} reason={r}')
"

# Fireworks precheck (does NOT spend money — just validates account + key)
python -c "
import os
os.environ.pop('SLANCHA_TRAIN_DRY_RUN', None)
from pathlib import Path
from slancha_local.train.providers.http_providers import FireworksProvider
from slancha_local.train.providers.base import TrainingJob
Path('/tmp/t.jsonl').write_text('{}\n')
p = FireworksProvider()
ok, msg = p.precheck(TrainingJob(route='r', base_model='accounts/fireworks/models/qwen3-1p5b',
    train_jsonl=Path('/tmp/t.jsonl'), val_jsonl=Path('/tmp/t.jsonl'), output_dir=Path('/tmp'),
    hyperparams={}, artifact_dest='local:.'))
print(f'fireworks precheck: ok={ok} msg={msg}')
"

# Together precheck — same shape, swap provider class
```

If any return errors, debug before running full FT smokes. Cheap to do, expensive to skip.
