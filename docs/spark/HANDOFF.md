# Spark handoff — slancha-local train pipeline

> **You are the Spark-side Claude agent.** This file tells you what the Mac-side
> agent has been building and what to do next. Last updated `iter 3` of
> `/loop full auto build mode for slancha-local train pipeline`.

## Context (read first)

- Project repo: `git@github.com:SlanchaAi/slancha-local.git`. Private.
- Current branch: `main`. Latest commit: see `git log --oneline -10`.
- Mac-side agent has shipped 20+ commits over 3 iterations. v0.0.1 inference is feature-complete (proxy, classifier, 6 backends, gallery, TUI, brag, export, train-bundle CLI).
- Tests: 147 unit+integration green + 17 adversarial regression. Lint clean.
- **Read order before doing anything:**
  1. `~/Source/slancha-local/CHANGELOG.md`
  2. `~/Source/slancha-local/docs/train/SPARK-RUNBOOK.md`
  3. `~/Source/slancha-local/docs/train/MULTIMODAL.md`
  4. `git log --oneline -20` from the slancha-local checkout

## Your job

Stand up the train-pipeline production substrate on Spark. The Mac side wrote
the code; you wire it to GPUs.

### Phase A — receiver + bundle ingest (do first)

1. `git clone git@github.com:SlanchaAi/slancha-local.git ~/Source/slancha-local`
2. `cd ~/Source/slancha-local && source ~/cu130_env.sh && source ~/venv_cu130/bin/activate`
3. `uv pip install -e ".[dev]"` (or fall back to `pip install -e ".[dev]"` if uv missing in cu130_env)
4. Smoke: `python -m pytest tests -q` — should be 147 green.
5. systemd unit `slancha-train-receiver.service` per `docs/train/SPARK-RUNBOOK.md` step 5.
6. `systemctl --user daemon-reload && systemctl --user enable --now slancha-train-receiver`
7. Verify `curl localhost:8765/healthz` returns `{"status":"ok",...}`.
8. (Optional) Cloudflare tunnel: `train.laulpogan.com` → `127.0.0.1:8765`. Pattern in main `CLAUDE.md`.

### Phase B — GPU verify + axolotl smoke (before any real FT)

1. `nvidia-smi` — note `memory.free` baseline. Should show >100GB free idle.
2. `python -c "import torch; print(torch.cuda.get_arch_list())"` — confirm `sm_120` is in the list (GB10 capability).
3. `python -c "from slancha_local.train.spark_runner import precheck_gpu; print(precheck_gpu(40_000))"` — should return `(True, ...)`.
4. Synthetic FT smoke: emit a 200-sample dummy bundle, run `slancha train-bundle` → axolotl on Qwen3-0.6B (smallest), 1 epoch, lora_r=8. Verify it completes without OOM. **DO NOT use Qwen3-8B for the smoke** — only the production runs should saturate.
5. Spot-check `nvidia-smi` mid-run; abort with `kill -9` if memory.free drops below 20GB. The watchdog in `train/spark_runner.py::precheck_gpu` only gates the launch; runtime monitoring is your responsibility.

### Phase C — feedback loop back to Mac

Mac side has `slancha export` which produces a tar.gz bundle. Patterns to receive these:

1. **HTTP push** (preferred when tunnel up): Mac runs `curl -F file=@bundle.tar.gz https://train.laulpogan.com/v1/traces/bulk` → receiver stores at `~/.slancha-train/storage/<bundle_id>/`.
2. **git-as-wire** (fallback, per memory `git_as_wire_validated.md`): Mac commits bundle metadata to a private wire-repo, Spark sync daemon pulls, kicks job.
3. **mcp_agent_mail** (substrate-up, per memory `spark_mcp_agent_mail.md`): SSH tunnel `localhost:8775 → spark:8765`. Send/receive coordination messages.

Whichever channel you confirm working first, document the chosen channel + the exact commands in `docs/spark/CHANNEL.md` and commit. Future iterations of this handoff will assume that channel.

## What's still in flight on Mac

Mac-side iter 4+ queue (will land in commits as you read this):
- ClickHouse storage backend behind `SLANCHA_TRAIN_STORAGE=clickhouse` flag
- ComfyUI image-gen backend stub + `POST /v1/images/generations`
- Variant pick wired into `proxy/chat.py` decision pipeline (Thompson sampling)
- piper TTS + whisper.cpp ASR backends

You don't need to wait for these. Phases A–C above are independent.

## Don't

- Don't re-run a smoke FT that just OOM'd; bail and trim hyperparams (lora_r 16→8, max_seq_len 2048→1024, batch_size 4→2) before retry.
- Don't promote a LoRA from a smoke run. Smoke = correctness only. Promotion gated on `slancha_local.train.eval.evaluate()` win-rate ≥ 0.55 per `RouteSpec.eval`.
- Don't rebase or force-push `main` of slancha-local — Mac-side agent uses linear history.
- Don't break the privacy red lines (ADR-002): default install must make zero non-loopback calls. Anything new that calls out gates on explicit opt-in.

## Reply to Mac

Once Phase A (receiver running) is green, write a one-line status to
`docs/spark/STATUS.md` and commit + push. Mac side will see it in the next
`/loop` iteration.

Format:
```
2026-05-09T??:??:??Z [phase-a] receiver up · localhost:8765/healthz ok · sm_120 confirmed · 147 tests green
```

## Open questions for Mac (no rush)

- Should the receiver enforce auth? Currently open on 127.0.0.1; Cloudflare tunnel adds zero auth.
- Log rotation strategy on `~/.slancha-train/storage/` once it grows past 100GB.
- Whether `feedback` traces (👍/👎) should be a separate route from `prompt+response` traces, or interleaved.
