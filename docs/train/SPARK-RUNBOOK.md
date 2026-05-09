# Spark migration runbook — slancha-train

Target host: `promaxgb10-d325.local` (DGX Spark GB10, ARM64, CUDA 13).
Env: `source ~/cu130_env.sh && source ~/venv_cu130/bin/activate` (per `~/.claude/CLAUDE.md` "Technical defaults").

## What lives on Spark

- **Trace receiver** (`slancha_local.train.receiver:build_receiver_app`)
  - systemd unit `slancha-train-receiver.service`
  - Listens on `127.0.0.1:8765` (Cloudflare tunnel exposes via `train.laulpogan.com` per existing pattern)
  - Storage at `/home/admin/.slancha-train/storage/<bundle_id>/`
- **Train pipeline** (`slancha_local.train.bundle` + `spark_runner`)
  - Run via cron (`0 * * * *` — hourly) or webhook
  - Picks up bundles from storage, clusters, splits train/val, kicks axolotl
- **GPU watchdog** — `nvidia-smi --query-gpu=memory.free` pre-check before any FT launch. Bail if <40GB free.

## Why Spark

- 121GB unified memory ceiling (per memory `finetuning_corpus.md`)
- max_seq_len 2048, lora_r 16-32 max for SFT/DPO
- ARM64 + CUDA 13: prebuilt wheels in `venv_cu130`; flash-attn not always pre-built (check before enabling)
- Already runs Theo's grant FT, Willard plotter FT — pipeline patterns transferable

## Migration steps

1. `git clone git@github.com:SlanchaAi/slancha-local.git ~/Source/slancha-local`
2. `cd ~/Source/slancha-local && uv venv .venv --python 3.12 && uv pip install -e ".[dev]"`
3. `pip install axolotl[flash-attn]` *or* `pip install axolotl` if flash-attn fails on sm_120
4. Verify: `python -c "import torch; print(torch.cuda.get_arch_list())"` — must include `sm_120` for GB10
5. systemd unit at `~/.config/systemd/user/slancha-train-receiver.service`:
   ```ini
   [Unit]
   After=network-online.target
   Wants=network-online.target
   [Service]
   Type=simple
   ExecStart=/home/admin/Source/slancha-local/.venv/bin/uvicorn slancha_local.train.receiver:app --host 127.0.0.1 --port 8765
   Environment=PYTHONUNBUFFERED=1
   Environment=SLANCHA_TRAIN_STORAGE=/home/admin/.slancha-train/storage
   Restart=on-failure
   RestartSec=15
   [Install]
   WantedBy=default.target
   ```
6. `systemctl --user daemon-reload && systemctl --user enable --now slancha-train-receiver`
7. (Optional) Cloudflare tunnel route for `train.laulpogan.com` → `127.0.0.1:8765`. See main `CLAUDE.md` "Public uplink".

## GPU pre-check (load-bearing)

Always before launching axolotl:

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
# bail if first GPU < 40000 MB (40GB)
```

The training run sizes itself to leave headroom; default `max_seq_len=2048`, `lora_r=16`, `batch_size=4`, `gradient_accumulation_steps=4` ≈ 40-60GB peak.

## Coordination from this Mac

Two paths to coordinate without SSH key auth:

1. **git-as-wire** (per memory `git_as_wire_validated.md`): I commit changes here, the Spark sync daemon pulls and runs.
2. **mcp_agent_mail** (per memory `spark_mcp_agent_mail.md`): SSH tunnel `localhost:8775 → spark:8765`. Once tunnel up, agents on both sides send mail.

Either works for "drop a bundle, kick a job, get back a GGUF" pattern.

## Future state (not yet built)

- Multi-route FT: one LoRA per `classifier.route × language` cluster; serve via vLLM with multi-LoRA hot-swap (per axolotl + vLLM docs)
- DPO once we have feedback (👍/👎 captured via the trace's `feedback` field)
- Auto-promotion: every quarter, the best LoRA per route gets pushed back as a downloadable artifact for users running slancha-local
- Multimodal extension: see `docs/train/MULTIMODAL.md`
