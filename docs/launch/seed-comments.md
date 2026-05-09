# Pre-written launch-day seed comments

Show HN top comments are highest-leverage real estate. Post these from the project-team account ~5–20 minutes after the OP goes live. **Always disclose authorship.** Never sock-puppet.

---

## Seed 1 — Technical primer (post T+5min)

> Author here. Quick technical primer for those interested:
>
> The classifier is mmBERT-small INT8 (Apache 2.0) producing 512-dim embeddings + 6 treelite-compiled XGBoost heads on top (domain, difficulty, language, jailbreak, PII, tool-calling). Inference path is ~4ms on a M2 Max CPU, embedding → 6 head predictions → rule selector picks the model from your local catalog.
>
> The rule selector is a flat ruleset (`src/slancha_local/classifier/local.py`) — first match wins. We chose this over a learned end-to-end policy because (a) it's auditable, (b) you can patch it in your own deployment without retraining, (c) the trace headers can name the actual rule that fired in plain English.
>
> RouterBench was the only repro-able third-party benchmark we found for this category — it has known limitations (single-turn, ground-truth-correct-model assumed) but it's reproducible.
>
> Happy to talk implementation tradeoffs.

---

## Seed 2 — Privacy specifics (post T+8min)

> On privacy: I've been on the other side of this — I've audited tools that claimed to be "local" and were quietly making cloud calls. The way to verify any tool's claim is `tcpdump -i any -n` while it runs. We pinned a screenshot at `docs/no-phone-home.png` (see the repo) of the default install routing 100 prompts with zero outbound packets except DNS for 127.0.0.1.
>
> The tool itself has `slancha doctor --capture` which prints exactly what it would send on the next request. Default: nothing.
>
> If you find an outbound packet in default config, it's a release-blocking bug. Open an issue and we'll patch within 24h.

---

## Seed 3 — Comparison (post T+12min)

> Common comparison question; quick framing:
>
> - **LiteLLM:** rule-based gateway. You write the rules. Great for unified-API. Different mechanism than ours (no learned classifier).
> - **TensorZero:** loop-complete OSS, 11.3K stars. ML-team-with-Docker-Compose audience. Bandit selection per declared function. Different deployment shape.
> - **NotDiamond:** hosted classifier, no BYOK to inference providers. Different business model.
> - **OpenRouter / Vercel / Cloudflare:** cross-provider cloud gateways. Don't help if you have local models.
>
> Our specific niche: prosumer / single-machine / local-models-already-on-disk. The brew install + zero-signup is the entry point; trace headers + reproducible bench are the differentiators.
>
> If you fit a different niche, one of the above is probably the right call.

---

## Seed 4 — Performance (post T+20min, after we see the thread tone)

> RouterBench score on our reference rig (RTX 4090 + Ollama + qwen3:8b + codestral:22b + llama-3.3:70b):
>
> ```
> [PASTE BENCH SCORECARD WHEN AVAILABLE]
> ```
>
> p95 routing overhead: ~12ms (CPU classifier) on top of whatever your local backend takes. We're targeting <2ms p99 in the Rust port (v0.2, 4 weeks out — see ADR-001 in the repo).
>
> Run on your hardware: `slancha bench --samples 1000`. Posts to leaderboard if you `--upload`.

---

## Anticipated comments + canned responses

Reuse `slancha-business/strategy/slancha-local-plans/ITER2-viral-mechanics.md` §3 for the full set.

Common cases:

- **"Why send my embedding to your servers?"** → it doesn't unless you opt in. Local classifier is default. `slancha doctor --capture` proves it.
- **"What's your moat if you open-source?"** → released = snapshot; cloud = always 1–2 quarters ahead. Llama playbook. Closed competitors lose to forks; we lose to ourselves a quarter from now.
- **"How is this different from LiteLLM?"** → LiteLLM is rule-based; you write the rules. We're learned-classifier; we decide for you. Both can coexist.
- **"How is this different from TensorZero?"** → they're loop-complete OSS for ML-teams-with-bandwidth-to-run-Docker-Compose. We're brew-install for prosumers. Different deployment shape.
- **"RouterBench is gameable"** → fair. It's reproducible, not perfect. Better methodology PRs welcome.
- **"Wheel is huge"** → 250MB. Includes weights so install is one step. v0.1.1 ships separate `slancha-local-no-weights` for CI.
- **"Windows?"** → WSL works. Native is on the v0.4 list.
- **"PII / jailbreak detector edge cases?"** → trained heads, ~3% FP rate on adversarial set. PRs to `tests/privacy/adversarial_prompts.json` welcome.
