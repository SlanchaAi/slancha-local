# ADR-002 — Privacy red lines (what we will never do)

**Status:** ACCEPTED
**Date:** 2026-05-09

---

## Context

slancha-local is built for an audience (r/LocalLLaMA) that has been burned by tools claiming "local" while quietly making cloud calls. The single biggest risk to launch reception is being perceived as another such tool. Marketing claims aren't enough; we need *committed-in-writing red lines* that we publish, link from the launch post, and reference any time someone asks "what won't you do?"

This ADR codifies those red lines and the consequence policy if we ever cross one.

---

## Decision: Five red lines

### Red line 1 — Default install makes zero outbound network calls

**Rule:** the default `slancha-local` install, after `slancha serve`, makes zero outbound network connections. The only network calls allowed are:
- to `127.0.0.1` (loopback) — for backend communication with Ollama / llama.cpp / vLLM / MLX / LM Studio.
- DNS resolution of `127.0.0.1` (which is a no-op).

That's it. No telemetry pings. No version-check pings. No license-validation pings. No "anonymous usage statistics." No CDN fetches. No analytics. No crash reporting.

**Verification:** CI test in `tests/integration/test_chat_e2e_mocked.py::test_default_install_makes_zero_calls_to_api_slancha_ai` asserts no calls hit `api.slancha.ai`. Any outbound call to a non-loopback address is a **release-blocking failure**.

**External verification:** see `docs/no-phone-home.md` for the tcpdump capture method anyone can run.

**Consequence of crossing:** 24h hotfix release. Public retrospective. Refund any paying customers who object.

### Red line 2 — Even on opt-in tiers, raw prompts never leave the box without explicit per-session consent

**Rule:** the `--share-prompts` and `--share-traces` flags are required for prompts/responses to be transmitted off-box. These flags are off by default. Setting them via env var **does not auto-persist**; the CLI prints a confirmation at startup and writes the consent record to `~/.slancha/consent.log` with timestamp, flag value, and CLI version.

**Verification:** unit test asserts the consent prompt fires; CI test asserts the consent log is appended.

**Consequence of crossing:** same as Red line 1.

### Red line 3 — Embeddings are not used to reverse-engineer prompts

**Rule:** even when we receive embeddings (cloud classifier opt-in tier), we commit to:
- Not training a "decoder" model that maps embeddings back to text.
- Not selling embeddings as a derived dataset.
- Not joining embeddings to other PII-bearing data sources to re-identify prompts.

We can train classifiers on embeddings (that's the whole point). We can release aggregated statistics. We cannot attempt prompt reconstruction.

**Verification:** internal policy + audit log on the cloud-classifier service. Not externally verifiable, which is why this red line is a commitment-in-writing.

**Consequence of crossing:** public retrospective. Restitution to opt-in users (refund + dataset-deletion option).

### Red line 4 — No dark patterns or upsell pressure

**Rule:** the local install will not include any of:
- Modal popups asking for an account.
- Time-limited "free trial" countdowns.
- Feature degradation that didn't exist on prior versions.
- Telemetry-driven personalized upsells.
- Email collection without explicit user-initiated signup.

The CLI may *suggest* upgrading to Pro tier in `slancha doctor` output (one line, dismissible) and at quota-cap moments. Beyond that, the upgrade path lives at slancha.ai/local — users who want it find it.

**Verification:** UX review at every major release. Public-feedback channel.

**Consequence of crossing:** rollback to prior UX in next patch release.

### Red line 5 — License-locked features are explicitly listed and never expand without notice

**Rule:** at any version, the list of features gated behind paid tiers is publicly enumerated in `docs/feature-matrix.md`. We commit to:
- Never moving an existing free feature behind a paid tier (only new features can be paid-only).
- Never reducing free-tier limits without 90 days notice and grandfathering existing free users.
- Never disabling a free-tier install via remote kill switch.

If we ever need to change the free/paid boundary, it requires a major version bump and a published rationale.

**Verification:** the feature matrix is in the repo; diffs are visible. Free users can pin to a specific version forever.

**Consequence of crossing:** public retrospective + grandfather affected users.

---

## Why publish red lines as an ADR

1. **Pre-commits us before the temptation arises.** When in 12 months a growth metric is shaky and someone proposes "let's just ping for crash reports," the ADR is the artifact that says "no, we wrote down that we wouldn't." Tied hands are useful.

2. **Becomes a marketing artifact.** The launch post links to this ADR. Cynics can read it and verify the wording.

3. **Defines red lines, not green lights.** We don't promise to do everything users want. We promise to *never* cross specific limits.
