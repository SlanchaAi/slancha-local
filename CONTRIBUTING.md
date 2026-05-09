# Contributing

slancha-local is Apache 2.0. Issues and PRs welcome.

## Quick start (developer)

```bash
git clone https://github.com/SlanchaAi/slancha-local.git
cd slancha-local
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# macOS users: install libomp for treelite (the local classifier)
brew install libomp
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

pytest tests -v        # 74 tests + 17 adversarial regression
ruff check src tests
ruff format --check src tests
```

## High-impact contributions

These are the gaps where a thoughtful PR moves the project forward most:

### A — Adversarial prompts that break the classifier

The `tests/privacy/adversarial_prompts.json` regression set is intentionally small (17 entries). PRs that add prompts which **should pass but don't** (or vice-versa) are gold — they let us see classifier weaknesses we missed.

Format:

```json
{
  "id": "your-prompt-id",
  "prompt": "Your prompt text here",
  "expected": {"jailbreak": false, "domain": "computer science"}
}
```

The set is the source of truth for `slancha bench` per-head accuracy.

### B — Backends

Phase 1 ships Ollama + llama.cpp. Phase 2 wants vLLM, MLX, LM Studio, generic OpenAI-compat. The pattern is `src/slancha_local/backends/<name>.py` implementing the `Backend` ABC. Probe + chat + chat_stream methods. ~100 LOC each.

### C — Classifier improvements

The v1 classifier is imperfect (see "Known issues" in README). Retraining the binary heads on multilingual + r/LocalLLaMA-style prompts would lift the bench number. PRs welcome but the training pipeline is currently in `slancha-api` (private repo); reach out if interested in collaborating.

### D — Frontend integrations

Drop-in setup guides for cline, aider, Continue.dev, OpenWebUI, Cursor, etc. These live in `docs/integrations/` (TBD). Each: install slancha-local, point base_url at it, paste the screenshot of the trace header.

## Code style

- Python 3.11+
- `ruff check + ruff format` are the canonical formatters
- Type hints on public APIs
- Pydantic v2 for any wire schemas
- Tests for new functionality (we hold ≥80% line coverage as a soft target)

## Commit style

Conventional Commits-ish:

- `feat(scope): ...` — new functionality
- `fix(scope): ...` — bug fix
- `docs(scope): ...` — docs only
- `test(scope): ...` — test only
- `chore(scope): ...` — build / tooling

PR description mentions which test added/changed.

## License

By contributing, you agree your contribution is licensed under Apache 2.0.

## Privacy red lines

If your PR causes the default install to make any non-loopback network call, the PR is a release-blocker until either (a) the call is removed, or (b) the call is gated behind explicit opt-in. See [ADR-002](docs/adr/002-privacy-red-lines.md). The CI test `test_default_install_makes_zero_calls_to_api_slancha_ai` enforces this.
