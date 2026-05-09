"""pytest conftest — global env hygiene.

Forces SLANCHA_TRAIN_DRY_RUN=1 by default so tests never hit the real
Fireworks/Together/OpenAI APIs even if the dev env has the keys set.
Also clears any user judge endpoints so eval tests deterministically
return ('tie', '...') without network.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _slancha_test_env(monkeypatch):
    monkeypatch.setenv("SLANCHA_TRAIN_DRY_RUN", "1")
    # Clear any judge-config that would trip a real network call.
    for var in (
        "SLANCHA_JUDGE_URL",
        "SLANCHA_JUDGE_API_KEY",
        "SLANCHA_JUDGE_MODEL",
        "SLANCHA_API_KEY",
        "OPENAI_API_KEY",
        "FIREWORKS_API_KEY",
        "FIREWORKS_ACCOUNT_ID",
        "TOGETHER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
