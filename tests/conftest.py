"""pytest conftest — global env hygiene + cross-repo contract bootstrap.

Forces SLANCHA_TRAIN_DRY_RUN=1 by default so tests never hit the real
Fireworks/Together/OpenAI APIs even if the dev env has the keys set.
Also clears any user judge endpoints so eval tests deterministically
return ('tie', '...') without network.

Cross-repo guards: slancha-local ships public (Apache-2.0) and can't take a
hard dependency on private slancha-mesh / slancha-api, so the shared wire
contracts (heartbeat shape, pref shape) are re-implemented here and pinned by
cross-repo tests. Those tests need the sibling repo on disk — a standard
sibling checkout (~/Source/slancha-*) or an explicit env path. We add
slancha-mesh to sys.path at conftest import (before collection) so
test_mesh_cross_repo_compat's top-level `import mesh.registry` resolves; absent
→ that test skips cleanly (no false green). slancha-api is NOT path-injected
(its top-level `app` package name is too generic to shadow safely) — the pref
guard reads its source by path instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _sibling_repo(name: str, env_var: str) -> Path | None:
    """Locate a sibling slancha repo: explicit env path, else ~/Source/<name>."""
    explicit = os.environ.get(env_var)
    if explicit and Path(explicit).is_dir():
        return Path(explicit)
    sibling = Path(__file__).resolve().parent.parent.parent / name
    return sibling if sibling.is_dir() else None


_mesh_repo = _sibling_repo("slancha-mesh", "SLANCHA_MESH_PATH")
if _mesh_repo is not None and str(_mesh_repo) not in sys.path:
    sys.path.insert(0, str(_mesh_repo))


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
