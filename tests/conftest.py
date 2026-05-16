"""pytest conftest — global env hygiene + sibling-repo bridging.

Forces SLANCHA_TRAIN_DRY_RUN=1 by default so tests never hit the real
Fireworks/Together/OpenAI APIs even if the dev env has the keys set.
Also clears any user judge endpoints so eval tests deterministically
return ('tie', '...') without network.

Sibling-repo bridging: when ~/Source/slancha-mesh exists alongside this
repo, add it to sys.path so `tests/test_mesh_cross_repo_compat.py` can
`from mesh.registry import HeartbeatPostRequest` without requiring a
separate `pip install slancha-mesh`. This is intentionally a TEST-TIME
bridge — production callers don't need slancha-mesh; only the
cross-repo verification tests do.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _bridge_slancha_mesh() -> None:
    """Best-effort: add ../slancha-mesh to sys.path if it's a checkout.

    Honored only if MESH_PATH env is unset (operator-set wins) AND the
    sibling checkout exists. Idempotent — won't double-add.
    """
    env_path = os.environ.get("SLANCHA_MESH_PATH")
    candidate = Path(env_path) if env_path else Path(__file__).resolve().parent.parent.parent / "slancha-mesh"
    if not candidate.is_dir():
        return
    if not (candidate / "mesh" / "__init__.py").exists():
        return
    p = str(candidate)
    if p not in sys.path:
        sys.path.insert(0, p)


_bridge_slancha_mesh()


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
