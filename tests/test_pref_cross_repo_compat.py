"""Cross-repo guard: slancha-local's accepted pref shape must stay in sync
with slancha-api's SlanchaPref (the gateway contract slancha-local mirrors).

slancha-local re-implements the pref shape rather than importing slancha-api
(it ships public; slancha-api is private). That copy can drift silently. This
guard reads slancha-api's source by **AST** — no import, so no `http_sfv`
dependency and no `app` package __init__ side effects — and fails when the two
diverge:

  - weights axis set differs → a new gateway axis would make slancha-local
    422 a rule the gateway accepts (forward-incompat), or
  - a field slancha-local reads off the pref disappears / is renamed upstream.

Skips cleanly when slancha-api isn't on disk (sibling checkout or
SLANCHA_API_PATH) — the alignment is only *enforced* where it can be checked.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from slancha_local.proxy.pref import _ALLOWED_AXES, _HEADER_TO_BODY_FIELD_MAP


def _slancha_api_pref_source() -> Path | None:
    roots: list[Path] = []
    explicit = os.environ.get("SLANCHA_API_PATH")
    if explicit:
        roots.append(Path(explicit))
    roots.append(Path(__file__).resolve().parent.parent.parent / "slancha-api")
    for r in roots:
        p = r / "app" / "mesh" / "pref.py"
        if p.is_file():
            return p
    return None


_API_PREF = _slancha_api_pref_source()
requires_api = pytest.mark.skipif(
    _API_PREF is None,
    reason="slancha-api not on disk; pref cross-repo guard skipped",
)


def _api_tree() -> ast.Module:
    return ast.parse(_API_PREF.read_text())


def _api_allowed_axes(tree: ast.Module) -> set[str]:
    """Extract the `allowed = {...}` axis set from SlanchaPref._validate_weights."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Set):
            if any(isinstance(t, ast.Name) and t.id == "allowed" for t in node.targets):
                vals = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
                if vals:
                    return vals
    return set()


def _api_slanchapref_fields(tree: ast.Module) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SlanchaPref":
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    return set()


@requires_api
def test_weights_axes_match_slancha_api():
    api_axes = _api_allowed_axes(_api_tree())
    assert api_axes, "could not locate `allowed = {...}` axis set in slancha-api pref.py"
    # Exact parity: if the gateway adds/removes an axis, slancha-local must
    # follow or it 422s (or silently accepts) a rule the gateway disagrees on.
    assert set(_ALLOWED_AXES) == api_axes, (
        f"weights axes drift — local={sorted(_ALLOWED_AXES)} api={sorted(api_axes)}. "
        "Sync slancha_local/proxy/pref.py::_ALLOWED_AXES with slancha-api."
    )


@requires_api
def test_mapped_fields_exist_in_slancha_api():
    api_fields = _api_slanchapref_fields(_api_tree())
    assert "weights" in api_fields, "parsed the wrong class — SlanchaPref.weights not found"
    # Every field slancha-local reads off the pref must still exist upstream.
    mapped = set(_HEADER_TO_BODY_FIELD_MAP.values()) | {
        "weights",
        "quality_weight",
        "max_latency_ms_p95",
        "max_cost_per_1m_usd",
        "allow_fallbacks",
    }
    missing = mapped - api_fields
    assert not missing, (
        f"slancha-local maps pref fields absent from slancha-api SlanchaPref: {sorted(missing)}"
    )
