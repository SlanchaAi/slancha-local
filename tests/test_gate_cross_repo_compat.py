"""Cross-repo guard: slancha_local.train.gate must stay in sync with
slancha-mesh's ``mesh.eval.gate`` and ``mesh.eval.runner.EvalPass``.

slancha-local re-implements the promotion-gate shape rather than
importing slancha-mesh — same discipline as ``mesh/heartbeat.py``
(which mirrors the mesh wire format) and ``tests/test_pref_cross_repo_compat.py``
(which mirrors slancha-api's pref shape). The mirror can drift silently;
this guard fails when it does.

Two layers of check:

1. **AST parity** — read mesh's source by ast (no import, no pulling
   mesh's dependencies into slancha-local's test env) and assert:
       • ``GateThresholds`` field names + defaults match,
       • ``PromotionVerdict`` field names match,
       • ``EvalPass.to_row()`` keys match :data:`EVAL_ROW_FIELDS`.

2. **Behavior parity** — if mesh is importable (it might not be: mesh
   has its own deps slancha-local doesn't pull), run both ``decide()``
   functions over a curated fixture pair and assert identical accept
   bool + identical reject reasons + identical per-domain deltas.

Skips cleanly when ``slancha-mesh`` isn't on disk — alignment is only
enforced where it can be checked.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

from slancha_local.train.gate import (
    EVAL_ROW_FIELDS,
    GateThresholds,
    PromotionVerdict,
)
from slancha_local.train.gate import decide as local_decide


def _slancha_mesh_root() -> Path | None:
    roots: list[Path] = []
    explicit = os.environ.get("SLANCHA_MESH_PATH")
    if explicit:
        roots.append(Path(explicit))
    # sibling of slancha-local (the conventional layout: ~/src/slancha-{local,mesh})
    repo_root = Path(__file__).resolve().parent.parent
    roots.append(repo_root.parent / "slancha-mesh")
    # legacy: nested under tests/
    roots.append(repo_root / "tests" / "slancha-mesh")
    for r in roots:
        if (r / "mesh" / "eval" / "gate.py").is_file():
            return r
    return None


_MESH_ROOT = _slancha_mesh_root()
requires_mesh = pytest.mark.skipif(
    _MESH_ROOT is None,
    reason="slancha-mesh not on disk; gate cross-repo guard skipped",
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _dataclass_fields(tree: ast.Module, cls_name: str) -> dict[str, ast.expr | None]:
    """Return ``{field_name: default_ast_or_None}`` for the named dataclass."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            out: dict[str, ast.expr | None] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    out[stmt.target.id] = stmt.value
            return out
    return {}


def _to_row_keys(tree: ast.Module, cls_name: str) -> tuple[str, ...]:
    """Extract the ordered list of ``"key": value`` keys returned by
    ``cls_name.to_row()``. Tolerates either a dict literal or
    ``**asdict(self)`` patterns."""
    for cls in ast.walk(tree):
        if not (isinstance(cls, ast.ClassDef) and cls.name == cls_name):
            continue
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "to_row":
                for sub in ast.walk(fn):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                        return tuple(
                            k.value
                            for k in sub.value.keys
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)
                        )
    return ()


def _default_value(node: ast.expr | None) -> object:
    """Evaluate a literal default (int / float / bool / None). Returns a
    sentinel for unhandled forms — fine since the parity check uses ==."""
    if node is None:
        return _MISSING
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return _UNEVAL


_MISSING = object()
_UNEVAL = object()


@requires_mesh
def test_gate_thresholds_field_parity():
    tree = _parse(_MESH_ROOT / "mesh" / "eval" / "gate.py")
    mesh_fields = _dataclass_fields(tree, "GateThresholds")
    assert mesh_fields, "could not locate GateThresholds in mesh.eval.gate"

    local_fields = {f.name: f.default for f in GateThresholds.__dataclass_fields__.values()}
    assert set(local_fields) == set(mesh_fields), (
        f"GateThresholds field drift: local={sorted(local_fields)} "
        f"mesh={sorted(mesh_fields)}. Sync slancha_local/train/gate.py."
    )

    for name, mesh_default_node in mesh_fields.items():
        mesh_default = _default_value(mesh_default_node)
        if mesh_default is _UNEVAL:
            # Mesh uses a module-level DEFAULT_* constant; resolve it
            # from the same tree so we still check parity.
            if isinstance(mesh_default_node, ast.Name) and mesh_default_node.id.startswith("DEFAULT_"):
                for top in tree.body:
                    if (
                        isinstance(top, ast.Assign)
                        and len(top.targets) == 1
                        and isinstance(top.targets[0], ast.Name)
                        and top.targets[0].id == mesh_default_node.id
                    ):
                        mesh_default = _default_value(top.value)
                        break
        if mesh_default in (_MISSING, _UNEVAL):
            continue
        assert local_fields[name] == mesh_default, (
            f"GateThresholds.{name} default drift: local={local_fields[name]!r} mesh={mesh_default!r}"
        )


@requires_mesh
def test_promotion_verdict_field_parity():
    tree = _parse(_MESH_ROOT / "mesh" / "eval" / "gate.py")
    mesh_fields = set(_dataclass_fields(tree, "PromotionVerdict"))
    assert mesh_fields, "could not locate PromotionVerdict in mesh.eval.gate"
    local_fields = set(PromotionVerdict.__dataclass_fields__)
    assert local_fields == mesh_fields, (
        f"PromotionVerdict field drift: local={sorted(local_fields)} "
        f"mesh={sorted(mesh_fields)}. Sync slancha_local/train/gate.py."
    )


@requires_mesh
def test_eval_row_schema_matches_eval_pass_to_row():
    tree = _parse(_MESH_ROOT / "mesh" / "eval" / "runner.py")
    mesh_keys = _to_row_keys(tree, "EvalPass")
    assert mesh_keys, "could not locate EvalPass.to_row() dict literal in mesh.eval.runner"
    assert tuple(EVAL_ROW_FIELDS) == mesh_keys, (
        f"EVAL_ROW_FIELDS drift: local={list(EVAL_ROW_FIELDS)} mesh={list(mesh_keys)}. "
        "Sync slancha_local/train/gate.py::EVAL_ROW_FIELDS with mesh.eval.runner.EvalPass.to_row()."
    )


@requires_mesh
@pytest.mark.parametrize(
    "champ,chall,thr",
    [
        # 1: clean accept
        (
            {
                "router_version": "v1",
                "n_eval": 200,
                "judge_model": "j",
                "mean_score": 0.6,
                "per_domain_mean": {"general": 0.6, "code": 0.6},
            },
            {
                "router_version": "v2",
                "n_eval": 200,
                "judge_model": "j",
                "mean_score": 0.7,
                "per_domain_mean": {"general": 0.7, "code": 0.65},
            },
            None,
        ),
        # 2: per-domain cliff rejects even with mean lift
        (
            {
                "router_version": "v1",
                "n_eval": 200,
                "judge_model": "j",
                "mean_score": 0.6,
                "per_domain_mean": {"general": 0.5, "code": 0.7},
            },
            {
                "router_version": "v2",
                "n_eval": 200,
                "judge_model": "j",
                "mean_score": 0.65,
                "per_domain_mean": {"general": 0.9, "code": 0.40},
            },
            None,
        ),
        # 3: under-min-n on champion
        (
            {
                "router_version": "v1",
                "n_eval": 50,
                "judge_model": "j",
                "mean_score": 0.6,
                "per_domain_mean": {},
            },
            {
                "router_version": "v2",
                "n_eval": 200,
                "judge_model": "j",
                "mean_score": 0.7,
                "per_domain_mean": {},
            },
            None,
        ),
        # 4: judge mismatch rejects by default
        (
            {
                "router_version": "v1",
                "n_eval": 200,
                "judge_model": "ja",
                "mean_score": 0.6,
                "per_domain_mean": {"general": 0.6},
            },
            {
                "router_version": "v2",
                "n_eval": 200,
                "judge_model": "jb",
                "mean_score": 0.7,
                "per_domain_mean": {"general": 0.7},
            },
            None,
        ),
        # 5: judge mismatch accepted when opted in
        (
            {
                "router_version": "v1",
                "n_eval": 200,
                "judge_model": "ja",
                "mean_score": 0.6,
                "per_domain_mean": {"general": 0.6},
            },
            {
                "router_version": "v2",
                "n_eval": 200,
                "judge_model": "jb",
                "mean_score": 0.7,
                "per_domain_mean": {"general": 0.7},
            },
            {"require_judge_match": False},
        ),
    ],
)
def test_decide_behavior_matches_mesh(champ, chall, thr):
    """Behavior parity: same inputs → same accept/reject + same reasons +
    same per-domain deltas. The ``decided_at`` field is excluded (clock)."""
    sys.path.insert(0, str(_MESH_ROOT))
    try:
        from mesh.eval import gate as mesh_gate  # type: ignore
    except ImportError:
        pytest.skip(f"slancha-mesh present at {_MESH_ROOT} but not importable (missing deps)")
    finally:
        if str(_MESH_ROOT) in sys.path:
            sys.path.remove(str(_MESH_ROOT))

    thresholds_kwargs = thr or {}
    local_v = local_decide(champ, chall, GateThresholds(**thresholds_kwargs))
    mesh_v = mesh_gate.decide(champ, chall, mesh_gate.GateThresholds(**thresholds_kwargs))

    assert local_v.accept == mesh_v.accept
    assert tuple(local_v.reject_reasons) == tuple(mesh_v.reject_reasons)
    assert local_v.per_domain_deltas == mesh_v.per_domain_deltas
    assert local_v.mean_delta == mesh_v.mean_delta
    assert local_v.champion_version == mesh_v.champion_version
    assert local_v.challenger_version == mesh_v.challenger_version
    assert local_v.n_eval_champion == mesh_v.n_eval_champion
    assert local_v.n_eval_challenger == mesh_v.n_eval_challenger
    assert local_v.thresholds == mesh_v.thresholds
