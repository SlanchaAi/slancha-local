"""Unit tests for the cluster-head selector (phase 2d, 7th-head READ path).

Three guardrails per onyx-ridge spec (events 270d6b58 / d924e5f5):

1. SAFE BY DEFAULT — no ACTIVE pointer ⇒ selector is fully inert; the
   classifier behaves exactly as today.
2. CONFIDENCE GATED — predictions below
   ``SLANCHA_CLUSTER_HEAD_CONF_THRESHOLD`` are ignored.
3. CLUSTER→CAP MAPPING — sidecar JSON co-located with the .bin in the
   same version dir; missing/malformed/schema-mismatched sidecars make
   the selector inert (NEVER crash classifier startup).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from slancha_local.classifier.cluster_head import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    HEAD_FILENAME,
    SCHEMA_VERSION,
    SIDECAR_FILENAME,
    ClusterHeadSelector,
    ClusterRouteHint,
    load_from_store,
)
from slancha_local.classifier.local import _apply_cluster_hint
from slancha_local.classifier_client.models import LocalModelDescriptor
from slancha_local.train.pointer_store import PointerStore

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeHead:
    """A scripted ClusterHead — returns whatever the test sets."""

    def __init__(self, cid: int, conf: float) -> None:
        self._cid = cid
        self._conf = conf

    def predict(self, x: np.ndarray) -> tuple[int, float]:
        return self._cid, self._conf


def _store(tmp_path: Path) -> PointerStore:
    return PointerStore(root=tmp_path / "store", keep_versions=5)


def _write_artifact(store: PointerStore, version: str, head_bytes: bytes, mapping: Any) -> None:
    """Place a synthetic head + sidecar into a versioned dir + flip ACTIVE.

    ``mapping`` may be the full sidecar dict (used as-is) or a
    ``{cid: cap}`` shortcut (wrapped in the v1 envelope).
    """
    if isinstance(mapping, dict) and "schema_version" in mapping:
        sidecar = mapping
    else:
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "routes": {str(k): v for k, v in (mapping or {}).items()},
        }
    files_map = {
        HEAD_FILENAME: head_bytes,
        SIDECAR_FILENAME: json.dumps(sidecar).encode("utf-8"),
    }
    store.write_candidate("classifier-head", version, files_map)
    store.promote("classifier-head", version)


# ---------------------------------------------------------------------------
# ClusterHeadSelector unit tests (pure, no treelite)
# ---------------------------------------------------------------------------


class TestClusterHeadSelector:
    def test_high_confidence_with_mapping_returns_hint(self):
        head = _FakeHead(cid=2, conf=0.85)
        sel = ClusterHeadSelector(head, {0: "general", 1: "coding", 2: "math"}, head_version="v1")
        hint = sel.predict(np.zeros((1, 4), dtype=np.float32))
        assert hint is not None
        assert hint.cluster_id == 2
        assert hint.cap == "math"
        assert hint.confidence == pytest.approx(0.85)
        assert hint.head_version == "v1"
        # reason() is for the decision trace; must be human-readable +
        # mention the version, cid, conf, and cap.
        s = hint.reason()
        for chunk in ("cluster-head", "v=v1", "cid=2", "conf=0.85", "cap=math"):
            assert chunk in s

    def test_low_confidence_returns_none(self):
        head = _FakeHead(cid=0, conf=DEFAULT_CONFIDENCE_THRESHOLD - 0.01)
        sel = ClusterHeadSelector(head, {0: "coding"})
        assert sel.predict(np.zeros((1, 4))) is None

    def test_exactly_at_threshold_returns_hint(self):
        # Boundary semantics: >= threshold counts as confident.
        head = _FakeHead(cid=0, conf=DEFAULT_CONFIDENCE_THRESHOLD)
        sel = ClusterHeadSelector(head, {0: "coding"})
        assert sel.predict(np.zeros((1, 4))) is not None

    def test_high_confidence_unmapped_cluster_returns_none(self):
        # Warm-up window: a freshly-spawned cluster has no mapping entry
        # yet; selector must fall through silently rather than emit a
        # garbage hint.
        head = _FakeHead(cid=99, conf=0.99)
        sel = ClusterHeadSelector(head, {0: "coding", 1: "math"})
        assert sel.predict(np.zeros((1, 4))) is None

    def test_env_threshold_override_lower(self, monkeypatch):
        # Lower the bar → a previously-rejected mid-conf prediction now
        # fires. Confirms the env var is honored.
        monkeypatch.setenv("SLANCHA_CLUSTER_HEAD_CONF_THRESHOLD", "0.5")
        head = _FakeHead(cid=0, conf=0.6)
        sel = ClusterHeadSelector(head, {0: "coding"})
        assert sel.predict(np.zeros((1, 4))) is not None

    def test_env_threshold_override_higher(self, monkeypatch):
        # Raise the bar → a previously-firing high-conf prediction stops
        # firing.
        monkeypatch.setenv("SLANCHA_CLUSTER_HEAD_CONF_THRESHOLD", "0.95")
        head = _FakeHead(cid=0, conf=0.9)
        sel = ClusterHeadSelector(head, {0: "coding"})
        assert sel.predict(np.zeros((1, 4))) is None

    @pytest.mark.parametrize("bad", ["not_a_number", "-0.1", "1.5", ""])
    def test_env_threshold_invalid_falls_back_to_default(self, monkeypatch, bad):
        # Operator typos must NOT silently disable the gate — fall back
        # to the default 0.7 rather than treat the typo as "no
        # threshold".
        monkeypatch.setenv("SLANCHA_CLUSTER_HEAD_CONF_THRESHOLD", bad)
        head_low = _FakeHead(cid=0, conf=DEFAULT_CONFIDENCE_THRESHOLD - 0.01)
        head_hi = _FakeHead(cid=0, conf=DEFAULT_CONFIDENCE_THRESHOLD + 0.01)
        sel_lo = ClusterHeadSelector(head_low, {0: "coding"})
        sel_hi = ClusterHeadSelector(head_hi, {0: "coding"})
        assert sel_lo.predict(np.zeros((1, 4))) is None
        assert sel_hi.predict(np.zeros((1, 4))) is not None


# ---------------------------------------------------------------------------
# load_from_store — sidecar/store loading
# ---------------------------------------------------------------------------


class TestLoadFromStore:
    def test_no_active_returns_none(self, tmp_path):
        # Fresh store, nothing promoted ⇒ inert selector.
        store = _store(tmp_path)
        assert load_from_store(store) is None

    def test_sidecar_missing_returns_none_logs_warning(self, tmp_path, caplog):
        # Promote a head WITHOUT a sidecar — sidecar required for the
        # cluster_id→cap mapping; without it the head's int output is
        # meaningless. Selector inert, warning logged.
        store = _store(tmp_path)
        store.write_candidate(
            "classifier-head",
            "20260101T000000Z",
            {HEAD_FILENAME: b"\x00\x01\x02"},
        )
        store.promote("classifier-head", "20260101T000000Z")
        with caplog.at_level("WARNING"):
            assert load_from_store(store) is None
        assert any("sidecar" in r.message for r in caplog.records)

    def test_sidecar_invalid_json_returns_none(self, tmp_path, caplog):
        store = _store(tmp_path)
        store.write_candidate(
            "classifier-head",
            "20260101T000000Z",
            {
                HEAD_FILENAME: b"\x00",
                SIDECAR_FILENAME: b"{ not json",
            },
        )
        store.promote("classifier-head", "20260101T000000Z")
        with caplog.at_level("WARNING"):
            assert load_from_store(store) is None

    def test_sidecar_wrong_schema_version_returns_none(self, tmp_path, caplog):
        store = _store(tmp_path)
        sidecar = {"schema_version": "v999", "routes": {"0": "coding"}}
        store.write_candidate(
            "classifier-head",
            "20260101T000000Z",
            {
                HEAD_FILENAME: b"\x00",
                SIDECAR_FILENAME: json.dumps(sidecar).encode(),
            },
        )
        store.promote("classifier-head", "20260101T000000Z")
        with caplog.at_level("WARNING"):
            assert load_from_store(store) is None
        # Schema mismatch IS the failure mode — message should mention it.
        msgs = " ".join(r.message for r in caplog.records)
        assert "schema_version" in msgs

    def test_sidecar_non_int_key_returns_none(self, tmp_path):
        store = _store(tmp_path)
        sidecar = {"schema_version": SCHEMA_VERSION, "routes": {"not_an_int": "coding"}}
        store.write_candidate(
            "classifier-head",
            "20260101T000000Z",
            {
                HEAD_FILENAME: b"\x00",
                SIDECAR_FILENAME: json.dumps(sidecar).encode(),
            },
        )
        store.promote("classifier-head", "20260101T000000Z")
        assert load_from_store(store) is None

    def test_sidecar_value_not_string_returns_none(self, tmp_path):
        store = _store(tmp_path)
        sidecar = {"schema_version": SCHEMA_VERSION, "routes": {"0": 42}}
        store.write_candidate(
            "classifier-head",
            "20260101T000000Z",
            {
                HEAD_FILENAME: b"\x00",
                SIDECAR_FILENAME: json.dumps(sidecar).encode(),
            },
        )
        store.promote("classifier-head", "20260101T000000Z")
        assert load_from_store(store) is None

    def test_sidecar_root_not_dict_returns_none(self, tmp_path):
        store = _store(tmp_path)
        store.write_candidate(
            "classifier-head",
            "20260101T000000Z",
            {
                HEAD_FILENAME: b"\x00",
                SIDECAR_FILENAME: b'["not", "a", "dict"]',
            },
        )
        store.promote("classifier-head", "20260101T000000Z")
        assert load_from_store(store) is None

    def test_treelite_missing_returns_none_does_not_crash(self, tmp_path, caplog):
        # libomp is not installed on this Mac, so importing treelite
        # fails at native-lib load time (ImportError or OSError).
        # Either way the selector must be inert + log warning, NOT
        # raise — this is the SAFE-BY-DEFAULT guardrail under partial
        # install.
        store = _store(tmp_path)
        _write_artifact(store, "20260101T000000Z", b"\x00fake head", {0: "coding"})
        with caplog.at_level("WARNING"):
            sel = load_from_store(store)
        # On a host with treelite + libomp working, deserializing the
        # fake head bytes will raise inside treelite; on a host without
        # them, the import raises. Either way the selector is inert.
        # We tolerate both outcomes — the contract is "no crash, return
        # None on any failure".
        assert sel is None


# ---------------------------------------------------------------------------
# _apply_cluster_hint — model selection from a hinted cap
# ---------------------------------------------------------------------------


def _coder():
    return LocalModelDescriptor(
        backend="ollama",
        id="codestral:22b",
        ctx_window=32768,
        capabilities=["en", "coding"],
    )


def _general():
    return LocalModelDescriptor(
        backend="ollama",
        id="qwen3:8b",
        ctx_window=32768,
        capabilities=["en"],
    )


def _reasoner():
    return LocalModelDescriptor(
        backend="ollama",
        id="deepseek-r1:14b",
        ctx_window=16384,
        capabilities=["en", "hard"],
    )


def _hint(cap: str, cid: int = 0, conf: float = 0.9, ver: str = "20260101T000000Z") -> ClusterRouteHint:
    return ClusterRouteHint(cluster_id=cid, cap=cap, confidence=conf, head_version=ver)


class TestApplyClusterHint:
    def test_coding_hint_picks_coder(self):
        out = _apply_cluster_hint(_hint("coding"), [_coder(), _general()])
        assert out is not None
        target, fallbacks, reason, conf = out
        assert target == "local:ollama:codestral:22b"
        assert "local:ollama:qwen3:8b" in fallbacks
        assert "cluster-head" in reason
        assert "cap=coding" in reason
        assert conf == pytest.approx(0.9)

    def test_math_hint_picks_hard_capable_reasoner(self):
        out = _apply_cluster_hint(_hint("math"), [_general(), _reasoner()])
        assert out is not None
        assert out[0] == "local:ollama:deepseek-r1:14b"

    def test_general_hint_picks_non_coder(self):
        out = _apply_cluster_hint(_hint("general"), [_coder(), _general()])
        assert out is not None
        # general means "not a coder" — matches existing rule selector
        assert out[0] == "local:ollama:qwen3:8b"

    def test_no_model_satisfies_cap_returns_none(self):
        # coding hint but no coder available ⇒ fall through to rules
        out = _apply_cluster_hint(_hint("coding"), [_general()])
        assert out is None

    def test_no_available_models_returns_none(self):
        out = _apply_cluster_hint(_hint("coding"), [])
        assert out is None

    def test_unknown_cap_returns_none_logs_warning(self, caplog):
        # Operator shipped a sidecar with cap="legendary" ⇒ never apply
        # the override, log a warning so the misconfig is visible.
        with caplog.at_level("WARNING"):
            out = _apply_cluster_hint(_hint("legendary"), [_coder(), _general()])
        assert out is None
        assert any("unknown cap" in r.message for r in caplog.records)

    def test_reason_string_carries_full_provenance(self):
        out = _apply_cluster_hint(_hint("coding", cid=7, conf=0.83, ver="20260601T120000Z"), [_coder()])
        assert out is not None
        reason = out[2]
        # Operators reading a trace need to know exactly which head
        # version + cluster + conf produced the override decision.
        for chunk in ("cluster-head", "v=20260601T120000Z", "cid=7", "conf=0.83", "cap=coding"):
            assert chunk in reason
