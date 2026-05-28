"""Tests for :mod:`slancha_local.train.head_retrain`.

Layered to match the module's three seams:

* The supervised-set derivation tests run **anywhere** — they only need
  numpy + the existing cluster module (which is already required by the
  default install).
* The training + verify-load tests are skip-marked behind
  ``requires_promote_extra`` — they need ``lightgbm`` and ``treelite``
  (the ``slancha-local[promote]`` extra). The skip pattern mirrors
  ``tests/privacy/test_adversarial.py`` (treelite-on-libomp).

Keeping the seams clean means we can review derive_supervised_set
end-to-end without touching the optional-extra path.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from slancha_local.train.cluster import (
    ClusterSnapshot,
    cluster_by_route,
    snapshot_from_clusters,
)
from slancha_local.train.head_retrain import (
    HeadRetrainError,
    HeadRetrainResult,
    derive_supervised_set,
    retrain_cluster_head,
    train_cluster_head,
    verify_load,
)

# ---------------------------------------------------------------------------
# Optional-extra skip marker
# ---------------------------------------------------------------------------

try:
    import lightgbm  # noqa: F401
    import treelite  # noqa: F401

    _HAS_PROMOTE_EXTRA = True
except (ImportError, OSError):
    _HAS_PROMOTE_EXTRA = False

requires_promote_extra = pytest.mark.skipif(
    not _HAS_PROMOTE_EXTRA,
    reason="lightgbm + treelite not installed (install slancha-local[promote])",
)


# ---------------------------------------------------------------------------
# Helpers (mirror tests/unit/test_train_cluster.py)
# ---------------------------------------------------------------------------


def _emb(vec: np.ndarray) -> str:
    return base64.b64encode(vec.astype(np.float32).tobytes()).decode()


def _make_modes(
    *,
    route: str,
    centers: list[np.ndarray],
    per_mode: int,
    noise: float = 0.02,
    seed: int = 0,
) -> list[dict]:
    """Build well-separated Gaussian blobs so KMeans recovers each mode
    cleanly (low noise is deliberate — we're testing the head pipeline,
    not the clustering's robustness)."""
    rng = np.random.default_rng(seed)
    traces: list[dict] = []
    for mode_i, center in enumerate(centers):
        for j in range(per_mode):
            vec = center + rng.standard_normal(center.shape) * noise
            traces.append(
                {
                    "request_id": f"{route}-{mode_i}-{j}",
                    "embedding_b64": _emb(vec),
                    "classifier": {"route": route},
                    "prompt": "p",
                    "response": "r",
                    "consent_at_capture": True,
                }
            )
    return traces


def _two_route_dataset(per_mode: int = 8) -> tuple[list[dict], ClusterSnapshot]:
    """Two routes, two modes each => 4 clusters total, fully separated.

    Returns ``(traces, snapshot)``. The snapshot is the canonical prior
    derived from the first fit, so a re-cluster on the same traces
    reproduces the exact same cluster ids.
    """
    e1 = np.array([1.0, 0.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0, 0.0])
    e4 = np.array([0.0, 0.0, 0.0, 1.0])
    traces = _make_modes(
        route="math_hard", centers=[e1, e2], per_mode=per_mode, seed=1
    ) + _make_modes(
        route="code_easy", centers=[e3, e4], per_mode=per_mode, seed=2
    )
    clusters = cluster_by_route(traces, n_clusters_per_route=2, min_cluster_size=2)
    snapshot = snapshot_from_clusters(clusters, prior=ClusterSnapshot())
    return traces, snapshot


# ---------------------------------------------------------------------------
# derive_supervised_set — pure numpy, runs anywhere
# ---------------------------------------------------------------------------


class TestDeriveSupervisedSet:
    def test_happy_path_two_routes_two_modes(self):
        traces, snapshot = _two_route_dataset(per_mode=8)
        x_arr, y, label_table = derive_supervised_set(
            traces,
            snapshot,
            n_clusters_per_route=2,
            min_cluster_size=2,
            min_samples_per_class=2,
        )

        assert x_arr.dtype == np.float32
        assert y.dtype == np.int32
        assert x_arr.shape[0] == y.shape[0]
        # Most traces should be captured (allow occasional KMeans dropouts)
        assert x_arr.shape[0] >= len(traces) - 2
        assert x_arr.shape[1] == 4  # embedding_dim
        # 2 routes * 2 modes = 4 surviving classes
        assert len(label_table) == 4
        # label_table is in label-index order
        for k, row in enumerate(label_table):
            assert row["label"] == k
            assert row["route"] in {"math_hard", "code_easy"}
            assert isinstance(row["cluster_id"], int)
        # y indices fall in [0, n_classes)
        assert y.min() >= 0
        assert y.max() < len(label_table)
        # Every class is represented
        assert set(y.tolist()) == set(range(len(label_table)))

    def test_drops_small_classes(self):
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        traces = _make_modes(route="r", centers=[e1, e2], per_mode=6, seed=0)
        # Add a singleton mode that should be dropped at min_samples=3
        traces += _make_modes(
            route="r",
            centers=[np.array([5.0, 5.0])],
            per_mode=1,
            seed=99,
        )
        clusters = cluster_by_route(traces, n_clusters_per_route=3, min_cluster_size=1)
        snapshot = snapshot_from_clusters(clusters, prior=ClusterSnapshot())

        x_arr, y, label_table = derive_supervised_set(
            traces,
            snapshot,
            n_clusters_per_route=3,
            min_cluster_size=1,
            min_samples_per_class=3,
        )
        # The singleton's class must have been filtered out
        # (it has 1 sample, threshold is 3).
        for row in label_table:
            count = int((y == row["label"]).sum())
            assert count >= 3, f"label {row} survived with only {count} samples"
        assert x_arr.shape[0] == y.shape[0]
        assert x_arr.shape[0] < len(traces)  # singleton dropped

    def test_empty_traces_raises(self):
        with pytest.raises(HeadRetrainError, match="no traces"):
            derive_supervised_set([], ClusterSnapshot())

    def test_all_classes_below_threshold_raises(self):
        # 6 singletons; nothing meets min_samples=2
        e1 = np.array([1.0, 0.0])
        traces = _make_modes(route="r", centers=[e1], per_mode=1, seed=0)
        traces += _make_modes(
            route="r2", centers=[np.array([0.0, 1.0])], per_mode=1, seed=1
        )
        clusters = cluster_by_route(traces, n_clusters_per_route=1, min_cluster_size=1)
        snapshot = snapshot_from_clusters(clusters, prior=ClusterSnapshot())
        with pytest.raises(HeadRetrainError, match="no cluster has at least"):
            derive_supervised_set(
                traces,
                snapshot,
                n_clusters_per_route=1,
                min_cluster_size=1,
                min_samples_per_class=5,
            )

    def test_missing_embedding_b64_skipped(self):
        traces, snapshot = _two_route_dataset(per_mode=6)
        # Strip embeddings from half the traces
        for t in traces[::2]:
            t["embedding_b64"] = ""
        x_arr, y, _ = derive_supervised_set(
            traces,
            snapshot,
            n_clusters_per_route=2,
            min_cluster_size=2,
            min_samples_per_class=2,
        )
        # Roughly half remain
        assert x_arr.shape[0] == y.shape[0]
        assert x_arr.shape[0] < len(traces)
        assert x_arr.shape[0] > 0

    def test_all_embeddings_missing_raises(self):
        traces, snapshot = _two_route_dataset(per_mode=6)
        for t in traces:
            t["embedding_b64"] = ""
        with pytest.raises(HeadRetrainError, match="embedding_b64"):
            derive_supervised_set(
                traces,
                snapshot,
                n_clusters_per_route=2,
                min_cluster_size=2,
                min_samples_per_class=2,
            )

    def test_inconsistent_embedding_dim_raises(self):
        traces, snapshot = _two_route_dataset(per_mode=6)
        # Corrupt one trace's embedding to a different width
        traces[0]["embedding_b64"] = _emb(np.zeros(8, dtype=np.float32))
        with pytest.raises(HeadRetrainError, match="inconsistent embedding dims"):
            derive_supervised_set(
                traces,
                snapshot,
                n_clusters_per_route=2,
                min_cluster_size=2,
                min_samples_per_class=2,
            )

    def test_label_table_stable_order(self):
        """label_table is sorted on (route, cluster_id) so callers can rely
        on the same training set producing the same label assignment."""
        traces, snapshot = _two_route_dataset(per_mode=8)
        _, _, lt1 = derive_supervised_set(
            traces,
            snapshot,
            n_clusters_per_route=2,
            min_cluster_size=2,
            min_samples_per_class=2,
        )
        _, _, lt2 = derive_supervised_set(
            traces,
            snapshot,
            n_clusters_per_route=2,
            min_cluster_size=2,
            min_samples_per_class=2,
        )
        assert lt1 == lt2
        keys = [(r["route"], r["cluster_id"]) for r in lt1]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# train_cluster_head — needs [promote] extra
# ---------------------------------------------------------------------------


@requires_promote_extra
class TestTrainClusterHead:
    def test_happy_path_returns_serialized_bytes(self):
        rng = np.random.default_rng(0)
        # 3 classes, 30 samples each, 8-dim embeddings, well separated
        centers = [rng.standard_normal(8) * 3.0 for _ in range(3)]
        x_rows = []
        y_list = []
        for cid, center in enumerate(centers):
            for _ in range(30):
                x_rows.append(center + rng.standard_normal(8) * 0.1)
                y_list.append(cid)
        x_arr = np.asarray(x_rows, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.int32)

        head_bytes = train_cluster_head(x_arr, y, n_classes=3, num_iterations=20)
        assert isinstance(head_bytes, bytes)
        assert len(head_bytes) > 100  # treelite .bin is not tiny

    def test_empty_training_set_raises(self):
        x_arr = np.zeros((0, 4), dtype=np.float32)
        y = np.asarray([], dtype=np.int32)
        with pytest.raises(HeadRetrainError, match="empty training set"):
            train_cluster_head(x_arr, y, n_classes=3)

    def test_label_out_of_range_raises(self):
        x_arr = np.zeros((3, 4), dtype=np.float32)
        y = np.asarray([0, 1, 5], dtype=np.int32)
        with pytest.raises(HeadRetrainError, match="out of range"):
            train_cluster_head(x_arr, y, n_classes=3)


# ---------------------------------------------------------------------------
# verify_load — needs [promote] extra
# ---------------------------------------------------------------------------


@requires_promote_extra
class TestVerifyLoad:
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        x_rows, y_list = [], []
        centers = [rng.standard_normal(8) * 3.0 for _ in range(2)]
        for cid, center in enumerate(centers):
            for _ in range(20):
                x_rows.append(center + rng.standard_normal(8) * 0.1)
                y_list.append(cid)
        x_arr = np.asarray(x_rows, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.int32)
        head_bytes = train_cluster_head(x_arr, y, n_classes=2, num_iterations=10)
        verify_load(head_bytes)  # must not raise

    def test_empty_bytes_raises(self):
        with pytest.raises(HeadRetrainError, match="empty head bytes"):
            verify_load(b"")

    def test_corrupted_bytes_raises(self):
        with pytest.raises(HeadRetrainError, match="treelite failed to load"):
            verify_load(b"not a treelite model" * 32)


# ---------------------------------------------------------------------------
# retrain_cluster_head — end-to-end, needs [promote] extra
# ---------------------------------------------------------------------------


@requires_promote_extra
class TestRetrainClusterHead:
    def test_end_to_end(self):
        traces, snapshot = _two_route_dataset(per_mode=12)
        result = retrain_cluster_head(
            traces,
            snapshot,
            n_clusters_per_route=2,
            min_cluster_size=2,
            min_samples_per_class=2,
            num_iterations=20,
        )
        assert isinstance(result, HeadRetrainResult)
        assert isinstance(result.head_bytes, bytes)
        assert len(result.head_bytes) > 100
        assert result.n_classes == 4  # 2 routes x 2 modes
        assert result.n_samples >= len(traces) - 2
        assert result.embedding_dim == 4
        assert len(result.label_table) == 4
        # Label table covers both routes
        routes = {row["route"] for row in result.label_table}
        assert routes == {"math_hard", "code_easy"}

    def test_propagates_derive_error(self):
        """If the derive step fails (empty traces), retrain surfaces the
        same error class — caller doesn't have to special-case."""
        with pytest.raises(HeadRetrainError):
            retrain_cluster_head([], ClusterSnapshot())


# ---------------------------------------------------------------------------
# Missing-extra error messaging (runs anywhere; uses monkeypatch to simulate)
# ---------------------------------------------------------------------------


class TestMissingExtraMessaging:
    """Surface a clear message pointing at [promote] when an optional dep
    is missing — operators shouldn't have to read tracebacks."""

    def test_train_without_lightgbm(self, monkeypatch):
        # Force lightgbm import to fail
        import sys

        monkeypatch.setitem(sys.modules, "lightgbm", None)
        x_arr = np.zeros((4, 2), dtype=np.float32)
        y = np.asarray([0, 0, 1, 1], dtype=np.int32)
        with pytest.raises(HeadRetrainError, match="lightgbm.*\\[promote\\]"):
            train_cluster_head(x_arr, y, n_classes=2)

    def test_verify_load_without_treelite(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "treelite", None)
        with pytest.raises(HeadRetrainError, match="treelite.*\\[promote\\]"):
            verify_load(b"\x00\x01\x02\x03" * 16)
