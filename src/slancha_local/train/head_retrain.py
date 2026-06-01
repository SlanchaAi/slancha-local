"""Classifier-head retrain pipeline for the closed cluster-head loop.

This module is the **training half** of P2b.3 phase 2 (per boss spec event
``270d6b58``). It turns a stream of router traces + a stable
:class:`ClusterSnapshot` into a fresh treelite-serialized classifier head
(the 7th, additive "cluster" head — see ``classifier/local.py``).

The module is split into three layers so each is testable in isolation:

1. :func:`derive_supervised_set` — pure numpy pipeline. Takes traces +
   snapshot, returns ``(X, y, label_table)``. No optional deps; runs in
   any environment with numpy.

2. :func:`train_cluster_head` — wraps LightGBM multiclass training +
   treelite conversion + serialization. Requires the optional
   ``slancha-local[promote]`` extra (``lightgbm`` + ``treelite``).

3. :func:`retrain_cluster_head` — top-level entry point. Combines the
   two halves and returns a :class:`HeadRetrainResult` ready for the
   pointer-store writer.

:func:`verify_load` is the "load the bytes we just wrote, before
flipping ACTIVE" smoke test — it catches a corrupt or
treelite-unloadable .bin BEFORE it becomes the live artifact.

This module deliberately does NOT touch the pointer store, the dispatcher,
the scorer, or the gate — those compose on top in
``train.promote_head`` (phase 2c). Keeping the seam clean makes each piece
reviewable in isolation and lets tests fake any one without standing up
the others.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from slancha_local.train.cluster import (
    ClusterSnapshot,
    _decode_embedding,
    cluster_by_route,
)

logger = logging.getLogger(__name__)


class HeadRetrainError(RuntimeError):
    """Raised when retraining fails for a reason callers can handle (bad
    inputs, training divergence, treelite conversion failure, smoke-load
    failure)."""


@dataclass
class HeadRetrainResult:
    """Output of :func:`retrain_cluster_head`.

    ``head_bytes`` is a treelite-serialized .bin ready to write to disk
    (e.g. via :meth:`PointerStore.write_candidate`).

    ``label_table`` is a list of ``{"label": int, "route": str,
    "cluster_id": int}`` rows in label-index order, so consumers can
    decode the head's output back to the (route, cluster_id) pair.
    """

    head_bytes: bytes
    label_table: list[dict]
    n_classes: int
    n_samples: int
    embedding_dim: int


# -------- supervised set derivation (pure, no optional deps) --------


def _flatten_cluster_assignment(
    traces: list[dict],
    snapshot: ClusterSnapshot,
    *,
    n_clusters_per_route: int,
    min_cluster_size: int,
) -> list[tuple[str, int]]:
    """Run cluster_by_route over the traces using ``snapshot`` as prior,
    return per-trace ``(route, cluster_id)`` tuples.

    Per-trace tuples drive both the membership count (for the
    min_samples_per_class filter) and the final label assignment
    downstream. The label_table itself is built later from the surviving
    pairs, not here.
    """
    clusters = cluster_by_route(
        traces,
        n_clusters_per_route=n_clusters_per_route,
        min_cluster_size=min_cluster_size,
        prior=snapshot,
    )
    per_trace: list[tuple[str, int]] = [("", -1)] * len(traces)
    for c in clusters:
        for idx in c.trace_indices:
            per_trace[idx] = (c.route, c.cluster_id)
    return per_trace


def derive_supervised_set(
    traces: list[dict],
    snapshot: ClusterSnapshot,
    *,
    n_clusters_per_route: int = 4,
    min_cluster_size: int = 2,
    min_samples_per_class: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build ``(X, y, label_table)`` for cluster-head training.

    * Re-runs :func:`cluster_by_route` over ``traces`` with ``snapshot``
      as prior so each trace gets a stable cluster id (clusters that
      survived a prior pass keep their ids — this is the whole point of
      P1.5). Clustering hyperparams (``n_clusters_per_route``,
      ``min_cluster_size``) should match the values used to build
      ``snapshot``; otherwise stable-id matching may not recover the
      same cluster topology.
    * Flattens ``(route, cluster_id)`` pairs into a single global label
      index so a single multiclass head can learn the full space.
    * Drops cluster classes with fewer than ``min_samples_per_class``
      member traces — multiclass training on a single-example class is
      degenerate and the resulting head over-fits noise.
    * Decodes each trace's ``embedding_b64`` into a row of ``X``.

    Returns
    -------
    X : ``(n_samples, embedding_dim)`` float32
    y : ``(n_samples,)`` int32 — label indices into ``label_table``
    label_table : list of ``{"label": int, "route": str, "cluster_id": int}``
        sorted by label index, so ``label_table[k]["route"]`` and
        ``label_table[k]["cluster_id"]`` describe class ``k``.

    Raises
    ------
    HeadRetrainError
        If ``traces`` is empty, no trace carries an ``embedding_b64``,
        every cluster falls below ``min_samples_per_class`` (the
        training set is empty), or only one cluster survives (a
        single-class head is degenerate — refuses early).
    """
    if not traces:
        raise HeadRetrainError("no traces supplied")

    # Filter out traces with no embedding upfront — cluster_by_route
    # decodes every trace's embedding_b64 unconditionally, and an empty
    # string produces a zero-length array that breaks np.stack.
    embedded_traces: list[dict] = [t for t in traces if t.get("embedding_b64")]
    if not embedded_traces:
        raise HeadRetrainError(
            "no traces carry an embedding_b64. Check the trace producer is emitting embeddings."
        )

    # Pre-check embedding widths — cluster_by_route would crash with an
    # unhelpful numpy error if rows are mismatched.
    seen_widths = {_decode_embedding(t["embedding_b64"]).shape[0] for t in embedded_traces}
    if len(seen_widths) > 1:
        raise HeadRetrainError(f"inconsistent embedding dims across traces: {sorted(seen_widths)}")

    per_trace = _flatten_cluster_assignment(
        embedded_traces,
        snapshot,
        n_clusters_per_route=n_clusters_per_route,
        min_cluster_size=min_cluster_size,
    )

    # Count members per (route, cluster_id) pair; drop classes too small
    # to train on.
    counts: Counter[tuple[str, int]] = Counter(t for t in per_trace if t[1] >= 0)
    surviving = sorted([pair for pair, n in counts.items() if n >= min_samples_per_class])
    if not surviving:
        raise HeadRetrainError(
            f"no cluster has at least {min_samples_per_class} samples; "
            f"largest cluster has {max(counts.values(), default=0)}. "
            "Either lower min_samples_per_class or collect more traces."
        )
    # A discriminative multiclass head needs >=2 classes. Training
    # LightGBM with num_class=1 would yield a degenerate model that
    # always predicts the lone class — meaningless as a classifier and
    # poison if it flowed through eval as a 'candidate'. Refuse here
    # rather than produce a garbage head.
    if len(surviving) < 2:
        raise HeadRetrainError(
            "need >=2 surviving clusters to train a discriminative head; "
            f"got {len(surviving)} "
            f"(only ({surviving[0][0]!r}, cid={surviving[0][1]}) survived "
            f"the min_samples_per_class={min_samples_per_class} filter). "
            "Either lower min_samples_per_class or collect a more diverse "
            "trace set covering multiple routes/clusters."
        )

    pair_to_label = {pair: idx for idx, pair in enumerate(surviving)}
    label_table = [
        {"label": idx, "route": route, "cluster_id": cid} for idx, (route, cid) in enumerate(surviving)
    ]

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for trace_idx, pair in enumerate(per_trace):
        if pair not in pair_to_label:
            continue
        b64 = embedded_traces[trace_idx].get("embedding_b64")
        if not b64:
            continue
        emb = _decode_embedding(b64).astype(np.float32, copy=False)
        rows.append(emb)
        labels.append(pair_to_label[pair])

    if not rows:
        raise HeadRetrainError(
            "no surviving training samples carry an embedding_b64. "
            "Check the trace producer is emitting embeddings."
        )

    # Uniform embedding dim — if one row is the wrong width, training
    # would crash deep in lightgbm with an unhelpful trace; surface it
    # here.
    widths = {r.shape[0] for r in rows}
    if len(widths) != 1:
        raise HeadRetrainError(f"inconsistent embedding dims across traces: {sorted(widths)}")

    X = np.vstack(rows).astype(np.float32, copy=False)  # noqa: N806
    y = np.asarray(labels, dtype=np.int32)
    return X, y, label_table


# -------- training (needs the [promote] extra) --------


def _missing_promote_extra(missing: str) -> HeadRetrainError:
    return HeadRetrainError(
        f"{missing} not installed — required for cluster-head retraining. "
        "Install with `pip install slancha-local[promote]`."
    )


def train_cluster_head(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    n_classes: int,
    *,
    num_iterations: int = 100,
    random_state: int = 42,
) -> bytes:
    """Train a LightGBM multiclass head and return the treelite-serialized
    .bin bytes.

    Pure compute — no I/O, no globals. Tests can call this directly with
    a small synthetic dataset.

    Parameters
    ----------
    X : ``(n_samples, n_features)`` float32
    y : ``(n_samples,)`` int32 — label indices in ``[0, n_classes)``
    n_classes : number of classes; LightGBM requires this explicitly so
        a class with zero samples in ``y`` still gets a head slot.
    num_iterations : boosting rounds. Default 100 is conservative; the
        eval-row aggregator decides whether the resulting head is good
        enough to promote.
    random_state : seed; LightGBM's defaults are non-deterministic
        across machines without it.

    Raises
    ------
    HeadRetrainError
        If lightgbm or treelite is missing, or if training / conversion
        fails.
    """
    if y.size == 0:
        raise HeadRetrainError("empty training set")
    if y.max(initial=-1) >= n_classes or y.min(initial=0) < 0:
        raise HeadRetrainError(f"labels out of range: y in [{y.min()}, {y.max()}] but n_classes={n_classes}")

    try:
        import lightgbm as lgb
    except ImportError as e:
        raise _missing_promote_extra("lightgbm") from e
    try:
        import treelite  # noqa: F401
        from treelite import frontend as tl_frontend
    except ImportError as e:
        raise _missing_promote_extra("treelite") from e
    except OSError as e:
        # libomp / dylib load failures look like OSError from ctypes.
        raise HeadRetrainError(
            f"treelite native library failed to load: {e}. "
            "On macOS run `brew install libomp` and reinstall treelite."
        ) from e

    train_set = lgb.Dataset(X, label=y)
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "verbosity": -1,
        "seed": random_state,
        "deterministic": True,
        "feature_pre_filter": False,
    }
    try:
        booster = lgb.train(params, train_set, num_boost_round=num_iterations)
    except Exception as e:  # lightgbm raises a grab-bag of errors
        raise HeadRetrainError(f"LightGBM training failed: {e}") from e

    try:
        tl_model = tl_frontend.from_lightgbm(booster)
        # treelite >= 4 exposes serialize_bytes for in-memory round-trip;
        # fall back to a tempfile if the runtime version only has
        # path-based serialize.
        if hasattr(tl_model, "serialize_bytes"):
            data = tl_model.serialize_bytes()
            return bytes(data)
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            tmp_path = fh.name
        try:
            tl_model.serialize(tmp_path)
            with open(tmp_path, "rb") as rh:
                return rh.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        raise HeadRetrainError(f"treelite conversion failed: {e}") from e


def verify_load(head_bytes: bytes) -> None:
    """Smoke-load ``head_bytes`` via treelite to catch corruption BEFORE
    flipping the ACTIVE pointer.

    Raises :class:`HeadRetrainError` on any deserialization failure.
    """
    if not head_bytes:
        raise HeadRetrainError("empty head bytes")
    try:
        import treelite
    except ImportError as e:
        raise _missing_promote_extra("treelite") from e
    except OSError as e:
        raise HeadRetrainError(f"treelite native library failed to load: {e}") from e

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(head_bytes)
        tmp_path = fh.name
    try:
        try:
            treelite.Model.deserialize(tmp_path)
        except Exception as e:
            raise HeadRetrainError(f"treelite failed to load head bytes: {e}") from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# -------- top-level orchestration --------


def retrain_cluster_head(
    traces: list[dict],
    snapshot: ClusterSnapshot,
    *,
    n_clusters_per_route: int = 4,
    min_cluster_size: int = 2,
    min_samples_per_class: int = 5,
    num_iterations: int = 100,
    random_state: int = 42,
) -> HeadRetrainResult:
    """End-to-end: traces + snapshot → trained, verify-loaded head bytes.

    Calls :func:`derive_supervised_set`, then :func:`train_cluster_head`,
    then :func:`verify_load`. Returns a :class:`HeadRetrainResult` that
    the caller (e.g. ``promote_head``) writes into a
    :class:`PointerStore` candidate dir.

    The clustering hyperparams should match those used to build
    ``snapshot`` (typically forwarded by the orchestrator from the same
    ``bundle`` config that produced the prior fit).

    Raises :class:`HeadRetrainError` on any pipeline failure.
    """
    X, y, label_table = derive_supervised_set(  # noqa: N806
        traces,
        snapshot,
        n_clusters_per_route=n_clusters_per_route,
        min_cluster_size=min_cluster_size,
        min_samples_per_class=min_samples_per_class,
    )
    n_classes = len(label_table)
    logger.info(
        "retrain_cluster_head: %d samples, %d classes, embedding_dim=%d",
        X.shape[0],
        n_classes,
        X.shape[1],
    )
    head_bytes = train_cluster_head(
        X,
        y,
        n_classes,
        num_iterations=num_iterations,
        random_state=random_state,
    )
    verify_load(head_bytes)
    return HeadRetrainResult(
        head_bytes=head_bytes,
        label_table=label_table,
        n_classes=n_classes,
        n_samples=int(X.shape[0]),
        embedding_dim=int(X.shape[1]),
    )
