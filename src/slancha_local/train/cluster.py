"""Cluster traces by classifier route + KMeans on embeddings.

Two-level clustering:
1. Group traces by classifier.route (coarse, deterministic).
2. Within each route, KMeans on embedding vectors to find sub-modes.

The result feeds train/val splitting (round-robin across clusters keeps val
representative of the route distribution).
"""

from __future__ import annotations

import base64
import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TraceCluster:
    route: str
    cluster_id: int
    trace_indices: list[int]  # indices into the input list


def _decode_embedding(b64: str) -> np.ndarray:
    """Trace stores 512 float32s as base64. Decode to (512,)."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32).copy()


def cluster_by_route(
    traces: list[dict],
    *,
    n_clusters_per_route: int = 4,
    min_cluster_size: int = 2,
    random_state: int = 42,
) -> list[TraceCluster]:
    """Group by route, then KMeans within each route.

    Routes with fewer than ``min_cluster_size * n_clusters_per_route`` traces
    skip KMeans (one cluster).
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise RuntimeError(
            "scikit-learn required for clustering. Install slancha-local[classifier] "
            "or [dev], or use --no-cluster to skip clustering."
        ) from e

    by_route: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(traces):
        route = (t.get("classifier") or {}).get("route") or "unknown"
        by_route[route].append(i)

    out: list[TraceCluster] = []
    for route, idxs in by_route.items():
        if len(idxs) < n_clusters_per_route * min_cluster_size:
            out.append(TraceCluster(route=route, cluster_id=0, trace_indices=idxs))
            continue
        X = np.stack([_decode_embedding(traces[i]["embedding_b64"]) for i in idxs])  # noqa: N806
        try:
            km = KMeans(n_clusters=n_clusters_per_route, random_state=random_state, n_init=10)
            labels = km.fit_predict(X)
        except Exception as e:
            logger.warning("KMeans failed for route %s: %s — single cluster", route, e)
            out.append(TraceCluster(route=route, cluster_id=0, trace_indices=idxs))
            continue
        for cid in range(n_clusters_per_route):
            members = [idxs[j] for j, lab in enumerate(labels) if lab == cid]
            if members:
                out.append(TraceCluster(route=route, cluster_id=cid, trace_indices=members))
    return out
