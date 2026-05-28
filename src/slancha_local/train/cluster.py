"""Cluster traces by classifier route + KMeans on embeddings.

Two-level clustering:
1. Group traces by ``classifier.route`` (coarse, deterministic).
2. Within each route, KMeans on embedding vectors to find sub-modes.

The result feeds train/val splitting (round-robin across clusters keeps val
representative of the route distribution).

Stable identity (P1)
--------------------
The emergent taxonomy only matters if a "cluster" is the *same conceptual
cluster* across re-fits. Two passes ago "python-debug" might land as cluster 2;
after new traffic it might land as cluster 0 — and the head retrained against
"cluster 2" is then a head retrained against nothing. To prevent that:

* Callers pass in a :class:`ClusterSnapshot` from the previous fit.
* After fitting, each new centroid is matched against the snapshot's prior
  centroids by cosine similarity. If the best match clears
  ``match_threshold`` and is still unclaimed, the new cluster *inherits* the
  prior id. Otherwise it gets a fresh monotonically-increasing id (never
  recycles a retired id, even within the same route).
* :func:`snapshot_from_clusters` round-trips the result back into a
  snapshot for the next pass.

Capacity-bounded k (P1)
-----------------------
Specialists cost compute. Unbounded KMeans grows ``k`` to whatever the caller
asked for; ``cluster_by_route`` instead clamps every per-route ``k`` so the
total cluster count fits the fleet's ``node_capacity`` budget, distributed
proportional to per-route traffic with a floor of 1. When unset (``None``)
the legacy ``n_clusters_per_route`` knob remains the only ceiling.
"""

from __future__ import annotations

import base64
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TraceCluster:
    """A cluster of traces sharing a route + sub-mode.

    ``centroid`` is the mean embedding of the cluster members (``None`` for
    fallback single-cluster groups that skipped KMeans). It is the input to
    stable-id matching on the next pass.
    """

    route: str
    cluster_id: int
    trace_indices: list[int]  # indices into the input list
    centroid: np.ndarray | None = None


@dataclass
class ClusterSnapshot:
    """Persistent state for stable cluster identity across fits.

    ``centroids`` maps ``route -> {cluster_id: centroid}``.
    ``next_id_by_route`` records the next fresh id to allocate on each route
    so retired ids are never recycled (a head retrained against id 7 must
    not silently re-bind to a brand-new mode that happened to land in slot 7).
    """

    centroids: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    next_id_by_route: dict[str, int] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.centroids


def _decode_embedding(b64: str) -> np.ndarray:
    """Trace stores 512 float32s as base64. Decode to (512,)."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32).copy()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _allocate_capacity(
    sizes_by_route: dict[str, int],
    *,
    node_capacity: int,
    n_clusters_per_route: int,
) -> dict[str, int]:
    """Distribute ``node_capacity`` clusters across routes proportional to volume.

    Every represented route gets at least 1. The per-route allocation is
    additionally capped by ``n_clusters_per_route`` (the legacy ceiling), so a
    very large capacity on a small handful of busy routes still respects the
    operator's "no more than N sub-modes per route" knob. The final sum is
    clamped to ``node_capacity`` by trimming the largest allocations first.
    """
    if node_capacity <= 0:
        return {route: 0 for route in sizes_by_route}

    routes = list(sizes_by_route.keys())
    total = sum(sizes_by_route.values())
    n_routes = len(routes)

    if total == 0 or n_routes == 0:
        return {route: 0 for route in routes}

    # Every represented route gets at least 1, otherwise we'd silently drop
    # a route. Cap effective capacity to (n_routes .. node_capacity).
    effective_cap = max(n_routes, node_capacity)

    raw: dict[str, float] = {route: (sizes_by_route[route] / total) * effective_cap for route in routes}
    alloc: dict[str, int] = {
        route: max(1, min(n_clusters_per_route, int(math.floor(raw[route])))) for route in routes
    }

    # Largest-remainder top-up: distribute the leftover seats to the routes
    # with the largest fractional shortfall, up to n_clusters_per_route.
    used = sum(alloc.values())
    remaining = node_capacity - used
    if remaining > 0:
        remainders = sorted(
            ((raw[route] - alloc[route], route) for route in routes),
            reverse=True,
        )
        for _, route in remainders:
            if remaining <= 0:
                break
            if alloc[route] < n_clusters_per_route:
                alloc[route] += 1
                remaining -= 1

    # If we over-allocated (because of the min=1 floor when capacity < n_routes),
    # trim from the routes with the largest allocations first.
    overflow = sum(alloc.values()) - node_capacity
    while overflow > 0:
        # Sort by current allocation desc, route name for determinism.
        biggest = sorted(routes, key=lambda r: (-alloc[r], r))
        trimmed = False
        for route in biggest:
            if alloc[route] > 1:
                alloc[route] -= 1
                overflow -= 1
                trimmed = True
                if overflow <= 0:
                    break
        if not trimmed:
            # All routes already at their floor of 1 — we cannot honor the cap
            # without dropping a route, which the contract forbids. Log and stop.
            logger.warning(
                "node_capacity=%d cannot cover %d routes with floor=1; returning %d clusters total",
                node_capacity,
                n_routes,
                sum(alloc.values()),
            )
            break

    return alloc


def _assign_stable_ids(
    route: str,
    new_centroids: list[np.ndarray],
    new_members: list[list[int]],
    prior: ClusterSnapshot,
    *,
    match_threshold: float,
) -> list[tuple[int, np.ndarray, list[int]]]:
    """Match new centroids against ``prior`` and assign ids.

    Returns a list of ``(cluster_id, centroid, members)`` triples.

    Strategy: greedy proximity matching. New centroids are processed largest
    cluster first (most signal → most worth preserving identity for). For each,
    the best unclaimed prior centroid above ``match_threshold`` wins; otherwise
    a fresh id is allocated from ``prior.next_id_by_route[route]``. ``prior`` is
    mutated to advance ``next_id_by_route``; callers should treat it as
    consumed (build the new snapshot via :func:`snapshot_from_clusters`).
    """
    prior_centroids = dict(prior.centroids.get(route, {}))
    used: set[int] = set()
    next_id = prior.next_id_by_route.get(route, 0)
    # Honour ids that exist in the prior snapshot even if next_id wasn't tracked.
    if prior_centroids:
        next_id = max(next_id, max(prior_centroids.keys()) + 1)

    order = sorted(range(len(new_centroids)), key=lambda i: -len(new_members[i]))

    assignments: list[tuple[int, np.ndarray, list[int]]] = [None] * len(new_centroids)  # type: ignore[list-item]
    for i in order:
        centroid = new_centroids[i]
        best_id: int | None = None
        best_sim = match_threshold
        for cid, prev in prior_centroids.items():
            if cid in used:
                continue
            sim = _cosine_similarity(centroid, prev)
            if sim >= best_sim:
                best_sim = sim
                best_id = cid
        if best_id is None:
            assigned = next_id
            next_id += 1
        else:
            assigned = best_id
            used.add(best_id)
        assignments[i] = (assigned, centroid, new_members[i])

    prior.next_id_by_route[route] = next_id
    return assignments


def cluster_by_route(
    traces: list[dict],
    *,
    n_clusters_per_route: int = 4,
    min_cluster_size: int = 2,
    random_state: int = 42,
    prior: ClusterSnapshot | None = None,
    match_threshold: float = 0.75,
    node_capacity: int | None = None,
) -> list[TraceCluster]:
    """Group by route, then KMeans within each route.

    Routes with fewer than ``min_cluster_size * effective_k`` traces skip
    KMeans (one cluster). When ``prior`` is supplied, new clusters inherit
    prior ids by centroid-proximity match (``match_threshold`` is the cosine
    similarity floor for a match). When ``node_capacity`` is supplied, the
    total cluster count across all routes is bounded by that capacity,
    distributed proportional to per-route volume.
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

    if node_capacity is not None:
        sizes = {route: len(idxs) for route, idxs in by_route.items()}
        per_route_k = _allocate_capacity(
            sizes,
            node_capacity=node_capacity,
            n_clusters_per_route=n_clusters_per_route,
        )
    else:
        per_route_k = {route: n_clusters_per_route for route in by_route}

    # ``prior`` semantics: None ⇒ no stable-id matching, fresh ids per route
    # starting at 0. We use an empty scratch snapshot so the matching path is
    # uniform; callers who passed in a real ``prior`` see it advance.
    snapshot = prior if prior is not None else ClusterSnapshot()

    out: list[TraceCluster] = []
    for route, idxs in by_route.items():
        k = per_route_k.get(route, n_clusters_per_route)
        if k <= 0:
            # Capacity kill-switch: route gets no clusters this pass.
            continue
        if k == 1 or len(idxs) < min_cluster_size * max(k, 2):
            members = list(idxs)
            X = np.stack(  # noqa: N806
                [_decode_embedding(traces[i]["embedding_b64"]) for i in members]
            )
            centroid = X.mean(axis=0)
            [(cid, _, _)] = _assign_stable_ids(
                route,
                [centroid],
                [members],
                snapshot,
                match_threshold=match_threshold,
            )
            out.append(
                TraceCluster(
                    route=route,
                    cluster_id=cid,
                    trace_indices=members,
                    centroid=centroid,
                )
            )
            continue
        X = np.stack(  # noqa: N806
            [_decode_embedding(traces[i]["embedding_b64"]) for i in idxs]
        )
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X)
            centroids = km.cluster_centers_
        except Exception as e:
            logger.warning("KMeans failed for route %s: %s — single cluster", route, e)
            members = list(idxs)
            centroid = X.mean(axis=0)
            [(cid, _, _)] = _assign_stable_ids(
                route,
                [centroid],
                [members],
                snapshot,
                match_threshold=match_threshold,
            )
            out.append(
                TraceCluster(
                    route=route,
                    cluster_id=cid,
                    trace_indices=members,
                    centroid=centroid,
                )
            )
            continue

        new_centroids: list[np.ndarray] = []
        new_members: list[list[int]] = []
        for cid in range(k):
            members = [idxs[j] for j, lab in enumerate(labels) if lab == cid]
            if not members:
                continue
            new_centroids.append(centroids[cid])
            new_members.append(members)

        if not new_centroids:
            continue

        assignments = _assign_stable_ids(
            route,
            new_centroids,
            new_members,
            snapshot,
            match_threshold=match_threshold,
        )
        for stable_id, centroid, members in assignments:
            out.append(
                TraceCluster(
                    route=route,
                    cluster_id=stable_id,
                    trace_indices=members,
                    centroid=centroid,
                )
            )
    return out


def snapshot_from_clusters(
    clusters: list[TraceCluster],
    prior: ClusterSnapshot | None = None,
) -> ClusterSnapshot:
    """Build a :class:`ClusterSnapshot` from a fit result for the next pass.

    Clusters without a recorded centroid are skipped (they cannot anchor a
    stable id).

    ``next_id_by_route`` is the **all-time** high-water mark per route, not
    just ``max(surviving_id) + 1``. If a high-numbered cluster drops out of
    the fit (its mode lost all traffic), recomputing from survivors alone
    would regress the counter and let the next pass recycle that id — a
    specialist deployed against the retired cluster would then silently
    re-bind to an unrelated mode. The high-water is carried forward by
    taking the max of:

    * the prior snapshot's ``next_id_by_route[route]`` (if ``prior`` given), and
    * ``max(surviving_id) + 1`` for clusters present in this result.

    Callers who passed a ``prior`` into :func:`cluster_by_route` should pass
    the *same* ``prior`` here — it has already been advanced in place during
    assignment, so this is also the convenient way to propagate that
    advancement into the next snapshot.
    """
    centroids: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for c in clusters:
        if c.centroid is None:
            continue
        centroids[c.route][c.cluster_id] = c.centroid

    prior_next = dict(prior.next_id_by_route) if prior is not None else {}
    all_routes: set[str] = set(centroids.keys()) | set(prior_next.keys())
    next_id_by_route: dict[str, int] = {}
    for route in all_routes:
        survivors_high = max(centroids[route].keys()) + 1 if centroids.get(route) else 0
        next_id_by_route[route] = max(prior_next.get(route, 0), survivors_high)

    return ClusterSnapshot(
        centroids=dict(centroids),
        next_id_by_route=next_id_by_route,
    )
