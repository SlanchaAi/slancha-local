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
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SNAPSHOT_FORMAT_VERSION = 1
"""On-disk ClusterSnapshot format version. Bump on incompatible schema changes
(field additions are not "incompatible" — readers ignore unknown fields)."""

_SNAPSHOT_JSON_SUFFIX = ".json"
_SNAPSHOT_NPZ_SUFFIX = ".npz"


def _snapshot_pair(path: str | Path) -> tuple[Path, Path]:
    """Resolve ``path`` to the canonical ``(npz, json)`` pair.

    Accepts a stem (``cluster_snapshot``), a ``.npz`` path, or a ``.json``
    path; always returns both. Used by :meth:`ClusterSnapshot.save` and
    :meth:`ClusterSnapshot.load` so callers don't have to track suffixes.
    """
    p = Path(path).expanduser().resolve()
    if p.suffix == _SNAPSHOT_NPZ_SUFFIX:
        return p, p.with_suffix(_SNAPSHOT_JSON_SUFFIX)
    if p.suffix == _SNAPSHOT_JSON_SUFFIX:
        return p.with_suffix(_SNAPSHOT_NPZ_SUFFIX), p
    return p.with_suffix(_SNAPSHOT_NPZ_SUFFIX), p.with_suffix(_SNAPSHOT_JSON_SUFFIX)


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

    ``centroids`` maps ``route -> {cluster_id: centroid}`` for the **active**
    clusters (everyone present in the last fit). ``next_id_by_route`` records
    the next fresh id to allocate on each route so retired ids are never
    recycled (a head retrained against id 7 must not silently re-bind to a
    brand-new mode that happened to land in slot 7).

    Retained stickiness (P1.5)
    --------------------------
    ``retired_centroids`` is a per-route bounded LRU of ``(retired_id,
    centroid)`` pairs in **most-recent-first** order: index 0 is the cluster
    that retired most recently. When a fit produces a new centroid that has no
    match in ``centroids`` but does match an entry in ``retired_centroids``
    above the same ``match_threshold``, the retired id is **revived** — the
    new cluster inherits its old id, the entry is removed from the retired
    pool, and a specialist that had been deployed against that id reattaches
    automatically when its traffic returns.

    The pool is bounded per-route by ``retained_capacity`` (the cap is applied
    in :func:`snapshot_from_clusters`, not stored on the snapshot). Once the
    cap is exceeded the oldest entries (tail of the list) are evicted. A mode
    that returns *after* its retired entry has been evicted does not revive —
    it gets a fresh id — but ``next_id_by_route`` still guarantees the new id
    is strictly greater than any id ever issued for that route, so it cannot
    collide with the dead specialist's slot.
    """

    centroids: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    next_id_by_route: dict[str, int] = field(default_factory=dict)
    retired_centroids: dict[str, list[tuple[int, np.ndarray]]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.centroids and not self.retired_centroids

    # ------------------------------------------------------------------
    # Persistence (P2a)
    #
    # On disk: TWO files sharing the same stem (e.g. ``cluster_snapshot``):
    #   • ``<stem>.npz`` — centroid ndarrays only, keyed ``arr_<i>``.
    #     numpy-native, lossless float32, no pickle.
    #   • ``<stem>.json`` — sidecar with ``schema_version`` plus per-route
    #     metadata: active ``{cluster_id: arr_<i>}``, retired LRU
    #     ``[[cluster_id, arr_<i>], ...]``, and ``next_id``.
    #
    # Why split: per onyx-ridge's review note, the sidecar grows as later
    # loop phases attach things to clusters (specialist/adapter pointers
    # in P3, eval bindings, …). A versioned JSON sidecar lets the loader
    # migrate forward without round-tripping numpy. Cheap now, painful to
    # retrofit. The loader also tolerates older snapshots that pre-date
    # ``retired`` (treated as empty) so a P1.0-era artifact still loads.
    #
    # Both files are written via tmpfile-then-rename. We finish the npz
    # first so a partial state never satisfies the existence check the
    # loader does on the sidecar.
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialize this snapshot to ``<path>.npz`` + ``<path>.json``.

        ``path`` may include or omit the ``.npz`` suffix; the returned
        :class:`Path` always points at the canonical ``.npz`` so callers
        can ship/inspect either pair. Creates parent directories. Empty
        snapshots are still written (they produce a valid round-trippable
        pair).
        """
        npz_path, json_path = _snapshot_pair(path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        meta_routes: dict[str, dict] = {}

        def _stash(c: np.ndarray) -> str:
            key = f"arr_{len(arrays)}"
            arrays[key] = np.ascontiguousarray(c)
            return key

        all_routes: set[str] = (
            set(self.centroids.keys())
            | set(self.next_id_by_route.keys())
            | set(self.retired_centroids.keys())
        )
        for route in sorted(all_routes):
            active_map: dict[str, str] = {}
            for cid, vec in sorted(self.centroids.get(route, {}).items()):
                active_map[str(cid)] = _stash(vec)
            retired_pairs: list[list] = []
            for cid, vec in self.retired_centroids.get(route, []):
                retired_pairs.append([int(cid), _stash(vec)])
            meta_routes[route] = {
                "active": active_map,  # {str(cluster_id): "arr_<i>"}
                "retired": retired_pairs,  # [[cluster_id, "arr_<i>"], ...] newest-first
                "next_id": int(self.next_id_by_route.get(route, 0)),
            }

        sidecar = {
            "schema_version": SNAPSHOT_FORMAT_VERSION,
            "routes": meta_routes,
        }
        sidecar_bytes = json.dumps(sidecar, sort_keys=True, indent=2).encode("utf-8")

        npz_tmp = npz_path.with_suffix(npz_path.suffix + ".tmp")
        json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
        # np.savez auto-appends ``.npz`` when given a path; pass a file
        # handle so the .tmp suffix survives the round-trip.
        with open(npz_tmp, "wb") as fh:
            if arrays:
                np.savez(fh, **arrays)
            else:
                # np.savez refuses an empty kwargs dict; emit a sentinel array
                # so the .npz file still exists for downstream tooling.
                np.savez(fh, _empty=np.zeros(0, dtype=np.float32))
        json_tmp.write_bytes(sidecar_bytes)
        # npz first (sidecar is the "commit" file the loader keys off).
        npz_tmp.replace(npz_path)
        json_tmp.replace(json_path)
        return npz_path

    @classmethod
    def load(cls, path: str | Path) -> ClusterSnapshot:
        """Load a snapshot previously written by :meth:`save`.

        ``path`` may point at either ``<stem>.npz`` or ``<stem>.json`` (or
        the bare stem). Raises :class:`FileNotFoundError` if either file
        is missing, :class:`ValueError` on an unparseable sidecar or a
        ``schema_version`` newer than this code understands.
        """
        npz_path, json_path = _snapshot_pair(path)
        if not json_path.exists():
            raise FileNotFoundError(f"snapshot sidecar not found: {json_path}")
        if not npz_path.exists():
            raise FileNotFoundError(f"snapshot npz not found: {npz_path}")

        try:
            sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"snapshot sidecar at {json_path} is unparseable: {e}") from e

        schema_version = sidecar.get("schema_version", sidecar.get("version"))
        if not isinstance(schema_version, int) or schema_version > SNAPSHOT_FORMAT_VERSION:
            raise ValueError(
                f"snapshot at {json_path} schema_version={schema_version!r} is newer than this code "
                f"(supports up to v{SNAPSHOT_FORMAT_VERSION}); upgrade slancha-local"
            )

        centroids: dict[str, dict[int, np.ndarray]] = {}
        retired_centroids: dict[str, list[tuple[int, np.ndarray]]] = {}
        next_id_by_route: dict[str, int] = {}

        with np.load(npz_path) as data:
            for route, rmeta in sidecar.get("routes", {}).items():
                active: dict[int, np.ndarray] = {}
                for cid_s, arr_key in rmeta.get("active", {}).items():
                    active[int(cid_s)] = np.asarray(data[str(arr_key)]).copy()
                if active:
                    centroids[route] = active
                # Tolerant of P1.0-era snapshots that pre-date the retired pool.
                retired: list[tuple[int, np.ndarray]] = []
                for cid, arr_key in rmeta.get("retired", []) or []:
                    retired.append((int(cid), np.asarray(data[str(arr_key)]).copy()))
                if retired:
                    retired_centroids[route] = retired
                next_id_by_route[route] = int(rmeta.get("next_id", 0))

        return cls(
            centroids=centroids,
            next_id_by_route=next_id_by_route,
            retired_centroids=retired_centroids,
        )


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
    cluster first (most signal → most worth preserving identity for). For
    each new centroid:

    1. Best unclaimed entry in ``prior.centroids[route]`` (the *active* pool)
       above ``match_threshold`` wins → adopt its id.
    2. Otherwise, best unclaimed entry in ``prior.retired_centroids[route]``
       (the *retired* LRU pool) above the same threshold wins → revive its
       id. The matched entry is removed from ``prior.retired_centroids`` so
       the next snapshot doesn't double-list it.
    3. Otherwise a fresh id is allocated from
       ``prior.next_id_by_route[route]``.

    ``prior`` is mutated to advance ``next_id_by_route`` and (when a retired
    id revives) to prune ``retired_centroids[route]``; callers should treat
    it as consumed (build the new snapshot via
    :func:`snapshot_from_clusters`).
    """
    prior_centroids = dict(prior.centroids.get(route, {}))
    used_active: set[int] = set()
    retired_pool: list[tuple[int, np.ndarray]] = list(prior.retired_centroids.get(route, []))
    used_retired_idx: set[int] = set()

    next_id = prior.next_id_by_route.get(route, 0)
    # Honour ids that exist in the prior snapshot even if next_id wasn't tracked.
    if prior_centroids:
        next_id = max(next_id, max(prior_centroids.keys()) + 1)
    if retired_pool:
        next_id = max(next_id, max(cid for cid, _ in retired_pool) + 1)

    order = sorted(range(len(new_centroids)), key=lambda i: -len(new_members[i]))

    assignments: list[tuple[int, np.ndarray, list[int]]] = [None] * len(new_centroids)  # type: ignore[list-item]
    for i in order:
        centroid = new_centroids[i]
        # Pass 1: active prior centroids.
        best_id: int | None = None
        best_sim = match_threshold
        for cid, prev in prior_centroids.items():
            if cid in used_active:
                continue
            sim = _cosine_similarity(centroid, prev)
            if sim >= best_sim:
                best_sim = sim
                best_id = cid
        if best_id is not None:
            assigned = best_id
            used_active.add(best_id)
            assignments[i] = (assigned, centroid, new_members[i])
            continue

        # Pass 2: retired pool (revival).
        best_retired_idx: int | None = None
        best_sim = match_threshold
        for idx, (_cid, prev) in enumerate(retired_pool):
            if idx in used_retired_idx:
                continue
            sim = _cosine_similarity(centroid, prev)
            if sim >= best_sim:
                best_sim = sim
                best_retired_idx = idx
        if best_retired_idx is not None:
            revived_id = retired_pool[best_retired_idx][0]
            used_retired_idx.add(best_retired_idx)
            assignments[i] = (revived_id, centroid, new_members[i])
            continue

        # Pass 3: fresh id.
        assigned = next_id
        next_id += 1
        assignments[i] = (assigned, centroid, new_members[i])

    prior.next_id_by_route[route] = next_id

    # Commit retired-pool pruning back to the prior snapshot so the next
    # snapshot doesn't re-list revived entries.
    if used_retired_idx:
        kept = [pair for idx, pair in enumerate(retired_pool) if idx not in used_retired_idx]
        if kept:
            prior.retired_centroids[route] = kept
        else:
            prior.retired_centroids.pop(route, None)

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


DEFAULT_RETAINED_CAPACITY = 32
"""Default per-route cap for retired centroids carried into the next snapshot.

Bounded so the snapshot doesn't grow unbounded with churn; large enough that a
mode whose traffic comes back after a few quiet passes still revives its old
id rather than minting a fresh one (and orphaning its specialist). Override
with the ``retained_capacity`` kwarg on :func:`snapshot_from_clusters`.
"""


def snapshot_from_clusters(
    clusters: list[TraceCluster],
    prior: ClusterSnapshot | None = None,
    *,
    retained_capacity: int = DEFAULT_RETAINED_CAPACITY,
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

    * the prior snapshot's ``next_id_by_route[route]`` (if ``prior`` given),
    * ``max(surviving_id) + 1`` for clusters present in this result, and
    * ``max(retired_id) + 1`` across retained-pool entries.

    ``retired_centroids`` (P1.5) is the per-route LRU pool of retired
    centroids that the next :func:`cluster_by_route` will check when a new
    cluster has no match in the active set. It is built by:

    1. Carrying any **newly retired** clusters from ``prior.centroids[route]``
       — entries whose id is NOT in this fit's surviving ids — to the front
       of the list (most-recent-first).
    2. Appending the pre-existing retired entries from
       ``prior.retired_centroids[route]`` (which the fit may have pruned to
       drop revived ids).
    3. Truncating to ``retained_capacity`` per route. The tail (oldest) is
       evicted first. A mode that returns after its retired entry was
       evicted does not revive — but it gets a fresh id strictly greater
       than the all-time high-water, so it still cannot collide with the
       dead specialist's slot.

    ``retained_capacity=0`` disables stickiness entirely (no retained pool;
    legacy P1.0 behaviour).

    Callers who passed a ``prior`` into :func:`cluster_by_route` should pass
    the *same* ``prior`` here — it has already been advanced in place during
    assignment (``next_id_by_route`` advanced, ``retired_centroids`` pruned
    of revived ids), so this is also the convenient way to propagate that
    advancement into the next snapshot.
    """
    if retained_capacity < 0:
        raise ValueError(f"retained_capacity must be >= 0, got {retained_capacity}")

    centroids: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for c in clusters:
        if c.centroid is None:
            continue
        centroids[c.route][c.cluster_id] = c.centroid

    prior_next = dict(prior.next_id_by_route) if prior is not None else {}
    prior_active = prior.centroids if prior is not None else {}
    prior_retired = prior.retired_centroids if prior is not None else {}

    all_routes: set[str] = (
        set(centroids.keys()) | set(prior_next.keys()) | set(prior_active.keys()) | set(prior_retired.keys())
    )

    next_id_by_route: dict[str, int] = {}
    retired_centroids: dict[str, list[tuple[int, np.ndarray]]] = {}

    for route in all_routes:
        survivors = centroids.get(route, {})
        survivors_high = max(survivors.keys()) + 1 if survivors else 0
        retired_for_route = list(prior_retired.get(route, []))
        retired_high = (max(cid for cid, _ in retired_for_route) + 1) if retired_for_route else 0
        next_id_by_route[route] = max(
            prior_next.get(route, 0),
            survivors_high,
            retired_high,
        )

        if retained_capacity == 0:
            continue

        # Newly-retired (in prior active, not in current survivors), prepended
        # to the LRU so they sit at the front (most recent).
        newly_retired: list[tuple[int, np.ndarray]] = [
            (cid, c) for cid, c in prior_active.get(route, {}).items() if cid not in survivors
        ]
        # Deterministic order on newly-retired by descending cid so the most
        # recently issued id is at index 0 within the batch (tests + readers
        # don't rely on dict iteration order).
        newly_retired.sort(key=lambda pair: -pair[0])

        merged = newly_retired + retired_for_route
        if not merged:
            continue
        # Deduplicate by id (revival paths can leave the same id in both halves
        # if a caller misuses the API; first occurrence — i.e. newer — wins).
        seen: set[int] = set()
        deduped: list[tuple[int, np.ndarray]] = []
        for cid, c in merged:
            if cid in seen:
                continue
            seen.add(cid)
            deduped.append((cid, c))
        retired_centroids[route] = deduped[:retained_capacity]

    return ClusterSnapshot(
        centroids=dict(centroids),
        next_id_by_route=next_id_by_route,
        retired_centroids=retired_centroids,
    )
