"""Tests for the P1 cluster module: stable identity + capacity-bounded k.

The clustering primitive is the linchpin of the self-organizing loop
(``docs/SELF_ORGANIZING_LOOP_SCOPE.md``). Two properties matter:

1. **Stable identity** — a cluster keeps its id when new traffic arrives,
   so a fine-tuned specialist stays bound to "its" cluster across retrains.
2. **Capacity-bounded k** — total cluster count fits the fleet's specialist
   budget, distributed proportional to per-route volume with a floor of 1.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from slancha_local.train.cluster import (
    ClusterSnapshot,
    TraceCluster,
    cluster_by_route,
    snapshot_from_clusters,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emb(vec: np.ndarray) -> str:
    return base64.b64encode(vec.astype(np.float32).tobytes()).decode()


def _make_modes(
    *,
    route: str,
    centers: list[np.ndarray],
    per_mode: int,
    noise: float = 0.05,
    seed: int = 0,
) -> list[dict]:
    """Build traces from well-separated centers so KMeans can recover them.

    Each "mode" is a tight Gaussian blob around its center. ``cluster_by_route``
    has no idea which mode is which until it fits; we keep noise low so the
    fit is deterministic enough to test identity preservation.
    """
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


def _by_member_signature(clusters: list[TraceCluster]) -> dict[frozenset, tuple[str, int]]:
    """Map each cluster's member set to (route, cluster_id).

    Same prompts, same membership → so we can ask "did the conceptual cluster
    keep its id between fits even though the traces are differently ordered?".
    """
    return {frozenset(c.trace_indices): (c.route, c.cluster_id) for c in clusters}


# ---------------------------------------------------------------------------
# Stable identity
# ---------------------------------------------------------------------------


def test_stable_ids_persist_when_new_traffic_appended() -> None:
    """A cluster should keep its id after new traces are appended.

    The KMeans label ordering is implementation-defined; without the
    snapshot the same conceptual cluster routinely gets renumbered.
    Stable-id matching against prior centroids fixes that.
    """
    dim = 16
    centers = [
        np.eye(dim)[0] * 5,
        np.eye(dim)[1] * 5,
        np.eye(dim)[2] * 5,
    ]
    pass1_traces = _make_modes(route="general_qa", centers=centers, per_mode=8, seed=1)
    pass1 = cluster_by_route(
        pass1_traces,
        n_clusters_per_route=3,
        min_cluster_size=2,
    )
    snap = snapshot_from_clusters(pass1)
    assert not snap.is_empty()
    assert set(snap.centroids["general_qa"].keys()) == {c.cluster_id for c in pass1}

    # Build pass 2 with the original traces + extra traffic for the same 3 modes.
    extra = _make_modes(route="general_qa", centers=centers, per_mode=4, seed=2)
    pass2_traces = pass1_traces + extra
    pass2 = cluster_by_route(
        pass2_traces,
        n_clusters_per_route=3,
        min_cluster_size=2,
        prior=snap,
    )

    # Every prior cluster id should still be present in pass 2 — match by
    # which mode the cluster is closest to. We reconstruct mode→id from
    # both passes and compare the bijection.
    def mode_of_cluster(cluster: TraceCluster, traces: list[dict]) -> int:
        embeddings = np.stack(
            [
                np.frombuffer(base64.b64decode(traces[i]["embedding_b64"]), dtype=np.float32)
                for i in cluster.trace_indices
            ]
        )
        mean = embeddings.mean(axis=0)
        return int(np.argmax([np.dot(mean, c) for c in centers]))

    pass1_mode_to_id = {mode_of_cluster(c, pass1_traces): c.cluster_id for c in pass1}
    pass2_mode_to_id = {mode_of_cluster(c, pass2_traces): c.cluster_id for c in pass2}
    assert set(pass1_mode_to_id.keys()) == {0, 1, 2}
    assert pass1_mode_to_id == pass2_mode_to_id, (
        "stable-id contract broken: a mode's cluster_id changed across passes "
        f"(pass1={pass1_mode_to_id} pass2={pass2_mode_to_id})"
    )


def test_new_mode_gets_fresh_id_above_prior_max() -> None:
    """A genuinely new mode should not recycle a retired id."""
    dim = 16
    base_centers = [np.eye(dim)[0] * 5, np.eye(dim)[1] * 5]
    pass1_traces = _make_modes(route="general_qa", centers=base_centers, per_mode=8, seed=1)
    pass1 = cluster_by_route(pass1_traces, n_clusters_per_route=2, min_cluster_size=2)
    snap = snapshot_from_clusters(pass1)
    prior_max = max(c.cluster_id for c in pass1)

    new_center = np.eye(dim)[7] * 5
    pass2_traces = pass1_traces + _make_modes(route="general_qa", centers=[new_center], per_mode=8, seed=2)
    pass2 = cluster_by_route(
        pass2_traces,
        n_clusters_per_route=3,
        min_cluster_size=2,
        prior=snap,
    )

    pass2_ids = {c.cluster_id for c in pass2}
    fresh = pass2_ids - {c.cluster_id for c in pass1}
    assert fresh, "expected at least one fresh id for the new mode"
    assert min(fresh) > prior_max, f"new id {fresh} must be > prior_max={prior_max} (no recycling)"


def test_retired_id_not_recycled() -> None:
    """If a prior cluster disappears (no matching traffic), its id stays retired.

    A specialist may still be deployed against that id; a future cluster that
    happens to land near a different mode must not silently inherit it.
    """
    dim = 16
    snap = ClusterSnapshot(
        centroids={
            "general_qa": {
                3: np.eye(dim)[5] * 5,  # nothing in new traffic matches this
            }
        },
        next_id_by_route={"general_qa": 7},  # next fresh id starts at 7
    )
    traces = _make_modes(
        route="general_qa",
        centers=[np.eye(dim)[0] * 5, np.eye(dim)[1] * 5],
        per_mode=6,
        seed=3,
    )
    clusters = cluster_by_route(traces, n_clusters_per_route=2, min_cluster_size=2, prior=snap)
    ids = sorted(c.cluster_id for c in clusters)
    assert 3 not in ids, "id 3 (retired prior) was silently recycled"
    assert min(ids) >= 7, f"new ids {ids} must be >= next_id_by_route=7"


def test_match_threshold_rejects_distant_centroids() -> None:
    """A near-orthogonal new centroid must not inherit a prior id."""
    dim = 16
    snap = ClusterSnapshot(
        centroids={"general_qa": {0: np.eye(dim)[0] * 5}},
        next_id_by_route={"general_qa": 1},
    )
    traces = _make_modes(
        route="general_qa",
        centers=[np.eye(dim)[8] * 5],
        per_mode=8,
        seed=4,
    )
    clusters = cluster_by_route(
        traces,
        n_clusters_per_route=1,
        min_cluster_size=2,
        prior=snap,
        match_threshold=0.9,
    )
    assert len(clusters) == 1
    assert clusters[0].cluster_id != 0, "distant centroid wrongly inherited id 0"
    assert clusters[0].cluster_id >= 1


# ---------------------------------------------------------------------------
# Capacity-bounded k
# ---------------------------------------------------------------------------


def test_capacity_caps_total_clusters() -> None:
    """``node_capacity`` is a hard ceiling on total cluster count."""
    dim = 16
    # Three routes that each *want* 4 clusters → 12 — but we only have room for 5.
    centers = [np.eye(dim)[i] * 5 for i in range(4)]
    traces: list[dict] = []
    for route in ("alpha", "beta", "gamma"):
        traces.extend(_make_modes(route=route, centers=centers, per_mode=4, seed=hash(route) & 0xFF))

    clusters = cluster_by_route(
        traces,
        n_clusters_per_route=4,
        min_cluster_size=2,
        node_capacity=5,
    )
    assert len(clusters) <= 5, f"node_capacity=5 violated: emitted {len(clusters)} clusters"
    # Every route still represented (no silent drops).
    assert {c.route for c in clusters} == {"alpha", "beta", "gamma"}


def test_capacity_proportional_to_traffic() -> None:
    """Heavier routes get a bigger share of the cluster budget."""
    dim = 16
    centers = [np.eye(dim)[i] * 5 for i in range(4)]
    # 'heavy' has 4x the traffic of 'light' → heavier route should get more clusters.
    heavy = _make_modes(route="heavy", centers=centers, per_mode=16, seed=10)
    light = _make_modes(route="light", centers=centers[:1], per_mode=4, seed=11)

    clusters = cluster_by_route(
        heavy + light,
        n_clusters_per_route=4,
        min_cluster_size=2,
        node_capacity=5,
    )
    heavy_k = sum(1 for c in clusters if c.route == "heavy")
    light_k = sum(1 for c in clusters if c.route == "light")
    assert heavy_k >= light_k, f"heavy={heavy_k} should be >= light={light_k}"
    assert heavy_k + light_k <= 5


def test_capacity_zero_emits_nothing() -> None:
    """Zero capacity is a kill-switch: no clusters at all."""
    dim = 16
    traces = _make_modes(
        route="x",
        centers=[np.eye(dim)[0] * 5, np.eye(dim)[1] * 5],
        per_mode=4,
        seed=0,
    )
    clusters = cluster_by_route(traces, n_clusters_per_route=2, min_cluster_size=2, node_capacity=0)
    assert clusters == []


def test_capacity_unset_matches_legacy_behaviour() -> None:
    """Without ``node_capacity`` the legacy ``n_clusters_per_route`` rules.

    The bundle pipeline and downstream consumers were written against the
    pre-P1 behaviour; the kwarg defaults must not change semantics.
    """
    dim = 16
    centers = [np.eye(dim)[i] * 5 for i in range(3)]
    traces = _make_modes(route="x", centers=centers, per_mode=6, seed=0)
    clusters = cluster_by_route(traces, n_clusters_per_route=3, min_cluster_size=2)
    assert len(clusters) == 3
    assert {c.route for c in clusters} == {"x"}


# ---------------------------------------------------------------------------
# Snapshot round-trip
# ---------------------------------------------------------------------------


def test_snapshot_round_trip_preserves_centroids_and_next_id() -> None:
    dim = 16
    centers = [np.eye(dim)[i] * 5 for i in range(3)]
    traces = _make_modes(route="x", centers=centers, per_mode=8, seed=0)
    clusters = cluster_by_route(traces, n_clusters_per_route=3, min_cluster_size=2)
    snap = snapshot_from_clusters(clusters)
    assert set(snap.centroids["x"].keys()) == {c.cluster_id for c in clusters}
    assert snap.next_id_by_route["x"] == max(c.cluster_id for c in clusters) + 1


def test_snapshot_high_water_survives_dropped_top_id() -> None:
    """Boss-reported regression: dropping the highest-id mode then minting a
    new one across two round-trips must NOT recycle the retired id.

    Three passes:
        P1 modes A,B,C -> ids 0,1,2 ; snap.next_id = 3
        P2 only A,B    -> ids {0,1} ; snap.next_id MUST stay >=3 (not regress to 2)
        P3 A,B,D       -> D's id MUST be > 2 (3 or higher)

    Without ``prior`` propagation through :func:`snapshot_from_clusters`,
    the P2 snapshot regresses ``next_id`` to ``max({0,1})+1 = 2`` and the
    new mode D in P3 silently re-binds to id 2 — exactly what stable-ids
    is supposed to prevent.
    """
    dim = 16
    base = [np.eye(dim)[0] * 5, np.eye(dim)[1] * 5, np.eye(dim)[2] * 5]

    # P1 — three modes.
    p1_traces = _make_modes(route="x", centers=base, per_mode=8, seed=1)
    p1 = cluster_by_route(p1_traces, n_clusters_per_route=3, min_cluster_size=2)
    snap1 = snapshot_from_clusters(p1)
    all_time_max_p1 = max(c.cluster_id for c in p1)
    assert snap1.next_id_by_route["x"] == all_time_max_p1 + 1

    # Which id mapped to mode C (the one we're going to retire)? It's the
    # cluster whose centroid is closest to base[2].
    def mode_of(cluster: TraceCluster, traces: list[dict]) -> int:
        embeddings = np.stack(
            [
                np.frombuffer(base64.b64decode(traces[i]["embedding_b64"]), dtype=np.float32)
                for i in cluster.trace_indices
            ]
        )
        mean = embeddings.mean(axis=0)
        return int(np.argmax([np.dot(mean, c) for c in base]))

    mode_to_id_p1 = {mode_of(c, p1_traces): c.cluster_id for c in p1}
    retired_id = mode_to_id_p1[2]  # id originally bound to mode C

    # P2 — only modes A and B (C has gone silent).
    p2_traces = _make_modes(route="x", centers=base[:2], per_mode=8, seed=2)
    p2 = cluster_by_route(p2_traces, n_clusters_per_route=2, min_cluster_size=2, prior=snap1)
    snap2 = snapshot_from_clusters(p2, prior=snap1)
    # next_id must NOT regress: stay at least at the P1 high-water.
    assert snap2.next_id_by_route["x"] >= snap1.next_id_by_route["x"], (
        f"next_id regressed across round-trip with dropout: "
        f"snap1.next_id={snap1.next_id_by_route['x']} "
        f"snap2.next_id={snap2.next_id_by_route['x']}"
    )

    # P3 — A, B, and a brand new mode D in a never-seen direction.
    new_mode = np.eye(dim)[9] * 5
    p3_traces = _make_modes(route="x", centers=[base[0], base[1], new_mode], per_mode=8, seed=3)
    p3 = cluster_by_route(p3_traces, n_clusters_per_route=3, min_cluster_size=2, prior=snap2)
    p3_ids = {c.cluster_id for c in p3}
    fresh = p3_ids - {c.cluster_id for c in p2}
    assert fresh, "expected a fresh id for the brand-new mode D in P3"
    assert retired_id not in fresh, (
        f"retired id {retired_id} was silently recycled by new mode D "
        f"(round-trip dropout regression — boss-reported)"
    )
    assert min(fresh) > all_time_max_p1, (
        f"new mode D should get id > all-time max {all_time_max_p1}, got {fresh}"
    )


def test_snapshot_carries_prior_high_water_for_routes_with_no_survivors() -> None:
    """If a route has zero survivors but the prior tracked it, the
    high-water for that route still must not regress.
    """
    dim = 8
    prior = ClusterSnapshot(
        centroids={"x": {5: np.eye(dim)[0] * 3.0}},
        next_id_by_route={"x": 6},
    )
    snap = snapshot_from_clusters([], prior=prior)
    assert snap.next_id_by_route["x"] == 6


def test_snapshot_skips_clusters_without_centroid() -> None:
    bare = [
        TraceCluster(route="x", cluster_id=2, trace_indices=[0]),
        TraceCluster(
            route="x",
            cluster_id=5,
            trace_indices=[1],
            centroid=np.ones(4, dtype=np.float32),
        ),
    ]
    snap = snapshot_from_clusters(bare)
    assert snap.centroids["x"] == {5: pytest.approx(np.ones(4, dtype=np.float32))}
    assert snap.next_id_by_route["x"] == 6


# ---------------------------------------------------------------------------
# Retained-centroid stickiness (P1.5)
# ---------------------------------------------------------------------------


def test_retired_centroid_revives_when_mode_returns() -> None:
    """A mode that goes quiet and comes back keeps its original id.

    Three passes:

    * P1: modes A, B, C → ids 0, 1, 2 ; snapshot_from_clusters retains C in
      retired_centroids when it falls off in P2.
    * P2: only A, B (C has gone silent) → ids {0, 1}.
    * P3: A, B, and C again → C revives id 2 from the retired pool instead
      of getting a fresh id like 3.
    """
    dim = 16
    base = [np.eye(dim)[0] * 5, np.eye(dim)[1] * 5, np.eye(dim)[2] * 5]

    p1_traces = _make_modes(route="x", centers=base, per_mode=8, seed=1)
    p1 = cluster_by_route(p1_traces, n_clusters_per_route=3, min_cluster_size=2)
    snap1 = snapshot_from_clusters(p1)
    assert set(c.cluster_id for c in p1) == {0, 1, 2}
    c_cluster_p1 = next(c for c in p1 if c.centroid is not None and int(np.argmax(c.centroid)) == 2)
    retired_id = c_cluster_p1.cluster_id

    # P2: drop mode C.
    p2_traces = _make_modes(route="x", centers=base[:2], per_mode=8, seed=2)
    p2 = cluster_by_route(p2_traces, n_clusters_per_route=2, min_cluster_size=2, prior=snap1)
    snap2 = snapshot_from_clusters(p2, prior=snap1)
    # mode C must be in the retired pool now.
    assert "x" in snap2.retired_centroids
    assert retired_id in {cid for cid, _ in snap2.retired_centroids["x"]}

    # P3: mode C returns.
    p3_traces = _make_modes(route="x", centers=base, per_mode=8, seed=3)
    p3 = cluster_by_route(p3_traces, n_clusters_per_route=3, min_cluster_size=2, prior=snap2)
    p3_ids = {c.cluster_id for c in p3}
    assert retired_id in p3_ids, (
        f"mode C should have revived its retired id {retired_id} from the pool; got ids {p3_ids}"
    )
    # And the new snapshot should NOT carry C as retired any more (it's active).
    snap3 = snapshot_from_clusters(p3, prior=snap2)
    retired_after = {cid for cid, _ in snap3.retired_centroids.get("x", [])}
    assert retired_id not in retired_after


def test_retired_pool_capped_evicts_oldest() -> None:
    """retained_capacity caps the per-route pool; oldest entries evict first.

    With ``retained_capacity=1``, retiring two modes in succession leaves only
    the most recently retired one in the pool. The earlier-retired mode can no
    longer revive (it gets a fresh id), but ``next_id_by_route`` still
    prevents recycling of its dead id.
    """
    dim = 8

    def cs(i: int) -> np.ndarray:
        return np.eye(dim)[i].astype(np.float32) * 5.0

    # Bootstrap a prior with two active modes A(id=0) and B(id=1).
    prior = ClusterSnapshot(
        centroids={"x": {0: cs(0), 1: cs(1)}},
        next_id_by_route={"x": 2},
    )

    # P2: only mode B survives. A gets retired into the pool, cap=1 keeps it.
    p2_traces = _make_modes(route="x", centers=[cs(1)], per_mode=6, seed=10)
    p2 = cluster_by_route(p2_traces, n_clusters_per_route=1, min_cluster_size=2, prior=prior)
    snap2 = snapshot_from_clusters(p2, prior=prior, retained_capacity=1)
    assert [cid for cid, _ in snap2.retired_centroids["x"]] == [0]

    # P3: only a brand-new mode C(id=2 baseline target — but high-water=2 so it's 2). B retires.
    p3_traces = _make_modes(route="x", centers=[cs(2)], per_mode=6, seed=11)
    p3 = cluster_by_route(p3_traces, n_clusters_per_route=1, min_cluster_size=2, prior=snap2)
    # New mode C must NOT match retired A (different direction); fresh id, not 0.
    assert all(c.cluster_id != 0 for c in p3)
    snap3 = snapshot_from_clusters(p3, prior=snap2, retained_capacity=1)
    # Pool cap = 1; newly-retired B (id=1) is more recent than A (id=0), so A evicts.
    retired_ids = [cid for cid, _ in snap3.retired_centroids["x"]]
    assert retired_ids == [1], f"expected only newest-retired id [1] under cap=1, got {retired_ids}"

    # P4: mode A returns. With A evicted from the pool, no revival — fresh id.
    p4_traces = _make_modes(route="x", centers=[cs(0)], per_mode=6, seed=12)
    snap3_high_water_before_p4 = snap3.next_id_by_route["x"]
    p4 = cluster_by_route(p4_traces, n_clusters_per_route=1, min_cluster_size=2, prior=snap3)
    p4_ids = {c.cluster_id for c in p4}
    assert 0 not in p4_ids, "mode A should NOT revive id 0 after being evicted from the retired pool"
    # And the new id must respect the all-time high-water (captured pre-fit so
    # the in-place advancement during cluster_by_route doesn't move the goalposts).
    assert min(p4_ids) >= snap3_high_water_before_p4


def test_retired_capacity_zero_disables_stickiness() -> None:
    """retained_capacity=0 → no retained pool; legacy P1.0 behaviour.

    A mode that returns after being absent does NOT revive; it gets a fresh
    id strictly above the all-time high-water.
    """
    dim = 8

    def cs(i: int) -> np.ndarray:
        return np.eye(dim)[i].astype(np.float32) * 5.0

    prior = ClusterSnapshot(
        centroids={"x": {0: cs(0), 1: cs(1)}},
        next_id_by_route={"x": 2},
    )
    p2_traces = _make_modes(route="x", centers=[cs(1)], per_mode=6, seed=20)
    p2 = cluster_by_route(p2_traces, n_clusters_per_route=1, min_cluster_size=2, prior=prior)
    snap2 = snapshot_from_clusters(p2, prior=prior, retained_capacity=0)
    assert snap2.retired_centroids == {}

    # Mode A returns; no pool → fresh id, not 0.
    p3_traces = _make_modes(route="x", centers=[cs(0)], per_mode=6, seed=21)
    snap2_high_water_before_p3 = snap2.next_id_by_route["x"]
    p3 = cluster_by_route(p3_traces, n_clusters_per_route=1, min_cluster_size=2, prior=snap2)
    p3_ids = {c.cluster_id for c in p3}
    assert 0 not in p3_ids
    assert min(p3_ids) >= snap2_high_water_before_p3


def test_retired_centroid_no_revival_below_threshold() -> None:
    """A new mode that doesn't clear match_threshold against any retired
    centroid does not revive — it gets a fresh id.
    """
    dim = 8

    def cs(i: int) -> np.ndarray:
        return np.eye(dim)[i].astype(np.float32) * 5.0

    # Mode A retires at id 0; cap=4.
    prior = ClusterSnapshot(
        centroids={"x": {0: cs(0), 1: cs(1)}},
        next_id_by_route={"x": 2},
    )
    p2_traces = _make_modes(route="x", centers=[cs(1)], per_mode=6, seed=30)
    p2 = cluster_by_route(p2_traces, n_clusters_per_route=1, min_cluster_size=2, prior=prior)
    snap2 = snapshot_from_clusters(p2, prior=prior)
    assert [cid for cid, _ in snap2.retired_centroids["x"]] == [0]

    # Brand-new orthogonal mode in P3 should NOT match retired A — orthogonal
    # vectors have cosine similarity 0, well below threshold=0.75.
    p3_traces = _make_modes(route="x", centers=[cs(4)], per_mode=6, seed=31)
    snap2_high_water_before_p3 = snap2.next_id_by_route["x"]
    p3 = cluster_by_route(
        p3_traces,
        n_clusters_per_route=1,
        min_cluster_size=2,
        prior=snap2,
        match_threshold=0.75,
    )
    p3_ids = {c.cluster_id for c in p3}
    assert 0 not in p3_ids
    assert min(p3_ids) >= snap2_high_water_before_p3


def test_revived_id_pruned_from_retired_pool_in_place() -> None:
    """When _assign_stable_ids revives a retired id, the prior snapshot's
    retired_centroids[route] entry is pruned so the next snapshot doesn't
    re-list it. This is the in-place mutation contract that lets
    snapshot_from_clusters use the same `prior` it was given.
    """
    dim = 8

    def cs(i: int) -> np.ndarray:
        return np.eye(dim)[i].astype(np.float32) * 5.0

    prior = ClusterSnapshot(
        centroids={"x": {1: cs(1)}},
        next_id_by_route={"x": 5},  # id 0 was previously issued and then retired
        retired_centroids={"x": [(0, cs(0))]},
    )

    p_traces = _make_modes(route="x", centers=[cs(0), cs(1)], per_mode=6, seed=40)
    p = cluster_by_route(p_traces, n_clusters_per_route=2, min_cluster_size=2, prior=prior)
    p_ids = {c.cluster_id for c in p}
    assert 0 in p_ids and 1 in p_ids, f"mode A should revive id 0; mode B should keep id 1. Got {p_ids}"
    # Pool was mutated in place — revived id no longer present.
    assert prior.retired_centroids.get("x", []) == []


def test_retained_capacity_negative_rejected() -> None:
    with pytest.raises(ValueError, match="retained_capacity must be >= 0"):
        snapshot_from_clusters([], retained_capacity=-1)
