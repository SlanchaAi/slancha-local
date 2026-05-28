"""Build train+val JSONL bundles from clustered traces.

Output format = axolotl-compatible chat JSONL:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Train/val split is round-robin per cluster: each cluster contributes a
proportional number of val samples, so val is a representative cross-section.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

from slancha_local.train.cluster import ClusterSnapshot, cluster_by_route, snapshot_from_clusters

logger = logging.getLogger(__name__)


SNAPSHOT_FILENAME = "cluster_snapshot.npz"
"""Default name for the bundle-local cluster snapshot file.

Placed inside the bundle output directory so a downstream consumer that
ships the bundle ships the snapshot with it. Stable across runs so each
``slancha train-bundle`` invocation rolls forward the same file."""


@dataclass
class BundleStats:
    train_count: int
    val_count: int
    skipped_no_response: int
    skipped_no_consent: int
    routes: list[str]
    clusters: int
    snapshot_path: Path | None = None
    snapshot_revived_ids: int = 0
    snapshot_retired_routes: int = 0


def _trace_to_chat(trace: dict) -> dict | None:
    """Convert a trace into axolotl chat-format. Returns None if untrainable."""
    prompt = trace.get("prompt")
    response = trace.get("response")
    if not prompt or not response:
        return None
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "request_id": trace.get("request_id"),
            "route": (trace.get("classifier") or {}).get("route"),
            "domain": (trace.get("classifier") or {}).get("domain"),
            "difficulty": (trace.get("classifier") or {}).get("difficulty"),
            "language": (trace.get("classifier") or {}).get("language"),
            "executed_target": (trace.get("execution") or {}).get("executed_target"),
        },
    }


def build_train_bundle(
    traces: list[dict],
    *,
    out_dir: Path,
    val_fraction: float = 0.1,
    n_clusters_per_route: int = 4,
    cluster: bool = True,
    random_state: int = 42,
    snapshot_in: Path | None = None,
    snapshot_out: Path | None | bool = True,
) -> BundleStats:
    """Cluster, split, and emit train.jsonl + val.jsonl.

    ``snapshot_in`` — optional path to a prior :class:`ClusterSnapshot`
    saved by :meth:`ClusterSnapshot.save`. When present, cluster ids are
    carried forward (and retired-pool ids may be revived) so a head
    trained against id 7 stays bound to id 7 across runs.

    ``snapshot_out`` — controls where the post-fit snapshot is written:
    ``True`` (default) writes to ``out_dir/cluster_snapshot.npz`` so the
    snapshot ships alongside the bundle. ``False`` / ``None`` skips
    writing. A :class:`~pathlib.Path` writes to that exact location.

    Snapshot writing is skipped automatically when ``cluster=False`` (no
    KMeans → no centroids worth persisting).

    Returns :class:`BundleStats` describing what was emitted.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_snapshot: ClusterSnapshot | None = None
    if snapshot_in is not None:
        prior_snapshot = ClusterSnapshot.load(snapshot_in)

    # Filter: must have prompt + response (consent + capture both)
    eligible: list[dict] = []
    skipped_no_response = 0
    skipped_no_consent = 0
    for t in traces:
        if not t.get("consent_at_capture", False):
            skipped_no_consent += 1
            continue
        if not t.get("prompt") or not t.get("response"):
            skipped_no_response += 1
            continue
        eligible.append(t)

    if not eligible:
        return BundleStats(
            train_count=0,
            val_count=0,
            skipped_no_response=skipped_no_response,
            skipped_no_consent=skipped_no_consent,
            routes=[],
            clusters=0,
        )

    # Cluster
    if cluster:
        clusters = cluster_by_route(
            eligible,
            n_clusters_per_route=n_clusters_per_route,
            random_state=random_state,
            prior=prior_snapshot,
        )
    else:
        clusters = []
        from collections import defaultdict

        by_route: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(eligible):
            r = (t.get("classifier") or {}).get("route") or "unknown"
            by_route[r].append(i)
        from slancha_local.train.cluster import TraceCluster

        for route, idxs in by_route.items():
            clusters.append(TraceCluster(route=route, cluster_id=0, trace_indices=idxs))

    # Round-robin val sampling per cluster
    rng = random.Random(random_state)
    train_records: list[dict] = []
    val_records: list[dict] = []
    for c in clusters:
        members = list(c.trace_indices)
        rng.shuffle(members)
        n_val = max(1, math.floor(len(members) * val_fraction)) if len(members) >= 5 else 0
        val_idx = set(members[:n_val])
        for member_i in members:
            chat = _trace_to_chat(eligible[member_i])
            if not chat:
                continue
            chat["metadata"]["cluster"] = f"{c.route}#{c.cluster_id}"
            (val_records if member_i in val_idx else train_records).append(chat)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    # Persist post-fit snapshot so the next run can revive cluster ids.
    snapshot_path: Path | None = None
    revived_ids = 0
    retired_routes = 0
    if cluster and snapshot_out is not False and snapshot_out is not None:
        target = out_dir / SNAPSHOT_FILENAME if snapshot_out is True else Path(snapshot_out)
        new_snapshot = snapshot_from_clusters(clusters, prior=prior_snapshot)
        if prior_snapshot is not None:
            # Diagnostic counters — how much stickiness paid off this run.
            prior_active_ids = {(r, cid) for r, m in prior_snapshot.centroids.items() for cid in m}
            new_active_ids = {(r, cid) for r, m in new_snapshot.centroids.items() for cid in m}
            revived_ids = len(prior_active_ids & new_active_ids)
        retired_routes = sum(1 for v in new_snapshot.retired_centroids.values() if v)
        snapshot_path = new_snapshot.save(target)
        logger.info(
            "wrote cluster snapshot path=%s active_routes=%d retired_routes=%d revived_ids=%d",
            snapshot_path,
            len(new_snapshot.centroids),
            retired_routes,
            revived_ids,
        )

    return BundleStats(
        train_count=len(train_records),
        val_count=len(val_records),
        skipped_no_response=skipped_no_response,
        skipped_no_consent=skipped_no_consent,
        routes=sorted({c.route for c in clusters}),
        clusters=len(clusters),
        snapshot_path=snapshot_path,
        snapshot_revived_ids=revived_ids,
        snapshot_retired_routes=retired_routes,
    )
