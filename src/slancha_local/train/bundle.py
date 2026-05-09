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

from slancha_local.train.cluster import cluster_by_route

logger = logging.getLogger(__name__)


@dataclass
class BundleStats:
    train_count: int
    val_count: int
    skipped_no_response: int
    skipped_no_consent: int
    routes: list[str]
    clusters: int


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
) -> BundleStats:
    """Cluster, split, and emit train.jsonl + val.jsonl.

    Returns BundleStats describing what was emitted.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    return BundleStats(
        train_count=len(train_records),
        val_count=len(val_records),
        skipped_no_response=skipped_no_response,
        skipped_no_consent=skipped_no_consent,
        routes=sorted({c.route for c in clusters}),
        clusters=len(clusters),
    )
