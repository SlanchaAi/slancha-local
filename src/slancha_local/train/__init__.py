"""Training-bundle pipeline: cluster traces → split train/val → emit JSONL.

The classifier+model FT pipeline itself runs server-side on Spark (CUDA).
This module is the *client-side* prep: takes ~/.slancha/traces, clusters
embeddings, splits per-cluster into train/val, emits axolotl-compatible
JSONL. Used by `slancha train-bundle` and (server-side) re-used by the
receiver before kicking off FT runs.
"""

from slancha_local.train.bundle import build_train_bundle
from slancha_local.train.cluster import (
    ClusterSnapshot,
    TraceCluster,
    cluster_by_route,
    snapshot_from_clusters,
)

__all__ = [
    "ClusterSnapshot",
    "TraceCluster",
    "build_train_bundle",
    "cluster_by_route",
    "snapshot_from_clusters",
]
