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
from slancha_local.train.eval_row import (
    EvalSample,
    aggregate_eval_pass,
    append_eval_row,
    read_eval_row,
)
from slancha_local.train.gate import (
    EVAL_ROW_FIELDS,
    GateThresholds,
    PromotionVerdict,
    decide,
)

__all__ = [
    "EVAL_ROW_FIELDS",
    "ClusterSnapshot",
    "EvalSample",
    "GateThresholds",
    "PromotionVerdict",
    "TraceCluster",
    "aggregate_eval_pass",
    "append_eval_row",
    "build_train_bundle",
    "cluster_by_route",
    "decide",
    "read_eval_row",
    "snapshot_from_clusters",
]
