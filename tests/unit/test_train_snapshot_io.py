"""Round-trip tests for ClusterSnapshot.save / load and train-bundle integration.

The snapshot file pair (``<stem>.npz`` + ``<stem>.json``) is the contract by
which a training bundle ships stable cluster identity to a downstream consumer.
A read-after-write mismatch silently breaks the stickiness / id-stability
guarantees of P1 + P1.5, so these tests assert bitwise equality of every
centroid plus deep structural equality of the metadata.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from slancha_local.train.bundle import SNAPSHOT_FILENAME, build_train_bundle
from slancha_local.train.cluster import (
    SNAPSHOT_FORMAT_VERSION,
    ClusterSnapshot,
)


def _rand_centroid(seed: int, dim: int = 512) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _assert_snapshot_eq(a: ClusterSnapshot, b: ClusterSnapshot) -> None:
    assert a.next_id_by_route == b.next_id_by_route
    assert set(a.centroids.keys()) == set(b.centroids.keys())
    for route, m in a.centroids.items():
        assert set(m.keys()) == set(b.centroids[route].keys()), route
        for cid, vec in m.items():
            np.testing.assert_array_equal(vec, b.centroids[route][cid])
    assert set(a.retired_centroids.keys()) == set(b.retired_centroids.keys())
    for route, pairs in a.retired_centroids.items():
        b_pairs = b.retired_centroids[route]
        assert len(pairs) == len(b_pairs), route
        for (a_cid, a_vec), (b_cid, b_vec) in zip(pairs, b_pairs, strict=True):
            assert a_cid == b_cid
            np.testing.assert_array_equal(a_vec, b_vec)


def test_snapshot_roundtrip_preserves_all_fields(tmp_path: Path):
    snap = ClusterSnapshot(
        centroids={
            "general_qa": {0: _rand_centroid(1), 3: _rand_centroid(2)},
            "code": {7: _rand_centroid(3)},
        },
        next_id_by_route={"general_qa": 5, "code": 8, "extinct_route": 12},
        retired_centroids={
            "general_qa": [(2, _rand_centroid(10)), (1, _rand_centroid(11))],
            "extinct_route": [(11, _rand_centroid(20))],
        },
    )
    out = snap.save(tmp_path / "snap.npz")
    assert out.suffix == ".npz"
    assert out.exists()
    assert out.with_suffix(".json").exists()

    loaded = ClusterSnapshot.load(out)
    _assert_snapshot_eq(snap, loaded)


def test_snapshot_roundtrip_via_stem_path(tmp_path: Path):
    snap = ClusterSnapshot(
        centroids={"r": {0: _rand_centroid(42)}},
        next_id_by_route={"r": 1},
    )
    # Pass the bare stem; save resolves to .npz / .json pair.
    out = snap.save(tmp_path / "snap")
    assert (tmp_path / "snap.npz").exists()
    assert (tmp_path / "snap.json").exists()

    loaded = ClusterSnapshot.load(tmp_path / "snap.json")
    _assert_snapshot_eq(snap, loaded)
    loaded2 = ClusterSnapshot.load(tmp_path / "snap")
    _assert_snapshot_eq(snap, loaded2)
    assert out == (tmp_path / "snap.npz").resolve()


def test_snapshot_empty_roundtrips(tmp_path: Path):
    snap = ClusterSnapshot()
    assert snap.is_empty()
    snap.save(tmp_path / "empty.npz")
    loaded = ClusterSnapshot.load(tmp_path / "empty.npz")
    assert loaded.is_empty()


def test_snapshot_sidecar_carries_schema_version(tmp_path: Path):
    """Per onyx-ridge: a versioned sidecar lets future loop phases migrate
    old snapshots instead of silently misreading them."""
    snap = ClusterSnapshot(
        centroids={"r": {0: _rand_centroid(1)}},
        next_id_by_route={"r": 1},
    )
    snap.save(tmp_path / "snap.npz")
    sidecar = json.loads((tmp_path / "snap.json").read_text())
    assert sidecar["schema_version"] == SNAPSHOT_FORMAT_VERSION
    assert "routes" in sidecar
    assert sidecar["routes"]["r"]["next_id"] == 1


def test_snapshot_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ClusterSnapshot.load(tmp_path / "does_not_exist.npz")


def test_snapshot_load_missing_npz(tmp_path: Path):
    # Sidecar exists but the npz half is gone — corruption surface.
    (tmp_path / "snap.json").write_text(json.dumps({"schema_version": 1, "routes": {}}))
    with pytest.raises(FileNotFoundError, match="npz"):
        ClusterSnapshot.load(tmp_path / "snap.json")


def test_snapshot_rejects_future_schema_version(tmp_path: Path):
    (tmp_path / "snap.npz")  # placeholder; we write it via numpy below.
    np.savez(tmp_path / "snap.npz", _empty=np.zeros(0, dtype=np.float32))
    (tmp_path / "snap.json").write_text(
        json.dumps({"schema_version": SNAPSHOT_FORMAT_VERSION + 99, "routes": {}})
    )
    with pytest.raises(ValueError, match="newer"):
        ClusterSnapshot.load(tmp_path / "snap.npz")


def test_snapshot_loader_tolerant_of_missing_retired(tmp_path: Path):
    """Per onyx-ridge: snapshots written by P1.0-era code (pre-retired pool)
    must still load — treat a missing ``retired`` key as an empty list."""
    centroid = _rand_centroid(7)
    np.savez(tmp_path / "snap.npz", arr_0=centroid)
    (tmp_path / "snap.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "routes": {
                    "general_qa": {
                        "active": {"0": "arr_0"},
                        # Note: NO "retired" key — simulating an older writer.
                        "next_id": 1,
                    }
                },
            }
        )
    )
    loaded = ClusterSnapshot.load(tmp_path / "snap.npz")
    assert loaded.next_id_by_route == {"general_qa": 1}
    np.testing.assert_array_equal(loaded.centroids["general_qa"][0], centroid)
    assert loaded.retired_centroids == {}


def test_snapshot_writer_is_atomic_no_tmp_leak(tmp_path: Path):
    snap = ClusterSnapshot(
        centroids={"r": {0: _rand_centroid(1)}},
        next_id_by_route={"r": 1},
    )
    snap.save(tmp_path / "snap.npz")
    # tmpfile-then-rename: no leftover .tmp files in the directory.
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], leftovers


# ---------------------------------------------------------------------------
# train-bundle integration: snapshot rolls forward across two passes
# ---------------------------------------------------------------------------


def _trace_with_embedding(rid: str, route: str, embedding: np.ndarray, *, consent: bool = True) -> dict:
    return {
        "request_id": rid,
        "ts": "2026-05-09T10:00:00.000Z",
        "mode": "local",
        "embedding_b64": base64.b64encode(embedding.tobytes()).decode(),
        "classifier": {
            "domain": "general",
            "difficulty": "easy",
            "language": "en",
            "jailbreak": False,
            "pii": False,
            "tool_calling": False,
            "route": route,
            "confidence": 0.7,
        },
        "decision": {"target": "local:ollama:qwen3:8b", "fallbacks": [], "reason": "r"},
        "execution": {
            "executed_target": "local:ollama:qwen3:8b",
            "tokens_in": 5,
            "tokens_out": 5,
            "latency_ms": 100,
            "status": "ok",
        },
        "prompt": f"prompt {rid}",
        "response": f"reply {rid}",
        "feedback": None,
        "consent_at_capture": consent,
        "schema_version": 1,
    }


def _two_mode_traces(seed_a: int, seed_b: int, n: int = 20) -> list[dict]:
    """Half traces near mode A, half near mode B. Forces ≥2 stable clusters."""
    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)
    mode_a = rng_a.standard_normal(512).astype(np.float32) * 10.0
    mode_b = rng_b.standard_normal(512).astype(np.float32) * 10.0
    traces = []
    for i in range(n // 2):
        emb = mode_a + rng_a.standard_normal(512).astype(np.float32) * 0.01
        traces.append(_trace_with_embedding(f"a{i}", "general_qa", emb))
    for i in range(n // 2):
        emb = mode_b + rng_b.standard_normal(512).astype(np.float32) * 0.01
        traces.append(_trace_with_embedding(f"b{i}", "general_qa", emb))
    return traces


def test_bundle_writes_snapshot_alongside_jsonl(tmp_path: Path):
    traces = _two_mode_traces(1, 2, n=20)
    out = tmp_path / "bundle"
    stats = build_train_bundle(traces, out_dir=out, n_clusters_per_route=2)
    assert stats.snapshot_path is not None
    assert stats.snapshot_path == (out / SNAPSHOT_FILENAME).resolve()
    assert stats.snapshot_path.exists()
    assert stats.snapshot_path.with_suffix(".json").exists()

    snap = ClusterSnapshot.load(stats.snapshot_path)
    assert "general_qa" in snap.centroids
    assert len(snap.centroids["general_qa"]) >= 1


def test_bundle_no_snapshot_when_no_cluster(tmp_path: Path):
    traces = _two_mode_traces(3, 4, n=10)
    stats = build_train_bundle(traces, out_dir=tmp_path / "bundle", cluster=False)
    assert stats.snapshot_path is None
    assert not (tmp_path / "bundle" / SNAPSHOT_FILENAME).exists()


def test_bundle_no_snapshot_when_snapshot_out_false(tmp_path: Path):
    traces = _two_mode_traces(5, 6, n=20)
    stats = build_train_bundle(
        traces, out_dir=tmp_path / "bundle", n_clusters_per_route=2, snapshot_out=False
    )
    assert stats.snapshot_path is None
    assert not (tmp_path / "bundle" / SNAPSHOT_FILENAME).exists()


def test_bundle_snapshot_round_trip_preserves_ids_across_passes(tmp_path: Path):
    """Second pass with the prior snapshot reuses cluster ids for the same
    underlying modes — this is the stable-id guarantee in action across the
    save/load boundary."""
    traces = _two_mode_traces(11, 22, n=20)
    out = tmp_path / "bundle"

    pass1 = build_train_bundle(traces, out_dir=out, n_clusters_per_route=2)
    snap1 = ClusterSnapshot.load(pass1.snapshot_path)
    ids1 = set(snap1.centroids.get("general_qa", {}).keys())
    assert ids1, "first pass must produce at least one cluster"

    # Same traces again, with snapshot_in pointing at pass1's output.
    pass2 = build_train_bundle(
        traces,
        out_dir=tmp_path / "bundle2",
        n_clusters_per_route=2,
        snapshot_in=pass1.snapshot_path,
    )
    snap2 = ClusterSnapshot.load(pass2.snapshot_path)
    ids2 = set(snap2.centroids.get("general_qa", {}).keys())
    assert ids2 == ids1, (
        f"stable cluster ids must survive save → load round-trip (pass1={sorted(ids1)} pass2={sorted(ids2)})"
    )
    assert pass2.snapshot_revived_ids == len(ids1)
