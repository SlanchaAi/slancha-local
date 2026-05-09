"""Storage abstraction: JSONL (default) + ClickHouse (graceful fallback) + receiver wiring."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from slancha_local.train.receiver import build_receiver_app
from slancha_local.train.storage import (
    ClickHouseStorage,
    JSONLStorage,
    resolve_storage,
)


def _make_bundle_tar(traces: list[dict]) -> bytes:
    """Build an in-memory bundle tar.gz with manifest.json + traces.jsonl."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        manifest = {
            "bundle_id": "test-bundle-123",
            "n_traces": len(traces),
            "exported_at": "2026-05-09T00:00:00Z",
        }
        manifest_bytes = json.dumps(manifest).encode()
        info = tarfile.TarInfo(name="bundle/manifest.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))

        traces_bytes = ("\n".join(json.dumps(t) for t in traces) + "\n").encode()
        info2 = tarfile.TarInfo(name="bundle/traces.jsonl")
        info2.size = len(traces_bytes)
        tar.addfile(info2, io.BytesIO(traces_bytes))
    return buf.getvalue()


def _trace(target: str = "local:ollama:qwen3:8b") -> dict:
    return {
        "request_id": "r1",
        "ts": "2026-05-09T00:00:00Z",
        "mode": "local",
        "classifier": {
            "domain": "general",
            "difficulty": "easy",
            "language": "en",
            "jailbreak": False,
            "pii": False,
            "tool_calling": False,
            "route": "general_easy",
            "confidence": 0.7,
        },
        "decision": {"target": target, "fallbacks": [], "reason": "rule"},
        "execution": {
            "executed_target": target,
            "tokens_in": 1,
            "tokens_out": 1,
            "latency_ms": 100,
            "status": "ok",
        },
        "consent_at_capture": False,
        "schema_version": 1,
    }


# ---------- JSONL storage ----------


def test_jsonl_storage_write_bundle(tmp_path: Path):
    s = JSONLStorage()
    bundle_dir = tmp_path / "test-bundle-123"
    traces = [_trace(), _trace()]
    traces_bytes = ("\n".join(json.dumps(t) for t in traces) + "\n").encode()
    meta = s.write_bundle(
        bundle_id="test-bundle-123",
        manifest={"bundle_id": "test-bundle-123", "n_traces": 2},
        traces_bytes=traces_bytes,
        bundle_dir=bundle_dir,
    )
    assert meta["bundle_id"] == "test-bundle-123"
    assert meta["n_traces"] == 2
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "traces.jsonl").exists()
    assert (bundle_dir / "received_at.txt").exists()


# ---------- resolve_storage flag ----------


def test_resolve_storage_default_jsonl(monkeypatch):
    monkeypatch.delenv("SLANCHA_TRAIN_STORAGE", raising=False)
    s = resolve_storage()
    assert isinstance(s, JSONLStorage)


def test_resolve_storage_clickhouse_flag(monkeypatch):
    monkeypatch.setenv("SLANCHA_TRAIN_STORAGE", "clickhouse")
    s = resolve_storage()
    assert isinstance(s, ClickHouseStorage)


def test_resolve_storage_unknown_flag_falls_back(monkeypatch):
    monkeypatch.setenv("SLANCHA_TRAIN_STORAGE", "bogus")
    s = resolve_storage()
    assert isinstance(s, JSONLStorage)


# ---------- ClickHouse graceful fallback ----------


def test_clickhouse_storage_no_lib_falls_back_to_jsonl(tmp_path: Path):
    """Without clickhouse-connect installed, write_bundle still succeeds via JSONL."""
    s = ClickHouseStorage()
    bundle_dir = tmp_path / "ch-test"
    traces_bytes = (json.dumps(_trace()) + "\n").encode()

    # Force ImportError path
    with patch.dict("sys.modules", {"clickhouse_connect": None}):
        meta = s.write_bundle(
            bundle_id="ch-test",
            manifest={"bundle_id": "ch-test"},
            traces_bytes=traces_bytes,
            bundle_dir=bundle_dir,
        )
    assert meta["bundle_id"] == "ch-test"
    assert (bundle_dir / "traces.jsonl").exists()
    assert "clickhouse" in meta
    assert "skipped" in meta["clickhouse"] or "failed" in meta["clickhouse"]


def test_clickhouse_storage_with_mock_client_inserts(tmp_path: Path):
    """Verify the fan-out path inserts rows into the mocked CH client."""
    s = ClickHouseStorage()
    bundle_dir = tmp_path / "ch-mock"
    traces = [_trace(), _trace(target="local:ollama:codestral:22b")]
    traces_bytes = ("\n".join(json.dumps(t) for t in traces) + "\n").encode()

    fake_client = MagicMock()
    s._client = fake_client
    s._connect_attempted = True

    meta = s.write_bundle(
        bundle_id="ch-mock",
        manifest={"bundle_id": "ch-mock"},
        traces_bytes=traces_bytes,
        bundle_dir=bundle_dir,
    )
    assert meta["clickhouse"] == "inserted 2"
    fake_client.insert.assert_called_once()
    call_args = fake_client.insert.call_args
    assert call_args[0][0] == "slancha_traces"
    assert len(call_args[0][1]) == 2  # 2 rows


def test_clickhouse_storage_insert_failure_keeps_jsonl(tmp_path: Path):
    """If CH insert raises, the bundle still lands in JSONL and we record the failure."""
    s = ClickHouseStorage()
    bundle_dir = tmp_path / "ch-fail"
    traces_bytes = (json.dumps(_trace()) + "\n").encode()

    fake_client = MagicMock()
    fake_client.insert.side_effect = RuntimeError("boom")
    s._client = fake_client
    s._connect_attempted = True

    meta = s.write_bundle(
        bundle_id="ch-fail",
        manifest={"bundle_id": "ch-fail"},
        traces_bytes=traces_bytes,
        bundle_dir=bundle_dir,
    )
    assert (bundle_dir / "traces.jsonl").exists()  # JSONL durable
    assert "failed" in meta["clickhouse"]


# ---------- Receiver wiring ----------


def test_receiver_healthz_reports_storage_backend(tmp_path: Path):
    app = build_receiver_app(storage_root=tmp_path, storage=JSONLStorage())
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["storage_backend"] == "JSONLStorage"


def test_receiver_bulk_round_trip(tmp_path: Path):
    app = build_receiver_app(storage_root=tmp_path, storage=JSONLStorage())
    client = TestClient(app)
    bundle = _make_bundle_tar([_trace(), _trace()])
    r = client.post("/v1/traces/bulk", files={"file": ("bundle.tar.gz", bundle, "application/gzip")})
    assert r.status_code == 200
    body = r.json()
    assert body["bundle_id"] == "test-bundle-123"
    assert body["n_traces"] == 2
    assert (tmp_path / "test-bundle-123" / "traces.jsonl").exists()


def test_receiver_bulk_rejects_empty(tmp_path: Path):
    app = build_receiver_app(storage_root=tmp_path, storage=JSONLStorage())
    client = TestClient(app)
    r = client.post("/v1/traces/bulk", files={"file": ("empty.tar.gz", b"", "application/gzip")})
    assert r.status_code == 400


def test_receiver_bulk_rejects_bad_tar(tmp_path: Path):
    app = build_receiver_app(storage_root=tmp_path, storage=JSONLStorage())
    client = TestClient(app)
    r = client.post(
        "/v1/traces/bulk",
        files={"file": ("bad.tar.gz", b"not a tar file", "application/gzip")},
    )
    assert r.status_code == 400


def test_receiver_uses_resolve_storage_when_unset(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SLANCHA_TRAIN_STORAGE", raising=False)
    app = build_receiver_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.json()["storage_backend"] == "JSONLStorage"


def test_receiver_resolves_clickhouse_when_flagged(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SLANCHA_TRAIN_STORAGE", "clickhouse")
    app = build_receiver_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.json()["storage_backend"] == "ClickHouseStorage"
