"""Train-trace storage backends — JSONL (default) and ClickHouse (flag).

Receiver writes incoming bundles via the abstract `Storage` interface.
Default: write to JSONL files under `~/.slancha-train/storage/<bundle_id>/`.
With SLANCHA_TRAIN_STORAGE=clickhouse, also fan-out to ClickHouse for
queryable analytics at scale (JSONL stays as durable backing store).

ClickHouse is optional. If `clickhouse-connect` isn't installed, the
backend logs a warning and falls back to JSONL-only mode — no install
required for default users.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Storage(ABC):
    @abstractmethod
    def write_bundle(
        self,
        *,
        bundle_id: str,
        manifest: dict,
        traces_bytes: bytes,
        bundle_dir: Path,
    ) -> dict[str, Any]:
        """Persist a bundle. Returns metadata dict with at least `n_traces`."""


class JSONLStorage(Storage):
    """Default — write manifest.json + traces.jsonl to bundle_dir."""

    def write_bundle(
        self,
        *,
        bundle_id: str,
        manifest: dict,
        traces_bytes: bytes,
        bundle_dir: Path,
    ) -> dict[str, Any]:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (bundle_dir / "traces.jsonl").write_bytes(traces_bytes)
        (bundle_dir / "received_at.txt").write_text(datetime.now(UTC).isoformat() + "\n")
        n_lines = traces_bytes.count(b"\n")
        return {
            "bundle_id": bundle_id,
            "stored_at": str(bundle_dir),
            "n_traces": n_lines,
            "received_at": datetime.now(UTC).isoformat(),
        }


class ClickHouseStorage(Storage):
    """JSONL on disk + fan-out to ClickHouse for queryable analytics.

    Requires `clickhouse-connect` (optional dep). On import failure or
    connection error, degrades gracefully to JSONL-only.

    Schema (created on first write):
        CREATE TABLE IF NOT EXISTS slancha_traces (
            bundle_id String,
            received_at DateTime DEFAULT now(),
            request_id String,
            ts DateTime64,
            mode LowCardinality(String),
            domain LowCardinality(String),
            difficulty LowCardinality(String),
            language LowCardinality(String),
            jailbreak UInt8,
            pii UInt8,
            tool_calling UInt8,
            route String,
            confidence Float32,
            target String,
            executed_target String,
            tokens_in UInt32,
            tokens_out UInt32,
            latency_ms UInt32,
            status LowCardinality(String),
            consent_at_capture UInt8,
            schema_version UInt16,
            raw String  -- the full JSON row, for backfill
        ) ENGINE = MergeTree
        ORDER BY (route, ts);
    """

    def __init__(self) -> None:
        self._fallback = JSONLStorage()
        self._client = None
        self._connect_attempted = False

    def _connect(self) -> Any:
        if self._connect_attempted:
            return self._client
        self._connect_attempted = True
        try:
            import clickhouse_connect  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "clickhouse-connect not installed; falling back to JSONL-only. "
                "Install with: pip install clickhouse-connect"
            )
            return None
        host = os.environ.get("CLICKHOUSE_HOST", "127.0.0.1")
        port = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
        user = os.environ.get("CLICKHOUSE_USER", "default")
        password = os.environ.get("CLICKHOUSE_PASSWORD", "")
        database = os.environ.get("CLICKHOUSE_DATABASE", "default")
        try:
            self._client = clickhouse_connect.get_client(
                host=host, port=port, username=user, password=password, database=database
            )
            self._ensure_schema()
        except Exception as e:
            logger.warning("ClickHouse connect failed (%s); falling back to JSONL-only", e)
            self._client = None
        return self._client

    def _ensure_schema(self) -> None:
        if self._client is None:
            return
        ddl = """
        CREATE TABLE IF NOT EXISTS slancha_traces (
            bundle_id String,
            received_at DateTime DEFAULT now(),
            request_id String,
            ts DateTime64,
            mode LowCardinality(String),
            domain LowCardinality(String),
            difficulty LowCardinality(String),
            language LowCardinality(String),
            jailbreak UInt8,
            pii UInt8,
            tool_calling UInt8,
            route String,
            confidence Float32,
            target String,
            executed_target String,
            tokens_in UInt32,
            tokens_out UInt32,
            latency_ms UInt32,
            status LowCardinality(String),
            consent_at_capture UInt8,
            schema_version UInt16,
            raw String
        ) ENGINE = MergeTree ORDER BY (route, ts)
        """
        self._client.command(ddl)

    def write_bundle(
        self,
        *,
        bundle_id: str,
        manifest: dict,
        traces_bytes: bytes,
        bundle_dir: Path,
    ) -> dict[str, Any]:
        # always JSONL first — durable
        meta = self._fallback.write_bundle(
            bundle_id=bundle_id,
            manifest=manifest,
            traces_bytes=traces_bytes,
            bundle_dir=bundle_dir,
        )
        client = self._connect()
        if client is None:
            meta["clickhouse"] = "skipped (no client)"
            return meta
        try:
            rows = []
            for line in traces_bytes.splitlines():
                if not line.strip():
                    continue
                t = json.loads(line)
                cls = t.get("classifier", {})
                dec = t.get("decision", {})
                ex = t.get("execution", {})
                rows.append(
                    [
                        bundle_id,
                        t.get("request_id", ""),
                        t.get("ts", ""),
                        t.get("mode", ""),
                        cls.get("domain", ""),
                        cls.get("difficulty", ""),
                        cls.get("language", ""),
                        int(bool(cls.get("jailbreak"))),
                        int(bool(cls.get("pii"))),
                        int(bool(cls.get("tool_calling"))),
                        cls.get("route", ""),
                        float(cls.get("confidence", 0.0)),
                        dec.get("target", ""),
                        ex.get("executed_target", ""),
                        int(ex.get("tokens_in", 0) or 0),
                        int(ex.get("tokens_out", 0) or 0),
                        int(ex.get("latency_ms", 0) or 0),
                        ex.get("status", ""),
                        int(bool(t.get("consent_at_capture"))),
                        int(t.get("schema_version", 1)),
                        line.decode("utf-8", errors="replace"),
                    ]
                )
            if rows:
                client.insert(
                    "slancha_traces",
                    rows,
                    column_names=[
                        "bundle_id",
                        "request_id",
                        "ts",
                        "mode",
                        "domain",
                        "difficulty",
                        "language",
                        "jailbreak",
                        "pii",
                        "tool_calling",
                        "route",
                        "confidence",
                        "target",
                        "executed_target",
                        "tokens_in",
                        "tokens_out",
                        "latency_ms",
                        "status",
                        "consent_at_capture",
                        "schema_version",
                        "raw",
                    ],
                )
            meta["clickhouse"] = f"inserted {len(rows)}"
        except Exception as e:
            logger.warning("ClickHouse insert failed (%s); JSONL still durable", e)
            meta["clickhouse"] = f"failed: {type(e).__name__}"
        return meta


def resolve_storage() -> Storage:
    """Pick storage backend per SLANCHA_TRAIN_STORAGE env. Default: JSONL."""
    flag = os.environ.get("SLANCHA_TRAIN_STORAGE", "").lower()
    if flag == "clickhouse":
        return ClickHouseStorage()
    return JSONLStorage()
