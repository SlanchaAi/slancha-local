"""Trace-bundle receiver — runs on Spark; ingests slancha export bundles.

Endpoint: POST /v1/traces/bulk  (tar.gz upload)
Storage: ~/.slancha-train/storage/<bundle_id>/{manifest.json, traces.jsonl}

Designed to live on Spark as a systemd service. For now, a FastAPI stub
that future iterations will productionize (auth, rate-limit, schema-validate).
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

logger = logging.getLogger(__name__)


def build_receiver_app(*, storage_root: Path) -> FastAPI:
    storage_root = Path(storage_root).expanduser().resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="slancha-train-receiver", version="0.0.1")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "storage_root": str(storage_root)}

    @app.post("/v1/traces/bulk")
    async def bulk(file: UploadFile) -> dict:
        body = await file.read()
        if not body:
            raise HTTPException(status_code=400, detail="empty body")
        try:
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
                names = tar.getnames()
                manifest_member = next(
                    (m for m in tar.getmembers() if m.name.endswith("/manifest.json")),
                    None,
                )
                traces_member = next(
                    (m for m in tar.getmembers() if m.name.endswith("/traces.jsonl")),
                    None,
                )
                if not manifest_member or not traces_member:
                    raise HTTPException(
                        status_code=400, detail=f"missing manifest/traces in tarball: {names}"
                    )
                manifest = json.loads(tar.extractfile(manifest_member).read().decode())
                traces_bytes = tar.extractfile(traces_member).read()
        except (tarfile.TarError, OSError) as e:
            raise HTTPException(status_code=400, detail=f"bad tar.gz: {e}") from e

        bundle_id = manifest.get("bundle_id") or f"unknown-{int(datetime.now(UTC).timestamp())}"
        bundle_dir = storage_root / bundle_id
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

    @app.get("/v1/bundles")
    async def list_bundles() -> dict:
        bundles = []
        for d in sorted(storage_root.iterdir()):
            if not d.is_dir():
                continue
            mp = d / "manifest.json"
            if not mp.exists():
                continue
            try:
                manifest = json.loads(mp.read_text())
                bundles.append(
                    {
                        "bundle_id": d.name,
                        "n_traces": manifest.get("n_traces", 0),
                        "exported_at": manifest.get("exported_at"),
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue
        return {"bundles": bundles}

    return app
