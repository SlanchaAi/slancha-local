"""GET /v1/decisions/last + /v1/decisions/{id} — read-only decision history."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


def _read_recent(root: Path, n: int) -> list[dict]:
    out: list[dict] = []
    if not root.exists():
        return out
    for f in sorted(root.glob("*.jsonl"), reverse=True):
        try:
            text = f.read_text()
        except OSError:
            continue
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= n:
                return out
    return out


def _find_by_id(root: Path, request_id: str) -> dict | None:
    if not root.exists():
        return None
    for f in root.glob("*.jsonl"):
        try:
            text = f.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t.get("request_id") == request_id:
                return t
    return None


@router.get("/v1/decisions/last")
async def last(request: Request, n: int = 10) -> dict:
    n = max(1, min(n, 100))
    root = request.app.state.settings.traces_root
    return {"decisions": _read_recent(root, n)}


@router.get("/v1/decisions/{request_id}")
async def by_id(request_id: str, request: Request) -> dict:
    root = request.app.state.settings.traces_root
    found = _find_by_id(root, request_id)
    if not found:
        raise HTTPException(status_code=404, detail="decision not found")
    return found
