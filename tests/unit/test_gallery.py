"""Gallery: stats computation + HTML render smoke."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from slancha_local.gallery.stats import compute_view
from slancha_local.gallery.web import build_gallery_app


def _trace(target: str, domain: str, difficulty: str, latency_ms: int = 100, ts: str | None = None) -> dict:
    return {
        "request_id": f"r-{target}-{domain}-{difficulty}",
        "ts": ts or datetime.now(UTC).isoformat(),
        "mode": "local",
        "embedding_b64": "AAAA",
        "classifier": {
            "domain": domain,
            "difficulty": difficulty,
            "language": "en",
            "jailbreak": False,
            "pii": False,
            "tool_calling": False,
            "route": f"{domain}_{difficulty}",
            "confidence": 0.7,
        },
        "decision": {"target": target, "fallbacks": [], "reason": "rule"},
        "execution": {
            "executed_target": target,
            "tokens_in": 1,
            "tokens_out": 1,
            "latency_ms": latency_ms,
            "status": "ok",
        },
        "prompt": None,
        "response": None,
        "feedback": None,
        "consent_at_capture": False,
        "schema_version": 1,
    }


def _seed(root: Path, traces: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    by_date: dict[str, list[dict]] = {}
    for t in traces:
        by_date.setdefault(t["ts"][:10], []).append(t)
    for date, ts in by_date.items():
        path = root / f"{date}.jsonl"
        with open(path, "a") as f:
            for t in ts:
                f.write(json.dumps(t) + "\n")


def test_compute_view_aggregates_per_model(tmp_path: Path):
    root = tmp_path / "traces"
    _seed(
        root,
        [
            _trace("local:ollama:qwen3:8b", "general", "easy", latency_ms=150),
            _trace("local:ollama:qwen3:8b", "general", "easy", latency_ms=250),
            _trace("local:ollama:codestral:22b", "computer science", "medium", latency_ms=400),
        ],
    )
    view = compute_view(root, days=30)
    assert view.total_routed == 3
    assert view.total_local == 3
    assert view.distinct_models == 2
    qwen = next(m for m in view.models if "qwen3" in m.target)
    assert qwen.use_count == 2
    assert qwen.avg_latency_ms == 200


def test_compute_view_window_filter(tmp_path: Path):
    root = tmp_path / "traces"
    old_ts = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    new_ts = datetime.now(UTC).isoformat()
    _seed(
        root,
        [
            _trace("local:ollama:qwen3:8b", "general", "easy", ts=old_ts),
            _trace("local:ollama:qwen3:8b", "general", "easy", ts=new_ts),
        ],
    )
    view = compute_view(root, days=7)
    assert view.total_routed == 1


def test_gallery_html_renders(tmp_path: Path):
    root = tmp_path / "traces"
    _seed(root, [_trace("local:ollama:qwen3:8b", "general", "easy")])
    app = build_gallery_app(traces_root=root)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "slancha gallery" in body
    assert "qwen3:8b" in body
    assert "<style>" in body  # inline CSS
    assert "models" in body  # the pill


def test_gallery_html_empty_state(tmp_path: Path):
    root = tmp_path / "traces"
    root.mkdir()
    app = build_gallery_app(traces_root=root)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "No routed traces" in r.text


def test_gallery_healthz(tmp_path: Path):
    app = build_gallery_app(traces_root=tmp_path)
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.text == "ok"


def test_compute_view_top_combos_sorted(tmp_path: Path):
    root = tmp_path / "traces"
    _seed(
        root,
        [
            _trace("local:ollama:qwen3:8b", "creative", "easy"),
            _trace("local:ollama:qwen3:8b", "general", "medium"),
            _trace("local:ollama:qwen3:8b", "general", "medium"),
            _trace("local:ollama:qwen3:8b", "general", "medium"),
        ],
    )
    view = compute_view(root, days=30)
    qwen = view.models[0]
    # general+medium count=3 should be first
    assert qwen.top_combos[0] == ("general", "medium", 3)
