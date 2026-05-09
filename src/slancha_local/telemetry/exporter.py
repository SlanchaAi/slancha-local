"""Trace bundle exporter — opt-in workflow for FT-credit upload.

Reads ~/.slancha/traces/*.jsonl, applies PII redaction, writes a
date-stamped tarball. The tarball is auditable (`tar tvf bundle.tar.gz`)
before any upload — no surprises.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

# Conservative PII redaction patterns. Mirrors the patterns in the
# adversarial set; documented in ADR-002 §3.
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"), "<EMAIL>"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<CARD>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "<API_KEY>"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "<AWS_KEY>"),
    (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), "<GH_TOKEN>"),
    (re.compile(r"\b\+?\d{1,3}[- ]?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"), "<PHONE>"),
]


def _redact(text: str | None) -> str | None:
    if text is None:
        return None
    out = text
    for pat, repl in _PII_PATTERNS:
        out = pat.sub(repl, out)
    return out


def _redact_trace(trace: dict) -> dict:
    """Apply redaction to a trace dict in-place. Returns the same dict."""
    if "prompt" in trace:
        trace["prompt"] = _redact(trace.get("prompt"))
    if "response" in trace:
        trace["response"] = _redact(trace.get("response"))
    return trace


def _iter_traces(root: Path, since: datetime | None) -> list[dict]:
    out: list[dict] = []
    if not root.exists():
        return out
    for f in sorted(root.glob("*.jsonl")):
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
            if since is not None:
                try:
                    ts = datetime.fromisoformat(t["ts"].replace("Z", "+00:00"))
                except (ValueError, KeyError):
                    continue
                if ts < since:
                    continue
            out.append(t)
    return out


def export_bundle(
    *,
    traces_root: Path,
    out_path: Path,
    since: datetime | None = None,
) -> tuple[int, int]:
    """Bundle redacted traces into a .tar.gz at out_path. Returns (n_traces, n_bytes).

    The bundle layout:
      slancha-traces-<UTC-ts>/
        manifest.json         { "schema_version": 1, "exported_at": "...", "n_traces": N }
        traces.jsonl          (one redacted JSON object per line)

    Pre-upload audit: `tar tvf <bundle>` lists what's in it. `tar -xOf <bundle> traces.jsonl`
    streams the redacted JSONL to stdout for inspection.
    """
    traces = _iter_traces(traces_root, since=since)
    redacted = [_redact_trace(t) for t in traces]

    now = datetime.now(UTC)
    bundle_id = f"slancha-traces-{now.strftime('%Y%m%dT%H%M%SZ')}"
    manifest = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "exported_at": now.isoformat(),
        "n_traces": len(redacted),
        "redaction": {
            "version": 1,
            "patterns_applied": ["email", "ssn", "card", "openai-key", "aws-key", "gh-token", "phone"],
        },
    }
    traces_bytes = b"\n".join(json.dumps(t).encode() for t in redacted) + b"\n"
    manifest_bytes = json.dumps(manifest, indent=2).encode() + b"\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        ti_m = tarfile.TarInfo(name=f"{bundle_id}/manifest.json")
        ti_m.size = len(manifest_bytes)
        ti_m.mtime = int(time.time())
        tar.addfile(ti_m, io.BytesIO(manifest_bytes))

        ti_t = tarfile.TarInfo(name=f"{bundle_id}/traces.jsonl")
        ti_t.size = len(traces_bytes)
        ti_t.mtime = int(time.time())
        tar.addfile(ti_t, io.BytesIO(traces_bytes))

    return len(redacted), out_path.stat().st_size
