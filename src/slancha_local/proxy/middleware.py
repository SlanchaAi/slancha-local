"""DecisionTraceHeaderMiddleware — emits slancha-decision-trace on every chat response."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class DecisionTraceHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        trace = getattr(request.state, "decision_trace", None)
        if trace:
            response.headers["slancha-decision-trace"] = trace
        return response


_ASCII_REPLACE = {
    "—": "--",  # em-dash
    "–": "-",  # en-dash
    "‘": "'",
    "’": "'",  # single quotes
    "“": '"',
    "”": '"',  # double quotes
    "…": "...",  # ellipsis
}


def _ascii_safe(s: str) -> str:
    """HTTP headers are latin-1; replace common unicode then drop unencodable chars."""
    for src, dst in _ASCII_REPLACE.items():
        s = s.replace(src, dst)
    return s.encode("ascii", errors="replace").decode("ascii")


def format_trace(
    *,
    picked: str,
    reason: str,
    fallbacks: list[str],
    domain: str | None,
    difficulty: str | None,
    jailbreak: bool,
    pii: bool,
    tool_calling: bool,
    confidence: float | None,
    classifier_ms: float,
    total_overhead_ms: float,
) -> str:
    parts = [
        f"picked={_ascii_safe(picked)}",
        f'reason="{_ascii_safe(reason)}"',
        f"fallbacks=[{','.join(_ascii_safe(f) for f in fallbacks)}]",
        f"domain={_ascii_safe(domain or 'unknown')}",
        f"difficulty={difficulty or 'unknown'}",
        f"jailbreak={'yes' if jailbreak else 'no'}",
        f"pii={'yes' if pii else 'no'}",
        f"tool={'yes' if tool_calling else 'no'}",
        f"confidence={confidence:.2f}" if confidence is not None else "confidence=na",
        f"classifier_ms={classifier_ms:.1f}",
        f"total_overhead_ms={total_overhead_ms:.1f}",
    ]
    return " | ".join(parts)
