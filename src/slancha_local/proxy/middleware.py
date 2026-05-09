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
        f"picked={picked}",
        f'reason="{reason}"',
        f"fallbacks=[{','.join(fallbacks)}]",
        f"domain={domain or 'unknown'}",
        f"difficulty={difficulty or 'unknown'}",
        f"jailbreak={'yes' if jailbreak else 'no'}",
        f"pii={'yes' if pii else 'no'}",
        f"tool={'yes' if tool_calling else 'no'}",
        f"confidence={confidence:.2f}" if confidence is not None else "confidence=na",
        f"classifier_ms={classifier_ms:.1f}",
        f"total_overhead_ms={total_overhead_ms:.1f}",
    ]
    return " | ".join(parts)
