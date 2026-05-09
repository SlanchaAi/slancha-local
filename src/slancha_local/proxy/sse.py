"""SSE-stream parser for OpenAI-compat chat streams.

Parses incrementally as bytes arrive from the backend and accumulates:
- the assistant message content (joined deltas)
- a count of content-bearing deltas (rough token-count proxy)
- the final usage dict if the backend sent one

Robust to: chunks splitting mid-line, multi-line `data:` blocks, the `[DONE]`
sentinel, malformed JSON deltas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class StreamAccumulator:
    """Stateful incremental SSE parser."""

    _buffer: bytes = b""
    content: str = ""
    delta_count: int = 0
    finish_reason: str | None = None
    usage_in: int = 0
    usage_out: int = 0

    # `delta.content` events accumulate into `content`. Other delta types
    # (tool_calls, function_call) are passed through but not counted.
    _seen_done: bool = field(default=False, init=False)

    def feed(self, chunk: bytes) -> None:
        """Accumulate a chunk; parse any complete `data: ...\\n\\n` blocks."""
        if self._seen_done:
            return
        self._buffer += chunk
        # SSE blocks are separated by blank lines (\n\n); be lenient about \r\n.
        # We process complete blocks and leave any trailing partial in the buffer.
        norm = self._buffer.replace(b"\r\n", b"\n")
        blocks = norm.split(b"\n\n")
        # Last item may be partial — keep it.
        complete, tail = blocks[:-1], blocks[-1]
        for block in complete:
            self._consume_block(block)
        self._buffer = tail

    def _consume_block(self, block: bytes) -> None:
        # A block is potentially multiple `data: ...` lines plus optional
        # `event: ...`, `id: ...`, etc. OpenAI-compat only uses `data:`.
        for line in block.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if payload == b"[DONE]":
                self._seen_done = True
                return
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                continue
            self._consume_event(evt)

    def _consume_event(self, evt: dict) -> None:
        choices = evt.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or {}
            content_chunk = delta.get("content")
            if isinstance(content_chunk, str) and content_chunk:
                self.content += content_chunk
                self.delta_count += 1
            if "finish_reason" in choice and choice["finish_reason"]:
                self.finish_reason = choice["finish_reason"]
        usage = evt.get("usage")
        if isinstance(usage, dict):
            self.usage_in = int(usage.get("prompt_tokens", 0) or 0) or self.usage_in
            self.usage_out = int(usage.get("completion_tokens", 0) or 0) or self.usage_out

    @property
    def tokens_out_estimate(self) -> int:
        """Best-available token count: usage if the backend reported it, else delta count."""
        return self.usage_out or self.delta_count
