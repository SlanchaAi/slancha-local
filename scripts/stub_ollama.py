"""Tiny stub Ollama for live testing without installing real Ollama.

Run: python scripts/stub_ollama.py --port 11434
Responds to:
  GET  /api/tags                    → fixed model list
  POST /v1/chat/completions         → fixed OpenAI-compat response
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

TAGS = {
    "models": [
        {"model": "qwen3:8b", "name": "qwen3:8b"},
        {"model": "codestral:22b", "name": "codestral:22b"},
        {"model": "llama-3.3:70b", "name": "llama-3.3:70b"},
    ]
}

CHAT_RESPONSE = {
    "id": "chatcmpl-stub-001",
    "object": "chat.completion",
    "created": 1715275000,
    "model": "qwen3:8b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "[stub-ollama] hello back from the local stack.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 11, "total_tokens": 23},
}


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quiet
        pass

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._send_json(200, TAGS)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)  # discard body
        if self.path == "/v1/chat/completions":
            self._send_json(200, CHAT_RESPONSE)
        else:
            self._send_json(404, {"error": "not found"})

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    server = HTTPServer((args.host, args.port), StubHandler)
    print(f"[stub-ollama] listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
