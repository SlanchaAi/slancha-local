#!/usr/bin/env bash
# slancha-local v0.0.1 smoke test on Spark (DGX GB10, Linux ARM64, CUDA 13).
#
# What this verifies:
#   - clone + venv + editable install on linux/aarch64
#   - libomp/libgomp present (treelite dep)
#   - 176 tests pass under linux Python (Mac dev parity)
#   - `slancha doctor` runs to completion
#   - proxy starts on 127.0.0.1:8765
#   - /health/detailed responds 200
#   - decision-trace header emitted on a real chat completion (if any local
#     backend is up; otherwise reports "no backend, header verified absent")
#
# Run with:   bash scripts/spark_smoke.sh
# Exit 0 = pass; non-zero = something needs investigation.

set -euo pipefail

REPO_DIR="${SLANCHA_SMOKE_DIR:-$HOME/Source/slancha-local}"
PORT="${SLANCHA_SMOKE_PORT:-8765}"
HOST="127.0.0.1"

log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# 1. Clone or pull
if [ ! -d "$REPO_DIR/.git" ]; then
  log "cloning slancha-local into $REPO_DIR"
  git clone git@github.com:SlanchaAi/slancha-local.git "$REPO_DIR"
else
  log "pulling slancha-local"
  git -C "$REPO_DIR" pull --ff-only
fi

cd "$REPO_DIR"

# 2. Detect Python toolchain
log "Python toolchain"
if command -v uv >/dev/null 2>&1; then
  log "uv detected — uv venv + install"
  uv venv .venv --python 3.12 || true
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  log "uv not found — falling back to python3 -m venv"
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -e ".[dev]"
fi

# 3. Activate
# shellcheck source=/dev/null
source .venv/bin/activate

# 4. libomp / libgomp probe (treelite needs it)
log "treelite native dep probe"
if ldconfig -p 2>/dev/null | grep -q libgomp; then
  log "libgomp present"
else
  log "WARNING libgomp missing — install via: sudo apt-get install -y libgomp1"
  log "classifier will degrade to RulesFallbackClassifier — proxy still works"
fi

# 5. Run tests
log "pytest"
SLANCHA_TRACES_ROOT=/tmp/slancha-test-traces python -m pytest tests -q --tb=short

# 6. doctor
log "slancha doctor"
slancha doctor || log "doctor exited non-zero (often expected when no backends running)"

# 7. Start proxy in background
log "starting proxy on $HOST:$PORT"
slancha serve --host "$HOST" --port "$PORT" >/tmp/slancha-serve.log 2>&1 &
PROXY_PID=$!
trap 'kill $PROXY_PID 2>/dev/null || true' EXIT

# Wait for /healthz to come up (max 10s)
for i in $(seq 1 20); do
  if curl -sf -m 1 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    log "proxy responsive after ${i}x500ms"
    break
  fi
  sleep 0.5
done

# 8. health detailed
log "health/detailed"
curl -sf -m 5 "http://$HOST:$PORT/health/detailed" | python -m json.tool || {
  log "health/detailed FAILED — see /tmp/slancha-serve.log"
  tail -50 /tmp/slancha-serve.log
  exit 1
}

# 9. Real chat completion (best-effort — no backend available is acceptable)
log "chat completion smoke"
RESP_FILE=/tmp/slancha-resp.json
HEADERS_FILE=/tmp/slancha-headers.txt
HTTP_CODE=$(curl -s -o "$RESP_FILE" -D "$HEADERS_FILE" -w '%{http_code}' \
  -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}],"stream":false}' || echo "curl-failed")

log "HTTP $HTTP_CODE"
if grep -i '^slancha-decision-trace:' "$HEADERS_FILE" >/dev/null 2>&1; then
  log "decision-trace header observed:"
  grep -i '^slancha-decision-trace:' "$HEADERS_FILE"
else
  log "decision-trace header absent (likely no backend running — install Ollama and pull a model to verify end-to-end)"
fi

if [ "$HTTP_CODE" = "200" ]; then
  log "completion 200 — full E2E green"
elif [ "$HTTP_CODE" = "503" ]; then
  log "503 (no healthy backend). Install Ollama + pull a model: 'curl https://ollama.com/install.sh | sh && ollama pull qwen3:8b'"
else
  log "unexpected HTTP $HTTP_CODE — body:"
  cat "$RESP_FILE"
fi

log "smoke complete."
