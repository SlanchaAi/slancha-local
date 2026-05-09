# Spark smoke status

> Spark side fills this in after running `scripts/spark_smoke.sh`.
> Mac side reads it on next session start.

## Latest

```
2026-05-09T17:18:27Z [smoke] pass · 176 tests green · proxy up (port 8766; 8765 reserved by mcp_agent_mail) · /health 200 · /health/detailed 200 · /v1/chat/completions 503 (no backend, expected)
host: promaxgb10-d325 · Linux aarch64 6.17.0-1014-nvidia · Python 3.12.3
gpu: NVIDIA GB10 (memory.free reports N/A — unified memory architecture)
deps: libgomp present, treelite OK, no uv (script fell back to python3 -m venv), no ollama, no llama.cpp
findings:
  - port 8765 already bound by mcp_agent_mail substrate per memory; smoke needs a different default port or port-probe loop
  - decision-trace HTTP header NOT emitted on 503 error path (cloud escalation disabled). Header observed absent on the 503 response. Likely DecisionTraceHeaderMiddleware skips setting the header when the handler raises early. Should be set even on errors so the gallery / brag / why CLI can introspect failed decisions
  - cu130_env.sh + venv_cu130 from CLAUDE.md "Technical defaults" do NOT exist on this Spark; system python3 is fine for smoke. Update the doc, or stand them up if intended for FT runs
```

## Format for new entries (prepend, don't overwrite)

```
2026-MM-DDTHH:MM:SSZ [smoke] <pass | fail | partial> · 176 tests <green | red> · proxy <up | down> · decision-trace <observed | absent> · backend <ollama | none>
notes: free text
```
