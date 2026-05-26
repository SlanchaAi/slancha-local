# Fix: slancha-local mesh heartbeat must advertise a tailnet URL, not loopback

> 2026-05-25. Companion to the slancha-mesh tailnet migration
> (`slancha-mesh@feat/tailnet-transport`,
> `docs/MESH_TAILNET_SURVEY_2026_05_25.md`). This is the **producer-side**
> fix: slancha-mesh now accepts per-specialist tailnet URLs; slancha-local
> is the heartbeat producer and currently advertises loopback.

## Status — IMPLEMENTED 2026-05-25

Fixes #1 and #3 landed, plus a blocker the original draft missed and a
scope cut:

- **#1 advertise host ≠ bind** — `mesh_advertise_host` in `config.py`;
  `parse_magicdns_name` / `resolve_magicdns_name` / `resolve_advertise_host`
  / `build_node_url` in `mesh/heartbeat.py` (re-implemented, no slancha-mesh
  import). Tests in `tests/test_mesh_tailnet.py`.
- **#3 bind off loopback** — `SLANCHA_BIND_HOST=0.0.0.0` (config already
  passed `bind_host` through; documented).
- **BLOCKER (not in original draft): `MeshHeartbeatLoop` was never wired.**
  Defined + tested but never instantiated (`mesh_lifespan.py` gone, only a
  stale `.pyc`). Added a FastAPI `lifespan` in `proxy/main.py` that builds
  the loop (opt-in via `SLANCHA_MESH_REGISTRY_URL`), warms the capability
  cache, starts the daemon thread, and stops it on shutdown.
  `CapabilityProbe.cached()` added for the daemon thread's sync read.
- **#2 per-specialist `node_url` — DROPPED.** Code-confirmed topology:
  slancha-local is a single-endpoint proxy (`heartbeat.py` docstring + the
  internal `mesh_fallback`/`LocalCatalog` dispatch). The mesh router dials
  one `/v1/chat/completions`; per-specialist URLs only matter for the
  bare-model-server producer (`mesh.serve --tailnet`), a different repo.

Verified end-to-end: with a registry + `SLANCHA_MESH_ADVERTISE_HOST` set,
the captured heartbeat POST carried `node_url=http://<magicdns>:<port>`
(advertise host, not loopback) and `loaded_models` from the live catalog.

> **Known follow-up (separate):** `catalog_fn` advertises every healthy
> backend model as a `domain="general"` specialist, including embedders
> (e.g. `nomic-embed-text`). Filtering embedders out is the same class of
> bug as the rules-classifier "first local" mispick — tracked as the
> separate classifier papercut, not fixed here.

## The bug

`src/slancha_local/mesh/heartbeat.py` advertises a single `node_url` that
defaults to the **bind** address:

```python
# heartbeat.py docstring + caller convention
node_url=f"http://{settings.bind_host}:{settings.bind_port}"
# config.py:59-60
bind_host: str = "127.0.0.1"
bind_port: int = 8000
```

In the **old** topology this was fine: `slancha-local-proxy` ran on the
home box co-located with the models, and the per-host Cloudflare tunnel
exposed the proxy. The registry handed the router a loopback URL and the
router dialed localhost.

In the **new** topology the proxy moved to a cloud **gateway**
(`tag:gateway`) that reaches home nodes **over a Tailscale/Headscale
tailnet by MagicDNS**. A loopback `node_url` is unreachable from the
gateway. The registry hands the gateway a URL it cannot dial → the node
silently never receives mesh traffic.

Two concrete gaps in `heartbeat.py`:

1. **`node_url` is the bind address.** Bind (`0.0.0.0` / loopback, where the
   server listens) and advertise (a MagicDNS name the gateway dials) are
   conflated. They must be separated — bind broad, advertise a routable name.
2. **`build_heartbeat_payload` omits per-specialist `node_url`.** Each
   `loaded_models[]` entry has only `{specialist_id, model_id, loaded_at,
   estimated_tps}`. slancha-mesh just added an **optional
   `LoadedModel.node_url`** so a node serving several specialists on
   distinct ports (vLLM `:8003`, HF `:8004`) binds each to the right port.
   Without it, all specialists collapse onto one node-level URL.

## The fix

Mirror what `slancha-mesh/mesh/tailnet.py` does node-side. Keep it
**config-gated and control-plane-agnostic** (Headscale == Tailscale; only
the `tailscale up --login-server` flag differs, which is onboarding, not
runtime).

### 1. Add an advertise host, separate from bind host

`config.py`:

```python
# Where the proxy / model servers LISTEN. Set 0.0.0.0 on a tailnet so the
# gateway can reach them over WireGuard.
bind_host: str = Field(default="127.0.0.1")
bind_port: int = Field(default=8000)

# What the registry advertises to the gateway. None → auto-discover the
# node's MagicDNS name via `tailscale status --json` (Self.DNSName). Set
# explicitly to override. Env: SLANCHA_MESH_ADVERTISE_HOST.
mesh_advertise_host: str | None = Field(default=None)
```

Resolution helper (same shape as `mesh/tailnet.py:resolve_magicdns_name`,
never-raises subprocess contract):

```python
def resolve_advertise_host(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    try:
        out = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=4.0, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    try:
        name = json.loads(out.stdout).get("Self", {}).get("DNSName", "")
    except (ValueError, AttributeError):
        return None
    return name.rstrip(".") or None  # FQDN minus trailing dot
```

Build the advertised `node_url` from the advertise host (falling back to
the bind host so non-tailnet dev is unchanged):

```python
host = resolve_advertise_host(settings.mesh_advertise_host) or settings.bind_host
node_url = f"http://{host}:{settings.bind_port}"
```

### 2. Carry per-specialist `node_url` in the heartbeat

Add the port/url to `LoadedSpecialist` and emit it. slancha-mesh accepts
the field as optional; older registries ignore it (back-compat).

```python
@dataclass
class LoadedSpecialist:
    specialist_id: str
    model_id: str
    domain: str
    difficulty_tiers: list[str] = field(default_factory=lambda: ["medium"])
    estimated_tps: float | None = None
    node_url: str | None = None   # this specialist's own tailnet URL+port

# build_heartbeat_payload → loaded_models[] entry:
{
    "specialist_id": s.specialist_id,
    "model_id": s.model_id,
    "loaded_at": now,
    "estimated_tps": s.estimated_tps,
    "node_url": s.node_url,       # NEW — None is fine; registry falls back
}
```

> **Scope check — confirm before doing #2:** per-specialist `node_url`
> matters only if the gateway dials the **model ports directly**
> (vLLM `:8003`, HF `:8004`), which is the migration's stated design
> ("reaches home specialist nodes ... on the model ports"). If instead the
> gateway dials the slancha-local **proxy** (one `:8000` endpoint that
> dispatches internally), the single node-level `node_url` from #1 is
> sufficient and #2 is unnecessary. **Decide which: is slancha-local in the
> data path on the home box, or does the gateway hit the model servers
> directly?** That determines whether #2 is needed and who emits the
> heartbeat (slancha-local vs the model server via `mesh.serve --tailnet`).

### 3. Bind the listener off loopback on a tailnet

`cli.py:38-39` already passes `host=settings.bind_host`. On a tailnet node,
set `SLANCHA_BIND_HOST=0.0.0.0` (or the tailnet IP). The tailnet ACL
(`tag:gateway -> tag:specialist:<ports>`, deny-by-default) is the access
control — do **not** add a public tunnel/Funnel.

## Onboarding (unchanged node steps, both control planes)

```bash
# Tailscale SaaS:
sudo tailscale up --auth-key=<KEY> --advertise-tags=tag:specialist
# Headscale (self-hosted):
sudo tailscale up --auth-key=<KEY> --advertise-tags=tag:specialist \
  --login-server=https://<headscale-host>
```

Key source: Tailscale admin console, `headscale preauthkeys create`, or
slancha-api `POST /api/v1/mesh/hosts` (returns the join command +
`model_ports`).

## Tests to add

- `build_heartbeat_payload` emits `loaded_models[].node_url` when set;
  `None`/absent round-trips (back-compat).
- `resolve_advertise_host`: explicit override wins; parses `Self.DNSName`
  (trailing dot stripped) from a captured `tailscale status --json`;
  returns `None` on missing binary / non-zero exit / unparseable.
- node_url uses advertise host when resolvable, bind host otherwise.

These mirror `slancha-mesh/mesh/tests/test_tailnet.py` +
`test_tailnet_serve.py` — keep the golden heartbeat shape in sync; the
slancha-mesh wire contract locks `HeartbeatPostRequest` to
`{heartbeat, node_url}` (don't add top-level keys) and `loaded_models[]`
entries grow only optional fields.

---

## Separate, unrelated: `serve_v8_hf.py` model-name cosmetic bug

Flagged during the mesh survey, out of scope there, and **not in this
repo** — `serve_v8_hf.py` is the HF-transformers OpenAI-compatible server
on the Spark GB10 (port `:8004`), not part of slancha-local (a proxy).
Recorded here so it isn't lost:

- **Symptom:** chat responses report `"model": "paul-voice"` instead of the
  served id `"paul-voice-v8"`.
- **Likely cause:** the `ChatCompletion` response object's `model` field is
  built from a hardcoded/default string rather than the served model id (or
  the request's `model`). Common in hand-rolled HF OpenAI-compat servers.
- **Fix:** set the response `model` to the served-model-name
  (`paul-voice-v8`) — echo the configured served id (or `request.model`
  when it matches a served alias). Confirm against the actual file on the
  Spark; align the served id with the catalog card
  `mesh/catalog/paul-voice-v8.toml` and slancha-api's
  `_MODEL_PORTS["paul-voice-v8"] = 8004`.
- **Why it matters beyond cosmetics:** usage telemetry + the routing-
  transparency UI key on the response `model`; a wrong id misattributes
  traffic between the two voice specialists.
