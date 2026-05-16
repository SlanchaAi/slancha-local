"""Optional mesh integration: register this slancha-local instance as a
node with a slancha-mesh registry (https://github.com/SlanchaAi/slancha-mesh).

Opt-in via SLANCHA_MESH_REGISTRY_URL env var. When set, the proxy
starts a background heartbeat loop that posts NodeHeartbeat shapes
matching slancha-mesh spec §5. The mesh router then routes domain-
matched requests to this slancha-local's /v1/chat/completions endpoint;
slancha-local internally picks the right local backend.

When unset (default), there is ZERO behavior change — heartbeat loop
never starts.

Zero dependency on slancha-mesh: the heartbeat shape is built as a
plain dict matching the wire format. No pydantic models cross repos.
"""

from slancha_local.mesh.heartbeat import (
    MeshHeartbeatLoop,
    build_heartbeat_payload,
)

__all__ = ["MeshHeartbeatLoop", "build_heartbeat_payload"]
