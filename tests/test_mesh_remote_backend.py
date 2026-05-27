"""RemoteMeshBackend — a discovered mesh specialist as a dispatchable Backend.

A discovered specialist is just an OpenAI-compat endpoint at its (host-pinned)
node_url. Wrapping it as a Backend with id == specialist_id lets it drop into
the existing BackendRegistry, so the proxy's dispatch path routes to remote
mesh specialists with no change to chat.py.
"""

from __future__ import annotations

from slancha_local.backends.mesh_remote import RemoteMeshBackend, backends_from_discovery
from slancha_local.mesh.discovery import DiscoveredSpecialist, DiscoveryResult


def _result() -> DiscoveryResult:
    return DiscoveryResult(
        specialists={
            "demo-model": DiscoveredSpecialist(
                specialist_id="demo-model", model_id="vendor/demo-model", domain="writing",
                capabilities=("streaming", "system_prompt"),
                node_urls=("http://mac.ts.net:8004",),
            ),
            "no-url": DiscoveredSpecialist(specialist_id="no-url", node_urls=()),  # skipped
        }
    )


def test_factory_builds_one_backend_per_specialist_with_urls():
    backends = backends_from_discovery(_result())
    ids = {b.id for b in backends}
    assert ids == {"demo-model"}  # "no-url" (no node_urls) skipped


def test_backend_id_is_specialist_id_and_base_url_from_node_url():
    be = backends_from_discovery(_result())[0]
    assert isinstance(be, RemoteMeshBackend)
    assert be.id == "demo-model"
    assert be._base_url == "http://mac.ts.net:8004"  # OpenAICompat appends /v1/...


async def test_probe_uses_discovered_data_without_network():
    be = backends_from_discovery(_result())[0]
    cap = await be.probe()  # must NOT make a network call
    assert cap.healthy is True
    assert cap.id == "demo-model"
    assert cap.base_url == "http://mac.ts.net:8004"
    assert cap.models[0].model_id == "vendor/demo-model"
    assert "streaming" in cap.models[0].capabilities


def test_distinct_ids_so_registry_keys_do_not_collide():
    result = DiscoveryResult(specialists={
        "a": DiscoveredSpecialist(specialist_id="a", node_urls=("http://x:8003",)),
        "b": DiscoveredSpecialist(specialist_id="b", node_urls=("http://y:8003",)),
    })
    backends = backends_from_discovery(result)
    assert {b.id for b in backends} == {"a", "b"}
