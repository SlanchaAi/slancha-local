"""Pull-discovery client — slancha-local side.

Lets a slancha-local router (local or the cloud gateway) build its routing
table by walking the tailnet for tag:specialist peers and pulling each node's
/models, instead of (or alongside) the push-heartbeat path. Re-implemented
against the wire shape — ZERO import of slancha-mesh (same discipline as
mesh/heartbeat.py). The security property: node_url is host-pinned to the
peer actually dialed, so a node cannot advertise another's address.
"""

from __future__ import annotations

import json

from slancha_local.mesh.discovery import (
    DiscoveryResult,
    discover_specialists,
    parse_specialist_peers,
    pin_host,
)

_STATUS = {
    "Self": {"DNSName": "gw.taila.ts.net.", "Online": True, "Tags": ["tag:gateway"]},
    "Peer": {
        "k1": {"DNSName": "mac.taila.ts.net.", "Online": True, "Tags": ["tag:specialist"]},
        "k2": {"DNSName": "off.taila.ts.net.", "Online": False, "Tags": ["tag:specialist"]},
        "k3": {"DNSName": "laptop.taila.ts.net.", "Online": True, "Tags": ["tag:laptop"]},
    },
}


def _models(specialist_id: str, port: int, domain: str = "code", host: str = "evil.example") -> dict:
    return {
        "object": "list",
        "data": [{
            "id": specialist_id, "object": "model",
            "routing_meta": {
                "model_id": f"vendor/{specialist_id}", "domain": domain,
                "capabilities": ["streaming"], "quality": {"router_observed": 4.0},
                "node_urls": [f"http://{host}:{port}"],
            },
        }],
    }


def test_parse_peers_online_specialists_only():
    peers = parse_specialist_peers(_STATUS, include_self=False)
    hosts = {p.host for p in peers}
    assert hosts == {"mac.taila.ts.net"}  # offline + wrong-tag excluded


def test_parse_peers_accepts_json_string():
    assert any(p.host == "mac.taila.ts.net" for p in parse_specialist_peers(json.dumps(_STATUS)))


def test_parse_peers_empty_on_garbage():
    assert parse_specialist_peers("nope") == []
    assert parse_specialist_peers({}) == []


def test_pin_host_forces_dialed_peer():
    assert pin_host("http://evil.example:8003/v1", "mac.ts.net") == "http://mac.ts.net:8003/v1"


def test_discover_host_pins_and_aggregates():
    def fetch(host, port):
        return _models("demo-model", 8004, domain="writing") if host == "mac.taila.ts.net" else None

    result = discover_specialists(_STATUS, fetch=fetch, include_self=False)
    assert isinstance(result, DiscoveryResult)
    spec = result.specialists["demo-model"]
    assert spec.domain == "writing"
    assert spec.node_urls == ("http://mac.taila.ts.net:8004",)  # NOT evil.example
    assert result.reachable == ["mac.taila.ts.net"]


def test_discover_marks_unreachable():
    result = discover_specialists(_STATUS, fetch=lambda h, p: None, include_self=False)
    assert result.unreachable == ["mac.taila.ts.net"]
    assert result.specialists == {}


def test_discover_merges_specialist_across_nodes():
    status = {
        "Self": {"DNSName": "a.ts.net.", "Online": True, "Tags": ["tag:specialist"]},
        "Peer": {"k": {"DNSName": "b.ts.net.", "Online": True, "Tags": ["tag:specialist"]}},
    }
    result = discover_specialists(status, fetch=lambda h, p: _models("coder", 8003))
    assert set(result.specialists["coder"].node_urls) == {
        "http://a.ts.net:8003", "http://b.ts.net:8003",
    }
