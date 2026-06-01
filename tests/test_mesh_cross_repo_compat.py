"""Cross-repo round-trip: slancha-local heartbeat → slancha-mesh pydantic.

Catches the bug class paul-mac surfaced 2026-05-16: slancha-local's
build_heartbeat_payload produced dicts that passed isolated unit tests
but failed slancha-mesh's HeartbeatPostRequest.model_validate with
422 ValidationError. Single-repo tests aren't enough; this file
explicitly validates the payload via slancha-mesh's pydantic models.

GATED on slancha-mesh being importable. When slancha-mesh isn't
installed (CI environments that test slancha-local in isolation),
the test SKIPs cleanly rather than failing collection.

Install slancha-mesh for these tests:
    pip install -e /path/to/slancha-mesh
or in CI:
    pip install slancha-mesh

The dependency direction is INTENTIONALLY ONE-WAY: slancha-local is a
slancha-mesh consumer; slancha-mesh has no awareness of slancha-local.
Testing the consumer against the producer's schema is the right
isolation; the alternative (factoring shared models into a third
slancha-mesh-protocol package) is a bigger refactor for v0.0.6+.
"""

from __future__ import annotations

import pytest

from slancha_local.mesh.heartbeat import (
    LoadedSpecialist,
    build_heartbeat_payload,
    probe_arch,
)

try:
    from mesh.registry import HeartbeatPostRequest  # type: ignore[import-not-found]

    HAS_MESH = True
except ImportError:
    HAS_MESH = False

requires_mesh = pytest.mark.skipif(
    not HAS_MESH,
    reason="slancha-mesh not installed; cross-repo round-trip skipped",
)


@requires_mesh
def test_default_payload_validates_against_mesh_schema():
    """Bare-defaults payload + probe_arch → HeartbeatPostRequest.model_validate green."""
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
    )
    # Must not raise
    HeartbeatPostRequest.model_validate(p)


@requires_mesh
def test_payload_with_one_specialist_validates():
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[
            LoadedSpecialist(
                specialist_id="qwen3-8b",
                model_id="Qwen/Qwen3-8B",
                domain="general",
                estimated_tps=42.5,
            )
        ],
    )
    HeartbeatPostRequest.model_validate(p)


@requires_mesh
def test_payload_arch_unknown_string_still_rejected():
    """Regression-lock: passing arch='unknown' explicitly should STILL fail
    mesh validation. probe_arch's job is to avoid this by default; if a
    caller bypasses the default, the protocol gate catches it.
    """
    from pydantic import ValidationError

    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
        arch="unknown",  # type: ignore[arg-type]
    )
    with pytest.raises(ValidationError):
        HeartbeatPostRequest.model_validate(p)


@requires_mesh
@pytest.mark.parametrize("arch", ["aarch64", "x86_64", "apple-silicon"])
def test_payload_all_valid_archs_validate(arch):
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
        arch=arch,
    )
    HeartbeatPostRequest.model_validate(p)


@requires_mesh
def test_payload_health_degraded_validates():
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
        health="degraded",
    )
    HeartbeatPostRequest.model_validate(p)


@requires_mesh
def test_payload_with_runtime_probe_arch_validates():
    """End-to-end: real probe_arch() on the actual host → valid payload.
    Catches the case where probe_arch falls back to an invalid label."""
    detected = probe_arch()
    assert detected in {"aarch64", "x86_64", "apple-silicon"}
    p = build_heartbeat_payload(
        node_id="n1",
        node_url="http://x",
        friendly_name="laptop",
        loaded=[],
    )
    HeartbeatPostRequest.model_validate(p)
