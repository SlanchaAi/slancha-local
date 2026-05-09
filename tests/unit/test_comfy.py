"""ComfyUI backend + /v1/images/generations endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from slancha_local.backends.comfy import DEFAULT_WORKFLOW, ComfyBackend, ImageRequest
from slancha_local.proxy.images import router as images_router

# ---------- ImageRequest ----------


@pytest.mark.parametrize(
    "size,expected",
    [
        ("512x512", (512, 512)),
        ("1024x1024", (1024, 1024)),
        ("768x1024", (768, 1024)),
        ("garbage", (512, 512)),
        ("", (512, 512)),
    ],
)
def test_image_request_width_height(size: str, expected: tuple[int, int]):
    req = ImageRequest(prompt="cat", size=size)
    assert req.width_height() == expected


# ---------- Workflow patching ----------


def test_load_default_workflow_when_no_override():
    b = ComfyBackend(base_url="http://x")
    w = b.load_workflow(None)
    assert w is not DEFAULT_WORKFLOW  # deep-copied
    assert w["6"]["class_type"] == "CLIPTextEncode"


def test_load_workflow_from_path(tmp_path: Path):
    custom = {"99": {"class_type": "CustomNode", "inputs": {}}}
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(custom))
    b = ComfyBackend(base_url="http://x", default_workflow=str(p))
    w = b.load_workflow(None)
    assert "99" in w


def test_patch_workflow_splices_prompt_and_size():
    b = ComfyBackend(base_url="http://x")
    w = b.load_workflow(None)
    req = ImageRequest(prompt="a fox in snow", size="768x1024", n=2, seed=42, steps=30)
    patched = b.patch_workflow(w, req)
    # First non-empty CLIPTextEncode (positive) gets the prompt
    assert patched["6"]["inputs"]["text"] == "a fox in snow"
    # Negative stays empty
    assert patched["7"]["inputs"]["text"] == ""
    # Latent gets size + batch
    assert patched["5"]["inputs"]["width"] == 768
    assert patched["5"]["inputs"]["height"] == 1024
    assert patched["5"]["inputs"]["batch_size"] == 2
    # Sampler gets seed + steps
    assert patched["3"]["inputs"]["seed"] == 42
    assert patched["3"]["inputs"]["steps"] == 30


# ---------- Probe ----------


@pytest.mark.asyncio
async def test_probe_healthy(monkeypatch):
    b = ComfyBackend(base_url="http://comfy.test")

    fake_response = MagicMock()
    fake_response.json.return_value = {"system": {"os": "linux"}, "devices": [{"name": "cuda:0"}]}
    fake_response.raise_for_status = MagicMock()

    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.get = AsyncMock(return_value=fake_response)

    with patch("slancha_local.backends.comfy.httpx.AsyncClient", return_value=fake_client):
        cap = await b.probe()

    assert cap.healthy
    assert cap.id == "comfy"
    assert cap.models[0].model_id == "comfy:cuda:0"
    assert "image_generation" in cap.models[0].capabilities


@pytest.mark.asyncio
async def test_probe_unreachable_returns_unhealthy(monkeypatch):
    import httpx as _httpx

    b = ComfyBackend(base_url="http://no.such.host")

    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.get = AsyncMock(side_effect=_httpx.ConnectError("nope"))

    with patch("slancha_local.backends.comfy.httpx.AsyncClient", return_value=fake_client):
        cap = await b.probe()

    assert not cap.healthy
    assert cap.models == ()


# ---------- Image extraction ----------


def test_extract_images_walks_outputs():
    b = ComfyBackend(base_url="http://x")
    outputs = {
        "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
        "10": {"images": [{"filename": "b.png", "subfolder": "", "type": "output"}]},
    }
    imgs = b._extract_images(outputs)
    assert len(imgs) == 2
    assert imgs[0]["filename"] == "a.png"


def test_image_url_construction():
    b = ComfyBackend(base_url="http://comfy:8188")
    url = b._image_url({"filename": "x.png", "subfolder": "sub", "type": "output"})
    assert "filename=x.png" in url
    assert "subfolder=sub" in url
    assert url.startswith("http://comfy:8188/view")


# ---------- /v1/images/generations endpoint ----------


def _build_test_app(image_backend=None) -> FastAPI:
    app = FastAPI()
    app.include_router(images_router)
    if image_backend is not None:
        app.state.image_backend = image_backend
    app.state.trace_writer = None  # skip trace persistence in tests
    return app


def test_images_endpoint_404_when_disabled():
    app = _build_test_app(image_backend=None)
    client = TestClient(app)
    r = client.post("/v1/images/generations", json={"prompt": "cat"})
    assert r.status_code == 404
    assert "not enabled" in r.json()["detail"]


def test_images_endpoint_validates_body():
    app = _build_test_app(image_backend=MagicMock())
    client = TestClient(app)
    r = client.post("/v1/images/generations", json={})  # missing prompt
    assert r.status_code == 422


def test_images_endpoint_happy_path():
    fake_backend = MagicMock()
    fake_backend.generate_image = AsyncMock(
        return_value={"created": 1700000000, "data": [{"url": "http://comfy/view?filename=x.png"}]}
    )
    app = _build_test_app(image_backend=fake_backend)
    client = TestClient(app)
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "a fox in snow", "size": "768x768", "seed": 7},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"][0]["url"].endswith("x.png")
    assert "request_id" in body
    fake_backend.generate_image.assert_awaited_once()
    # ImageRequest got the seed
    call_arg = fake_backend.generate_image.await_args[0][0]
    assert call_arg.seed == 7
    assert call_arg.size == "768x768"


def test_images_endpoint_504_on_timeout():
    fake_backend = MagicMock()
    fake_backend.generate_image = AsyncMock(side_effect=TimeoutError("comfy slow"))
    app = _build_test_app(image_backend=fake_backend)
    client = TestClient(app)
    r = client.post("/v1/images/generations", json={"prompt": "cat"})
    assert r.status_code == 504
