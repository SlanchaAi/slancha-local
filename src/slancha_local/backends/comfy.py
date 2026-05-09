"""ComfyUI image-gen backend — opt-in.

ComfyUI runs as a local HTTP server (default :8188). API surface used:
- GET /system_stats  → probe (returns {system: {os, ram_total, ...}, devices: [...]})
- POST /prompt       → enqueue a workflow (returns {prompt_id, ...})
- GET /history/{id}  → poll for completion (returns {<id>: {outputs: {...}}})
- GET /view?...      → fetch generated image bytes

Workflow shape: ComfyUI workflows are JSON graphs of node ids → {class_type, inputs}.
We bundle a minimal text2img workflow template; users can override by setting
SLANCHA_COMFY_DEFAULT_WORKFLOW to a JSON file path.

Privacy red line: this backend is opt-in (settings.comfy_enabled defaults False).
Disabled = zero network calls, same as the rest of the proxy.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from slancha_local.backends.base import Backend, BackendCapability, BackendModel
from slancha_local.proxy.models import ChatCompletionRequest

logger = logging.getLogger(__name__)


# Minimal SD1.5-style text2img workflow. Users can override via
# SLANCHA_COMFY_DEFAULT_WORKFLOW path. Node ids are arbitrary integers.
DEFAULT_WORKFLOW: dict[str, Any] = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "model.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "<PROMPT>", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "slancha", "images": ["8", 0]},
    },
}


@dataclass
class ImageRequest:
    """OpenAI-compat image-generation params."""

    prompt: str
    n: int = 1
    size: str = "512x512"
    response_format: str = "url"  # or "b64_json"
    model: str | None = None
    seed: int | None = None
    steps: int | None = None
    workflow_path: str | None = None  # override per-request

    def width_height(self) -> tuple[int, int]:
        try:
            w, h = self.size.lower().split("x")
            return int(w), int(h)
        except (ValueError, AttributeError):
            return 512, 512


class ComfyBackend(Backend):
    """ComfyUI image-gen adapter. Implements Backend ABC for parity with chat backends."""

    id = "comfy"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8188",
        default_workflow: str | None = None,
        poll_interval_s: float = 1.0,
        timeout_s: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_workflow = default_workflow
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s

    async def probe(self) -> BackendCapability:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=2.0) as c:
                r = await c.get("/system_stats")
                r.raise_for_status()
                stats = r.json()
            # ComfyUI doesn't expose a list of checkpoints via /system_stats;
            # we surface a single synthetic model id sourced from devices/checkpoints.
            device_name = (stats.get("devices") or [{}])[0].get("name", "comfy")
            return BackendCapability(
                id=self.id,
                healthy=True,
                base_url=self.base_url,
                models=(
                    BackendModel(
                        backend_id=self.id,
                        model_id=f"comfy:{device_name}",
                        ctx_window=0,  # not applicable
                        capabilities=("image_generation",),
                        est_throughput_tps=None,
                    ),
                ),
            )
        except (httpx.HTTPError, ValueError, KeyError):
            return BackendCapability(id=self.id, healthy=False, base_url=self.base_url, models=())

    async def chat(self, model_id: str, request: ChatCompletionRequest) -> dict[str, Any]:
        """ComfyBackend doesn't speak chat. Stub raises NotImplementedError so the
        registry doesn't pick it for chat routes. Image generation flows through
        `generate_image()` instead."""
        raise NotImplementedError("ComfyBackend handles images only; use generate_image()")

    async def chat_stream(self, model_id: str, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        raise NotImplementedError("ComfyBackend handles images only; use generate_image()")
        yield b""  # unreachable; here so static analysis sees this as an async-iter

    def load_workflow(self, override_path: str | None = None) -> dict[str, Any]:
        """Load workflow from disk if configured; else return DEFAULT_WORKFLOW (deep-copied)."""
        path = override_path or self.default_workflow
        if path and Path(path).exists():
            return json.loads(Path(path).read_text())
        return json.loads(json.dumps(DEFAULT_WORKFLOW))  # deep copy

    def patch_workflow(self, workflow: dict[str, Any], req: ImageRequest) -> dict[str, Any]:
        """Splice prompt + size + seed + steps into the workflow before submission.

        Walks the workflow graph and updates the FIRST positive CLIPTextEncode (for prompt),
        the FIRST EmptyLatentImage (for size + batch), and the FIRST KSampler (for seed/steps).
        Workflows with multiple positive prompts won't be fully covered — users should
        provide a custom workflow_path then.
        """
        w, h = req.width_height()
        positive_patched = False
        for _node_id, node in workflow.items():
            ct = node.get("class_type")
            inp = node.get("inputs", {})
            if ct == "CLIPTextEncode" and not positive_patched and inp.get("text") != "":
                inp["text"] = req.prompt
                positive_patched = True
            elif ct == "EmptyLatentImage":
                inp["width"] = w
                inp["height"] = h
                inp["batch_size"] = max(1, req.n)
            elif ct == "KSampler":
                if req.seed is not None:
                    inp["seed"] = req.seed
                if req.steps is not None:
                    inp["steps"] = req.steps
        return workflow

    async def generate_image(self, req: ImageRequest) -> dict[str, Any]:
        """End-to-end: load+patch workflow, submit, poll, fetch first image, return OpenAI-shape response."""
        workflow = self.load_workflow(req.workflow_path)
        workflow = self.patch_workflow(workflow, req)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as c:
            r = await c.post("/prompt", json={"prompt": workflow})
            r.raise_for_status()
            prompt_id = r.json().get("prompt_id")
            if not prompt_id:
                return {"data": [], "error": "comfy did not return prompt_id"}
            outputs = await self._poll_history(c, prompt_id)
            images = self._extract_images(outputs)
            if req.response_format == "b64_json":
                data = []
                for img_meta in images:
                    img_bytes = await self._fetch_image_bytes(c, img_meta)
                    import base64

                    data.append({"b64_json": base64.b64encode(img_bytes).decode()})
            else:
                data = [{"url": self._image_url(img_meta)} for img_meta in images]
            return {"created": int(time.time()), "data": data}

    async def _poll_history(self, client: httpx.AsyncClient, prompt_id: str) -> dict:
        deadline = time.time() + self.timeout_s
        import asyncio

        while time.time() < deadline:
            r = await client.get(f"/history/{prompt_id}")
            if r.status_code == 200:
                history = r.json()
                if prompt_id in history and history[prompt_id].get("outputs"):
                    return history[prompt_id]["outputs"]
            await asyncio.sleep(self.poll_interval_s)
        raise TimeoutError(f"comfy job {prompt_id} timed out after {self.timeout_s}s")

    def _extract_images(self, outputs: dict) -> list[dict]:
        """Outputs is {<node_id>: {"images": [{"filename": ..., "subfolder": ..., "type": ...}, ...]}}."""
        images: list[dict] = []
        for _node_id, node_out in outputs.items():
            for img in node_out.get("images", []):
                images.append(img)
        return images

    def _image_url(self, img_meta: dict) -> str:
        sub = img_meta.get("subfolder", "")
        typ = img_meta.get("type", "output")
        fn = img_meta.get("filename", "")
        return f"{self.base_url}/view?filename={fn}&subfolder={sub}&type={typ}"

    async def _fetch_image_bytes(self, client: httpx.AsyncClient, img_meta: dict) -> bytes:
        params = {
            "filename": img_meta.get("filename", ""),
            "subfolder": img_meta.get("subfolder", ""),
            "type": img_meta.get("type", "output"),
        }
        r = await client.get("/view", params=params)
        r.raise_for_status()
        return r.content
