# Multimodal extension — design

slancha-local v0.1 routes only chat/text. The classifier is text-only.
Extending to **diffusion (image gen)** + **voice (TTS/ASR)** + **vision** is plausible
because the gateway architecture is mode-agnostic — it's the *classifier* and *backend
adapters* that need to be teach-aware.

## Routes the classifier already exposes

The 6-head classifier emits per-prompt:
- `tool_calling: bool` — already there; mode hint adjacent
- `vision: capability` (in BackendModel only, inferred from model name)

What's missing: a **mode head** (`text | image | audio_in | audio_out`).

## Proposed extension (Phase 1.2 candidate)

### A — New endpoint surface

Add OpenAI-compat `POST /v1/images/generations` + `/v1/audio/speech` + `/v1/audio/transcriptions`. Pattern matches existing `/v1/chat/completions`:

```
POST /v1/images/generations { "prompt": "...", "size": "1024x1024", "n": 1 }
  → classifier reads prompt
  → classifier picks image backend (e.g. local:diffusers:flux-schnell)
  → backend dispatches; trace header `slancha-decision-trace` still emitted
```

### B — New backends

| Mode | Backend | Probe | Notes |
|---|---|---|---|
| Image | ComfyUI | `GET /system_stats` | Most prosumer-popular. JSON workflow API. |
| Image | diffusers (HF) | local Python lib | High-quality but heavy; suit Spark only |
| Image | SwarmUI | `GET /api/get_models` | Auto1111-compat |
| Audio TTS | piper | local CLI | Fast, low-VRAM, good for many voices |
| Audio TTS | xtts-v2 | HTTP server | Higher quality, ~3GB VRAM |
| Audio ASR | whisper.cpp | `POST /inference` | Same quality as openai-whisper, native binary |
| Vision input | (handled inside chat backends if model supports) | - | qwen-vl, llava, etc. |

Same `Backend` ABC, just `chat → generate` and `chat_stream → generate_stream`.

### C — Classifier extension

Two paths:

1. **Routing-only by URL.** `POST /v1/images/...` always picks an image backend; classifier just picks among image backends (e.g. fast vs quality). Cheap, ships in a week.
2. **Mode classification head.** Train a 4-class head (text / image / audio-out / audio-in) on prompt embeddings. Lets `POST /v1/chat/completions` "be smart" about routing image-y prompts ("draw me a cat") to a diffusion backend automatically — the **viral magic moment**. Costlier; needs training data.

Recommend: ship #1 immediately when we add the endpoints; train the head when we have ~5K mode-tagged examples (likely from the trace upload pipeline).

### D — Trace schema impact

Add `mode: "text" | "image" | "audio_in" | "audio_out"` to trace JSONL. Default `text` for backward compat.

### E — Spark unique advantages for multimodal

GB10 has 121GB unified memory + sm_120. That's:

- **Flux schnell** (12GB VRAM) trivially fits with room for several at once
- **xtts-v2** voice clone (3GB) leaves headroom for chat + image + audio simultaneously
- **whisper-large-v3** (3GB)

So the same Spark box can host 3-4 multimodal backends concurrently with proper LRU eviction. ComfyUI's "smart memory" feature handles this; vLLM doesn't. Architecture implication: Spark hosts a ComfyUI-like supervisor for image; vLLM/llama.cpp for chat; piper/xtts for audio. slancha-local routes across them.

### F — Viral mechanic candidate

`slancha generate "make a poster for my Pokémon collection: qwen3, codestral, llama-3.3"` — sends to image backend, returns PNG, displays inline (TUI ANSI image protocol or terminal-image library). The 3-second screen recording of `slancha generate ...` → ASCII-art preview is its own viral artifact.

## Out of scope for v0.1

- Video gen (compute too heavy)
- Embedding-only mode for RAG (different abstraction; defer)
- Cross-modal routing (text → image → text loops) — too speculative

## Phase plan

- v0.2: Image gen backend (ComfyUI), `POST /v1/images/generations`, mode=image traces
- v0.3: Audio TTS backend (piper), `POST /v1/audio/speech`
- v0.4: Audio ASR (whisper.cpp), `POST /v1/audio/transcriptions`
- v0.5: Mode-classifier head trained on accumulated multimodal traces
- v0.6: `slancha generate` CLI + terminal-image preview

Each ships independent of the next. v0.2 alone is enough to flip the launch story from "smart router for chat" to "smart router for your multimodal local stack."
