# Roadmap — beyond text chat

Where Sparsify goes next, with the engineering truth attached to each item.
Ordered by how directly each builds on what's already proven.

## 1. Context on storage (started 2026-07-08)

The thesis applied to context: the KV cache is "active memory" too.

- **Done:** persistent KV cache across chat turns — each turn prefills
  only the unseen suffix (prefix-matched, trimmed on divergence, verified
  identical to vanilla mlx-lm chat). On a paged model this avoids
  re-reading experts for the whole history every turn.
- **Next:** save/load the KV cache to SSD (`mlx-lm` ships
  `save_prompt_cache`/`load_prompt_cache`) → sessions survive restarts;
  a 24/7 service resumes conversations instantly.
- **Honest limit:** storage does NOT make context unlimited. Two hard
  walls remain: (a) the model's trained context window (Qwen3-30B: 256k),
  and (b) attention compute grows with context length regardless of where
  the KV bytes live. What storage buys: cheap persistence, instant
  resume, and many parallel long sessions. "Unlimited memory" beyond the
  window is a retrieval problem (see agents, below), not a KV problem.

## 2. Agentic serving (24/7 assistant)

The pieces Sparsify already has: a login service that runs 24/7, an
OpenAI-compatible API, and (soon) persistent sessions.

- **Next:** `tools`/function-calling passthrough on
  `/v1/chat/completions` — Qwen3/Mixtral-class models have tool-call
  templates; the server needs to render tool schemas into the template
  and parse tool-call responses. With that, any agent framework
  (LangChain, OpenClaw-style assistants, custom loops) can point at
  `localhost:7777` and run fully local.
- **Position:** Sparsify stays the *runtime*, not the agent — the same
  way Ollama powers agents without being one. Session persistence +
  tools + 24/7 service is the complete substrate.

## 3. Multimodal (image → audio → video)

- **Image:** the `mlx-vlm` project runs vision-language models on Apple
  Silicon. Integration path: a second engine type behind the same server
  (`/v1/chat/completions` already accepts image content parts in the
  OpenAI schema; web UI gets a file-drop). Caveat to state up front:
  today's strong open VLMs are mostly **dense**, so paging gains are
  limited until MoE VLMs (e.g. DeepSeek-VL2 family) have solid MLX ports.
- **Audio in:** `mlx-whisper` for speech-to-text is mature — transcribe,
  then chat; cheap to add to the web UI (mic button) and CLI.
- **Audio out / video:** no production-grade MLX paths yet; revisit when
  the ecosystem lands them. We do not ship placeholders.

## 4. Router research mode (explicitly non-exact)

The runtime's core guarantee is exact model output — inference-time
routing changes break that and (per published MoE literature) usually
hurt quality, since routers are trained jointly with experts. What
Sparsify can own honestly: its telemetry already captures per-token
routing traces; a clearly-labeled opt-in research mode (expert dropout
sweeps, routing-temperature studies, utilization dashboards) would make
Sparsify the *instrument* for router science without contaminating the
default runtime. Any quality claim requires eval-suite evidence, not
vibes.

## 5. The 120B-in-8GB milestone

Architecturally ready: GLM-4.5-Air (106B stored, ~12B active, ~2 GB
backbone) is in the catalog. The claim to publish is whatever a real run
measures — memory bounding is expected to hold (it's the same math
proven on Mixtral); decode speed will be SSD-bound and needs the
prefetch milestone plus fast storage. First public benchmark: pull it on
a 16 GB machine, measure, publish the number either way.
