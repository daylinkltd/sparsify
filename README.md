<div align="center">

<img src="site/assets/logo.svg" width="72" alt="Sparsify logo — a grid of expert cells, three lit">

# Sparsify

**Run Mixture-of-Experts models bigger than your RAM.**

*Your model is 26 GB. Your RAM budget is 3 GB. It runs anyway — with byte-identical output.*

![platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![backend](https://img.shields.io/badge/backend-MLX-orange)

</div>

---

Every local LLM runtime — Ollama, llama.cpp, LM Studio — assumes the whole
model must sit in RAM. For Mixture-of-Experts models that assumption wastes
almost everything: Mixtral 8x7B stores 26 GB of experts but *touches* only a
fraction per token. Sparsify treats your SSD as a first-class memory tier:
expert weights stay on disk and page into a bounded RAM cache **only when the
model's router selects them** — virtual memory, applied to intelligence.

```
Stored intelligence   →  SSD          (Mixtral 8x7B: 26.3 GB)
Active experts        →  RAM cache    (your budget, e.g. 3 GB)
Backbone              →  RAM          (~1 GB)
```

## Measured results — never simulated

On a 16 GB MacBook Air with models on an external USB SSD:

| Model (4-bit MLX) | Stored | Sparsify RSS | Result |
|---|---|---|---|
| **Mixtral 8x7B** | **26.3 GB** | **3.33 GB, flat** | runs correctly on a 16 GB machine |
| Qwen3-30B-A3B | 16.3 GB | 4.15 GB | correct · 1.8–2.6 tok/s (SSD-bound) |
| OLMoE-1B-7B *(fits budget)* | 3.9 GB | 4.2 GB | **154 tok/s — vanilla mlx-lm does 151.5** |
| OLMoE-1B-7B *(1 GB budget)* | 3.9 GB | 1.34 GB | output **token-identical to full-RAM inference** |

Two facts define the system: when a model's experts fit your budget it runs
at **native mlx-lm speed** (the runtime adds zero overhead), and when they
don't, output stays **exactly identical** — verified by golden tests with
cache evictions active — while decode speed scales with your SSD and budget.
Raw logs ship in [`docs/measurements/`](docs/measurements).

## Install

```bash
curl -fsSL https://github.com/daylinkltd/sparsify/releases/latest/download/install.sh | sh
```

(or from a checkout: `git clone https://github.com/daylinkltd/sparsify && cd sparsify && ./install.sh`)

One command: checks your platform, creates an isolated install in
`~/.sparsify`, puts `sparsify` on your PATH, and starts the API service on
**localhost:7777** (runs at login, Ollama-style).

## Quickstart

```bash
sparsify models                  # browse the catalog (11 MoE models)
sparsify pull olmoe:1b-7b        # 3.9 GB starter MoE
sparsify run  olmoe:1b-7b        # full-screen chat TUI, auto RAM budget

open http://localhost:7777       # web chat UI — model picker, live telemetry

curl localhost:7777/v1/chat/completions \
  -d '{"model":"olmoe:1b-7b","messages":[{"role":"user","content":"hi"}]}'
```

The API is OpenAI-compatible (`/v1/chat/completions` with SSE streaming,
`/v1/models`, `/health`), loads models on demand per request, and returns
measured paging telemetry with every response. Short names work everywhere:
`sparsify run qwen3` finds `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`.

## How it works

1. The model loads **lazily** — zero weight bytes read.
2. Surgery replaces every expert projection (any leaf tensor with a leading
   expert dimension — no per-architecture code) with a paged module. Router,
   attention, activations: untouched upstream mlx-lm.
3. The backbone materializes; experts stay on SSD.
4. When the router picks experts, exactly those weight slices are read — one
   contiguous `pread` each, in parallel — into a byte-budgeted LRU cache.
5. If the whole expert set fits your budget, everything loads once and the
   original code path runs — that's why resident speed equals vanilla mlx-lm.
6. The KV cache persists across chat turns (only new tokens prefill), and
   every metric — RSS, cache hits, SSD bytes, tok/s — streams live.

## The honest part

- Decode speed for models **larger than your budget** is bounded by
  `miss-bytes-per-token ÷ SSD-speed`. We measured our USB test drive at
  ~0.5 GB/s for expert-sized reads; internal NVMe is 5–10× that.
- We replayed 129k real routing decisions against LFU/CLOCK/SLRU: none beat
  LRU. The misses are genuine routing churn — so we publish the physics
  instead of pretending a cache trick fixes it. Next lever: async prefetch.
- Every number we publish is labeled measured, derived, or estimated.
  Simulated benchmarks are banned by [VISION.md](VISION.md).

## Status & roadmap

Working today: expert paging (verified exact), hybrid residency, parallel
reads, persistent KV cache, full-screen TUI with message queueing, web UI,
OpenAI API, login service, idempotent pulls, self-healing model registry.

Next ([docs/roadmap-vision.md](docs/roadmap-vision.md)): tools/function
calling for agents, KV-cache save/load to SSD, async expert prefetch, the
GLM-4.5-Air (106B stored) milestone on 16 GB hardware, mlx-vlm images and
mlx-whisper voice, CUDA backend for Linux/Windows.

## Verification

```bash
pytest                                    # unit + integration
pytest -m e2e tests/test_e2e_golden.py    # output-equivalence golden tests
```

The golden tests are the contract: paged output must equal full-RAM output
token-for-token; multi-turn must equal vanilla mlx-lm chat; dense models
must pass through byte-identical.

## License

MIT © Daylink Ltd
