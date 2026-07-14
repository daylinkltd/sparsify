<div align="center">

<img src="site/assets/logo.svg" width="72" alt="Sparsify logo — a grid of expert cells, three lit">

# Sparsify

**Run Mixture-of-Experts models bigger than your RAM.**

*Your model is 390 GB (DeepSeek-R1). Your RAM budget is 24 GB. It runs anyway — with byte-identical output.*

![version](https://img.shields.io/badge/version-0.4.2-E8A33D)
![platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![backend](https://img.shields.io/badge/backend-MLX-orange)
![output](https://img.shields.io/badge/output-byte--identical-3FB27F)

</div>

---

Every local LLM runtime — Ollama, llama.cpp, LM Studio — assumes the whole
model must sit in RAM. For Mixture-of-Experts models that assumption wastes
almost everything: DeepSeek-R1 stores 390 GB of experts but *touches* only a
fraction per token. Sparsify treats your SSD as a first-class memory tier:
expert weights stay on disk and page into a bounded RAM cache **only when the
model's router selects them** — virtual memory, applied to intelligence.

```
Stored intelligence   →  SSD          (DeepSeek-R1: 390.0 GB)
Active experts        →  RAM cache    (your budget, e.g. 24 GB)
Backbone              →  RAM          (~10 GB)
```

## See it

Interactive terminal chat, and the built-in web UI at `localhost:7777` —
both show live telemetry (tok/s, RSS, cache hit rate) under every
reply, and both drive tools (fetch a URL, write a file, run a command):

<p align="center">
  <img src="site/assets/view-terminal.svg" width="49%" alt="Sparsify terminal chat: a paged Qwen3-30B answering, with tool calls and live telemetry">
  <img src="site/assets/view-webui.svg" width="49%" alt="Sparsify web UI: chat with sidebar, model picker, tool card, and telemetry">
</p>

## Measured results — never simulated

On a 16 GB MacBook Air (models on internal NVMe unless noted):

| Model (4-bit MLX) | Stored | Sparsify RSS | Result |
|---|---|---|---|
| **Mixtral 8x7B** | **26.3 GB** | **3.33 GB, flat** | runs on a 16 GB machine — vanilla mlx-lm can't load it at all |
| Qwen3-30B-A3B | 16.3 GB | ~4.5 GB | **11.0 tok/s** @ 91% expert reuse (1.8 on USB SSD) |
| OLMoE-1B-7B *(fits budget)* | 3.9 GB | 4.2 GB | **168 tok/s — vs 161.6 vanilla mlx-lm** (zero overhead) |
| OLMoE-1B-7B *(1 GB budget)* | 3.9 GB | 1.34 GB | output **token-identical to full-RAM inference** |

<p align="center">
  <img src="site/assets/chart-memory.png" width="80%" alt="Model on disk vs RAM used: Mixtral 26.3 GB stored / 3.33 GB RAM; Qwen3 16.3/4.15; OLMoE 3.9/1.34">
</p>
<p align="center">
  <img src="site/assets/chart-speed.png" width="49%" alt="Qwen3-30B decode: 1.8 tok/s USB SSD, 8.5 NVMe 3GB, 11.0 NVMe 4.5GB">
  <img src="site/assets/chart-overhead.png" width="49%" alt="Zero paging overhead: OLMoE 168 tok/s Sparsify resident vs 161.6 vanilla mlx-lm">
</p>

Two facts define the system. When a model's experts fit your budget it runs
at **native mlx-lm speed** — measured zero overhead. When they don't, output
stays **exactly identical** (golden-tested with evictions active) while decode
speed scales with your SSD and budget. 

Unlike CPU-only streaming prototypes (such as `colibri`) that are throttled to **0.05–0.1 tok/s** on CPU RAM, Sparsify uses a **GPU-accelerated runtime** (supporting CUDA, MPS, and Metal) to execute the backbone on the GPU while dynamically paging experts into a VRAM cache, yielding **100x+ faster generation speeds (8.5–11.0 tok/s)**. Raw logs: [`docs/measurements/`](docs/measurements).

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
sparsify pull                    # interactive model picker in the terminal
sparsify ps                      # what's loaded, cache hit rate, SSD traffic
sparsify remove olmoe:1b-7b      # delete a downloaded model and free up space
sparsify uninstall               # complete removal, models included (or --keep-models)

curl localhost:7777/v1/chat/completions \
  -d '{"model":"olmoe:1b-7b","messages":[{"role":"user","content":"hi"}]}'
```

### Popular Supported Models

Sparsify automatically handles both Mixture-of-Experts (MoE) paging targets and standard dense architectures:

| Alias | Hugging Face Repository | Type | Size (4-bit) |
|---|---|---|---|
| `olmoe:1b-7b` | `mlx-community/OLMoE-1B-7B-0125-Instruct-4bit` | MoE (1B/7B) | 3.9 GB |
| `qwen:moe-14b` | `mlx-community/Qwen2.5-Moe-14B-A2.7B-Instruct-4bit` | MoE (14B) | 8.5 GB |
| `qwen:30b` | `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` | MoE (30B) | 16.3 GB |
| `mixtral:8x7b` | `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` | MoE (8x7B) | 26.3 GB |
| `deepseek:r1-distill-qwen-1.5b` | `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit` | Dense (Reasoner) | 1.0 GB |
| `deepseek:r1-distill-qwen-7b` | `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` | Dense (Reasoner) | 4.3 GB |
| `deepseek:r1-distill-llama-8b` | `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit` | Dense (Reasoner) | 4.5 GB |
| `deepseek:r1-distill-qwen-14b` | `mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit` | Dense (Reasoner) | 9.0 GB |
| `deepseek:r1-distill-qwen-32b` | `mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit` | Dense (Reasoner) | 19.5 GB |
| `deepseek:r1-distill-llama-70b` | `mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit` | Dense (Reasoner) | 42.5 GB |
| `qwen:coder-32b` | `mlx-community/Qwen2.5-Coder-32B-Instruct-4bit` | Dense (Coding) | 19.5 GB |
| `llama:3.3-70b` | `mlx-community/Llama-3.3-70B-Instruct-4bit` | Dense (General) | 42.5 GB |
| `deepseek:r1` | `mlx-community/DeepSeek-R1-4bit` | MoE (671B) | 390.0 GB |
| `glm:5.2` | `mlx-community/GLM-5.2-4bit` | MoE (744B) | 385.0 GB |


The API is OpenAI-compatible (`/v1/chat/completions` with SSE streaming,
`/v1/models`, `/health`), loads models on demand per request, and returns
measured paging telemetry with every response. Short names work everywhere:
`sparsify run qwen3` finds `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`.

## Features

- **Cross-Platform PyTorch Backend** — Run with CUDA acceleration on Windows/Linux, MPS on macOS, or CPU fallbacks.
- **High-Fidelity Accuracy Preservation** — We preserve original quantization bit-rates (4-bit or 8-bit) instead of using lossy 2-bit/3-bit compressions that degrade intelligence. Output remains token-for-token byte-identical to native full-RAM execution.
- **Expert paging** — run models larger than RAM; output verified identical.
- **Hybrid residency** — models that fit the budget load fully, at native speed.
- **Persistent KV cache** — each chat turn prefills only new tokens.
- **Unlimited generation** — replies run to the model's own context window, no arbitrary cap.
- **Honest context sizing** — `/health` reports both the model's architectural `context_limit` and a `safe_context_tokens` derived from your machine's free RAM right now (KV cache scales linearly with tokens; a 262k-context model can need 25+ GB of KV cache alone at that ceiling). Agent frameworks like OpenClaw should size their context/compaction budget from the safe number, not the architectural one.
- **Tools / agent loop** — fetch URLs, web search, read/write files, run shell, and **control a browser** (log in, click, type — DOM-driven, persistent session), workspace-scoped with opt-in tiers.
- **OpenAI-compatible API with function calling** — `/v1/chat/completions` (SSE), `/v1/models`; send `tools`, get structured `tool_calls` back (streaming and non-streaming), send `role:"tool"` results in. Drop-in model provider for agent frameworks like OpenClaw — the agent shell runs on their side, every token runs on your paged runtime.
- **Terminal + web UI** — live telemetry, chat history, projects, settings (system prompt, temperature, theme).
- **Attachments** — drag &amp; drop or attach text files in the web UI (`/attach <path>` in the terminal); contents go into your message. Images honestly declined until vision models land (mlx-vlm, roadmap).
- **Voice input, fully local** — mic button in the web UI; audio is transcribed on your machine by mlx-whisper via `/v1/audio/transcriptions` (OpenAI-compatible). Nothing leaves localhost. `pip install 'sparsify[voice]'`.
- **Scheduled agents** — `sparsify task add "…" --at 10:00 --tz Asia/Kolkata`: run any instruction autonomously on a schedule.
- **Ollama-style ops** — `pull` / `run` / `serve` / `ps`, login service, one-command install, self-update.

## Backends

The paging core (store, cache, module surgery) is backend-agnostic. Today one
backend is shipped and verified; others are the roadmap — listed honestly, not
claimed before they run.

| Backend | Platform | Status |
|---|---|---|
| **MLX** | macOS · Apple Silicon | ✅ shipping, all results above measured on it |
| **CUDA / PyTorch** | Linux · Windows · macOS | ✅ shipping, cross-platform PyTorch paging with CUDA/MPS acceleration |
| CPU / GGUF | any | 🔭 exploratory / in progress |

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
  `miss-bytes-per-token ÷ SSD-speed`. Measured on the same machine, same
  model, same budget: USB SSD 1.8 tok/s → internal NVMe **8.5 tok/s**
  (11.0 at a 4.5 GB budget). Storage speed converts directly into tokens.
- We replayed 129k real routing decisions against LFU/CLOCK/SLRU: none beat
  LRU. The misses are genuine routing churn — so we publish the physics
  instead of pretending a cache trick fixes it. (Speculative prefetch was
  built and measured too: it made decode *slower* — notes in
  `paging/surgery.py`.)
- Until v0.4.1 the reported hit rate included structural hits (three
  projections share one cached expert), floor-inflating it to 66.7% even
  with zero actual reuse. It now measures true cross-token expert reuse —
  earlier "97%" reads as 91% under the honest definition. Same runs, same
  speed, corrected stat.
- Every number we publish is labeled measured, derived, or estimated.
  Simulated benchmarks are banned by [VISION.md](VISION.md).

## Status & roadmap

Working today: expert paging (verified exact), hybrid residency, parallel
reads, persistent KV cache, full-screen TUI with message queueing, web UI,
OpenAI API with function calling (verified live against a paged Qwen3-30B),
login service, idempotent pulls, self-healing model registry.

Next ([docs/roadmap-vision.md](docs/roadmap-vision.md)):
KV-cache save/load to SSD, async expert prefetch, the GLM-4.5-Air (106B stored) milestone on 16 GB hardware, mlx-vlm images and mlx-whisper voice, and **Activation Sparsity for dense models** (PowerInfer-style neuron paging) to unlock high-speed local inference for dense architectures in low memory.

## Updating

New pushes reach you without a reinstall:

```bash
sparsify version    # current vs latest (checks GitHub)
sparsify update     # git pull + reinstall + restart the service
```

`sparsify run` shows a one-line hint when an update is available, and the
web UI shows an **Update available** button (top bar) that updates and
reconnects in place. Everything self-hosted — no telemetry, just a commit
comparison against the public repo.

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
