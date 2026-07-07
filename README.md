# Sparsify

**A storage-backed runtime for Mixture-of-Experts models.**

Traditional runtimes (Ollama, llama.cpp, mlx-lm, vLLM) assume the whole
model lives in RAM. Sparsify treats the SSD as a first-class memory tier:
expert weights stay on disk and are paged into a byte-budgeted RAM cache
only when the model's router actually selects them.

```
Total intelligence  →  SSD          (e.g. Mixtral 8x7B: 26.3 GB)
Active experts      →  RAM cache    (configurable budget, e.g. 3 GB)
Backbone            →  RAM          (attention/norms/embeddings, ~1 GB)
```

## Status — what is real today

Everything below is measured on a 16 GB M-series Mac with models on an
external USB SSD. Nothing is simulated.

- **Correctness (verified)** — storage-backed output is *exactly identical*
  to full-RAM inference (golden test on OLMoE-1B-7B with active evictions;
  paged projections bit-identical to full-tensor `gather_qmm`).
- **Memory bounding (measured)** — Mixtral 8x7B, 26.3 GB on disk, generates
  correct text in **~3.3 GB process RSS** on a 16 GB machine. Qwen3-30B-A3B
  (16.3 GB) runs in **4.15 GB RSS**.
- **Universality** — no per-architecture code: any mlx-lm MoE whose expert
  projections are expert-stacked linears is detected structurally
  (Mixtral, Qwen3-MoE, OLMoE tested). Dense models pass through untouched,
  byte-identical. Both on-disk layouts are supported (stacked tensors and
  per-expert tensors à la upstream Mixtral).
- **Throughput (measured, slow)** — decode is SSD-bound when the expert
  working set exceeds the cache budget. This is the current engineering
  frontier; see the roadmap in `docs/`.

## Quick start

```bash
uv pip install -e ".[dev]"

sparsify pull qwen:30b-a3b        # download & register (idempotent)
sparsify list                     # local models
sparsify run qwen:30b-a3b         # interactive chat, auto RAM budget
sparsify run qwen:30b-a3b --memory-limit 3   # explicit 3 GB expert cache

sparsify start                    # background API service on localhost:7777
curl localhost:7777/v1/chat/completions \
  -d '{"model":"qwen:30b-a3b","messages":[{"role":"user","content":"hi"}]}'
sparsify serve <model>            # or run the server in the foreground
```

The API is OpenAI-compatible (`/v1/chat/completions` with streaming,
`/v1/models`, `/health`) and loads models on demand per request, Ollama-style.
One model is resident at a time; responses include measured paging telemetry
under `"sparsify"`.

The expert-cache budget defaults to **auto** (half of measured free RAM at
startup, 1 GiB floor). `--memory-limit N` pins it and persists per model.

## How it works

1. The model is loaded **lazily** (no weight bytes read).
2. `sparsify.paging.attach_paging` walks the module tree and replaces every
   expert projection (any leaf linear with a leading expert dimension) with
   a `PagedSwitchLinear`. The router, top-k selection, GLU activation and
   shared experts remain unmodified upstream code.
3. `mx.eval` then materializes only the backbone.
4. During inference each MoE block resolves its router indices, and the
   selected experts are fetched through a byte-budgeted LRU `ExpertCache`
   backed by `SafetensorsExpertStore` — one contiguous `pread` per expert
   tensor slice, any dtype including bfloat16.
5. Telemetry reports measured RSS, active/peak Metal memory, cache
   hits/misses/evictions and SSD bytes/latency per token.

## Project structure

```
src/sparsify/
├── cli.py              # pull / list / run / serve / inspect / stats
├── paging/             # the runtime core
│   ├── store.py        #   safetensors range reads (pread per expert)
│   ├── cache.py        #   byte-budgeted LRU expert cache
│   ├── modules.py      #   PagedSwitchLinear + ExpertGroup
│   └── surgery.py      #   structural detection & module replacement
├── runtime/
│   ├── chat_generation.py  # SparsifyEngine (streaming, telemetry)
│   ├── model_registry.py   # local model registry
│   └── tui.py              # interactive chat UI
├── profiler/           # research instruments (GGUF/system profiling)
└── experiments/        # dense-transformer research (falsified; archived)
```

## Verification

```bash
pytest                       # unit + integration (fast)
pytest -m e2e tests/test_e2e_golden.py   # golden output-equivalence tests
```

The golden tests are the contract: paged output must equal full-RAM output
token-for-token, and dense models must be byte-identical to unmodified
mlx-lm.

## Scientific honesty

Every reported number is labeled measured / derived / estimated. Simulated
results are not allowed in benchmarks or demos. See `VISION.md`.

## License

MIT
