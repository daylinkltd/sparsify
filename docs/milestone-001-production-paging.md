# Milestone 001 — Production Expert Paging

**Date:** 2026-07-07 · **Machine:** 16 GB M-series MacBook Air (Mac17,3), models on Transcend ESD310C USB SSD · **Stack:** mlx 0.31.2, mlx-lm 0.31.3

Every number in this document is **measured** unless explicitly marked *derived*. Raw logs: `docs/measurements/2026-07-07/`.

## What was built

The research-phase paging prototype (model-proxy wrapper + instance
`__call__` monkeypatching + per-layer graph surgery) was replaced wholesale
with `sparsify.paging`:

| Component | Responsibility |
|---|---|
| `SafetensorsExpertStore` | One contiguous `pread` per expert tensor slice, straight from the documented safetensors container. Any dtype (incl. bfloat16). Two layouts: expert-stacked tensors (sliced) and per-expert tensors (upstream Mixtral style), the latter resolved by tracing the model's own `sanitize()` converter — no per-architecture naming tables. |
| `ExpertCache` | Byte-budgeted LRU at whole-expert granularity. In-flight entries are protected from self-eviction; overshoot is measured and reported, never hidden. |
| `PagedSwitchLinear` / `ExpertGroup` | Drop-in replacements for mlx-lm's `SwitchLinear`/`QuantizedSwitchLinear`. Identical `gather_qmm`/`gather_mm` computation with indices remapped into the gathered stack. Router, top-k, GLU, shared experts: untouched upstream code. |
| `attach_paging` (surgery) | Structural detection — any leaf module whose weight has a leading expert dimension. Zero hardcoded architecture names. Dense models: returns None, model runs unmodified. |
| `SparsifyEngine` | Lazy load → surgery → materialize backbone only. Auto RAM budget (half of measured free RAM, 1 GiB floor) unless pinned. Streaming telemetry: RSS, active/peak Metal memory, cache hits/misses/evictions, SSD bytes and read latency. |

Removed: the entire prototype layer (`registry`, `MoeCache`, `storage`,
`loader` with its top-1-unquantized placeholder FFN, `patcher`, `sharder`,
offline `optimizer` re-sharding, analyzers, dashboard, `runtime`/`prototype`
CLI groups and their tests). `pull` is now idempotent (`--force` to
re-download) and no longer rewrites model directories.

## Correctness (success criterion: output identical to full-RAM)

- **Golden end-to-end:** OLMoE-1B-7B-0125 4-bit, paged with a 1 GB expert
  budget (evictions active throughout), reproduces unmodified mlx-lm
  full-RAM output **exactly**, token for token. `tests/test_e2e_golden.py`.
- **Module-level:** paged projections are **bit-identical**
  (`mx.array_equal`) to the full-tensor `gather_qmm` reference across
  decode/repeated/prefill index patterns. `tests/test_paging.py`.
- **Dense passthrough:** Llama-3.2-1B output byte-identical to unmodified
  mlx-lm; paging correctly refuses to attach.
- **Store:** pread slices bit-identical to MLX's own reader on real shards.
- Mixtral 8x7B cannot be baselined un-paged on this machine (26.3 GB > 16 GB
  RAM — the point of the project); its correctness evidence is the verified
  identity of every layer the runtime touches plus coherent, factually
  correct generations.

## Memory (success criterion: bounded by configured budget)

| Model (4-bit) | On disk | Full-RAM baseline | Sparsify RSS | Expert budget | Cache hit rate |
|---|---|---|---|---|---|
| OLMoE-1B-7B | 3.9 GB | **3.89 GB** active (measured) | **1.34 GB** active | 1 GB | 83 % |
| Qwen3-30B-A3B | 16.3 GB | ≥ 16.3 GB (*derived from weight bytes; not runnable un-paged on this machine*) | **4.15 GB** | 3 GB | 77 % |
| Mixtral 8x7B | 26.3 GB | ≥ 26.3 GB (*derived; not runnable*) | **3.33 GB** flat across all 25 tokens | 2.97 GB (auto) | 67 % overall |

Peak Metal memory for Mixtral: 4.91 GB (includes transient per-token
gather stacks). Cache residency tracked at 3.17 GB vs 2.97 GB budget — the
documented in-flight protection overshoot (~1 block's experts), visible in
telemetry.

## Throughput — the honest bottleneck

Decode is SSD-bound whenever the per-token expert working set exceeds the
cache budget:

- Mixtral, 3 GB budget: **6.3 GB SSD reads per decode token → ~25 s/token**.
  Total run: 189.9 GB read for 25 tokens, 0.38 GB/s effective.
- Root cause (measured, classic): layer-sequential access over a 25.4 GB
  expert set with a 3 GB LRU cache → each layer's experts are evicted
  before the next token revisits them → ~0 % *decode* hit rate. The 67 %
  overall rate is prefill reuse.
- Qwen3-30B (2.7 MB experts, 128-way routing) fares better: 77 % hits,
  ~0.2 tok/s cold.

This is not an implementation bug; it is the next research/engineering
frontier, and it is exactly measurable with the telemetry now in place.

## Remaining assumptions blocking "any size on any machine"

1. **Backbone + one MoE block's active experts must fit in RAM.** True for
   every current open MoE; would break on e.g. enormous shared experts.
2. **Expert tensors must be individually addressable on disk** (safetensors
   stacked or per-expert). GGUF is not yet wired into the store.
3. **Prefill reads the full routed expert set once** (up to the whole model
   for long prompts). Unavoidable without speculative routing; costs
   minutes on USB SSDs for Mixtral-class models.
4. **Throughput** scales with `miss_bytes_per_token / SSD_bandwidth`; on a
   16 GB machine with a USB SSD, Mixtral decode is ~25 s/token today.

## Roadmap — Milestone 002

1. **Cyclic-aware eviction:** LRU is provably worst-case for layer-cyclic
   access. Per-layer budget partitions (cache `budget/n_layers` per block)
   or MRU/CLOCK hybrids turn 0 % decode hits into `budget/working_set`.
2. **Async prefetch pipeline:** while block L computes, read block L+1's
   likely experts on a background thread (router-of-L+1 isn't known yet,
   but expert reuse across tokens is — start with "previous token's
   experts" and measure). Overlaps SSD latency with compute; biggest
   single win available.
3. **Read coalescing + parallel preads:** adjacent expert ids in a stacked
   tensor are contiguous on disk; a single 200 MB read beats seven 30 MB
   reads on USB SSDs. Measure per-device.
4. **Routing-trace telemetry:** persist (layer, expert) traces per token to
   ground prefetch policy in measured locality rather than intuition.
5. **GGUF store backend** behind the same `ExpertStore` interface.
6. **KV-cache persistence across turns** (currently re-prefills each turn).
