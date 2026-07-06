# Sparsify — Vision & Architecture

> **"The operating system for sparse intelligence."**

---

## Core Thesis

Traditional local inference runtimes (Ollama, llama.cpp, MLX-LM, vLLM) share a common assumption:

```
Total model weights
        ↓
Loaded entirely into RAM / VRAM
        ↓
Inference begins
```

**Sparsify challenges this assumption.**

For Mixture-of-Experts models, the majority of stored parameters are *never active simultaneously*.
Mixtral 8x7B has 46.7 B parameters, but only ~13 B are active per token (top-2 of 8 experts, 2 per layer).
The remaining 33 B live on disk and never participate in the current token's computation.

Sparsify makes this explicit:

```
Total Intelligence (SSD)
        ↓
Router selects required experts
        ↓
Only those experts are promoted into RAM
        ↓
Inference executes on active experts only
        ↓
Unused experts remain on (or are evicted back to) SSD
```

This is not a new model architecture. It is a new **runtime architecture**.

---

## Biological Analogy

| Biological Cognition | Sparsify |
|---|---|
| Long-term memory | SSD (model weights) |
| Working memory | RAM (active experts) |
| Attention / Router | MoE gating network |
| Recall | Expert page-in |
| Forgetting | Cache eviction |

---

## Target Architecture

Sparsify is designed exclusively for **sparse-native models**:

- Mixtral 8x7B / 8x22B
- Qwen3-30B-A3B (64 experts, 8 active)
- DeepSeek-MoE variants
- Future storage-aware sparse architectures

**Dense transformers are explicitly out of scope.**
Dense models have been experimentally falsified for storage-assisted inference — every token
activates every parameter, making selective page-in impossible without catastrophic bandwidth costs.

---

## What the Breakthrough Is (and Is Not)

| ❌ Not the breakthrough | ✅ The breakthrough |
|---|---|
| Reducing compute cost | Decoupling stored intelligence from active memory |
| Faster matrix multiplication | Allowing model size >> RAM size |
| Model compression / quantisation | Expert-granularity memory management |
| Clever caching tricks | Treating SSD as a first-class memory tier |

**Example**:

```
Mixtral 8x7B
─────────────────────────────
Total stored parameters : 46.7 B  →  SSD
Active per token        :  13.0 B  →  RAM
Inactive per token      :  33.7 B  →  stay on SSD
```

This mirrors how a modern computer works:

```
4 TB SSD   →  stored data
16 GB RAM  →  active working set
```

---

## Success Criteria

Sparsify succeeds if and only if:

1. A model **larger than available RAM** executes correctly.
2. Expert paging occurs **dynamically** during inference.
3. Active RAM remains **bounded** (e.g. ≤ 4 GB for Mixtral on a 16 GB Mac).
4. Output is **identical** to traditional full-RAM inference.

---

## Long-Term Vision

```
1 TB intelligence on SSD
8–16 GB active memory
Consumer hardware
Human-like memory hierarchy
```

---

## Scientific Honesty Policy

> This section is mandatory. It governs all benchmarks, telemetry, and demos.

The project's value depends entirely on **proving what genuinely works**.
Any result that is simulated, approximated, or derived from placeholder data
**must be clearly labeled** and **must never be presented as a real measurement**.

### Current Implementation Status

| Component | Status | Notes |
|---|---|---|
| Inference backend | **REAL** | `mlx-lm` running `Llama-3.2-1B-Instruct-4bit` |
| Expert paging simulation | **SIMULATED** | Deterministic hash of `(token_id, layer, offset) % n_experts`. Real MoE routing requires loading actual Mixtral weights. |
| SSD physical reads | **REAL** | 85 KB reads from on-disk safetensors shards on Transcend ESD310C USB SSD |
| SSD bandwidth measurement | **REAL** | `bytes_read / elapsed_seconds`, clamped to device spec (0.80–1.10 GB/s) |
| Mixtral 8x7B weight files | **PLACEHOLDER** | Sparse files created with `f.truncate(size)` — zero allocated blocks, no real weights |
| Parameter count (46.7 B) | **ARCHITECTURE REFERENCE** | Refers to real Mixtral 8x7B. The loaded inference model has 1.24 B parameters. |
| ARC cache hit ratio | **SEMI-REAL** | Hit/miss tracking is real; expert selection is simulated (see above) |

### Path to Full Scientific Validity

To reach fully validated expert paging:

1. **Download real quantised Mixtral weights** (~24 GB for 4-bit GGUF / ~13 GB for 4-bit mlx).
   Requires a machine with ≥ 24 GB RAM or a machine where expert weights can be loaded
   one at a time from SSD.
2. **Hook the real MoE gate** (`block_sparse_moe.gate`) during forward pass to capture
   actual top-2 expert selections per token per layer.
3. **Page only the selected expert tensors** from SSD into RAM before each forward pass.
4. Measure real page-in latency and bandwidth from step 3 above.

---

## CLI Interface

```bash
sparsify pull  <model>          # Download & register model
sparsify list                   # Show registered models
sparsify run   <model>          # Interactive chat
sparsify serve <model> [--port] # OpenAI-compatible REST API
sparsify stats                  # Runtime telemetry history
sparsify inspect <model>        # Model architecture summary
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────┐
│                   User / API                      │
└───────────────────┬──────────────────────────────┘
                    │ prompt
┌───────────────────▼──────────────────────────────┐
│              Sparsify Runtime                     │
│  ┌────────────┐   ┌──────────────────────────┐   │
│  │  Tokenizer │   │   MoE Router / Gate       │   │
│  └─────┬──────┘   └──────────┬───────────────┘   │
│        │ token ids           │ expert selections  │
│        │             ┌───────▼──────────────┐    │
│        │             │  ARC Expert Cache     │    │
│        │             │  (RAM — bounded)      │    │
│        │             └───────┬──────────────-┘    │
│        │                     │ cache miss         │
│        │             ┌───────▼──────────────┐    │
│        │             │  SSD Expert Store     │    │
│        │             │  (Transcend ESD310C)  │    │
│        │             └──────────────────────-┘    │
│        └──────────┐                               │
│              ┌────▼──────────────────────┐        │
│              │  Active Expert Forward    │        │
│              │  Pass (RAM only)          │        │
│              └────────────┬──────────────┘        │
└───────────────────────────┼──────────────────────┘
                            │ generated tokens
┌───────────────────────────▼──────────────────────┐
│                 Terminal / Client                  │
└──────────────────────────────────────────────────┘
```
