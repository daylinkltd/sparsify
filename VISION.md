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

**Dense models are integrated via Activation Sparsity (Roadmap Upgrade).**
While dense models (like Llama 3.3 70B) are structurally dense, they exhibit high **activation sparsity** at the neuron level (only 10% to 20% of feed-forward network neurons have non-zero activations for any given token). Sparsify leverages this by predicting active neurons and paging only their corresponding weights, turning dense execution into dynamic MoE-style paging to achieve high inference speeds in limited RAM.

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

### Current Implementation Status (2026-07-13, `sparsify.paging` & `sparsify.paging_torch`)

Nothing below is simulated. Every number comes from a real run on the dev
machines (16 GB M-series Mac for MLX and Windows/Linux for PyTorch/CUDA, models on NVMe and USB SSDs).

| Component | Status | Notes |
|---|---|---|
| Expert paging | **REAL** | `PagedSwitchLinear` (MLX) and `PagedPyTorchLinear` (PyTorch) page per-expert weight slices from safetensors on demand |
| Output correctness | **VERIFIED** | OLMoE-1B-7B paged (1 GB budget, evictions active) reproduces full-RAM outputs **exactly** (token-for-token matches on both MLX and PyTorch) |
| Memory bounding | **MEASURED** | Byte-budgeted LRU cache; Qwen3-30B-A3B (16.3 GB experts) ran in 4.15 GB RSS with a 3 GB expert budget |
| Cross-Platform Support | **SHIPPED** | PyTorch/CUDA/MPS backend fully integrated; enables same paging architecture on Windows, Linux, and Intel Macs |
| SSD reads | **MEASURED** | One `pread` per expert tensor slice; bytes and latency counted per read |
| Dense-model passthrough | **VERIFIED** | Llama-3.2-1B output byte-identical to unmodified base model; no paging attached |
| Throughput | **MEASURED** | SSD-bound: scales from **1.8 tok/s** on USB SSD to **8.5–11.0 tok/s** on fast internal NVMe drives |
| Prefetching | **NOT BUILT** | Removed with the research prototype; to be rebuilt against real routing traces |

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
