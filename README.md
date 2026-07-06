# Sparsify

**Inference memory profiler and optimization research framework.**

Sparsify exists to answer one question:

> Can we dynamically discover and execute only the minimal subset of computation required to produce a given answer — while preserving model quality?

## What Sparsify Does (V1)

Sparsify V1 is a **research instrument** — not a production runtime. It provides:

1. **Model Profiler** — Static analysis of model memory breakdown (weights, KV cache estimates, per-layer decomposition)
2. **Inference Profiler** — Runtime measurement of memory, latency, and per-layer execution (coming Sprint 2)
3. **Benchmark Framework** — Quality and performance measurement (coming Sprint 3)
4. **Experiment Framework** — Controlled experiments like layer skipping and head pruning (coming Sprint 4)

## Quick Start

```bash
# Install with uv
uv pip install -e ".[dev]"

# Profile a GGUF model
sparsify profile-model path/to/model.gguf

# Show system info
sparsify info

# JSON output
sparsify profile-model path/to/model.gguf --json
```

## Supported Formats

- **GGUF** — Full metadata extraction + per-tensor memory analysis
- **Safetensors** — Tensor-level memory analysis

## Supported Backends (Research Profiling)

| Backend | Status | Profiling Depth |
|---------|--------|-----------------|
| MLX | Primary | Per-layer hooks, attention weights, Metal memory |
| llama.cpp | Secondary | Aggregate timings, KV cache metrics |
| Ollama | Planned | API-level metrics |
| vLLM | Deferred | GPU-focused, Phase 3+ |

## Architecture

```
Application
  → Sparsify (profiling harness + experiment runner)
    → Inference Backend (MLX, llama.cpp, ...)
      → Model (GGUF, Safetensors)
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[all]"

# Run tests
pytest

# Lint
ruff check src/
```

## Project Structure

```
src/sparsify/
├── cli.py              # CLI entry point
├── backends/           # Backend adapters (MLX, llama.cpp)
├── profiler/           # Profiling subsystem
├── benchmarks/         # Benchmark framework
├── experiments/        # Experiment framework
├── storage/            # SQLite persistence
├── visualization/      # Charts and export
└── utils/              # GGUF reader, config, helpers
```

## Philosophy

1. Measure before optimizing.
2. Build observability before optimization.
3. Validate hypotheses with experiments.
4. Favor incremental breakthroughs over theoretical perfection.

## License

MIT
