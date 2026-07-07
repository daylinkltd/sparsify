# Contributing to Sparsify

Thanks for looking. The codebase is deliberately small — `src/sparsify/paging/`
(the runtime core) is ~600 lines and readable in one sitting.

## Ground rules

1. **Measure everything.** Every performance claim in a PR must come from a
   real run, with the command that produced it. `experiments/bench_decode.py`
   is the standard decode benchmark.
2. **Label everything.** Numbers are *measured*, *derived*, or *estimated* —
   say which. Simulated benchmarks are banned (see `VISION.md`).
3. **Correctness is non-negotiable.** The golden tests
   (`pytest -m e2e tests/test_e2e_golden.py`) define the contract: paged
   output must equal full-RAM output token-for-token. If your change flips a
   golden test, the change is wrong (or the test's baseline needs a documented
   argument — floating-point kernel-shape effects are the known exception).
4. **Negative results are contributions.** We keep measured dead ends in the
   repo (e.g. partial residency, non-LRU eviction policies) so nobody
   re-walks them. If you tried something and it lost, PR the numbers.

## Where help matters most

- **Verify catalog models** — pull an untested MoE from `sparsify models`,
  run the golden tests against it, post the memory/tok/s numbers in an issue.
- **Async expert prefetch** — overlap SSD reads with compute. Design against
  the real routing traces in `experiments/` (129k-access dataset + policy
  simulator included).
- **CUDA/Linux backend** — `runtime/backend.py` is the seam; the paging core
  never touches MLX directly except in `store.wrap_raw` and the module layer.
- **GGUF store backend** — same `ExpertStore` interface, different container.

## Dev setup

```bash
git clone https://github.com/daylinkltd/sparsify && cd sparsify
python3 -m venv .venv && .venv/bin/pip install -e ".[all]" huggingface_hub
.venv/bin/python -m pytest tests/ -m "not e2e"     # fast suite
```

Model-dependent tests skip automatically when the model isn't on disk.

## PR checklist

- [ ] `pytest -m "not e2e"` passes
- [ ] golden tests pass if you touched the inference path
- [ ] new claims carry their measurement commands
- [ ] no per-architecture branches where structural detection can work
