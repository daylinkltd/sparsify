"""Decode-throughput benchmark: fixed prompt, fixed budget, measured splits.

Usage: python experiments/bench_decode.py <model> [budget_gb] [n_tokens]
Reports tok/s and where the time went (SSD I/O vs everything else).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparsify.runtime.model_registry import resolve_local  # noqa: E402
from sparsify.runtime.chat_generation import SparsifyEngine  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
n_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 30

hf_id, path = resolve_local(model)
print(f"model={hf_id} budget={budget}GB tokens={n_tokens}")

engine = SparsifyEngine(path, max_tokens=n_tokens, memory_limit_gb=budget)
store = engine.paging.store
cache = engine.paging.cache

# Turn 1 (prefill-heavy, warms the cache) — not measured.
for _ in engine.chat_stream([{"role": "user", "content": "hey"}], max_tokens=8):
    pass

# Steady-state decode measurement: clock starts at the FIRST generated
# token, so prompt prefill is excluded from all per-token numbers.
io0 = reads0 = bytes0 = h0 = m0 = t0 = None
n = 0
for _text, tel in engine.chat_stream(
    [{"role": "user", "content": "Write a short paragraph about the ocean."}],
    max_tokens=n_tokens,
):
    if t0 is None:
        io0, reads0, bytes0 = store.read_seconds, store.reads, store.bytes_read
        h0, m0 = cache.hits, cache.misses
        t0 = time.perf_counter()
        continue
    n += 1
elapsed = time.perf_counter() - t0

io = store.read_seconds - io0
reads = store.reads - reads0
gb = (store.bytes_read - bytes0) / 1e9
hits, misses = cache.hits - h0, cache.misses - m0

print(f"\ntokens            : {n} in {elapsed:.1f}s  ->  {n/elapsed:.2f} tok/s")
print(f"SSD I/O (thread-sum): {io:.1f}s over parallel workers — "
      f"{gb:.2f} GB in {reads} preads")
print(f"effective read bw : {gb/elapsed:.2f} GB/s of wall time")
print(f"expert requests   : {hits+misses} (hit {hits} / miss {misses}, "
      f"{hits/(hits+misses)*100 if hits+misses else 0:.1f}%)")
print(f"per token         : {(hits+misses)/n:.0f} expert lookups, "
      f"{misses/n:.1f} misses, {gb/n*1000:.0f} MB read, {reads/n:.0f} preads")
