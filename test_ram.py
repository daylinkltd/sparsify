import time
from pathlib import Path
import mlx.core as mx

# Print memory before loading
print(f"Memory before load: {mx.metal.get_active_memory() / 1e9:.2f} GB")

from sparsify.runtime.chat_generation import SparsifyEngine

print("Loading engine...")
t0 = time.time()
engine = SparsifyEngine(Path('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit'), memory_limit_gb=4)
print(f"Engine loaded in {time.time()-t0:.2f}s")
print(f"Memory after load: {mx.metal.get_active_memory() / 1e9:.2f} GB")

print("Generating 10 tokens...")
for i, (text, telemetry) in enumerate(engine.generate_stream("What is 2+2?")):
    print(f"Token {i}: {text!r}")
    if i == 9: break

print(f"Memory after generation: {mx.metal.get_active_memory() / 1e9:.2f} GB")
print(f"Peak memory during generation: {mx.metal.get_peak_memory() / 1e9:.2f} GB")
