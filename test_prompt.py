import time
from pathlib import Path
import mlx.core as mx
from sparsify.runtime.chat_generation import SparsifyEngine

print("Loading...")
t0 = time.time()
engine = SparsifyEngine(Path('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit'), memory_limit_gb=4)
print(f"Loaded in {time.time()-t0:.2f}s")

prompt = "Hello"
print(f"Generating for prompt {prompt!r}...")
t0 = time.time()
for text, telemetry in engine.generate_stream(prompt):
    print(f"Time to first token: {time.time()-t0:.2f}s | Token: {text!r}")
    break
