import time
from pathlib import Path
from sparsify.runtime.chat_generation import SparsifyEngine
print("Loading engine...")
t0 = time.time()
engine = SparsifyEngine(Path('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit'), memory_limit_gb=4)
print(f"Loaded in {time.time()-t0:.2f}s")
print("Generating...")
for i, (text, _) in enumerate(engine.generate_stream("hey")):
    print(f"Token {i}: {text!r}")
    if i == 2: break
print("Done")
