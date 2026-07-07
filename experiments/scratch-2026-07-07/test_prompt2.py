import mlx_lm
import mlx.core as mx
import time
import os

from sparsify.runtime.chat_generation import SparsifyEngine

print("Loading...")
t0 = time.time()
engine = SparsifyEngine("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit")
print(f"Loaded in {time.time()-t0:.2f}s")

prompt = "Hello, what is your name?"
print(f"Generating for prompt '{prompt}'...")

# Use engine's own generator
generator = engine.generate(prompt=prompt)
