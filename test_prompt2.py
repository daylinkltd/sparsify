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

tokens = []
for i in range(10):
    t_start = time.time()
    try:
        word, metrics = next(generator)
    except StopIteration:
        break
    t_end = time.time()
    print(f"Token {i}: '{word}' (took {t_end-t_start:.2f}s)")
