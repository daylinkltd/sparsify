import os
import mlx.core as mx
import time
from pathlib import Path
from sparsify.runtime.chat_generation import SparsifyEngine

print("Initializing...")
engine = SparsifyEngine(Path("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit"))

import sparsify.runtime.chat_generation
old_call = sparsify.runtime.chat_generation.PagingModelProxy.__call__

def profiled_call(self, inputs, cache=None, input_embeddings=None, **kwargs):
    t0 = time.time()
    res = old_call(self, inputs, cache, input_embeddings, **kwargs)
    print(f"Layer pass took {time.time()-t0:.2f}s")
    return res

sparsify.runtime.chat_generation.PagingModelProxy.__call__ = profiled_call

print("Generating...")
for i, (text, tel) in enumerate(engine.generate_stream("Hello")):
    print(f"Token {i}: {text!r}")
    if i >= 1:
        break
