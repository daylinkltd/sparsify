import mlx.core as mx
import time
from pathlib import Path
from sparsify.runtime.chat_generation import SparsifyEngine

print("Loading...")
engine = SparsifyEngine(Path('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit'), memory_limit_gb=4)
print("Generating...")
for text, telemetry in engine.generate_stream("Hello"):
    print(text)
    break
