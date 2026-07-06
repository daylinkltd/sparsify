import time
import mlx.core as mx
from pathlib import Path
from sparsify.runtime.chat_generation import SparsifyEngine

print("Loading engine...")
engine = SparsifyEngine(Path('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit'), memory_limit_gb=4)
print("Engine loaded.")
layer = engine.model.model.layers[0]

# Pre-load shard
print("Loading shard...")
shard = mx.load(str(Path('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit/model-00001-of-00013.safetensors')))
layer_weights = {}
for k, v in shard.items():
    if k.startswith("model.layers.0."):
        rel_k = k[len("model.layers.0."):]
        parts = rel_k.split(".")
        curr = layer_weights
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = v

print("Starting loop...")
t0 = time.time()
for i in range(32):
    t_iter = time.time()
    layer.update(layer_weights)
    print(f"Iter {i} update took {time.time()-t_iter:.4f}s")
print(f"32 layer.update() took {time.time()-t0:.2f}s")
