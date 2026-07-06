import mlx.core as mx
import time
import os

print(f"PID: {os.getpid()}")

t0 = time.time()
weights = mx.load("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit/model-00001-of-00005.safetensors", mmap=True)
print(f"Loaded in {time.time()-t0:.4f}s")

# Get a huge tensor
huge = weights["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"]
print(f"Huge shape: {huge.shape}")

# Active memory should be ~0
print(f"Active mem before eval: {mx.metal.get_active_memory() / 1e6} MB")

t0 = time.time()
slice_0 = huge[0]
slice_0_sum = mx.sum(slice_0)
mx.eval(slice_0_sum)
print(f"Evaluated slice 0 in {time.time()-t0:.4f}s")
print(f"Active mem after eval: {mx.metal.get_active_memory() / 1e6} MB")

