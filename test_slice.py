import time
from pathlib import Path
import mlx.core as mx
import mlx_lm

t0 = time.time()
model, tokenizer = mlx_lm.load('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit', lazy=True)
print(f"Loaded in {time.time()-t0:.2f}s")

layer = model.model.layers[0]
expert = layer.block_sparse_moe.experts[0]

t0 = time.time()
# Force a slice evaluation
w = expert.w1.weight
print(f"Weight shape: {w.shape}")
slice_val = w[mx.array([0, 1])]
mx.eval(slice_val)
print(f"Slice eval took {time.time()-t0:.2f}s")

t0 = time.time()
slice_val2 = w[mx.array([2, 3])]
mx.eval(slice_val2)
print(f"Slice eval 2 took {time.time()-t0:.2f}s")
