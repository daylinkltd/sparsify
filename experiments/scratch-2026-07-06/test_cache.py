import time
from pathlib import Path
import mlx.core as mx
import mlx_lm

t0 = time.time()
model, tokenizer = mlx_lm.load('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit', lazy=True)

layer = model.model.layers[0]
gate = layer.block_sparse_moe.switch_mlp.gate_proj
full_tensor = gate.weight

# Cache experts
cached_experts = {}

t0 = time.time()
# Load expert 0
e0 = full_tensor[0:1]
mx.eval(e0)
cached_experts[0] = e0
print(f"Eval expert 0 took: {time.time()-t0:.2f}s")

t0 = time.time()
# Load expert 2
e2 = full_tensor[2:3]
mx.eval(e2)
cached_experts[2] = e2
print(f"Eval expert 2 took: {time.time()-t0:.2f}s")

# Now concatenate in-memory tensors
t0 = time.time()
concat = mx.concatenate([cached_experts[0], cached_experts[2]], axis=0)
mx.eval(concat)
print(f"Concat in-memory took: {time.time()-t0:.2f}s")
