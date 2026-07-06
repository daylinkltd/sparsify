import time
from pathlib import Path
import mlx.core as mx
import mlx_lm

t0 = time.time()
model, tokenizer = mlx_lm.load('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit', lazy=True)
print(f"Loaded in {time.time()-t0:.2f}s")

layer = model.model.layers[0]
gate = layer.block_sparse_moe.switch_mlp.gate_proj

t0 = time.time()
full_tensor = gate.weight
print(f"Full tensor: {full_tensor.shape} {full_tensor.dtype}")

# Slice 1 expert
t1 = time.time()
small_tensor = full_tensor[0:1]
mx.eval(small_tensor)
print(f"Eval single slice took: {time.time()-t1:.2f}s")

# Concatenate 2 experts
t2 = time.time()
concat_tensor = mx.concatenate([full_tensor[0:1], full_tensor[2:3]], axis=0)
mx.eval(concat_tensor)
print(f"Eval concat slice took: {time.time()-t2:.2f}s")

# Concatenate 2 experts AGAIN
t3 = time.time()
concat_tensor2 = mx.concatenate([full_tensor[0:1], full_tensor[2:3]], axis=0)
mx.eval(concat_tensor2)
print(f"Eval concat slice 2 took: {time.time()-t3:.2f}s")
