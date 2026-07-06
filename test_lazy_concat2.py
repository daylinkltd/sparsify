import mlx.core as mx
import time

from mlx_lm.utils import load
model, _ = load("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit", lazy=True)
gate = model.layers[0].block_sparse_moe.switch_mlp.gate_proj.weight

# Test eval 1
t0 = time.time()
concat1 = mx.concatenate([gate[0:1], gate[5:6]], axis=0)
mx.eval(concat1)
print(f"Eval 1 took {time.time()-t0:.4f}s")

# Test eval 2 (different experts)
t0 = time.time()
concat2 = mx.concatenate([gate[2:3], gate[7:8]], axis=0)
mx.eval(concat2)
print(f"Eval 2 took {time.time()-t0:.4f}s")

# Test eval 3 (same experts)
t0 = time.time()
concat3 = mx.concatenate([gate[0:1], gate[5:6]], axis=0)
mx.eval(concat3)
print(f"Eval 3 took {time.time()-t0:.4f}s")
