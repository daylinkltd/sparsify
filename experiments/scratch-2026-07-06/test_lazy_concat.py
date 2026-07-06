import mlx.core as mx
import time

print("Loading lazy model...")
t0 = time.time()
from mlx_lm.utils import load
model, _ = load("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit", lazy=True)
gate = model.layers[0].block_sparse_moe.switch_mlp.gate_proj.weight

print(f"Loaded in {time.time()-t0:.4f}s. Gate shape: {gate.shape}")

print("Testing lazy slice concat...")
t0 = time.time()
# simulate taking 2 experts
expert0 = gate[0:1] # shape (1, 14336, 1024)
expert5 = gate[5:6]
concat = mx.concatenate([expert0, expert5], axis=0)
print(f"Concat shape: {concat.shape}")
print(f"Concat took {time.time()-t0:.4f}s")

print("Testing eval...")
t0 = time.time()
mx.eval(concat)
print(f"Eval took {time.time()-t0:.4f}s")
