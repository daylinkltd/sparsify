import mlx.core as mx
import time
from mlx_lm.utils import load

print("Loading lazy model...")
t0 = time.time()
model, _ = load("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit", lazy=True)
print(f"Loaded in {time.time()-t0:.4f}s")
print(f"Active mem after load: {mx.metal.get_active_memory() / 1e6} MB")

# Get a block
moe = model.layers[0].block_sparse_moe.switch_mlp
huge_gate = moe.gate_proj.weight

# Swap it out with a small tensor!
expert0 = huge_gate[0:1]
expert5 = huge_gate[5:6]
small_gate = mx.concatenate([expert0, expert5], axis=0)
moe.gate_proj.weight = small_gate

# Now test mx.take on it
inds = mx.array([[0, 1]]) # use mapped indices
taken = mx.take(moe.gate_proj.weight, inds, axis=0)

t0 = time.time()
mx.eval(taken)
print(f"Eval took {time.time()-t0:.4f}s")
print(f"Active mem after eval: {mx.metal.get_active_memory() / 1e6} MB")
