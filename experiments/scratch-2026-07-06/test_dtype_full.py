import mlx.core as mx
from safetensors import safe_open
from pathlib import Path
import mlx_lm

model_path = Path("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit")
print("Loading model...")
model, _ = mlx_lm.load(str(model_path), lazy=True)
layer = model.model.layers[0]

prefix = "model.layers.0."
layer_weights = {}

with safe_open(model_path / "model-00006-of-00013.safetensors", framework="numpy") as f:
    for k in f.keys():
        if k.startswith(prefix):
            rel_k = k[len(prefix):]
            parts = rel_k.split(".")
            curr = layer_weights
            for part in parts[:-1]:
                curr = curr.setdefault(part, {})
            curr[parts[-1]] = mx.array(f.get_tensor(k))

layer.update(layer_weights)

x = mx.zeros((1, 1, 4096))
inds = mx.zeros((1, 1), dtype=mx.uint32)
y = layer.block_sparse_moe.switch_mlp(x, inds)
mx.eval(y)
print("Evaluated successfully!")
