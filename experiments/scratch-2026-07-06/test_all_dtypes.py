import mlx.core as mx
from safetensors import safe_open
from pathlib import Path

path = Path("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit/model-00006-of-00013.safetensors")

with safe_open(path, framework="numpy") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        m = mx.array(t)
        print(f"{k}: {m.dtype}")
