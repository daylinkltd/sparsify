import safetensors
from safetensors import safe_open
import mlx.core as mx
import numpy as np
import time

with safe_open("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit/model-00001-of-00013.safetensors", framework="numpy") as f:
    t = f.get_slice("model.layers.8.self_attn.o_proj.weight")
    
    t0 = time.time()
    for i in range(10):
        s = t[0:1]
        x = mx.array(s)
        mx.eval(x)
    print("mx.array(s) numpy took:", time.time() - t0)

    t0 = time.time()
    for i in range(10):
        s = t[0:1]
        x = mx.array(np.ascontiguousarray(s))
        mx.eval(x)
    print("mx.array(np.ascontiguousarray(s)) took:", time.time() - t0)
