import safetensors
from safetensors import safe_open
import mlx.core as mx

with safe_open("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit/model-00001-of-00013.safetensors", framework="pt") as f:
    t = f.get_slice("model.layers.8.self_attn.o_proj.scales")
    s = t[0:1]
    
    t2 = f.get_slice("model.layers.8.self_attn.o_proj.weight")
    s2 = t2[0:1]

    print("mx array scales:", mx.array(s).dtype)
    print("mx array weight:", mx.array(s2).dtype)
