import time
from safetensors import safe_open
import mlx.core as mx
import mlx_lm
import json

model, tokenizer = mlx_lm.load('models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit', lazy=True)
full_tensor = model.model.layers[0].block_sparse_moe.switch_mlp.gate_proj.weight

path = 'models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit/model.safetensors.index.json'
with open(path) as f:
    index = json.load(f)
weight_name = "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"
file_name = index['weight_map'][weight_name]

with safe_open(f'models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit/{file_name}', framework="numpy") as f:
    st_tensor = f.get_slice(weight_name)
    np_slice = st_tensor[0:1]
    
mx_slice = mx.array(np_slice)
print("Safetensors shape:", mx_slice.shape, mx_slice.dtype)

mlx_slice = full_tensor[0:1]
mx.eval(mlx_slice)
print("MLX shape:", mlx_slice.shape, mlx_slice.dtype)

diff = mx.sum(mx.abs(mx_slice.astype(mx.float32) - mlx_slice.astype(mx.float32))).item()
print("Difference:", diff)
