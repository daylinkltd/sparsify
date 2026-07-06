from pathlib import Path
from mlx_lm import load
model, tokenizer = load("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit", lazy=True)
layer = model.model.layers[0]
moe = layer.block_sparse_moe.switch_mlp
print("gate_proj weight dtype:", moe.gate_proj.weight.dtype)
print("gate_proj scales dtype:", moe.gate_proj.scales.dtype)
print("gate_proj biases dtype:", moe.gate_proj.biases.dtype)
