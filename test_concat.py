from mlx_lm import load
import mlx.core as mx

model, _ = load("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit", lazy=True)
layer = model.model.layers[0]
moe = layer.block_sparse_moe.switch_mlp

weight = moe.gate_proj.weight
print("original weight dtype:", weight.dtype)

# Try slicing and concatenation
sliced1 = weight[1:2]
print("sliced1 dtype:", sliced1.dtype)

sliced2 = weight[3:4]
print("sliced2 dtype:", sliced2.dtype)

concat = mx.concatenate([sliced1, sliced2], axis=0)
print("concat dtype:", concat.dtype)

# What if we eval it?
mx.eval(concat)
print("eval concat dtype:", concat.dtype)

