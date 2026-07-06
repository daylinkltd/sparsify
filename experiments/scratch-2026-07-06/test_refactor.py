import mlx.core as mx
from mlx_lm.utils import load

model, _ = load("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit", lazy=True)

# Save original weights
original_experts = {}
for i, layer in enumerate(model.layers):
    if hasattr(layer, "block_sparse_moe"):
        moe = layer.block_sparse_moe.switch_mlp
        original_experts[i] = {
            "gate_proj": {"weight": moe.gate_proj.weight} if hasattr(moe.gate_proj, "weight") else {},
            "up_proj": {"weight": moe.up_proj.weight} if hasattr(moe.up_proj, "weight") else {},
            "down_proj": {"weight": moe.down_proj.weight} if hasattr(moe.down_proj, "weight") else {}
        }
        if hasattr(moe.gate_proj, "scales"): original_experts[i]["gate_proj"]["scales"] = moe.gate_proj.scales
        if hasattr(moe.gate_proj, "biases"): original_experts[i]["gate_proj"]["biases"] = moe.gate_proj.biases
        
print(f"Saved {len(original_experts)} layers of experts")
