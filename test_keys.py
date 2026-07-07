import json
with open("models/mlx-community--Mixtral-8x7B-Instruct-v0.1-4bit/model.safetensors.index.json") as f:
    d = json.load(f)
keys = list(d.get("weight_map", {}).keys())
moe_keys = [k for k in keys if "block_sparse_moe" in k][:10]
print("Moe keys sample:", moe_keys)
