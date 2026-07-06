with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

# Remove the dummy replacement block before the proxy
dummy_block = """        # Unload all switch_mlp weights from memory initially to save RAM
        dummy = mx.zeros((0,))
        for layer in self.model.model.layers:
            if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "switch_mlp"):
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    mod = getattr(layer.block_sparse_moe.switch_mlp, proj)
                    update_dict = {}
                    if hasattr(mod, "weight"): update_dict["weight"] = dummy
                    if hasattr(mod, "scales"): update_dict["scales"] = dummy
                    if hasattr(mod, "biases"): update_dict["biases"] = dummy
                    if hasattr(mod, "bias"): update_dict["bias"] = dummy
                    getattr(layer.block_sparse_moe.switch_mlp, proj).update(update_dict)
        mx.clear_cache()"""

content = content.replace(dummy_block, "")

# Add it after self.model.model = PagingModelProxy(self.model.model)
proxy_init = "self.model.model = PagingModelProxy(self.model.model)"

new_dummy_block = """        self.model.model = PagingModelProxy(self.model.model)
        
        # Unload all switch_mlp weights from memory initially to save RAM
        dummy = mx.zeros((0,))
        for layer in self.model.model.original_model.layers:
            if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "switch_mlp"):
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    mod = getattr(layer.block_sparse_moe.switch_mlp, proj)
                    update_dict = {}
                    if hasattr(mod, "weight"): update_dict["weight"] = dummy
                    if hasattr(mod, "scales"): update_dict["scales"] = dummy
                    if hasattr(mod, "biases"): update_dict["biases"] = dummy
                    if hasattr(mod, "bias"): update_dict["bias"] = dummy
                    getattr(layer.block_sparse_moe.switch_mlp, proj).update(update_dict)
        mx.clear_cache()"""

content = content.replace(proxy_init, new_dummy_block)

# Clean up debug prints
content = content.replace("print(f'{attr} small_tensor dtype: {small_tensor.dtype} vs {full_tensor.dtype}'); ", "")

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
print("Fixed order!")
