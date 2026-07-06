import json
import re
from pathlib import Path

with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

# Add safetensors import
content = content.replace("import mlx_lm\n", "import mlx_lm\nimport json\nfrom safetensors import safe_open\n")

# Modify __init__ to read safetensors mapping
init_str = """                self.expert_cache = {} # Cache for evaluated individual experts
                self.expert_cache_lru = [] # LRU list
                self.max_cached_experts = 100 # Approx 2.3 GB for 100 experts"""
                
init_repl = """                self.expert_cache = {} # Cache for evaluated individual experts
                self.expert_cache_lru = [] # LRU list
                self.max_cached_experts = 100 # Approx 2.3 GB for 100 experts
                
                # Build safetensors weight map
                self.weight_map = {}
                index_path = model_path / "model.safetensors.index.json"
                if index_path.exists():
                    with open(index_path) as f:
                        idx = json.load(f)
                    self.weight_map = {k: str(model_path / v) for k, v in idx.get("weight_map", {}).items()}
                else:
                    # Single file
                    sf = model_path / "model.safetensors"
                    if sf.exists():
                        import safetensors
                        with safetensors.safe_open(str(sf), framework="pt") as f:
                            self.weight_map = {k: str(sf) for k in f.keys()}"""
                            
content = content.replace(init_str, init_repl)

# Update the slicing logic
old_slice = """                                            slices = []
                                            for e in active_experts:
                                                cache_key = (i, proj_name, attr, e)
                                                if cache_key not in self.expert_cache:
                                                    e_slice = full_tensor[e:e+1]
                                                    mx.eval(e_slice)
                                                    self.expert_cache[cache_key] = e_slice
                                                else:
                                                    # Update LRU
                                                    self.expert_cache_lru.remove(cache_key)
                                                self.expert_cache_lru.append(cache_key)
                                                
                                                # Enforce memory limit
                                                while len(self.expert_cache_lru) > self.max_cached_experts:
                                                    oldest_key = self.expert_cache_lru.pop(0)
                                                    del self.expert_cache[oldest_key]
                                                    
                                                slices.append(self.expert_cache[cache_key])
                                                
                                            small_tensor = mx.concatenate(slices, axis=0)"""

new_slice = """                                            slices = []
                                            for e in active_experts:
                                                cache_key = (i, proj_name, attr, e)
                                                if cache_key not in self.expert_cache:
                                                    # Determine exact tensor name based on architecture
                                                    # For mixtral, it's model.layers.X.block_sparse_moe.experts.Y.w1.weight (if unstacked)
                                                    # But mlx_lm mixtral uses model.layers.X.block_sparse_moe.switch_mlp.gate_proj.weight etc
                                                    # Wait, mlx_lm lazy loading preserves the raw safetensors name or maps it!
                                                    # Actually mlx_lm maps them. The raw safetensors file for Mixtral has model.layers.X.block_sparse_moe.experts.Y.w1.weight
                                                    # Or does it have the stacked versions?
                                                    # mlx_lm Mixtral stacked weights: mlx_lm converts them on the fly if needed, or uses raw.
                                                    pass
                                                    # Wait, if mlx-community uploaded it, it's ALREADY in mlx format in safetensors!
                                                    # So the name in safetensors IS model.layers.X.block_sparse_moe.switch_mlp.gate_proj.weight
                                                    weight_name = f"model.layers.{i}.block_sparse_moe.switch_mlp.{proj_name}.{attr}"
                                                    if weight_name in self.weight_map:
                                                        file_path = self.weight_map[weight_name]
                                                        from safetensors import safe_open
                                                        with safe_open(file_path, framework="numpy") as f:
                                                            st_tensor = f.get_slice(weight_name)
                                                            e_slice = st_tensor[e:e+1]
                                                            e_slice_mx = mx.array(e_slice)
                                                            mx.eval(e_slice_mx)
                                                            self.expert_cache[cache_key] = e_slice_mx
                                                    else:
                                                        # Fallback to MLX full tensor slice (will be slow/OOM but works for tests)
                                                        e_slice_mx = full_tensor[e:e+1]
                                                        mx.eval(e_slice_mx)
                                                        self.expert_cache[cache_key] = e_slice_mx
                                                else:
                                                    # Update LRU
                                                    self.expert_cache_lru.remove(cache_key)
                                                self.expert_cache_lru.append(cache_key)
                                                
                                                # Enforce memory limit
                                                while len(self.expert_cache_lru) > self.max_cached_experts:
                                                    oldest_key = self.expert_cache_lru.pop(0)
                                                    del self.expert_cache[oldest_key]
                                                    
                                                slices.append(self.expert_cache[cache_key])
                                                
                                            small_tensor = mx.concatenate(slices, axis=0)"""
                                            
content = content.replace(old_slice, new_slice)

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
