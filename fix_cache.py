with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

# Inside __init__, add self.expert_cache = {}
init_str = """                self.original_model = model
                
                # Save original lazy experts"""
init_repl = """                self.original_model = model
                
                # Save original lazy experts
                self.expert_cache = {} # Cache for evaluated individual experts
                self.expert_cache_lru = [] # LRU list
                self.max_cached_experts = 100 # Approx 2.3 GB for 100 experts"""
content = content.replace(init_str, init_repl)

# Inside __call__, update the small_tensor creation
# Old code:
# small_tensor = mx.concatenate([full_tensor[e:e+1] for e in active_experts], axis=0)

old_slice = """                                            small_tensor = mx.concatenate([full_tensor[e:e+1] for e in active_experts], axis=0)"""
new_slice = """                                            slices = []
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
content = content.replace(old_slice, new_slice)

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
