with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    content = f.read()

old_slice_code = """                                                    weight_name = f"model.layers.{i}.block_sparse_moe.switch_mlp.{proj_name}.{attr}"
                                                    if weight_name in self.weight_map:
                                                        file_path = self.weight_map[weight_name]
                                                        from safetensors import safe_open
                                                        with safe_open(file_path, framework="numpy") as f:
                                                            st_tensor = f.get_slice(weight_name)
                                                            e_slice = st_tensor[e:e+1]
                                                            e_slice_mx = mx.array(e_slice)
                                                            mx.eval(e_slice_mx)
                                                            self.expert_cache[cache_key] = e_slice_mx"""

new_slice_code = """                                                    weight_name = f"model.layers.{i}.block_sparse_moe.switch_mlp.{proj_name}.{attr}"
                                                    if weight_name in self.weight_map:
                                                        file_path = self.weight_map[weight_name]
                                                        if file_path in self.safetensor_handles:
                                                            f = self.safetensor_handles[file_path]
                                                            st_tensor = f.get_slice(weight_name)
                                                            e_slice = st_tensor[e:e+1]
                                                            e_slice_mx = mx.array(e_slice)
                                                            mx.eval(e_slice_mx)
                                                            self.expert_cache[cache_key] = e_slice_mx
                                                        else:
                                                            e_slice_mx = full_tensor[e:e+1]
                                                            mx.eval(e_slice_mx)
                                                            self.expert_cache[cache_key] = e_slice_mx"""

content = content.replace(old_slice_code, new_slice_code)

with open("src/sparsify/runtime/chat_generation.py", "w") as f:
    f.write(content)
