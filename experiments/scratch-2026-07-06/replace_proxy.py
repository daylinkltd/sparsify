with open("src/sparsify/runtime/chat_generation.py", "r") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "class PagingModelProxy:" in line:
        start_idx = i
    if "# Replace the inner model with our paging proxy" in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    new_proxy = """        class PagingModelProxy:
            \"\"\"
            A proxy wrapper that intercepts model calls and dynamically slices the lazy expert 
            tensors so that only the active experts are evaluated by Metal.
            \"\"\"
            def __init__(self, model):
                import mlx.core as mx
                self.original_model = model
                
                # Save original lazy experts
                self.original_experts = {}
                for i, layer in enumerate(self.original_model.layers):
                    if hasattr(layer, "block_sparse_moe"):
                        moe = layer.block_sparse_moe.switch_mlp
                        self.original_experts[i] = {}
                        for proj_name in ["gate_proj", "up_proj", "down_proj", "fc1", "fc2"]:
                            if hasattr(moe, proj_name):
                                mod = getattr(moe, proj_name)
                                self.original_experts[i][proj_name] = {}
                                for attr in ["weight", "scales", "biases", "bias"]:
                                    if hasattr(mod, attr):
                                        self.original_experts[i][proj_name][attr] = getattr(mod, attr)

            def __getattr__(self, name):
                return getattr(self.original_model, name)
                
            def __call__(self, inputs, cache=None, input_embeddings=None, **kwargs):
                import time
                import mlx.core as mx
                if input_embeddings is None:
                    h = self.original_model.embed_tokens(inputs)
                else:
                    h = input_embeddings
                    
                mask = None
                if h.shape[1] > 1:
                    from mlx_lm.models.base import create_attention_mask
                    if cache is not None:
                        mask = create_attention_mask(h, cache[0])
                    else:
                        mask = create_attention_mask(h, None)

                if cache is None:
                    cache = [None] * len(self.original_model.layers)

                for i, layer in enumerate(self.original_model.layers):
                    c = cache[i]
                    
                    moe_block = None
                    if hasattr(layer, "block_sparse_moe"):
                        moe_block = layer.block_sparse_moe
                    elif hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                        moe_block = layer.mlp
                        
                    original_switch_mlp_call = None
                    
                    if moe_block and hasattr(moe_block, "gate"):
                        # 1. Gate forward pass to find active experts
                        gate_out = moe_block.gate(h)
                        
                        if hasattr(moe_block, "num_experts_per_tok") and moe_block.num_experts_per_tok > 1:
                            inds = mx.argpartition(-gate_out, kth=moe_block.num_experts_per_tok - 1, axis=-1)[..., :moe_block.num_experts_per_tok]
                        else:
                            inds = mx.argmax(gate_out, axis=-1)
                            
                        # Force evaluation of inds to know which experts to slice
                        mx.eval(inds)
                        active_experts = list(set(inds.flatten().tolist()))
                        
                        # 2. Map indices
                        import numpy as np
                        max_expert_id = getattr(moe_block, "num_experts", 8)
                        mapping = np.zeros(max_expert_id, dtype=np.int32)
                        for idx, exp_idx in enumerate(active_experts):
                            mapping[exp_idx] = idx
                        mapping_mx = mx.array(mapping)
                        
                        if hasattr(moe_block, "switch_mlp") and hasattr(moe_block.switch_mlp, "__call__"):
                            original_switch_mlp_call = moe_block.switch_mlp.__call__
                            
                            def mapped_switch_mlp_call(x, inds_args, *args, **kw):
                                mapped_inds = mapping_mx[inds_args]
                                return original_switch_mlp_call(x, mapped_inds, *args, **kw)
                                
                            moe_block.switch_mlp.__call__ = mapped_switch_mlp_call
                            
                            # 3. Swap the huge lazy tensors with small concatenated lazy slices
                            orig = self.original_experts[i]
                            for proj_name, proj_attrs in orig.items():
                                if hasattr(moe_block.switch_mlp, proj_name):
                                    mod = getattr(moe_block.switch_mlp, proj_name)
                                    for attr, full_tensor in proj_attrs.items():
                                        if hasattr(mod, attr):
                                            small_tensor = mx.concatenate([full_tensor[e:e+1] for e in active_experts], axis=0)
                                            setattr(mod, attr, small_tensor)
                                    
                    # 4. Evaluate layer (builds the graph using the small_tensor references)
                    h = layer(h, mask, c)
                    
                    # 5. Restore full lazy tensors immediately so the graph is valid for next tokens
                    if moe_block and hasattr(moe_block, "switch_mlp"):
                        if original_switch_mlp_call is not None:
                            moe_block.switch_mlp.__call__ = original_switch_mlp_call
                        orig = self.original_experts[i]
                        for proj_name, proj_attrs in orig.items():
                            if hasattr(moe_block.switch_mlp, proj_name):
                                mod = getattr(moe_block.switch_mlp, proj_name)
                                for attr, full_tensor in proj_attrs.items():
                                    if hasattr(mod, attr):
                                        setattr(mod, attr, full_tensor)
                                
                    if c is not None and hasattr(c, "state"):
                        mx.eval(h)
                        mx.eval(c.state)
                        
                    _ = mx.sum(h).item() # Force CPU-GPU sync
                    
                return self.original_model.norm(h)

"""
    new_lines = lines[:start_idx] + [new_proxy] + lines[end_idx:]
    with open("src/sparsify/runtime/chat_generation.py", "w") as f:
        f.writelines(new_lines)
    print("Replaced successfully!")
else:
    print("Could not find start/end indices")
