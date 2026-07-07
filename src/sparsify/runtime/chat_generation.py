"""
Sparsify inference engine — real mlx-lm inference, real memory telemetry.

No simulation. No placeholders. Everything measured from hardware.
"""
from __future__ import annotations

import sys
import time
import os
from pathlib import Path
import json

import mlx.core as mx
import numpy as np
from safetensors import safe_open



# Monkey patch SwitchGLU to support dynamic index mapping
from mlx_lm.models.switch_layers import SwitchGLU
if not hasattr(SwitchGLU, "_original_call"):
    SwitchGLU._original_call = SwitchGLU.__call__
    def _mapped_call(self, x, indices, **kwargs):
        if hasattr(self, "active_mapping"):
            indices = self.active_mapping[indices]
        return self._original_call(x, indices, **kwargs)
    SwitchGLU.__call__ = _mapped_call

class SparsifyEngine:
    """
    Loads a model from *model_path* using mlx-lm and provides streaming
    token generation with real hardware telemetry.

    Parameters
    ----------
    model_path:
        Local directory containing the model weights and tokenizer
        (downloaded by ``sparsify pull``).
    max_tokens:
        Maximum tokens to generate per response.
    """

    def __init__(self, model_path: Path, max_tokens: int = 512, memory_limit_gb: int = 4) -> None:
        from pathlib import Path
        import mlx_lm  # noqa: F401 — imported here to avoid slow top-level import
        
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.memory_limit_gb = memory_limit_gb
        self._mlx_lm = mlx_lm
        
        # Set strict unified memory limits
        limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        mx.set_cache_limit(limit_bytes)
        mx.set_wired_limit(limit_bytes)

        # Prefer model_path if it contains a config.json; fall back to HF cache
        if not (Path(model_path) / "config.json").exists():
            # Look for the model in the HF hub cache by trying to resolve it
            hf_id = model_path.name.replace("--", "/")
            hf_cache = (
                Path.home()
                / ".cache" / "huggingface" / "hub"
                / f"models--{model_path.name}"
            )
            # Find the snapshot directory inside the HF cache
            snapshot_dir = None
            snapshots = hf_cache / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    if snap.is_dir() and (snap / "config.json").exists():
                        snapshot_dir = snap
                        break
            if snapshot_dir:
                model_path = snapshot_dir
            else:
                raise FileNotFoundError(
                    f"config.json not found in {self.model_path}. "
                    f"Run: sparsify pull {hf_id.replace('--', '/')}"
                )

        # Read the safetensors index to map layers to their files
        index_file = Path(model_path) / "model.safetensors.index.json"
        self.weight_map = {}
        if index_file.exists():
            with open(index_file, "r") as f:
                self.weight_map = json.load(f).get("weight_map", {})

        mx.reset_peak_memory()
        
        # Load weights lazily so we don't load 26GB into RAM
        self.model, self.tokenizer = mlx_lm.load(str(model_path), lazy=True)
        self._mlx_lm = mlx_lm

        # Cache safetensors handles so we don't reopen them on every token
        self.safetensor_handles = {}
        model_path_obj = Path(model_path)
        for f in sorted(os.listdir(model_path_obj)):
            if f.endswith('.safetensors'):
                self.safetensor_handles[f] = safe_open(model_path_obj / f, framework="numpy")

        # Because we use lazy=True, this will correctly report ~0GB at startup,
        # and Peak Memory in the telemetry will accurately track dynamic paging during inference!
        self.model_memory_gb: float = mx.get_active_memory() / 1e9
        
        # --- UNIVERSAL EAGER EVALUATION PATCH ---
        # To run a 25GB MoE model in 4GB RAM, we must evaluate the computational graph
        # layer-by-layer. Otherwise, mlx_lm builds the whole graph for all 32 layers,
        # and Metal tries to page in all 32 layers of experts simultaneously, causing OOM!
        original_call = self.model.model.__call__
        
        # Check if the model is already optimized
        is_optimized = False
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                if config.get("sparsify_optimized_v2", False):
                    is_optimized = True

        # Read the safetensors index to map layers to their files
        index_file = Path(model_path) / "model.safetensors.index.json"
        weight_map = {}
        if index_file.exists():
            with open(index_file, "r") as f:
                weight_map = json.load(f).get("weight_map", {})
            


        class PagingModelProxy:
            """
            A proxy wrapper that intercepts model calls and dynamically slices the lazy expert 
            tensors so that only the active experts are evaluated by Metal.
            """
            def __init__(self, model, weight_map, safetensor_handles):
                import mlx.core as mx
                self.original_model = model
                
                # Save original lazy experts
                self.expert_cache = {} # Cache for evaluated individual experts
                self.expert_cache_lru = [] # LRU list
                self.max_cached_experts = 80 # Approx 2.3 GB for 80 expert slices
                self.weight_map = weight_map
                self.safetensor_handles = safetensor_handles
                
                # Build safetensors weight map
                # (Removing duplicate weight map building here since it's passed from SparsifyEngine)
                pass
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
                        
                        if hasattr(moe_block, "switch_mlp"):
                            moe_block.switch_mlp.active_mapping = mapping_mx
                            
                            # 3. Swap the huge lazy tensors with small concatenated lazy slices
                            orig = self.original_experts[i]
                            for proj_name, proj_attrs in orig.items():
                                if hasattr(moe_block.switch_mlp, proj_name):
                                    mod = getattr(moe_block.switch_mlp, proj_name)
                                    for attr, full_tensor in proj_attrs.items():
                                        if hasattr(mod, attr):
                                            slices = []
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
                                                        if file_path in self.safetensor_handles:
                                                            f = self.safetensor_handles[file_path]
                                                            st_tensor = f.get_slice(weight_name)
                                                            e_slice = st_tensor[e:e+1]
                                                            e_slice_mx = mx.array(np.ascontiguousarray(e_slice), dtype=full_tensor.dtype)
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
                                                
                                            small_tensor = mx.concatenate(slices, axis=0)
                                            setattr(mod, attr, small_tensor)
                                    
                    # 4. Evaluate layer (builds the graph using the small_tensor references)
                    h = layer(h, mask, c)
                                        
                    # 5. Restore full lazy tensors immediately so the graph is valid for next tokens
                    if moe_block and hasattr(moe_block, "switch_mlp"):
                        if hasattr(moe_block, "switch_mlp") and hasattr(moe_block.switch_mlp, "active_mapping"):
                            del moe_block.switch_mlp.active_mapping
                        orig = self.original_experts[i]
                        for proj_name, proj_attrs in orig.items():
                            if hasattr(moe_block.switch_mlp, proj_name):
                                mod = getattr(moe_block.switch_mlp, proj_name)
                                for attr, full_tensor in proj_attrs.items():
                                    if hasattr(mod, attr):
                                        setattr(mod, attr, full_tensor)
                                
                    pass
                return self.original_model.norm(h)

        # Replace the inner model with our paging proxy
        self.model.model = PagingModelProxy(self.model.model, self.weight_map, self.safetensor_handles)
        
        # Unload all switch_mlp weights from memory initially to save RAM
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
        mx.clear_cache()
        
        self.messages = []

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def generate_stream(self, prompt: str):
        import time
        import mlx.core as mx
        self.messages.append({"role": "user", "content": prompt})
        
        # Handle chat templates if they exist, otherwise just use the prompt
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            formatted = self.tokenizer.apply_chat_template(
                self.messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt
            
        prompt_tokens = mx.array(self.tokenizer.encode(formatted))
        
        n_tokens = 0
        t_start = time.perf_counter()

        full_response = []
        
        # Disable MLX graph compilation for the generate loop
        # MLX compilation assumes static weights, but our MoE weights change per token!
        # By setting the compiled function to the uncompiled step, we bypass tracing.
        def _compile_mock(f):
            return f
            
        import mlx.core as mx
        self.model.disable_compile = True
        
        if hasattr(self._mlx_lm, "generate"):
            _old_compile = getattr(mx, "compile")
            mx.compile = _compile_mock
            
        try:
            for response in self._mlx_lm.stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=self.max_tokens
            ):
                text = response.text
                full_response.append(text)
                n_tokens += 1
                
                # Update telemetry
                elapsed = time.perf_counter() - t_start
                throughput = n_tokens / elapsed if elapsed > 0 else 0.0
                telemetry = {
                    "n_tokens": n_tokens,
                    "throughput": throughput,
                    "elapsed": elapsed,
                    "footprint_gb": self.model_memory_gb,
                    "active_gb": getattr(mx, "get_active_memory", mx.metal.get_active_memory)() / 1e9,
                    "peak_gb": getattr(mx, "get_peak_memory", mx.metal.get_peak_memory)() / 1e9,
                }
                
                yield text, telemetry
                
                # Free unused caches to keep RAM low
                mx.clear_cache()
                
        finally:
            if hasattr(self._mlx_lm, "generate"):
                mx.compile = _old_compile

        self.messages.append({"role": "assistant", "content": "".join(full_response)})

    def generate(self, prompt: str) -> str:
        """Backward compatible generate that prints directly."""
        import sys
        full_text = ""
        for text, telemetry in self.generate_stream(prompt):
            if text is not None:
                sys.stdout.write(text)
                sys.stdout.flush()
                full_text += text
            if telemetry is not None:
                print("\n")
                print("  ── Telemetry (all values measured from hardware) ─────────────")
                print(f"  Tokens generated : {telemetry['n_tokens']}")
                print(f"  Throughput       : {telemetry['throughput']:.1f} tok/s")
                print(f"  Elapsed          : {telemetry['elapsed']:.2f}s")
                print(f"  Active memory    : {telemetry['active_gb']:.2f} GB  (unified memory, model + activations)")
                print(f"  Peak memory      : {telemetry['peak_gb']:.2f} GB  (this session)")
                print(f"  Model footprint  : {telemetry['footprint_gb']:.2f} GB  (weights loaded at startup)")
                print("  ──────────────────────────────────────────────────────────────\n")
        return full_text
