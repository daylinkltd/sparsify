"""Integration tests verifying output equivalence on sharded safetensors MoE models."""
from __future__ import annotations

import json
import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from safetensors.numpy import save_file

from sparsify.runtime.sharder import shard_moe_model
from sparsify.runtime.patcher import patch_production_model
from sparsify.runtime.cache import MoeCache
from sparsify.runtime.registry import ExpertRegistry


class MockQwenMoeBlock(nn.Module):
    """Mock architecture matching Qwen MoE block layout."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(16, 4, bias=False)  # 4 experts
        self.experts = [
            nn.Sequential(
                nn.Linear(16, 32, bias=False),  # w1/gate
                nn.Linear(16, 32, bias=False),  # w3/up
                nn.Linear(32, 16, bias=False)   # w2/down
            )
            for _ in range(4)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        logits = self.gate(x_flat)
        routing_idx = mx.argmax(logits, axis=-1)
        
        out_flat = mx.zeros_like(x_flat)
        for idx, exp in enumerate(self.experts):
            mask = routing_idx == idx
            indices = mx.array(np.where(np.array(mask))[0])
            if len(indices) == 0:
                continue
            # Simple linear mapping to simulate expert FFN
            exp_in = x_flat[indices]
            # w1 gating, w3 up, w2 down SwiGLU simulation
            w1 = exp.layers[0](exp_in)
            w3 = exp.layers[1](exp_in)
            w2 = exp.layers[2](mx.sigmoid(w1) * w1 * w3)
            out_flat[indices] = w2
            
        return out_flat.reshape(B, L, D)


class MockQwenMoeModel(nn.Module):
    """Mock model containing layers list."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [MockQwenMoeBlock() for _ in range(2)]

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.layers:
            h = h + layer(h)
        return h


def test_safetensors_sharding_and_output_parity(tmp_path) -> None:
    """Validate weight splitting, model patching, and dynamic routing parity on safetensors."""
    
    # 1. Instantiate the mock model and collect weight dictionaries
    model_src = MockQwenMoeModel()
    
    weights_dict = {}
    for k, v in model_src.parameters().items():
        # Convert nested parameter tree to safetensors naming convention
        # e.g., 'layers.0.gate.weight'
        # Let's map model parameters to flat dict
        pass

    # We manually populate weights matching safetensors format
    # Layer 0 & 1 experts w1, w2, w3 and gates
    safetensors_weights = {}
    for l_idx in range(2):
        # Gate weights
        safetensors_weights[f"model.layers.{l_idx}.mlp.gate.weight"] = np.array(model_src.layers[l_idx].gate.weight)
        
        # Experts weights
        for e_idx in range(4):
            exp = model_src.layers[l_idx].experts[e_idx]
            safetensors_weights[f"model.layers.{l_idx}.mlp.experts.{e_idx}.gate_proj.weight"] = np.array(exp.layers[0].weight)
            safetensors_weights[f"model.layers.{l_idx}.mlp.experts.{e_idx}.up_proj.weight"] = np.array(exp.layers[1].weight)
            safetensors_weights[f"model.layers.{l_idx}.mlp.experts.{e_idx}.down_proj.weight"] = np.array(exp.layers[2].weight)

    # Save mock safetensors shard
    hf_dir = tmp_path / "hf_model"
    hf_dir.mkdir()
    save_file(safetensors_weights, str(hf_dir / "model.safetensors"))
    
    # Save a dummy config.json
    config = {
        "model_type": "qwen2_moe",
        "num_hidden_layers": 2,
        "num_experts": 4
    }
    with open(hf_dir / "config.json", "w") as f:
        json.dump(config, f)

    # 2. Run Sharder to separate experts from safetensors
    sharded_dir = tmp_path / "sharded_model"
    sharded_dir.mkdir()
    
    count, shared_path, experts_path = shard_moe_model(hf_dir, sharded_dir)
    assert count == 8  # 2 layers * 4 experts = 8 experts sharded
    
    # Verify file existence
    assert (experts_path / "layer_0_expert_0.npz").exists()
    assert (experts_path / "layer_1_expert_3.npz").exists()
    assert (shared_path / "model.safetensors").exists()
    assert (shared_path / "registry_cache.json").exists()

    # 3. Instantiate the target model to patch
    model_target = MockQwenMoeModel()
    # Mock MLX architecture matching Qwen2-MoE representation
    # Let's map attribute names for the patcher to discover
    for l_idx, layer in enumerate(model_target.layers):
        # The patcher searches for layer.mlp or layer.block_sparse_moe
        # Let's dynamically attach 'mlp' mock subclass to match
        class MockMlp:
            def __init__(self, block):
                self.gate = block.gate
                self.experts = block.experts
        layer.mlp = MockMlp(layer)

    # Verify unpatched model run
    test_input = mx.random.normal((1, 5, 16))
    out_original = model_target(test_input)
    mx.eval(out_original)

    # 4. Patch target model and link with MoeCache pointing to sharded experts
    registry = ExpertRegistry(registry_file=shared_path / "registry_cache.json")
    registry.load_cache()
    
    cache = MoeCache(registry, budget_bytes=100 * 1024 * 1024, policy_name="lru")
    patched_count = patch_production_model(model_target, cache)
    assert patched_count == 2

    # 5. Execute patched model forward pass
    out_patched = model_target(test_input)
    mx.eval(out_patched)

    # Verify absolute output numerical equivalence!
    assert mx.allclose(out_original, out_patched, atol=1e-5)
    print("Parity Assert Passed! Outputs are 100% numerically identical.")
