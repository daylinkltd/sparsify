"""Model patcher to dynamically intercept and swap FFN layers in production models."""
from __future__ import annotations

import mlx.nn as nn

from sparsify.runtime.cache import MoeCache
from sparsify.runtime.loader import DynamicExpertLoaderLayer


def patch_production_model(model: nn.Module, cache: MoeCache) -> int:
    """Recursively search for MoE blocks in the model and replace them with DynamicExpertLoaderLayers.

    Reuses the existing router/gate projections while directing expert FFN weights to cache.
    Returns:
        Number of patched layers.
    """
    patched_count = 0

    if not hasattr(model, "layers"):
        return 0

    for l_idx, layer in enumerate(model.layers):
        # 1. Detect Mixtral style sparse block
        if hasattr(layer, "block_sparse_moe"):
            moe_block = getattr(layer, "block_sparse_moe")
            # Extract existing router & expert counts
            router = getattr(moe_block, "gate", None)
            if router is None:
                continue
                
            n_experts = len(getattr(moe_block, "experts", []))
            
            # Replace FFN with storage-native loader
            patched_block = DynamicExpertLoaderLayer(
                layer_index=l_idx,
                n_experts=n_experts,
                cache=cache,
                router=router
            )
            setattr(layer, "block_sparse_moe", patched_block)
            patched_count += 1

        # 2. Detect Qwen / DeepSeek style mlp block
        elif hasattr(layer, "mlp") and hasattr(getattr(layer, "mlp"), "experts"):
            moe_block = getattr(layer, "mlp")
            router = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)
            if router is None:
                continue
                
            n_experts = len(getattr(moe_block, "experts", []))
            
            patched_block = DynamicExpertLoaderLayer(
                layer_index=l_idx,
                n_experts=n_experts,
                cache=cache,
                router=router
            )
            setattr(layer, "mlp", patched_block)
            patched_count += 1

    return patched_count
