"""Dynamic Expert Loader executing FFN projections using cached dynamic weights."""
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from sparsify.runtime.cache import MoeCache


class DynamicExpertLoaderLayer(nn.Module):
    """FFN Block wrapper that intercepts execution, routes tokens, and loads experts dynamically."""

    def __init__(
        self,
        layer_index: int,
        n_experts: int,
        cache: MoeCache,
        router: nn.Linear,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.n_experts = n_experts
        self.cache = cache
        self.router = router

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens to top-1 expert, loading expert weights from Cache dynamically."""
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)

        # 1. Compute routing logits and top-1 indices
        gate_logits = self.router(x_flat)  # Shape: (B*L, n_experts)
        routing_idx = mx.argmax(gate_logits, axis=-1)  # Shape: (B*L,)

        out_flat = mx.zeros_like(x_flat)
        unique_experts = np.unique(np.array(routing_idx)).tolist()

        # 2. Dynamic loading and FFN execution loop
        for expert_idx in unique_experts:
            mask = routing_idx == expert_idx
            indices = mx.array(np.where(np.array(mask))[0])
            if len(indices) == 0:
                continue

            # Fetch expert weights from the cache (SSD read happens on cache miss)
            weights = self.cache.get_expert(self.layer_index, expert_idx)
            expert_inputs = x_flat[indices]

            # Execute SwiGLU FFN gating
            gate_out = mx.matmul(expert_inputs, weights["gate_proj.weight"].T)
            gate_out = mx.sigmoid(gate_out) * gate_out  # SiLU

            up_out = mx.matmul(expert_inputs, weights["up_proj.weight"].T)
            inter = gate_out * up_out

            down_out = mx.matmul(inter, weights["down_proj.weight"].T)

            # Scatter back to the sequence flat tensor
            out_flat[indices] = down_out

        return out_flat.reshape(B, L, D)
