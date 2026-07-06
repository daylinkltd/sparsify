"""Mixture-of-Experts (MoE) sequence model with dynamic expert weight swapping."""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from sparsify.prototype.lru_cache import ExpertLRUCache


class DynamicMoeBlock(nn.Module):
    """Transformer block using dynamic on-demand Mixture-of-Experts FFN routing."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int,
        layer_index: int,
        cache: ExpertLRUCache,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.cache = cache
        self.n_experts = n_experts

        # Multi-Head Self Attention
        self.attn_ln = nn.RMSNorm(d_model)
        self.self_attn = nn.MultiHeadAttention(d_model, n_heads)

        # FFN Router & RMSNorm
        self.ffn_ln = nn.RMSNorm(d_model)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self._call_impl = self.execute

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        return self._call_impl(x, mask=mask)

    def execute(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # 1. Multi-Head Self Attention (always in memory)
        attn_x = self.attn_ln(x)
        h = x + self.self_attn(attn_x, attn_x, attn_x, mask=mask)

        # 2. Dynamic Mixture-of-Experts FFN
        norm_h = self.ffn_ln(h)
        B, L, D = norm_h.shape
        h_flat = norm_h.reshape(-1, D)

        # Compute routing logits and indices (Top-1 routing)
        gate_logits = self.router(h_flat)  # Shape: (B*L, n_experts)
        routing_idx = mx.argmax(gate_logits, axis=-1)  # Shape: (B*L,)

        out_flat = mx.zeros_like(h_flat)
        unique_experts = np.unique(np.array(routing_idx)).tolist()

        # Dynamic loading loop
        for expert_idx in unique_experts:
            mask_exp = routing_idx == expert_idx
            indices = mx.array(np.where(np.array(mask_exp))[0])
            if len(indices) == 0:
                continue

            # Fetch expert weights from the LRU memory cache
            weights = self.cache.get_expert(self.layer_index, expert_idx)

            # Extract token embeddings routed to this expert
            expert_inputs = h_flat[indices]

            # Execute SwiGLU gating in Python using loaded arrays
            gate_out = mx.matmul(expert_inputs, weights["gate_proj.weight"].T)
            gate_out = mx.sigmoid(gate_out) * gate_out  # SiLU

            up_out = mx.matmul(expert_inputs, weights["up_proj.weight"].T)
            inter = gate_out * up_out

            down_out = mx.matmul(inter, weights["down_proj.weight"].T)

            # Scatter the output back into the sequence flat tensor
            out_flat[indices] = down_out

        # Reshape back to residual stream shape
        ffn_out = out_flat.reshape(B, L, D)
        return h + ffn_out


class MoeTransformer(nn.Module):
    """Sparsify Mixture-of-Experts Research Prototype Model (100M parameters total)."""

    def __init__(
        self,
        vocab_size: int = 4000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        n_experts: int = 16,
        cache: ExpertLRUCache | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Bounded working cache reference
        self.cache = cache or ExpertLRUCache()

        # Input and Output layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = [
            DynamicMoeBlock(d_model, n_heads, n_experts, idx, self.cache)
            for idx in range(n_layers)
        ]
        self.norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        
        # Self-attention mask for causal autoregressive decoding
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)

        for layer in self.layers:
            h = layer(h, mask=mask)
            
        h = self.norm(h)
        return self.output(h)
