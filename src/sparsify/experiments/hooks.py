"""Execution hooks for injecting perturbations and collecting telemetry from the MLX model graph."""
from __future__ import annotations

from typing import Any, Set, List
import numpy as np
import mlx.core as mx
from mlx.core.fast import scaled_dot_product_attention


class LayerBypassHook:
    """Hook to bypass a transformer block entirely while maintaining cache integrity.

    Designed as a lightweight Python wrapper to avoid nanobind/C++ submodule
    registration side-effects.
    """

    def __init__(self, original_layer: Any, layer_index: int) -> None:
        self.original_layer = original_layer
        self.layer_index = layer_index
        self.bypass_active = False
        self._call_impl = self._default_call
        
        # Telemetry for SP-009
        self.last_input_norm: float = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_layer, name)

    def _default_call(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        # Log input residual stream norm (cast to float32 to prevent float16 overflow)
        self.last_input_norm = float(mx.linalg.norm(x.astype(mx.float32)))
        
        if self.bypass_active:
            if cache is not None:
                # We must update the key-value cache to keep the context length aligned
                # across all layers, even if the representation output is discarded.
                # In standard llama layers, self.self_attn updates cache in-place.
                _ = self.original_layer.self_attn(
                    self.original_layer.input_layernorm(x), mask, cache
                )
            # Route residual input directly to output, bypassing self_attn output and FFN.
            return x
        else:
            return self.original_layer(x, mask, cache)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        return self._call_impl(x, mask, cache)


class AttentionHeadMaskHook:
    """Hook to mask specific attention heads and log head-level activation telemetry."""

    def __init__(self, original_attention: Any, layer_index: int) -> None:
        self.original_attention = original_attention
        self.layer_index = layer_index
        self.masked_heads: Set[int] = set()
        
        # Telemetry for SP-009
        self.last_output_norm: float = 0.0
        self.last_head_norms: List[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_attention, name)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, D = x.shape
        queries = self.original_attention.q_proj(x)
        keys = self.original_attention.k_proj(x)
        values = self.original_attention.v_proj(x)

        n_heads = self.original_attention.n_heads
        n_kv_heads = self.original_attention.n_kv_heads

        queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.original_attention.rope(queries, offset=cache.offset)
            keys = self.original_attention.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.original_attention.rope(queries)
            keys = self.original_attention.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.original_attention.scale, mask=mask
        )

        # Telemetry: Log head activity norms before masking and projection (cast to float32 to prevent overflow)
        self.last_head_norms = [
            float(mx.linalg.norm(output[:, h, :, :].astype(mx.float32))) for h in range(n_heads)
        ]

        # Apply head masking by multiplying the corresponding heads by 0
        if self.masked_heads:
            head_mask = np.ones((n_heads,), dtype=np.float32)
            for h in self.masked_heads:
                head_mask[h] = 0.0
            mx_mask = mx.array(head_mask).reshape(1, -1, 1, 1)
            output = output * mx_mask

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        r_attn = self.original_attention.o_proj(output)
        
        # Telemetry: Log total attention output norm (cast to float32 to prevent overflow)
        self.last_output_norm = float(mx.linalg.norm(r_attn.astype(mx.float32)))
        
        return r_attn


class ActivationSparsityHook:
    """Hook to monitor and enforce activation sparsity in the MLP/FFN layer."""

    def __init__(self, original_mlp: Any, layer_index: int) -> None:
        self.original_mlp = original_mlp
        self.layer_index = layer_index
        
        # Diagnostics
        self.last_sparsity_rate: float = 0.0
        self.last_activation_norms: mx.array | None = None
        
        # Telemetry for SP-009
        self.last_output_norm: float = 0.0
        self.last_gating_norm: float = 0.0
        
        # Control
        self.force_sparsity = False
        self.sparsity_threshold = 1e-4

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_mlp, name)

    def __call__(self, x: mx.array) -> mx.array:
        gate_out = self.original_mlp.gate_proj(x)
        up_out = self.original_mlp.up_proj(x)

        # SiLU activation on gate
        activated_gate = mx.sigmoid(gate_out) * gate_out
        
        # Calculate sparsity rate
        below_thresh = activated_gate < self.sparsity_threshold
        self.last_sparsity_rate = float(mx.mean(below_thresh))
        
        if self.force_sparsity:
            # Zero out any activations below the threshold
            activated_gate = mx.where(below_thresh, mx.zeros_like(activated_gate), activated_gate)

        # Perform SwiGLU gating
        inter = activated_gate * up_out
        self.last_activation_norms = mx.linalg.norm(inter, axis=-1)
        
        # Telemetry (cast to float32 to prevent float16 overflow)
        self.last_gating_norm = float(mx.linalg.norm(inter.astype(mx.float32)))
        
        r_ffn = self.original_mlp.down_proj(inter)
        self.last_output_norm = float(mx.linalg.norm(r_ffn.astype(mx.float32)))
        
        return r_ffn


def patch_model_for_experimentation(model: Any) -> tuple[list[LayerBypassHook], list[AttentionHeadMaskHook], list[ActivationSparsityHook]]:
    """Monkey-patch the model layers with Sparsify perturbation hooks.

    Returns lists of the active hooks for external control and telemetry logging.
    """
    bypass_hooks = []
    head_hooks = []
    sparsity_hooks = []

    # Get underlying model layers
    layers_container = getattr(model, "model", model)
    layers = layers_container.layers
    
    for idx, layer in enumerate(layers):
        # Avoid double-patching: check if layer is already patched
        if isinstance(layer, LayerBypassHook):
            bypass_hooks.append(layer)
            head_hooks.append(layer.self_attn)
            sparsity_hooks.append(layer.mlp)
            continue

        # 1. Patch the self attention module for head masking
        head_hook = AttentionHeadMaskHook(layer.self_attn, idx)
        layer.self_attn = head_hook
        head_hooks.append(head_hook)

        # 2. Patch the MLP block for activation gating sparsity
        sparsity_hook = ActivationSparsityHook(layer.mlp, idx)
        layer.mlp = sparsity_hook
        sparsity_hooks.append(sparsity_hook)

        # 3. Patch the entire transformer block for layer bypassing
        bypass_hook = LayerBypassHook(layer, idx)
        layers[idx] = bypass_hook
        bypass_hooks.append(bypass_hook)

    return bypass_hooks, head_hooks, sparsity_hooks
