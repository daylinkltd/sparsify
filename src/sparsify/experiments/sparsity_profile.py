"""Experiment 3 and 6: Static Component Sparsity and Activation Gating Sparsity."""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import mlx.core as mx

from sparsify.experiments.hooks import LayerBypassHook, ActivationSparsityHook
from sparsify.experiments.evaluator import evaluate_perplexity, evaluate_accuracy_probe


def run_static_sparsity_experiment(
    model: Any,
    tokenizer: Any,
    bypass_hooks: List[LayerBypassHook],
) -> Dict[str, Any]:
    """Execute Experiment 3: Measuring the active contribution of attention and FFN blocks.

    We measure the norm ratio of each block output to input: ||f(x)||_2 / ||x||_2.
    """
    attention_ratios: Dict[int, List[float]] = {i: [] for i in range(len(bypass_hooks))}
    ffn_ratios: Dict[int, List[float]] = {i: [] for i in range(len(bypass_hooks))}

    # Let's patch LayerBypassHooks to capture norms during inference
    original_calls = [hook.__call__ for hook in bypass_hooks]

    def make_norm_capture_call(hook: LayerBypassHook, original_call: Any):
        def norm_capture_call(x: mx.array, mask: mx.array | None = None, cache: Any | None = None) -> mx.array:
            # Step 1: Input norm
            x_norm = float(mx.mean(mx.linalg.norm(x, axis=-1)))
            
            # Step 2: Attention output contribution
            x_attn_in = hook.original_layer.input_layernorm(x)
            r_attn = hook.original_layer.self_attn(x_attn_in, mask, cache)
            attn_norm = float(mx.mean(mx.linalg.norm(r_attn, axis=-1)))
            
            # Step 3: MLP output contribution
            h = x + r_attn
            h_mlp_in = hook.original_layer.post_attention_layernorm(h)
            r_ffn = hook.original_layer.mlp(h_mlp_in)
            ffn_norm = float(mx.mean(mx.linalg.norm(r_ffn, axis=-1)))
            
            # Record ratios
            if x_norm > 0:
                attention_ratios[hook.layer_index].append(attn_norm / x_norm)
            h_norm = float(mx.mean(mx.linalg.norm(h, axis=-1)))
            if h_norm > 0:
                ffn_ratios[hook.layer_index].append(ffn_norm / h_norm)
                
            return h + r_ffn
        return norm_capture_call

    # Install captures
    for hook, orig in zip(bypass_hooks, original_calls):
        hook.__call__ = make_norm_capture_call(hook, orig)  # type: ignore[assignment]

    try:
        # Run inference on a few prompts to capture norm distributions
        prompts = [
            "Write a function to compute prime numbers.",
            "What are the core components of a database compiler?",
            "Explain the concept of attention in transformer neural networks.",
        ]
        for p in prompts:
            # We don't need to generate a lot, just run a forward pass of a few tokens
            tokens = tokenizer.encode(p)
            inputs = mx.array(tokens).reshape(1, -1)
            # Run model
            _ = model(inputs)
            mx.eval(_)
    finally:
        # Restore original __call__ methods
        for hook, orig in zip(bypass_hooks, original_calls):
            hook.__call__ = orig  # type: ignore[assignment]

    # Compute mean ratios per layer
    summary_results = []
    cold_components_count = 0
    total_components = len(bypass_hooks) * 2
    
    for idx in range(len(bypass_hooks)):
        mean_attn = float(np.mean(attention_ratios[idx])) if attention_ratios[idx] else 0.0
        mean_ffn = float(np.mean(ffn_ratios[idx])) if ffn_ratios[idx] else 0.0
        
        is_attn_cold = mean_attn < 0.02
        is_ffn_cold = mean_ffn < 0.02
        
        if is_attn_cold:
            cold_components_count += 1
        if is_ffn_cold:
            cold_components_count += 1
            
        summary_results.append({
            "layer_index": idx,
            "mean_attention_norm_ratio": mean_attn,
            "mean_ffn_norm_ratio": mean_ffn,
            "attention_is_cold": is_attn_cold,
            "ffn_is_cold": is_ffn_cold,
        })

    cold_ratio = cold_components_count / total_components
    success = cold_ratio >= 0.20

    return {
        "experiment_name": "Experiment 3: Static Model Sparsity (Hot/Cold Components)",
        "hypothesis": "Across standard tasks, a subset of components consistently outputs near-zero contribution.",
        "results": summary_results,
        "summary": {
            "total_components_monitored": total_components,
            "cold_components_detected": cold_components_count,
            "cold_components_percentage": cold_ratio * 100.0,
            "opportunity_confirmed": success,
        }
    }


def run_activation_sparsity_experiment(
    model: Any,
    tokenizer: Any,
    sparsity_hooks: List[ActivationSparsityHook],
    baseline_ppl: float,
) -> Dict[str, Any]:
    """Execute Experiment 6: Measuring MLP gate sparsity and its resilience to zero-masking."""
    
    # Step 1: Capture natural activation sparsity
    # Run a forward pass over some tokens
    prompts = [
        "In physics, energy is the quantitative property that is transferred to a body.",
        "A Personal Computer consists of various chips, a processor, memory, and controllers."
    ]
    for p in prompts:
        tokens = tokenizer.encode(p)
        inputs = mx.array(tokens).reshape(1, -1)
        _ = model(inputs)
        mx.eval(_)

    layer_sparsity_rates = []
    for hook in sparsity_hooks:
        layer_sparsity_rates.append({
            "layer_index": hook.layer_index,
            "natural_sparsity_rate": hook.last_sparsity_rate,
        })
        
    mean_natural_sparsity = float(np.mean([r["natural_sparsity_rate"] for r in layer_sparsity_rates]))

    # Step 2: Enforce/Force sparsity (zero out values below threshold) and check quality
    for hook in sparsity_hooks:
        hook.force_sparsity = True
        hook.sparsity_threshold = 1e-4

    try:
        ppl = evaluate_perplexity(model, tokenizer)
        acc = evaluate_accuracy_probe(model, tokenizer)
        
        rel_degradation = (ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
        
        success = (mean_natural_sparsity >= 0.70) and (rel_degradation < 0.01)
        
        result = {
            "experiment_name": "Experiment 6: Activation Gating Sparsity (MLP Gate Sparsity)",
            "hypothesis": "Gated MLP activations naturally exhibit high sparsity that can be zero-masked safely.",
            "baseline_perplexity": baseline_ppl,
            "sparsity_forced_perplexity": ppl,
            "sparsity_forced_accuracy": acc,
            "relative_perplexity_degradation": rel_degradation,
            "average_natural_sparsity_percentage": mean_natural_sparsity * 100.0,
            "layer_sparsity_rates": layer_sparsity_rates,
            "passed_success_criteria": success,
            "passed_failure_criteria": mean_natural_sparsity < 0.30 or rel_degradation > 0.05,
        }
    finally:
        # Restore hooks
        for hook in sparsity_hooks:
            hook.force_sparsity = False

    return result
