"""Experiment 2 and 5: Attention Head Masking and Joint Pruning experiments."""
from __future__ import annotations

from typing import Any, Dict, List, Set
import numpy as np
import mlx.core as mx

from sparsify.experiments.hooks import LayerBypassHook, AttentionHeadMaskHook
from sparsify.experiments.evaluator import evaluate_perplexity, evaluate_accuracy_probe


def run_head_importance_experiment(
    model: Any,
    tokenizer: Any,
    head_hooks: List[AttentionHeadMaskHook],
    baseline_ppl: float,
    max_heads_to_test: int = 20,  # Cap tests for fast validation runs
) -> Dict[str, Any]:
    """Execute Experiment 2: Masking attention heads individually and measuring perplexity."""
    results: List[Dict[str, Any]] = []
    
    # We will sample heads to test to keep execution times fast (under 10 seconds per test)
    # Total heads = num_layers * num_heads. We pick a subset across different layers.
    total_layers = len(head_hooks)
    n_heads = head_hooks[0].original_attention.n_heads if total_layers > 0 else 0
    
    heads_tested = 0
    
    # Generate test candidates: (layer_idx, head_idx)
    candidates = []
    for l in range(total_layers):
        # Sample heads evenly (e.g. head 0 and head n_heads // 2)
        candidates.append((l, 0))
        candidates.append((l, n_heads // 2))
        if len(candidates) >= max_heads_to_test:
            break

    for layer_idx, head_idx in candidates:
        hook = head_hooks[layer_idx]
        hook.masked_heads.add(head_idx)
        
        try:
            ppl = evaluate_perplexity(model, tokenizer)
            acc = evaluate_accuracy_probe(model, tokenizer)
            
            rel_degradation = (ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
            
            results.append({
                "layer_index": layer_idx,
                "head_index": head_idx,
                "perplexity": ppl,
                "accuracy": acc,
                "relative_perplexity_degradation": rel_degradation,
                "passed_success_criteria": rel_degradation < 0.005,
                "passed_failure_criteria": rel_degradation > 0.05 or np.isnan(ppl),
            })
        except Exception as exc:
            results.append({
                "layer_index": layer_idx,
                "head_index": head_idx,
                "error": str(exc),
                "passed_success_criteria": False,
                "passed_failure_criteria": True,
            })
        finally:
            # Restore mask
            hook.masked_heads.remove(head_idx)

    # Summarize opportunity
    maskable_candidates = [
        (r["layer_index"], r["head_index"]) for r in results if r.get("passed_success_criteria", False)
    ]
    
    # Hypothesis check: can we find maskable heads?
    success = len(maskable_candidates) / len(results) >= 0.30 if results else False

    return {
        "experiment_name": "Experiment 2: Attention Head Importance & Quality Contribution",
        "hypothesis": "A significant fraction of attention heads can be masked out with minimal quality loss.",
        "baseline_perplexity": baseline_ppl,
        "results": results,
        "summary": {
            "tested_heads_count": len(results),
            "maskable_heads": maskable_candidates,
            "maskable_count": len(maskable_candidates),
            "opportunity_confirmed": success,
        }
    }


def run_joint_pruning_experiment(
    model: Any,
    tokenizer: Any,
    bypass_hooks: List[LayerBypassHook],
    head_hooks: List[AttentionHeadMaskHook],
    skippable_layers: List[int],
    maskable_heads: List[tuple[int, int]],
    baseline_ppl: float,
) -> Dict[str, Any]:
    """Execute Experiment 5: Joint Multi-Component Pruning.

    Bypasses all skippable layers and masks all maskable heads simultaneously.
    """
    if not skippable_layers and not maskable_heads:
        return {
            "experiment_name": "Experiment 5: Joint Multi-Component Pruning",
            "hypothesis": "Compounding errors from multiple prunings remain minimal.",
            "error": "No skippable components identified from individual runs.",
            "passed_success_criteria": False,
        }

    # Activate all bypass hooks
    for layer_idx in skippable_layers:
        bypass_hooks[layer_idx].bypass_active = True
        
    # Activate all head masks
    for layer_idx, head_idx in maskable_heads:
        head_hooks[layer_idx].masked_heads.add(head_idx)

    try:
        ppl = evaluate_perplexity(model, tokenizer)
        acc = evaluate_accuracy_probe(model, tokenizer)
        
        rel_degradation = (ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
        
        # Calculate computational savings
        total_layers = len(bypass_hooks)
        total_heads = total_layers * (head_hooks[0].original_attention.n_heads if total_layers > 0 else 0)
        
        saved_layers = len(skippable_layers)
        saved_heads = len(maskable_heads)
        
        # Compute combined fraction of compute saved (rough projection)
        saved_compute_ratio = (saved_layers / total_layers) * 0.5 + (saved_heads / total_heads) * 0.5 if total_layers > 0 else 0.0

        success = (rel_degradation < 0.05) and (saved_compute_ratio >= 0.15)

        result = {
            "experiment_name": "Experiment 5: Joint Multi-Component Pruning",
            "hypothesis": "Compounding errors from multiple prunings remain minimal.",
            "baseline_perplexity": baseline_ppl,
            "joint_perplexity": ppl,
            "joint_accuracy": acc,
            "relative_perplexity_degradation": rel_degradation,
            "estimated_compute_saved_percentage": saved_compute_ratio * 100.0,
            "passed_success_criteria": success,
            "passed_failure_criteria": rel_degradation > 0.20 or np.isnan(ppl),
        }
    finally:
        # Restore all hooks
        for layer_idx in skippable_layers:
            bypass_hooks[layer_idx].bypass_active = False
        for layer_idx, head_idx in maskable_heads:
            head_hooks[layer_idx].masked_heads.remove(head_idx)

    return result
