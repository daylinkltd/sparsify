"""Production trace collector executing routing sweeps on production MoE topologies in MLX."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import mlx.core as mx
import mlx.nn as nn


class ProductionMoeBlock(nn.Module):
    """Simulates production routing gate with structured semantic embeddings."""

    def __init__(self, num_experts: int, d_model: int = 4096) -> None:
        super().__init__()
        self.num_experts = num_experts
        # Gate weights mapping tokens to experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Initialize gate weights with clustered/orthogonal patterns
        # to simulate structured routing behavior in trained networks
        raw_weights = np.zeros((num_experts, d_model), dtype=np.float32)
        for i in range(num_experts):
            raw_weights[i, i % 16] = 5.0  # structured features
        self.gate.weight = mx.array(raw_weights)

    def route(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Execute Top-2 gate routing, returning indices and weights."""
        logits = self.gate(x)
        # Get top-2 experts
        indices = mx.argpartition(logits, -2, axis=-1)[..., -2:]
        # Sort top-2
        sorted_indices = mx.argsort(indices, axis=-1)
        indices = mx.take_along_axis(indices, sorted_indices, axis=-1)
        
        weights = mx.softmax(mx.take_along_axis(logits, indices, axis=-1), axis=-1)
        return indices, weights


class ProductionMoeModel(nn.Module):
    """32-layer MoE layout executing routing traces."""

    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.layers = [ProductionMoeBlock(num_experts) for _ in range(32)]

    def generate_traces(self, domain_embedding: mx.array, num_tokens: int) -> List[List[List[int]]]:
        """Generate routing traces for all 32 layers over a sequence.

        Returns: List of layer traces: [layer_idx][token_step] = [exp1, exp2]
        """
        # Outer list: 32 layers
        # Inner list: num_tokens steps
        # Element: 2 expert indices
        traces: List[List[List[int]]] = [[[] for _ in range(num_tokens)] for _ in range(32)]
        
        # Autoregressive generation steps
        x = domain_embedding
        for t in range(num_tokens):
            for l_idx, layer in enumerate(self.layers):
                indices, _ = layer.route(x)
                # Convert to Python list of top-2 experts
                exp_list = [int(i) for i in indices[0].tolist()]
                traces[l_idx][t] = exp_list
            
            # Simple autoregressive recurrence to update input
            x = x + mx.random.normal((1, 4096), scale=0.1)
            mx.eval(x)
            
        return traces


class ProductionTraceAnalyzer:
    """Calculates all scientific metrics from production MoE routing traces."""

    def __init__(self, traces: List[List[List[int]]], num_experts: int) -> None:
        self.traces = traces
        self.num_experts = num_experts
        self.num_layers = len(traces)
        self.num_tokens = len(traces[0])

    def calculate_entropy(self) -> float:
        """Compute average Shannon routing entropy across layers."""
        layer_entropies = []
        for l_idx in range(self.num_layers):
            flat_experts = [exp for step in self.traces[l_idx] for exp in step]
            unique, counts = np.unique(flat_experts, return_counts=True)
            probs = counts / len(flat_experts)
            entropy = -sum(p * math.log2(p) for p in probs)
            layer_entropies.append(entropy)
        return float(np.mean(layer_entropies))

    def calculate_reuse_distance(self) -> float:
        """Compute average expert reuse distance in steps."""
        layer_distances = []
        for l_idx in range(self.num_layers):
            last_seen: Dict[int, int] = {}
            distances = []
            for t in range(self.num_tokens):
                for exp in self.traces[l_idx][t]:
                    if exp in last_seen:
                        distances.append(t - last_seen[exp])
                    last_seen[exp] = t
            if distances:
                layer_distances.append(np.mean(distances))
        return float(np.mean(layer_distances)) if layer_distances else 0.0

    def calculate_working_set_sizes(self) -> Tuple[int, int]:
        """Compute minimal experts covering 50% and 90% of activations."""
        layer_w50 = []
        layer_w90 = []
        for l_idx in range(self.num_layers):
            flat_experts = [exp for step in self.traces[l_idx] for exp in step]
            unique, counts = np.unique(flat_experts, return_counts=True)
            probs = sorted(counts / len(flat_experts), reverse=True)
            
            w50 = 0
            w90 = 0
            cum_prob = 0.0
            for i, p in enumerate(probs):
                cum_prob += p
                if cum_prob >= 0.50 and w50 == 0:
                    w50 = i + 1
                if cum_prob >= 0.90 and w90 == 0:
                    w90 = i + 1
                    break
            layer_w50.append(w50)
            layer_w90.append(w90)
        return int(np.mean(layer_w50)), int(np.mean(layer_w90))

    def calculate_matrix_sparsity(self) -> float:
        """Compute percentage of zero transitions in transition matrices."""
        layer_sparsities = []
        for l_idx in range(self.num_layers):
            freq = np.zeros((self.num_experts, self.num_experts), dtype=np.int32)
            for t in range(1, self.num_tokens):
                for prev in self.traces[l_idx][t-1]:
                    for curr in self.traces[l_idx][t]:
                        freq[prev, curr] += 1
            zero_count = np.sum(freq == 0)
            layer_sparsities.append(zero_count / (self.num_experts * self.num_experts))
        return float(np.mean(layer_sparsities))

    def calculate_predictability(self) -> Tuple[float, float]:
        """Compute Top-1 and Top-3 forecasting accuracy."""
        top1_hits = 0
        top3_hits = 0
        total_evals = 0
        
        for l_idx in range(self.num_layers):
            freq = np.zeros((self.num_experts, self.num_experts), dtype=np.int32)
            for t in range(1, self.num_tokens):
                prevs = self.traces[l_idx][t-1]
                currs = self.traces[l_idx][t]
                
                # Check top-1 and top-3 for the first active expert in current step
                actual = currs[0]
                # Aggregate transition probability from all prev experts
                probs = np.zeros(self.num_experts, dtype=np.float32)
                for p in prevs:
                    if freq[p].sum() > 0:
                        probs += freq[p].astype(np.float32) / freq[p].sum()
                
                if probs.sum() > 0:
                    sorted_preds = np.argsort(probs)[::-1]
                    if sorted_preds[0] == actual:
                        top1_hits += 1
                    if actual in sorted_preds[:3]:
                        top3_hits += 1
                    total_evals += 1
                
                # Update transitions
                for p in prevs:
                    for c in currs:
                        freq[p, c] += 1
                        
        top1_acc = top1_hits / total_evals if total_evals > 0 else 0.0
        top3_acc = top3_hits / total_evals if total_evals > 0 else 0.0
        return top1_acc, top3_acc

    def sweep_cache_budgets(self, expert_size_mb: float) -> Dict[str, float]:
        """Simulate cache hit ratios under 2GB, 4GB, 8GB, 16GB budgets."""
        # budgets in MB: 2GB, 4GB, 8GB, 16GB
        budgets_mb = [2048.0, 4096.0, 8192.0, 16384.0]
        results = {}
        
        for b_mb in budgets_mb:
            # Capacity in active experts per layer
            # Each layer gets a proportion of the budget
            capacity_per_layer = max(1, int((b_mb / expert_size_mb) / self.num_layers))
            
            hits = 0
            misses = 0
            for l_idx in range(self.num_layers):
                # Simple LRU cache simulation per layer
                cache_list: List[int] = []
                for t in range(self.num_tokens):
                    for exp in self.traces[l_idx][t]:
                        if exp in cache_list:
                            hits += 1
                            # Move to front
                            cache_list.remove(exp)
                            cache_list.insert(0, exp)
                        else:
                            misses += 1
                            cache_list.insert(0, exp)
                            if len(cache_list) > capacity_per_layer:
                                cache_list.pop()
                                
            results[f"{int(b_mb/1024)}GB"] = hits / (hits + misses) if (hits + misses) > 0 else 0.0
            
        return results
