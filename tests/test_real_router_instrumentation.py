"""Unit and integration tests for Sprint 9 real router instrumentation."""
from __future__ import annotations

import torch
import pytest

from sparsify.runtime.real_router_instrumentation import RealRouterTracer


def test_pytorch_moe_gating_hooks_and_tracer() -> None:
    """Validate PyTorch hook registration, domain runs, and metric solvers."""
    
    # 1. Instantiate the real tracer
    tracer = RealRouterTracer()
    assert tracer.num_layers == 2
    assert tracer.num_experts == 4
    
    # Register forward hooks on gate linear layer
    hooks = tracer.register_hooks()
    assert len(hooks) == 2
    
    # 2. Run a mock forward pass to log traces
    tracer.run_trace_collection("Mock domain query.")
    
    # Verify traces recorded
    for l in range(2):
        assert len(tracer.traces[l]) > 0
        assert len(tracer.traces[l][0]) == 2
        
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # 3. Compute metrics
    metrics = tracer.calculate_metrics()
    assert metrics["entropy"] >= 0.0
    assert metrics["avg_reuse_distance"] >= 0.0
    assert metrics["w50"] >= 1.0
    assert metrics["w90"] >= 1.0
    assert 0.0 <= metrics["top1_predictability"] <= 1.0
    assert 0.0 <= metrics["top3_predictability"] <= 1.0
    assert "8GB" in metrics["cache_sweeps"]
