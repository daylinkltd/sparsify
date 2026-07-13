"""Verification suite for sparsify.paging_torch.

Tests PyTorch expert store, cache, and model surgery.
"""
from __future__ import annotations

from pathlib import Path
import pytest

try:
    import torch
    import torch.nn as nn
    from safetensors.torch import save_file
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sparsify.paging_torch.store import PyTorchExpertStore
from sparsify.paging_torch.cache import PyTorchExpertCache
from sparsify.paging_torch.surgery import attach_pytorch_paging, PagedPyTorchLinear

pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="PyTorch/Transformers not installed"),
]


class DummyExpert(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, out_dim, bias=False)
        self.w2 = nn.Linear(out_dim, in_dim, bias=False)


class DummyMoEBlock(nn.Module):
    def __init__(self, num_experts: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList([DummyExpert(in_dim, out_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        weights = torch.softmax(logits, dim=-1)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out = out + weights[:, :, i:i+1] * expert.w2(expert.w1(x))
        return out


def test_pytorch_paging_e2e(tmp_path: Path):
    # 1. Create a dummy MoE model
    num_experts = 4
    in_dim = 8
    out_dim = 16
    model = DummyMoEBlock(num_experts, in_dim, out_dim)

    # Get reference outputs before surgery
    x = torch.randn(1, 2, in_dim)
    with torch.no_grad():
        ref_out = model(x)

    # 2. Save its expert weights to safetensors files
    # The surgery expects block_name.experts.idx.linear_name.weight
    # Here block_name is empty (root)
    tensors = {}
    for i, expert in enumerate(model.experts):
        tensors[f"experts.{i}.w1.weight"] = expert.w1.weight.data
        tensors[f"experts.{i}.w2.weight"] = expert.w2.weight.data

    # Add non-paged weights (router gate)
    tensors["gate.weight"] = model.gate.weight.data

    save_file(tensors, tmp_path / "model.safetensors")

    # 3. Apply paging surgery
    runtime = attach_pytorch_paging(model, tmp_path, budget_bytes=100 * 1024, device="cpu")
    assert runtime is not None
    assert runtime.paged_modules_count == num_experts * 2

    # Verify weights are empty (numel == 0)
    for expert in model.experts:
        assert expert.w1.original_linear.weight.numel() == 0
        assert expert.w2.original_linear.weight.numel() == 0

    # 4. Run forward pass on paged model
    with torch.no_grad():
        paged_out = model(x)

    # 5. Verify outputs are numerically identical
    assert torch.allclose(ref_out, paged_out, atol=1e-5)

    # 6. Verify weights are cleared back to 0-size parameters after forward pass finishes
    for expert in model.experts:
        assert expert.w1.original_linear.weight.numel() == 0
        assert expert.w2.original_linear.weight.numel() == 0

    runtime.close()
