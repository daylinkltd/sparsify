"""Model surgery for PyTorch: detect expert modules and wrap their linear layers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from sparsify.paging_torch.cache import PyTorchExpertCache
from sparsify.paging_torch.store import PyTorchExpertStore

_PARAM_CANDIDATES = ("weight", "bias")


class PyTorchExpertGroup:
    """Shared state for paged experts of one MoE block."""

    def __init__(self, uid: int, prefix: str, num_experts: int, proj_sources: dict) -> None:
        self.uid = uid
        self.prefix = prefix
        self.num_experts = num_experts
        self.proj_sources = proj_sources


class PagedPyTorchLinear(nn.Module):
    """Wrapper around a linear layer in a PyTorch MoE expert that loads weights dynamically."""

    def __init__(self, original_linear: nn.Module, cache: PyTorchExpertCache,
                 group: PyTorchExpertGroup, expert_idx: int, proj_name: str, param_map: dict) -> None:
        super().__init__()
        self.original_linear = original_linear
        self.cache = cache
        self.group = group
        self.expert_idx = expert_idx
        self.proj_name = proj_name
        self.param_map = param_map  # param_name -> disk_name

        self._param_dtypes = {}
        self._param_shapes = {}

        # Save type and shapes, then release weights to save VRAM at startup
        for param_name in list(self.param_map.keys()):
            param = getattr(self.original_linear, param_name, None)
            if param is not None:
                self._param_dtypes[param_name] = param.dtype
                self._param_shapes[param_name] = param.shape
                # Set to empty
                setattr(self.original_linear, param_name, nn.Parameter(
                    torch.empty(0, dtype=param.dtype, device=param.device),
                    requires_grad=False
                ))

    def release_weights(self) -> None:
        """Release the loaded weights back to empty tensors to clear VRAM."""
        for param_name in list(self.param_map.keys()):
            param = getattr(self.original_linear, param_name, None)
            if param is not None and param.numel() > 0:
                setattr(self.original_linear, param_name, nn.Parameter(
                    torch.empty(0, dtype=self._param_dtypes[param_name], device=param.device),
                    requires_grad=False
                ))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Fetch this expert from cache using the block group
        entries = self.cache.get_experts(self.group, [self.expert_idx])
        entry = entries[self.expert_idx]

        # Swap weights in
        for param_name, disk_name in self.param_map.items():
            loaded_tensor = entry[self.proj_name][param_name]
            if loaded_tensor.ndim == 3 and len(self._param_shapes[param_name]) == 2:
                loaded_tensor = loaded_tensor.squeeze(0)
            setattr(self.original_linear, param_name, nn.Parameter(
                loaded_tensor, requires_grad=False
            ))

        return self.original_linear(*args, **kwargs)


@dataclass
class PyTorchPagingRuntime:
    """Handle over a paged PyTorch model: owns the store and cache."""

    store: PyTorchExpertStore
    cache: PyTorchExpertCache
    paged_modules_count: int
    paged_bytes: int

    def configure(self, budget_bytes: int) -> None:
        self.cache.budget_bytes = budget_bytes

    def close(self) -> None:
        self.cache.close()

    def stats(self) -> dict:
        return {
            "replaced_modules": self.paged_modules_count,
            "paged_gb": round(self.paged_bytes / 1e9, 3),
            **self.cache.stats(),
            **self.store.stats(),
        }


def attach_pytorch_paging(model: nn.Module, model_path: Path, budget_bytes: int,
                          device: torch.device | str) -> PyTorchPagingRuntime | None:
    """Replace all detected expert projections with PyTorch-paged equivalent modules."""
    # 1. Scan for MoE Blocks (modules containing an 'experts' ModuleList)
    moe_blocks = []
    for name, module in model.named_modules():
        if hasattr(module, "experts") and isinstance(module.experts, nn.ModuleList):
            moe_blocks.append((name, module))

    if not moe_blocks:
        return None

    store = PyTorchExpertStore(model_path, device=device)
    cache = PyTorchExpertCache(store, budget_bytes)
    paged_count = 0
    paged_bytes = 0

    # 2. Iterate over MoE Blocks and perform surgery on expert linear layers
    for group_uid, (block_name, block) in enumerate(moe_blocks):
        block_paged_linears = []
        num_experts = len(block.experts)

        # Discover all projection layers and expert count
        proj_names = set()
        for expert in block.experts:
            for sub_name, submodule in expert.named_modules():
                if isinstance(submodule, nn.Linear):
                    proj_names.add(sub_name)

        # Build block-wide group projection sources
        proj_sources = {}
        for proj_name in proj_names:
            expert_0_path = f"{block_name}.experts.0.{proj_name}"
            param_map = {}
            for p in _PARAM_CANDIDATES:
                path_0 = f"{expert_0_path}.{p}".lstrip(".")
                if path_0 in store:
                    ref = [f"{block_name}.experts.{e}.{proj_name}.{p}".lstrip(".") for e in range(num_experts)]
                    param_map[p] = ("per_expert", ref)
            if param_map:
                proj_sources[proj_name] = param_map

        group = PyTorchExpertGroup(group_uid, block_name, num_experts, proj_sources)

        # Replace linear layers
        for expert_idx, expert in enumerate(block.experts):
            for sub_name, submodule in list(expert.named_modules()):
                if isinstance(submodule, nn.Linear):
                    proj_name = sub_name
                    
                    param_map = {}
                    for p in _PARAM_CANDIDATES:
                        param_path = f"{block_name}.experts.{expert_idx}.{sub_name}.{p}".lstrip(".")
                        if param_path in store:
                            param_map[p] = param_path

                    if "weight" in param_map:
                        # Compute size
                        shape, dtype, nbytes = store.tensor_info(param_map["weight"])
                        paged_bytes += nbytes
                        if "bias" in param_map:
                            _, _, b_bytes = store.tensor_info(param_map["bias"])
                            paged_bytes += b_bytes

                        # Replace Linear layer with wrapper
                        wrapper = PagedPyTorchLinear(
                            submodule, cache, group, expert_idx, proj_name, param_map
                        )
                        # Set it on the parent module of the Linear layer
                        parent_name, _, attr_name = sub_name.rpartition(".")
                        if parent_name:
                            parent_mod = expert.get_submodule(parent_name)
                        else:
                            parent_mod = expert
                        setattr(parent_mod, attr_name, wrapper)

                        block_paged_linears.append(wrapper)
                        paged_count += 1

        # Hook the forward pass of the block to release weights immediately after execution
        if block_paged_linears:
            original_forward = block.forward
            def make_hook(orig_forward, paged_linears):
                def hooked_forward(*args, **kwargs):
                    try:
                        return orig_forward(*args, **kwargs)
                    finally:
                        for pl in paged_linears:
                            pl.release_weights()
                return hooked_forward
            block.forward = make_hook(original_forward, block_paged_linears)

    return PyTorchPagingRuntime(
        store=store,
        cache=cache,
        paged_modules_count=paged_count,
        paged_bytes=paged_bytes
    )
