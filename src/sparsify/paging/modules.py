"""Paged drop-in replacements for mlx-lm's expert projection modules.

The swap happens at the *leaf* level (``SwitchLinear`` /
``QuantizedSwitchLinear``). Everything above — router, top-k selection,
sorting, GLU activation, shared experts — is untouched upstream code, which
is what keeps this architecture-agnostic across Mixtral / Qwen MoE /
DeepSeek / OLMoE and future mlx-lm MoE models.

Numerical contract: for the experts actually selected, the computation is
the same ``mx.gather_qmm`` / ``mx.gather_mm`` kernel over the same weight
values as the original module, with row indices remapped into the small
gathered stack. Output must be identical to full-RAM inference (verified in
tests/test_paging_equivalence.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class ExpertGroup:
    """Shared state for the projections of one MoE block (one layer).

    Owns index resolution (which experts does this token need?) so the
    gate/up/down projections of a block resolve the router output once.
    """

    def __init__(self, uid: int, prefix: str, num_experts: int,
                 proj_params: Dict[str, List[str]], cache) -> None:
        self.uid = uid
        self.prefix = prefix
        self.num_experts = num_experts
        self.proj_params = proj_params  # proj_name -> param names on disk
        self.cache = cache
        self._last_indices: Optional[mx.array] = None
        self._last_resolved: Optional[Tuple[Tuple[int, ...], mx.array]] = None

    def resolve(self, indices: mx.array) -> Tuple[Tuple[int, ...], mx.array]:
        """Materialize router indices and build the full->small remap table.

        Returns (sorted unique expert ids, remap array). The remap is
        monotonic, so indices pre-sorted by the caller stay sorted after
        remapping (required by ``sorted_indices=True`` fast paths).
        """
        if self._last_indices is indices and self._last_resolved is not None:
            return self._last_resolved
        ids = tuple(np.unique(np.array(indices, copy=False)).tolist())
        remap = np.zeros(self.num_experts, dtype=np.uint32)
        for small, full in enumerate(ids):
            remap[full] = small
        resolved = (ids, mx.array(remap))
        self._last_indices = indices
        self._last_resolved = resolved
        return resolved

    def gather_stack(self, proj_name: str, ids: Tuple[int, ...]) -> Dict[str, mx.array]:
        """Fetch *ids* through the cache and stack this projection's tensors."""
        entries = self.cache.get_experts(self, ids)
        params = self.proj_params[proj_name]
        if len(ids) == 1:
            return {p: entries[ids[0]][proj_name][p] for p in params}
        return {
            p: mx.concatenate([entries[e][proj_name][p] for e in ids], axis=0)
            for p in params
        }


class PagedSwitchLinear(nn.Module):
    """Expert projection whose weights live on SSD and page in on demand.

    Registers no mx.array parameters: the weight residency is owned by the
    ExpertCache, keeping the module out of ``model.parameters()`` entirely.
    """

    def __init__(self, group: ExpertGroup, proj_name: str, *,
                 quantized: bool, group_size: int = 64, bits: int = 4,
                 mode: str = "affine", has_scalar_biases: bool = False,
                 has_expert_bias: bool = False) -> None:
        super().__init__()
        # Plain-object attributes only (never mx.array) — see class docstring.
        self._group = group
        self._proj_name = proj_name
        self._quantized = quantized
        self._group_size = group_size
        self._bits = bits
        self._mode = mode
        self._has_scalar_biases = has_scalar_biases
        self._has_expert_bias = has_expert_bias

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        ids, remap = self._group.resolve(indices)
        stack = self._group.gather_stack(self._proj_name, ids)
        small_indices = remap[indices]

        if self._quantized:
            out = mx.gather_qmm(
                x,
                stack["weight"],
                stack["scales"],
                stack["biases"] if self._has_scalar_biases else None,
                rhs_indices=small_indices,
                transpose=True,
                group_size=self._group_size,
                bits=self._bits,
                mode=self._mode,
                sorted_indices=sorted_indices,
            )
        else:
            out = mx.gather_mm(
                x,
                stack["weight"].swapaxes(-1, -2),
                rhs_indices=small_indices,
                sorted_indices=sorted_indices,
            )
        if self._has_expert_bias:
            out = out + mx.expand_dims(stack["bias"][small_indices], -2)
        return out
