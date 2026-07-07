"""Model surgery: detect expert projections in a loaded model and replace
them with paged equivalents.

Detection is structural, not architecture-specific: any leaf module whose
``weight`` has a leading expert dimension (mlx-lm's ``SwitchLinear`` /
``QuantizedSwitchLinear``, or anything shaped like them) is paged, provided
its tensors exist on disk under the module's own tree path. Router gates,
attention, norms, embeddings and shared experts are untouched and stay
resident.

Must run on a model loaded with ``lazy=True``: the original expert arrays
are still unevaluated then, so dropping them means their bytes are never
read from disk at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import mlx.nn as nn

from sparsify.paging.cache import ExpertCache
from sparsify.paging.modules import ExpertGroup, PagedSwitchLinear
from sparsify.paging.store import SafetensorsExpertStore

_PARAM_CANDIDATES = ("weight", "scales", "biases", "bias")


def _is_switch_linear(module) -> bool:
    """Structural test: a linear layer stacked along a leading expert dim."""
    if not isinstance(module, nn.Module):
        return False
    weight = getattr(module, "weight", None)
    return weight is not None and getattr(weight, "ndim", 0) == 3


@dataclass
class PagingRuntime:
    """Handle over a paged model: owns the store/cache and reports telemetry."""

    store: SafetensorsExpertStore
    cache: ExpertCache
    groups: List[ExpertGroup]
    replaced_modules: int
    paged_bytes: int  # total expert bytes now living on SSD instead of RAM
    per_group_experts: Dict[str, int] = field(default_factory=dict)

    def stats(self) -> Dict:
        return {
            "replaced_modules": self.replaced_modules,
            "paged_gb": round(self.paged_bytes / 1e9, 3),
            "moe_blocks": len(self.groups),
            **self.cache.stats(),
            **self.store.stats(),
        }


def attach_paging(model: nn.Module, model_path: Path, budget_bytes: int) -> PagingRuntime | None:
    """Replace all detected expert projections with SSD-paged modules.

    Returns None when the model has no pageable expert structure (dense
    models run unmodified).
    """
    modules = dict(model.named_modules())
    targets: Dict[str, List[str]] = {}  # parent path -> [proj attr names]
    for path, module in modules.items():
        if not _is_switch_linear(module):
            continue
        parent, _, attr = path.rpartition(".")
        targets.setdefault(parent, []).append(attr)

    if not targets:
        return None

    store = SafetensorsExpertStore(model_path)
    cache = ExpertCache(store, budget_bytes)
    groups: List[ExpertGroup] = []
    replaced = 0
    paged_bytes = 0
    per_group_experts: Dict[str, int] = {}

    for uid, (parent_path, attrs) in enumerate(sorted(targets.items())):
        parent = modules[parent_path]
        proj_params: Dict[str, List[str]] = {}
        proj_meta: Dict[str, dict] = {}
        num_experts = None

        for attr in attrs:
            prefix = f"{parent_path}.{attr}"
            params = [p for p in _PARAM_CANDIDATES if f"{prefix}.{p}" in store]
            if "weight" not in params:
                raise RuntimeError(
                    f"Expert module {prefix} has no on-disk weight tensor; "
                    f"cannot page this model's format."
                )
            shape, _, _ = store.tensor_info(f"{prefix}.weight")
            if num_experts is None:
                num_experts = shape[0]
            elif num_experts != shape[0]:
                raise RuntimeError(
                    f"Inconsistent expert counts in block {parent_path}: "
                    f"{num_experts} vs {shape[0]} ({attr})"
                )
            for p in params:
                _, _, nbytes = store.tensor_info(f"{prefix}.{p}")
                paged_bytes += nbytes

            old = modules[prefix]
            proj_params[attr] = params
            proj_meta[attr] = {
                "quantized": "scales" in params,
                "group_size": getattr(old, "group_size", 64),
                "bits": getattr(old, "bits", 4),
                "mode": getattr(old, "mode", "affine"),
                "has_scalar_biases": "biases" in params,
                "has_expert_bias": "bias" in params,
            }

        group = ExpertGroup(uid, parent_path, num_experts, proj_params, cache)
        groups.append(group)
        per_group_experts[parent_path] = num_experts

        for attr in attrs:
            setattr(parent, attr, PagedSwitchLinear(group, attr, **proj_meta[attr]))
            replaced += 1

    return PagingRuntime(
        store=store,
        cache=cache,
        groups=groups,
        replaced_modules=replaced,
        paged_bytes=paged_bytes,
        per_group_experts=per_group_experts,
    )
