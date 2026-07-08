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


def _trace_sanitize(model, disk_names: List[str]) -> Dict[str, List[str]]:
    """Derive on-disk -> parameter-tree name mapping for per-expert layouts.

    Feeds the model's own ``sanitize()`` converter a probe dict where each
    on-disk tensor name carries a unique sentinel value, then reads which
    sentinels ended up in each row of every stacked output tensor. This
    reuses the model's authoritative conversion logic instead of hardcoding
    per-architecture naming conventions.
    """
    import mlx.core as mx
    import numpy as np

    if not hasattr(model, "sanitize"):
        return {}
    probe = {n: mx.full((1, 1), i, dtype=mx.float32)
             for i, n in enumerate(disk_names)}
    try:
        sanitized = model.sanitize(probe)
    except Exception as exc:
        raise RuntimeError(
            f"Could not trace {type(model).__name__}.sanitize() to resolve "
            f"per-expert tensor names: {exc}"
        ) from exc

    mapping: Dict[str, List[str]] = {}
    for key, arr in sanitized.items():
        if getattr(arr, "ndim", 0) < 2 or arr.shape[0] < 2:
            continue  # not an expert-stacked output
        vals = np.array(arr.reshape(arr.shape[0], -1)[:, 0]).astype(int)
        if ((vals < 0) | (vals >= len(disk_names))).any():
            continue  # sanitize computed on the values; not a pure restack
        mapping[key] = [disk_names[v] for v in vals]
    return mapping


@dataclass
class PagingRuntime:
    """Handle over a paged model: owns the store/cache and reports telemetry."""

    store: SafetensorsExpertStore
    cache: ExpertCache
    groups: List[ExpertGroup]
    replaced_modules: int
    paged_bytes: int  # total expert bytes now living on SSD instead of RAM
    per_group_experts: Dict[str, int] = field(default_factory=dict)
    resident_bytes_full: int = 0

    def configure(self, budget_bytes: int, log=None) -> None:
        """Decide between fully-resident and paged execution.

        All-or-nothing, by measurement: partial residency was tested and
        LOST (Qwen3 @4.5 GB: 0.57 tok/s hybrid vs 2.62 paged) — budget
        concentrated in a few resident blocks starves the paged cache of
        the rest. So: everything fits -> load it all, run at native
        mlx-lm speed (zero paging overhead); otherwise every block pages
        and the cache gets the whole budget.
        """
        total = sum(g.total_bytes for g in self.groups)
        if total <= budget_bytes:
            for i, g in enumerate(self.groups):
                if g.full is None:
                    if log and (i % 8 == 0 or i == len(self.groups) - 1):
                        log(f"loading experts into RAM… block {i + 1}/{len(self.groups)}")
                    g.load_full(self.store)
                    self.resident_bytes_full += g.total_bytes
            self.cache.budget_bytes = 256 * 1024 * 1024  # nothing left to page
        else:
            self.cache.budget_bytes = budget_bytes

    def close(self) -> None:
        self.cache.close()

    def stats(self) -> Dict:
        return {
            "replaced_modules": self.replaced_modules,
            "paged_gb": round(self.paged_bytes / 1e9, 3),
            "moe_blocks": len(self.groups),
            "resident_blocks": sum(1 for g in self.groups if g.full is not None),
            "resident_full_bytes": self.resident_bytes_full,
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
    sanitize_map: Dict[str, List[str]] | None = None  # built on first need

    def _natural(path: str):
        import re
        return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", path)]

    for uid, (parent_path, attrs) in enumerate(sorted(targets.items(), key=lambda kv: _natural(kv[0]))):
        parent = modules[parent_path]
        proj_sources: Dict[str, Dict[str, tuple]] = {}
        proj_meta: Dict[str, dict] = {}
        num_experts = None
        group_bytes = 0

        for attr in attrs:
            prefix = f"{parent_path}.{attr}"
            module_experts = modules[prefix].weight.shape[0]
            if num_experts is None:
                num_experts = module_experts
            elif num_experts != module_experts:
                raise RuntimeError(
                    f"Inconsistent expert counts in block {parent_path}: "
                    f"{num_experts} vs {module_experts} ({attr})"
                )

            if f"{prefix}.weight" in store:
                # Layout A: expert-stacked tensors, sliced per expert.
                sources = {
                    p: ("stacked", f"{prefix}.{p}")
                    for p in _PARAM_CANDIDATES if f"{prefix}.{p}" in store
                }
                disk_experts = store.tensor_info(f"{prefix}.weight")[0][0]
                if disk_experts != num_experts:
                    raise RuntimeError(
                        f"{prefix}.weight has {disk_experts} experts on disk "
                        f"but the model expects {num_experts}"
                    )
            else:
                # Layout B: one tensor per expert (e.g. Mixtral upstream
                # ``experts.N.w1``). Derive disk names by tracing the
                # model's own sanitize() converter.
                if sanitize_map is None:
                    sanitize_map = _trace_sanitize(model, store.names())
                sources = {}
                for p in _PARAM_CANDIDATES:
                    per_expert = sanitize_map.get(f"{prefix}.{p}")
                    if per_expert is None:
                        continue
                    if len(per_expert) != num_experts:
                        raise RuntimeError(
                            f"sanitize() maps {len(per_expert)} tensors onto "
                            f"{prefix}.{p}; the model expects {num_experts} experts"
                        )
                    sources[p] = ("per_expert", per_expert)
                if "weight" not in sources:
                    raise RuntimeError(
                        f"Expert module {prefix} has no resolvable on-disk "
                        f"weight tensors; cannot page this model's format."
                    )

            for p, (kind, ref) in sources.items():
                if kind == "stacked":
                    group_bytes += store.tensor_info(ref)[2]
                else:
                    group_bytes += sum(store.tensor_info(n)[2] for n in ref)

            old = modules[prefix]
            proj_sources[attr] = sources
            proj_meta[attr] = {
                "quantized": "scales" in sources,
                "group_size": getattr(old, "group_size", 64),
                "bits": getattr(old, "bits", 4),
                "mode": getattr(old, "mode", "affine"),
                "has_scalar_biases": "biases" in sources,
                "has_expert_bias": "bias" in sources,
            }

        group = ExpertGroup(uid, parent_path, num_experts, proj_sources, cache,
                            total_bytes=group_bytes)
        groups.append(group)
        per_group_experts[parent_path] = num_experts
        paged_bytes += group_bytes

        for attr in attrs:
            setattr(parent, attr, PagedSwitchLinear(group, attr, **proj_meta[attr]))
            replaced += 1

    # Speculative prefetch (SPARSIFY_PREFETCH=1, experimental): while block
    # k computes, block k+1's previous-token experts stage in the background.
    # MEASURED and default-OFF on 2026-07-08, on BOTH storage tiers:
    #   USB SSD  @2GB: 1.54 tok/s with prefetch vs 1.73 without
    #   NVMe     @3GB: 7.00 tok/s with prefetch vs 8.50 without
    # Rationale: recurrent experts (the predictable ones) are already
    # resident; the misses are routing churn no history predicts. The
    # machinery stays for router-aware predictors (future work), not for
    # last-token replay. I/O queue depth is likewise saturated at the
    # default 8 workers (NVMe @3GB: 8.50/8.45/8.53 tok/s at 8/16/24).
    import os
    if os.environ.get("SPARSIFY_PREFETCH", "").lower() not in ("", "0", "false", "no"):
        for a, b in zip(groups, groups[1:]):
            a.next_group = b

    return PagingRuntime(
        store=store,
        cache=cache,
        groups=groups,
        replaced_modules=replaced,
        paged_bytes=paged_bytes,
        per_group_experts=per_group_experts,
    )
