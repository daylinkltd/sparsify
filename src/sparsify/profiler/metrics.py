"""Profiling result data classes for model memory analysis.

Every data class exposes a ``to_dict()`` helper that returns a plain-Python
dictionary suitable for JSON serialisation or tabular display.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ComponentMemory:
    """Memory footprint of a single logical component (e.g. 'attention_q')."""

    name: str
    size_bytes: int
    percentage: float
    tensor_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayerProfile:
    """Per-layer memory breakdown."""

    layer_index: int
    total_size_bytes: int
    components: dict[str, ComponentMemory]
    percentage_of_model: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["components"] = {k: v.to_dict() for k, v in self.components.items()}
        return d


@dataclass
class KVCacheEstimate:
    """Estimated KV-cache memory for a given context length / batch size."""

    context_length: int
    batch_size: int
    bytes_per_token: int
    total_size_bytes: int
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelProfile:
    """Aggregate profiling result for a GGUF model file."""

    model_path: str
    model_name: str | None
    file_size_bytes: int
    architecture: str
    quantization: str | None
    parameter_count: int
    layer_count: int
    head_count: int
    kv_head_count: int
    embedding_dim: int
    ffn_dim: int
    context_length: int
    total_weight_bytes: int
    layer_profiles: list[LayerProfile]
    component_summary: dict[str, ComponentMemory]
    kv_cache_estimates: list[KVCacheEstimate]
    embedding_bytes: int
    output_head_bytes: int
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layer_profiles"] = [lp.to_dict() for lp in self.layer_profiles]
        d["component_summary"] = {k: v.to_dict() for k, v in self.component_summary.items()}
        d["kv_cache_estimates"] = [kv.to_dict() for kv in self.kv_cache_estimates]
        return d
