"""Storage Layer for loading specific expert weights dynamically from different file formats."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict
import numpy as np

import mlx.core as mx
from safetensors import safe_open
from gguf import GGUFReader

from sparsify.runtime.registry import ExpertMetadata


def load_expert_tensors(metadata: ExpertMetadata) -> Dict[str, mx.array]:
    """Load specific expert tensors from disk, measuring load time and read bandwidth.

    Supports:
    - `.npz` (MLX/NumPy archive)
    - `.safetensors` (Zero-copy HuggingFace standard format)
    - `.gguf` (Unified binary quantization format)
    """
    start_time = time.perf_counter()
    file_path = Path(metadata.file_path)
    weights: Dict[str, mx.array] = {}

    if not file_path.exists():
        raise FileNotFoundError(f"Weight file not found: {file_path}")

    # --- 1. MLX Shard Loading (.npz) ---
    if file_path.suffix == ".npz":
        loaded = mx.load(str(file_path))
        # Ensure standard keys: map whatever keys exist to the standard ones
        for k, v in loaded.items():
            # If keys contain 'weight' or 'bias'
            std_key = k
            if "gate_proj" in k:
                std_key = "gate_proj.weight" if "weight" in k else "gate_proj.bias"
            elif "up_proj" in k:
                std_key = "up_proj.weight" if "weight" in k else "up_proj.bias"
            elif "down_proj" in k:
                std_key = "down_proj.weight" if "weight" in k else "down_proj.bias"
            weights[std_key] = v

    # --- 2. Safetensors Loading (.safetensors) ---
    elif file_path.suffix == ".safetensors":
        with safe_open(str(file_path), framework="numpy") as f:
            for k in metadata.tensor_keys:
                # Load only the specific tensor from disk (zero-copy memory mapped)
                numpy_arr = f.get_tensor(k)
                # Map standard FFN layer names
                std_key = k.split(".")[-1]  # w1, w2, gate_proj, etc.
                if "gate_proj" in k or "w1" in k:
                    std_key = "gate_proj.weight"
                elif "up_proj" in k or "w3" in k:
                    std_key = "up_proj.weight"
                elif "down_proj" in k or "w2" in k:
                    std_key = "down_proj.weight"
                weights[std_key] = mx.array(numpy_arr)

    # --- 3. GGUF Loading (.gguf) ---
    elif file_path.suffix == ".gguf":
        reader = GGUFReader(str(file_path))
        for k in metadata.tensor_keys:
            # Find the GGUF tensor info
            tensor_info = None
            for t in reader.tensors:
                if t.name == k:
                    tensor_info = t
                    break
            if tensor_info is not None:
                # Seek to data offset and read bytes
                # Note: GGUFReader reads the full tensor dictionary internally if requested,
                # but we can fetch the specific tensor data slice to prevent full-file load.
                numpy_arr = reader.get_tensor_data(tensor_info)
                std_key = k.split(".")[-2]  # ffn_gate, ffn_up, ffn_down
                if "ffn_gate" in k:
                    std_key = "gate_proj.weight"
                elif "ffn_up" in k:
                    std_key = "up_proj.weight"
                elif "ffn_down" in k:
                    std_key = "down_proj.weight"
                weights[std_key] = mx.array(numpy_arr)

    else:
        raise ValueError(f"Unsupported weight serialization format: {file_path.suffix}")

    # Track loading telemetry performance
    load_time = time.perf_counter() - start_time
    metadata.load_count += 1
    metadata.load_time_total += load_time
    metadata.last_loaded_timestamp = datetime.now(timezone.utc).isoformat()

    return weights


from datetime import datetime, timezone
