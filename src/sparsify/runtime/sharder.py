"""Model Sharder to extract and separate expert weights from production safetensors."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from safetensors import safe_open
from safetensors.numpy import save_file

from sparsify.runtime.registry import ExpertRegistry


def shard_moe_model(model_dir: Path, output_dir: Path) -> Tuple[int, Path, Path]:
    """Parse safetensors files in model_dir, strip out expert weights, and save them individually.

    Returns:
        Tuple of (number of experts sharded, shared weights directory path, experts directory path).
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    
    shared_dir = output_dir / "shared"
    experts_dir = output_dir / "experts"
    
    shared_dir.mkdir(parents=True, exist_ok=True)
    experts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy config and non-weight files (like vocab, tokenizer)
    for f in model_dir.iterdir():
        if f.is_file() and f.suffix not in [".safetensors", ".bin", ".npz"]:
            shutil.copy(f, shared_dir / f.name)

    # 2. Scan and partition safetensors
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    expert_tensors: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    shared_tensors: Dict[str, Dict[str, np.ndarray]] = {}
    
    expert_tensor_keys_map: Dict[Tuple[int, int], List[str]] = {}

    for sf in safetensors_files:
        file_shared = {}
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                tensor_data = f.get_tensor(k)
                
                # Check for Qwen/Mixtral/DeepSeek expert pattern
                # Examples:
                # - model.layers.0.mlp.experts.2.gate_proj.weight
                # - model.layers.0.block_sparse_moe.experts.1.w1.weight
                if ".experts." in k:
                    parts = k.split(".")
                    layer_id = None
                    expert_id = None
                    
                    # Parse indexes
                    for i, p in enumerate(parts):
                        if p == "layers" and i + 1 < len(parts):
                            layer_id = int(parts[i+1])
                        if p == "experts" and i + 1 < len(parts):
                            expert_id = int(parts[i+1])
                            
                    if layer_id is not None and expert_id is not None:
                        key = (layer_id, expert_id)
                        if key not in expert_tensors:
                            expert_tensors[key] = {}
                            expert_tensor_keys_map[key] = []
                        
                        # Store standard projection names for the cache
                        # map w1/gate_proj -> gate_proj.weight, etc.
                        std_name = parts[-2]
                        if std_name in ["w1", "gate_proj"]:
                            std_name = "gate_proj.weight"
                        elif std_name in ["w3", "up_proj"]:
                            std_name = "up_proj.weight"
                        elif std_name in ["w2", "down_proj"]:
                            std_name = "down_proj.weight"
                        else:
                            std_name = f"{std_name}.weight"
                            
                        expert_tensors[key][std_name] = tensor_data
                        expert_tensor_keys_map[key].append(k)
                else:
                    # Keep shared weight
                    file_shared[k] = tensor_data
                    
        # Save shorn shared safetensors file
        if file_shared:
            save_file(file_shared, str(shared_dir / sf.name))

    # 3. Serialize grouped expert weights to individual NPZ files
    for (l_id, e_id), tensors in expert_tensors.items():
        expert_file = experts_dir / f"layer_{l_id}_expert_{e_id}.npz"
        # Save as npz archive
        np.savez(str(expert_file), **tensors)

    # 4. Generate local registry cache for pre-indexing
    registry = ExpertRegistry(registry_file=shared_dir / "registry_cache.json")
    for (l_id, e_id), keys in expert_tensor_keys_map.items():
        expert_file = experts_dir / f"layer_{l_id}_expert_{e_id}.npz"
        size = expert_file.stat().st_size
        registry.register_expert(
            layer_id=l_id,
            expert_id=e_id,
            file_path=expert_file,
            tensor_keys=["gate_proj.weight", "up_proj.weight", "down_proj.weight"],
            size_bytes=size
        )
    registry.save_cache()

    return len(expert_tensors), shared_dir, experts_dir
