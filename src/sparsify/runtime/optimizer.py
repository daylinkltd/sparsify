import json
import glob
import shutil
import gc
from pathlib import Path
from typing import Dict, Any, Union

import mlx.core as mx
from mlx_lm.utils import _get_classes, load_config

def optimize_moe_safetensors(model_path: Union[str, Path], out_path: Union[str, Path]) -> None:
    """
    Optimizes a downloaded MoE model's safetensors to pre-stack experts.
    This avoids the massive 45GB memory spike during mx.stack() in native mlx-lm loading,
    enabling it to run in <4GB of active RAM.
    """
    model_path = Path(model_path)
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config(model_path)
    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    
    # Check if model has a sanitize method; if not, no optimization needed
    if not hasattr(model, "sanitize"):
        print(f"Model {model_path.name} does not require MoE pre-stacking.")
        return

    # Fast check: see if it's already optimized using the new v2 scheme
    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
            if config_data.get("sparsify_optimized_v2", False):
                print(f"Model {model_path.name} is already optimized.")
                return
        except Exception:
            pass

    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
        
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
        
    print("Sanitizing weights (creating lazy stacking graphs)...")
    weights = model.sanitize(weights)
    
    # We will shard them into 2GB chunks to keep RAM well under 4GB during offline optimization
    max_file_size_bytes = 2 * (1 << 30)
    shards = []
    
    # First shard: ALL non-expert weights (base model)
    base_shard, base_shard_size = {}, 0
    expert_shard, expert_shard_size = {}, 0
    
    # Pop items to avoid keeping them in RAM
    for k in list(weights.keys()):
        v = weights.pop(k)
        if "switch_mlp" not in k:
            base_shard[k] = v
            base_shard_size += v.nbytes
        else:
            if expert_shard_size + v.nbytes > max_file_size_bytes and expert_shard:
                shards.append(expert_shard)
                expert_shard, expert_shard_size = {}, 0
            expert_shard[k] = v
            expert_shard_size += v.nbytes
            
    if base_shard:
        shards.insert(0, base_shard)  # Base shard is always first
    if expert_shard:
        shards.append(expert_shard)
        
    total_size = 0
    weight_map = {}
    
    import tempfile
    
    with tempfile.TemporaryDirectory(dir=model_path.parent) as temp_dir:
        temp_path = Path(temp_dir)
        
        for i, shard_dict in enumerate(shards):
            print(f"Optimizing shard {i+1}/{len(shards)}...")
            # Evaluate graph to materialize the stacked arrays
            mx.eval(list(shard_dict.values()))
            
            shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            shard_path = temp_path / shard_name
            
            mx.save_safetensors(str(shard_path), shard_dict)
            
            for k, v in shard_dict.items():
                total_size += v.nbytes
                weight_map[k] = shard_name
                
            # Free memory aggressively
            shard_dict.clear()
            mx.clear_cache()
            gc.collect()

        # Write the Hugging Face standard safetensors index
        index_data = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        with open(temp_path / "model.safetensors.index.json", "w") as f:
            json.dump(index_data, f, indent=2)

        with open(temp_path / "config.json", "w") as f:
            config["sparsify_optimized_v2"] = True
            json.dump(config, f, indent=2)
            
        # Copy all other non-safetensors config files
        for f in model_path.glob("*"):
            if not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json" and f.name != "config.json":
                if f.is_file():
                    shutil.copy(f, temp_path / f.name)
                    
        # If we successfully created everything, delete the old safetensors files
        for f in model_path.glob("model*.safetensors"):
            f.unlink()
        if (model_path / "model.safetensors.index.json").exists():
            (model_path / "model.safetensors.index.json").unlink()
            
        # Move the new optimized files into the model directory
        for f in temp_path.glob("*"):
            shutil.move(str(f), str(model_path / f.name))
