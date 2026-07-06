"""Expert Registry to index and manage MoE expert weight locations and metadata."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sparsify.utils.gguf_reader import read_gguf_metadata, list_tensors


@dataclass
class ExpertMetadata:
    """Metadata representing a single dynamic MoE expert block location and performance stats."""
    expert_id: int
    layer_id: int
    file_path: str
    tensor_keys: List[str]
    size_bytes: int
    load_count: int = 0
    load_time_total: float = 0.0
    last_loaded_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExpertRegistry:
    """Inode table style registry caching locations and stats of MoE expert blocks on disk."""

    def __init__(self, registry_file: Path | None = None) -> None:
        self.registry_file = registry_file or (Path.home() / ".sparsify" / "registry_cache.json")
        self.experts: Dict[Tuple[int, int], ExpertMetadata] = {}

    def load_cache(self) -> bool:
        """Load the cached registry index from disk if it exists."""
        if not self.registry_file.exists():
            return False
        try:
            with open(self.registry_file, "r") as f:
                data = json.load(f)
            for item in data.values():
                meta = ExpertMetadata(**item)
                self.experts[(meta.layer_id, meta.expert_id)] = meta
            return True
        except Exception:
            return False

    def save_cache(self) -> None:
        """Save the registry index cache to disk."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            f"{meta.layer_id}_{meta.expert_id}": meta.to_dict()
            for meta in self.experts.values()
        }
        with open(self.registry_file, "w") as f:
            json.dump(serializable, f, indent=2)

    def register_expert(
        self,
        layer_id: int,
        expert_id: int,
        file_path: Path,
        tensor_keys: List[str],
        size_bytes: int,
    ) -> None:
        """Add or update an expert metadata entry in the registry."""
        meta = ExpertMetadata(
            expert_id=expert_id,
            layer_id=layer_id,
            file_path=str(file_path),
            tensor_keys=tensor_keys,
            size_bytes=size_bytes
        )
        self.experts[(layer_id, expert_id)] = meta

    def get_expert(self, layer_id: int, expert_id: int) -> ExpertMetadata | None:
        """Look up expert metadata by layer and expert index."""
        return self.experts.get((layer_id, expert_id))

    def scan_directory(self, path: Path) -> int:
        """Scan a directory containing MoE model files and index all detected expert weights.

        Supports:
        - MLX shards (layer_{l}_expert_{e}.npz)
        - GGUF models containing ffn_gate / ffn_up / ffn_down
        - safetensors directory structures
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        # Scan for MLX npz shards
        npz_files = list(path.glob("**/*.npz"))
        for f in npz_files:
            name = f.stem
            # Expected format: layer_L_expert_E.npz
            if name.startswith("layer_") and "_expert_" in name:
                try:
                    parts = name.split("_")
                    layer_id = int(parts[1])
                    expert_id = int(parts[3])
                    size = f.stat().st_size
                    self.register_expert(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        file_path=f,
                        tensor_keys=["gate_proj.weight", "up_proj.weight", "down_proj.weight"],
                        size_bytes=size
                    )
                except (ValueError, IndexError):
                    continue

        # Scan for GGUF files
        gguf_files = list(path.glob("**/*.gguf"))
        for f in gguf_files:
            try:
                tensors = list_tensors(f)
                # Group GGUF tensors by layer and expert index
                # Example tensor name: blk.0.ffn_gate.0.weight
                for t in tensors:
                    if "ffn_gate" in t.name or "ffn_up" in t.name or "ffn_down" in t.name:
                        # Extract layer index and expert index
                        # blk.L.ffn_gate.E.weight
                        parts = t.name.split(".")
                        if len(parts) >= 5:
                            layer_id = int(parts[1])
                            expert_id = int(parts[3])
                            
                            # Check if already registered, otherwise create metadata
                            key = (layer_id, expert_id)
                            if key in self.experts:
                                meta = self.experts[key]
                                if t.name not in meta.tensor_keys:
                                    meta.tensor_keys.append(t.name)
                                    meta.size_bytes += t.size_bytes
                            else:
                                self.register_expert(
                                    layer_id=layer_id,
                                    expert_id=expert_id,
                                    file_path=f,
                                    tensor_keys=[t.name],
                                    size_bytes=t.size_bytes
                                )
            except Exception:
                continue

        # Scan for safetensors files
        st_files = list(path.glob("**/*.safetensors"))
        for f in st_files:
            try:
                # Read safetensors header without loading files
                with open(f, "rb") as fp:
                    header_size_bytes = fp.read(8)
                    header_size = int.from_bytes(header_size_bytes, byteorder="little")
                    header_json = fp.read(header_size).decode("utf-8")
                    header = json.loads(header_json)
                
                # Check for MoE keys: e.g. layers.L.feed_forward.experts.E....
                for t_name, t_info in header.items():
                    if "experts" in t_name or "mlp.experts" in t_name:
                        # Parse names: layers.0.mlp.experts.0.gate_proj.weight
                        # or model.layers.0.block_sparse_moe.experts.0.w1.weight
                        parts = t_name.split(".")
                        layer_id = None
                        expert_id = None
                        
                        # Find indices dynamically
                        for idx, p in enumerate(parts):
                            if p == "layers" and idx + 1 < len(parts):
                                layer_id = int(parts[idx+1])
                            if (p == "experts" or p == "block_sparse_moe") and idx + 1 < len(parts):
                                try:
                                    expert_id = int(parts[idx+1])
                                except ValueError:
                                    pass
                                    
                        if layer_id is not None and expert_id is not None:
                            key = (layer_id, expert_id)
                            t_size = (t_info["data_offsets"][1] - t_info["data_offsets"][0])
                            if key in self.experts:
                                meta = self.experts[key]
                                if t_name not in meta.tensor_keys:
                                    meta.tensor_keys.append(t_name)
                                    meta.size_bytes += t_size
                            else:
                                self.register_expert(
                                    layer_id=layer_id,
                                    expert_id=expert_id,
                                    file_path=f,
                                    tensor_keys=[t_name],
                                    size_bytes=t_size
                                )
            except Exception:
                continue

        # Save cache to avoid re-scanning
        self.save_cache()
        return len(self.experts)
