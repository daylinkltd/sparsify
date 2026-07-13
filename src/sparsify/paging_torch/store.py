"""Storage layer: per-expert range reads from safetensors shards for PyTorch."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# safetensors dtype -> (numpy read dtype, torch target dtype)
_DTYPES = {
    "F64": (np.float64, torch.float64),
    "F32": (np.float32, torch.float32),
    "F16": (np.float16, torch.float16),
    "BF16": (np.uint16, torch.bfloat16),
    "I64": (np.int64, torch.int64),
    "I32": (np.int32, torch.int32),
    "U32": (np.uint32, torch.int32),
    "I16": (np.int16, torch.int16),
    "U16": (np.uint16, torch.int16),
    "I8": (np.int8, torch.int8),
    "U8": (np.uint8, torch.uint8),
    "BOOL": (np.bool_, torch.bool),
}


class _Shard:
    """Parsed header + open file descriptor for one safetensors file."""

    def __init__(self, path: Path) -> None:
        self.fd = os.open(path, os.O_RDONLY)
        header_len = int.from_bytes(os.pread(self.fd, 8, 0), "little")
        header = json.loads(os.pread(self.fd, header_len, 8))
        header.pop("__metadata__", None)
        self.data_start = 8 + header_len
        # name -> (dtype str, shape, start offset within data region)
        self.tensors: Dict[str, Tuple[str, List[int], int]] = {
            name: (info["dtype"], info["shape"], info["data_offsets"][0])
            for name, info in header.items()
        }

    def __del__(self):
        try:
            os.close(self.fd)
        except OSError:
            pass


class PyTorchExpertStore:
    """Reads individual expert slices from a model's safetensors shards as PyTorch tensors."""

    def __init__(self, model_path: Path, device: torch.device | str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(device)

        index_file = self.model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                self._weight_map: Dict[str, str] = json.load(f)["weight_map"]
        elif (self.model_path / "model.safetensors").exists():
            shard = _Shard(self.model_path / "model.safetensors")
            self._weight_map = {k: "model.safetensors" for k in shard.tensors}
        else:
            raise FileNotFoundError(f"No safetensors weights found in {self.model_path}")

        self._shards: Dict[str, _Shard] = {}
        self._shards_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        # Telemetry
        self.reads = 0
        self.bytes_read = 0
        self.read_seconds = 0.0

    def _locate(self, tensor_name: str) -> Tuple[_Shard, str, List[int], int]:
        shard_name = self._weight_map[tensor_name]
        shard = self._shards.get(shard_name)
        if shard is None:
            with self._shards_lock:
                shard = self._shards.get(shard_name)
                if shard is None:
                    shard = _Shard(self.model_path / shard_name)
                    self._shards[shard_name] = shard
        dtype, shape, offset = shard.tensors[tensor_name]
        return shard, dtype, shape, offset

    def __contains__(self, tensor_name: str) -> bool:
        return tensor_name in self._weight_map

    def tensor_info(self, tensor_name: str) -> Tuple[List[int], str, int]:
        """Return (shape, dtype string, nbytes) without reading tensor data."""
        _, dtype, shape, _ = self._locate(tensor_name)
        np_dtype, _ = _DTYPES[dtype]
        return shape, dtype, math.prod(shape) * np.dtype(np_dtype).itemsize

    def _pread(self, shard: _Shard, offset: int, nbytes: int, what: str) -> bytes:
        t0 = time.perf_counter()
        raw = os.pread(shard.fd, nbytes, shard.data_start + offset)
        if len(raw) != nbytes:
            raise IOError(f"short read for {what}: {len(raw)} of {nbytes} bytes")
        dt = time.perf_counter() - t0
        with self._stats_lock:
            self.read_seconds += dt
            self.reads += 1
            self.bytes_read += nbytes
        return raw

    def wrap_raw(self, raw: bytes, shape: List[int], dtype: str) -> torch.Tensor:
        """Turn raw tensor bytes into a torch.Tensor on target device."""
        np_dtype, torch_type = _DTYPES[dtype]
        np_arr = np.frombuffer(raw, dtype=np_dtype).copy().reshape(shape)
        t = torch.from_numpy(np_arr)
        if dtype == "BF16":
            t = t.view(torch.bfloat16)
        else:
            t = t.to(torch_type)
        return t.to(self.device)

    def _read_range(self, shard: _Shard, offset: int, nbytes: int,
                    shape: List[int], dtype: str, what: str) -> torch.Tensor:
        raw = self._pread(shard, offset, nbytes, what)
        return self.wrap_raw(raw, shape, dtype)

    def read_expert_slice_raw(self, tensor_name: str, expert_idx: int
                              ) -> Tuple[bytes, List[int], str]:
        """Raw bytes of ``tensor[expert_idx:expert_idx+1]`` (one pread)."""
        shard, dtype, shape, offset = self._locate(tensor_name)
        np_dtype, _ = _DTYPES[dtype]
        row_bytes = math.prod(shape[1:]) * np.dtype(np_dtype).itemsize
        if not 0 <= expert_idx < shape[0]:
            raise IndexError(f"expert {expert_idx} out of range for {tensor_name} {shape}")
        raw = self._pread(shard, offset + expert_idx * row_bytes, row_bytes,
                          f"{tensor_name}[{expert_idx}]")
        return raw, [1] + shape[1:], dtype

    def read_tensor_raw(self, tensor_name: str) -> Tuple[bytes, List[int], str]:
        """Raw bytes of one whole tensor (one pread)."""
        shard, dtype, shape, offset = self._locate(tensor_name)
        np_dtype, _ = _DTYPES[dtype]
        nbytes = math.prod(shape) * np.dtype(np_dtype).itemsize
        raw = self._pread(shard, offset, nbytes, tensor_name)
        return raw, list(shape), dtype

    def read_expert_slice(self, tensor_name: str, expert_idx: int) -> torch.Tensor:
        """Read tensor[expert_idx:expert_idx+1] from disk as torch.Tensor."""
        shard, dtype, shape, offset = self._locate(tensor_name)
        np_dtype, _ = _DTYPES[dtype]
        row_bytes = math.prod(shape[1:]) * np.dtype(np_dtype).itemsize
        if not 0 <= expert_idx < shape[0]:
            raise IndexError(f"expert {expert_idx} out of range for {tensor_name} {shape}")
        return self._read_range(
            shard, offset + expert_idx * row_bytes, row_bytes, [1] + shape[1:],
            dtype, f"{tensor_name}[{expert_idx}]",
        )

    def read_tensor(self, tensor_name: str) -> torch.Tensor:
        """Read one whole tensor from disk as torch.Tensor."""
        shard, dtype, shape, offset = self._locate(tensor_name)
        np_dtype, _ = _DTYPES[dtype]
        nbytes = math.prod(shape) * np.dtype(np_dtype).itemsize
        return self._read_range(shard, offset, nbytes, shape,
                                dtype, tensor_name)

    def names(self) -> List[str]:
        return list(self._weight_map)

    def stats(self) -> Dict[str, float]:
        return {
            "reads": self.reads,
            "bytes_read": self.bytes_read,
            "read_seconds": round(self.read_seconds, 4),
        }
