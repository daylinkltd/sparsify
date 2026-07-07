"""Storage layer: per-expert range reads from safetensors shards.

Reads expert slices straight from the safetensors container (documented
format: little-endian u64 header length, JSON header with per-tensor
``dtype``/``shape``/``data_offsets``, then a flat data region). Because
expert weights are stacked on the *leading* axis and stored row-major, one
expert's slice is a single contiguous byte range — each read is one
``os.pread`` of exactly the bytes needed. No mmap tricks, no full-tensor
loads, no framework-conversion quirks (bfloat16 included).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

# safetensors dtype -> (numpy read dtype, mlx view dtype or None)
_DTYPES = {
    "F64": (np.float64, None),
    "F32": (np.float32, None),
    "F16": (np.float16, None),
    "BF16": (np.uint16, mx.bfloat16),  # numpy has no bf16; bit-reinterpret
    "I64": (np.int64, None),
    "I32": (np.int32, None),
    "U32": (np.uint32, None),
    "I16": (np.int16, None),
    "U16": (np.uint16, None),
    "I8": (np.int8, None),
    "U8": (np.uint8, None),
    "BOOL": (np.bool_, None),
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


class SafetensorsExpertStore:
    """Reads individual expert slices from a model's safetensors shards."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)

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

        # Telemetry — all values measured.
        self.reads = 0
        self.bytes_read = 0
        self.read_seconds = 0.0

    def _locate(self, tensor_name: str) -> Tuple[_Shard, str, List[int], int]:
        shard_name = self._weight_map[tensor_name]
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

    def read_expert_slice(self, tensor_name: str, expert_idx: int) -> mx.array:
        """Read ``tensor[expert_idx:expert_idx+1]`` from disk (one pread)."""
        shard, dtype, shape, offset = self._locate(tensor_name)
        np_dtype, mx_view = _DTYPES[dtype]
        itemsize = np.dtype(np_dtype).itemsize
        row_elems = math.prod(shape[1:])
        row_bytes = row_elems * itemsize
        if not 0 <= expert_idx < shape[0]:
            raise IndexError(f"expert {expert_idx} out of range for {tensor_name} {shape}")

        t0 = time.perf_counter()
        raw = os.pread(shard.fd, row_bytes, shard.data_start + offset + expert_idx * row_bytes)
        if len(raw) != row_bytes:
            raise IOError(f"short read for {tensor_name}[{expert_idx}]: "
                          f"{len(raw)} of {row_bytes} bytes")
        arr = mx.array(np.frombuffer(raw, dtype=np_dtype).reshape([1] + shape[1:]))
        if mx_view is not None:
            arr = arr.view(mx_view)
        self.read_seconds += time.perf_counter() - t0
        self.reads += 1
        self.bytes_read += row_bytes
        return arr

    def stats(self) -> Dict[str, float]:
        return {
            "reads": self.reads,
            "bytes_read": self.bytes_read,
            "read_seconds": round(self.read_seconds, 4),
        }
