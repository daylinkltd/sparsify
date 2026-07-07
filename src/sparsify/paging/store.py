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
import threading
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
        self._shards_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        # Telemetry — all values measured.
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
        """One raw range read. Thread-safe (pread carries its own offset;
        telemetry under a lock) and GIL-releasing — the parallel fetch path
        runs many of these concurrently on worker threads."""
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

    @staticmethod
    def wrap_raw(raw: bytes, shape: List[int], dtype: str) -> mx.array:
        """Turn raw tensor bytes into an mx.array (call from the MLX thread)."""
        np_dtype, mx_view = _DTYPES[dtype]
        arr = mx.array(np.frombuffer(raw, dtype=np_dtype).reshape(shape))
        if mx_view is not None:
            arr = arr.view(mx_view)
        return arr

    def read_expert_slice_raw(self, tensor_name: str, expert_idx: int
                              ) -> Tuple[bytes, List[int], str]:
        """Raw bytes of ``tensor[expert_idx:expert_idx+1]`` (one pread).
        Safe to call from any thread; contains no MLX operations."""
        shard, dtype, shape, offset = self._locate(tensor_name)
        np_dtype, _ = _DTYPES[dtype]
        row_bytes = math.prod(shape[1:]) * np.dtype(np_dtype).itemsize
        if not 0 <= expert_idx < shape[0]:
            raise IndexError(f"expert {expert_idx} out of range for {tensor_name} {shape}")
        raw = self._pread(shard, offset + expert_idx * row_bytes, row_bytes,
                          f"{tensor_name}[{expert_idx}]")
        return raw, [1] + shape[1:], dtype

    def read_tensor_raw(self, tensor_name: str) -> Tuple[bytes, List[int], str]:
        """Raw bytes of one whole tensor (one pread). Thread-safe, no MLX."""
        shard, dtype, shape, offset = self._locate(tensor_name)
        np_dtype, _ = _DTYPES[dtype]
        nbytes = math.prod(shape) * np.dtype(np_dtype).itemsize
        raw = self._pread(shard, offset, nbytes, tensor_name)
        return raw, list(shape), dtype

    def _read_range(self, shard: _Shard, offset: int, nbytes: int,
                    shape: List[int], np_dtype, mx_view, what: str) -> mx.array:
        raw = self._pread(shard, offset, nbytes, what)
        arr = mx.array(np.frombuffer(raw, dtype=np_dtype).reshape(shape))
        if mx_view is not None:
            arr = arr.view(mx_view)
        return arr

    def read_expert_slice(self, tensor_name: str, expert_idx: int) -> mx.array:
        """Read ``tensor[expert_idx:expert_idx+1]`` from disk (one pread).

        For expert-stacked tensors (leading expert dimension)."""
        shard, dtype, shape, offset = self._locate(tensor_name)
        np_dtype, mx_view = _DTYPES[dtype]
        row_bytes = math.prod(shape[1:]) * np.dtype(np_dtype).itemsize
        if not 0 <= expert_idx < shape[0]:
            raise IndexError(f"expert {expert_idx} out of range for {tensor_name} {shape}")
        return self._read_range(
            shard, offset + expert_idx * row_bytes, row_bytes, [1] + shape[1:],
            np_dtype, mx_view, f"{tensor_name}[{expert_idx}]",
        )

    def read_tensor(self, tensor_name: str) -> mx.array:
        """Read one whole tensor from disk (one pread).

        For layouts that store each expert as its own tensor."""
        shard, dtype, shape, offset = self._locate(tensor_name)
        np_dtype, mx_view = _DTYPES[dtype]
        nbytes = math.prod(shape) * np.dtype(np_dtype).itemsize
        return self._read_range(shard, offset, nbytes, shape,
                                np_dtype, mx_view, tensor_name)

    def names(self) -> List[str]:
        return list(self._weight_map)

    def stats(self) -> Dict[str, float]:
        return {
            "reads": self.reads,
            "bytes_read": self.bytes_read,
            "read_seconds": round(self.read_seconds, 4),
        }
