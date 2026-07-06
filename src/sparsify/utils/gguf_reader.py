"""GGUF metadata reader — extracts model metadata and tensor info without loading weights.

Uses the `gguf` Python package (GGUFReader) to parse the GGUF container header,
metadata key-value pairs, and tensor descriptors. No tensor data is mmap'd or
copied into Python memory, keeping the reader lightweight enough for profiling
multi-GB model files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from gguf import GGUFReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GGUF_FILE_TYPES: dict[int, str] = {
    0: "ALL_F32",
    1: "MOSTLY_F16",
    2: "MOSTLY_Q4_0",
    3: "MOSTLY_Q4_1",
    7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1",
    10: "MOSTLY_Q2_K",
    11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M",
    13: "MOSTLY_Q3_K_L",
    14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M",
    16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K",
    19: "MOSTLY_IQ2_XXS",
    20: "MOSTLY_IQ2_XS",
}

_LAYER_INDEX_RE = re.compile(r"(?:blk|layers?)\.([0-9]+)\.")

_COMPONENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"attn[_.]q"), "attention_q"),
    (re.compile(r"attn[_.]k"), "attention_k"),
    (re.compile(r"attn[_.]v"), "attention_v"),
    (re.compile(r"attn[_.](?:output|o)"), "attention_output"),
    (re.compile(r"ffn_gate"), "ffn_gate"),
    (re.compile(r"ffn_up"), "ffn_up"),
    (re.compile(r"ffn_down"), "ffn_down"),
    (re.compile(r"norm"), "norm"),
    (re.compile(r"token_embd|embed"), "embedding"),
    # 'output' but NOT when preceded by 'attn' (handled above)
    (re.compile(r"(?<!attn[_.])output"), "output"),
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GGUFMetadata:
    """Parsed GGUF file-level metadata."""

    file_path: Path
    file_size_bytes: int
    architecture: str
    model_name: str | None
    quantization_type: str | None
    block_count: int
    attention_head_count: int
    attention_head_count_kv: int
    embedding_length: int
    feed_forward_length: int
    context_length: int
    rope_dimension_count: int | None
    vocab_size: int | None
    raw_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["file_path"] = str(self.file_path)
        return d


@dataclass(slots=True)
class TensorInfo:
    """Descriptor for a single tensor stored in a GGUF file."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    n_elements: int
    size_bytes: int
    layer_index: int | None
    component: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_field_string(reader: GGUFReader, key: str, default: str | None = None) -> str | None:
    """Extract a UTF-8 string value from a GGUF metadata field."""
    field_obj = reader.fields.get(key)
    if field_obj is None:
        return default
    try:
        val = field_obj.contents()
        return str(val) if val is not None else default
    except Exception:
        return default


def _get_field_int(reader: GGUFReader, key: str, default: int = 0) -> int:
    """Extract an integer value from a GGUF metadata field."""
    field_obj = reader.fields.get(key)
    if field_obj is None:
        return default
    try:
        val = field_obj.contents()
        return int(val) if val is not None else default
    except Exception:
        return default


def _classify_component(tensor_name: str) -> str:
    """Map a tensor name to a high-level component category."""
    for pattern, component in _COMPONENT_PATTERNS:
        if pattern.search(tensor_name):
            return component
    return "other"


def _extract_layer_index(tensor_name: str) -> int | None:
    """Extract the transformer block index from a tensor name, if present."""
    m = _LAYER_INDEX_RE.search(tensor_name)
    return int(m.group(1)) if m else None


def _collect_raw_metadata(reader: GGUFReader) -> dict[str, Any]:
    """Collect all GGUF metadata fields into a plain Python dict.

    Best-effort: numeric scalars are stored as ``int``/``float``, strings as
    ``str``, and anything that cannot be decoded is stored as its ``repr``.
    """
    raw: dict[str, Any] = {}
    for key, field_obj in reader.fields.items():
        try:
            val = field_obj.contents()
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            elif isinstance(val, list) and len(val) > 100:
                val = val[:100] + [f"... and {len(val) - 100} more items"]
            raw[key] = val
        except Exception:
            raw[key] = repr(field_obj.parts)
    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_gguf_metadata(path: Path) -> GGUFMetadata:
    """Read GGUF metadata from *path* without loading tensor data.

    Parameters
    ----------
    path:
        Filesystem path to a ``.gguf`` file.

    Returns
    -------
    GGUFMetadata
        Parsed metadata including architecture, quantization, dimension
        parameters and the full raw metadata dict.
    """
    path = Path(path)
    reader = GGUFReader(str(path))

    architecture = _get_field_string(reader, "general.architecture", default="unknown") or "unknown"
    arch_prefix = f"{architecture}."

    file_type_int = _get_field_int(reader, "general.file_type", default=-1)
    quantization_type = GGUF_FILE_TYPES.get(file_type_int)

    return GGUFMetadata(
        file_path=path,
        file_size_bytes=path.stat().st_size,
        architecture=architecture,
        model_name=_get_field_string(reader, "general.name"),
        quantization_type=quantization_type,
        block_count=_get_field_int(reader, f"{arch_prefix}block_count"),
        attention_head_count=_get_field_int(reader, f"{arch_prefix}attention.head_count"),
        attention_head_count_kv=_get_field_int(reader, f"{arch_prefix}attention.head_count_kv"),
        embedding_length=_get_field_int(reader, f"{arch_prefix}embedding_length"),
        feed_forward_length=_get_field_int(reader, f"{arch_prefix}feed_forward_length"),
        context_length=_get_field_int(reader, f"{arch_prefix}context_length"),
        rope_dimension_count=(
            _get_field_int(reader, f"{arch_prefix}rope.dimension_count", default=-1)
            if reader.fields.get(f"{arch_prefix}rope.dimension_count") is not None
            else None
        ),
        vocab_size=(
            _get_field_int(reader, f"{arch_prefix}vocab_size", default=-1)
            if reader.fields.get(f"{arch_prefix}vocab_size") is not None
            else None
        ),
        raw_metadata=_collect_raw_metadata(reader),
    )


def list_tensors(path: Path) -> list[TensorInfo]:
    """List every tensor stored in a GGUF file with classification metadata.

    Parameters
    ----------
    path:
        Filesystem path to a ``.gguf`` file.

    Returns
    -------
    list[TensorInfo]
        One entry per tensor, including shape, dtype, byte-size, inferred
        layer index, and component classification.
    """
    path = Path(path)
    reader = GGUFReader(str(path))

    tensors: list[TensorInfo] = []
    for tensor in reader.tensors:
        name: str = tensor.name
        tensors.append(
            TensorInfo(
                name=name,
                shape=tuple(int(d) for d in tensor.shape),
                dtype=tensor.tensor_type.name,
                n_elements=int(tensor.n_elements),
                size_bytes=int(tensor.n_bytes),
                layer_index=_extract_layer_index(name),
                component=_classify_component(name),
            )
        )
    return tensors
