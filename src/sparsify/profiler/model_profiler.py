"""Static model profiler — builds a full memory profile from a GGUF file.

Combines GGUF metadata and tensor descriptors to produce per-layer breakdowns,
component summaries, and KV-cache memory estimates across common context lengths.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from sparsify.profiler.metrics import (
    ComponentMemory,
    KVCacheEstimate,
    LayerProfile,
    ModelProfile,
)
from sparsify.utils.gguf_reader import (
    GGUFMetadata,
    TensorInfo,
    list_tensors,
    read_gguf_metadata,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KV_CACHE_CONTEXT_LENGTHS: list[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_bytes(n: int) -> str:
    """Return a human-readable byte string (e.g. ``'1.23 GB'``)."""
    if n < 0:
        return f"-{_format_bytes(-n)}"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024.0  # type: ignore[assignment]
    return f"{n:.2f} TB"


def _build_component_memory(
    name: str,
    tensors: list[TensorInfo],
    total_bytes: int,
) -> ComponentMemory:
    """Aggregate a list of tensors into a single ``ComponentMemory``."""
    size = sum(t.size_bytes for t in tensors)
    pct = (size / total_bytes * 100.0) if total_bytes > 0 else 0.0
    return ComponentMemory(
        name=name,
        size_bytes=size,
        percentage=round(pct, 2),
        tensor_count=len(tensors),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def profile_model(path: Path) -> ModelProfile:
    """Build a complete static memory profile for the GGUF model at *path*.

    Parameters
    ----------
    path:
        Filesystem path to a ``.gguf`` file.

    Returns
    -------
    ModelProfile
        Fully populated profile including per-layer breakdowns, component
        summaries, and KV-cache estimates.
    """
    path = Path(path)
    metadata: GGUFMetadata = read_gguf_metadata(path)
    tensors: list[TensorInfo] = list_tensors(path)

    total_weight_bytes = sum(t.size_bytes for t in tensors)
    parameter_count = sum(t.n_elements for t in tensors)

    # -- Group tensors by layer index ----------------------------------------
    layers: dict[int, list[TensorInfo]] = defaultdict(list)
    non_layer_tensors: list[TensorInfo] = []
    for t in tensors:
        if t.layer_index is not None:
            layers[t.layer_index].append(t)
        else:
            non_layer_tensors.append(t)

    # -- Per-layer profiles --------------------------------------------------
    layer_profiles: list[LayerProfile] = []
    for idx in sorted(layers.keys()):
        layer_tensors = layers[idx]
        layer_size = sum(t.size_bytes for t in layer_tensors)

        # Group by component within the layer
        comp_groups: dict[str, list[TensorInfo]] = defaultdict(list)
        for t in layer_tensors:
            comp_groups[t.component].append(t)

        components = {
            comp: _build_component_memory(comp, ctensors, total_weight_bytes)
            for comp, ctensors in comp_groups.items()
        }

        pct = (layer_size / total_weight_bytes * 100.0) if total_weight_bytes > 0 else 0.0
        layer_profiles.append(
            LayerProfile(
                layer_index=idx,
                total_size_bytes=layer_size,
                components=components,
                percentage_of_model=round(pct, 2),
            )
        )

    # -- Aggregate component summary across ALL tensors ----------------------
    all_comp_groups: dict[str, list[TensorInfo]] = defaultdict(list)
    for t in tensors:
        all_comp_groups[t.component].append(t)

    component_summary = {
        comp: _build_component_memory(comp, ctensors, total_weight_bytes)
        for comp, ctensors in all_comp_groups.items()
    }

    # -- Embedding & output head sizes ---------------------------------------
    embedding_bytes = sum(t.size_bytes for t in tensors if t.component == "embedding")
    output_head_bytes = sum(t.size_bytes for t in tensors if t.component == "output")

    # -- KV cache estimates --------------------------------------------------
    head_dim = (
        metadata.embedding_length // metadata.attention_head_count
        if metadata.attention_head_count > 0
        else 0
    )
    n_layers = metadata.block_count
    n_kv_heads = metadata.attention_head_count_kv or metadata.attention_head_count
    # KV per token = 2 (K+V) * n_layers * n_kv_heads * head_dim * 2 bytes (F16)
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * 2

    kv_cache_estimates: list[KVCacheEstimate] = []
    for ctx_len in _KV_CACHE_CONTEXT_LENGTHS:
        if ctx_len > metadata.context_length and metadata.context_length > 0:
            break
        batch_size = 1
        total_kv = kv_bytes_per_token * ctx_len * batch_size
        kv_cache_estimates.append(
            KVCacheEstimate(
                context_length=ctx_len,
                batch_size=batch_size,
                bytes_per_token=kv_bytes_per_token,
                total_size_bytes=total_kv,
                dtype="F16",
            )
        )

    return ModelProfile(
        model_path=str(path),
        model_name=metadata.model_name,
        file_size_bytes=metadata.file_size_bytes,
        architecture=metadata.architecture,
        quantization=metadata.quantization_type,
        parameter_count=parameter_count,
        layer_count=metadata.block_count,
        head_count=metadata.attention_head_count,
        kv_head_count=n_kv_heads,
        embedding_dim=metadata.embedding_length,
        ffn_dim=metadata.feed_forward_length,
        context_length=metadata.context_length,
        total_weight_bytes=total_weight_bytes,
        layer_profiles=layer_profiles,
        component_summary=component_summary,
        kv_cache_estimates=kv_cache_estimates,
        embedding_bytes=embedding_bytes,
        output_head_bytes=output_head_bytes,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def format_profile_table(profile: ModelProfile, *, verbose: bool = False) -> str:
    """Render *profile* as a rich-formatted table string.

    Parameters
    ----------
    profile:
        A ``ModelProfile`` produced by :func:`profile_model`.
    verbose:
        If ``True``, include a per-layer breakdown section.

    Returns
    -------
    str
         ANSI-coloured text suitable for terminal display.
    """
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=100)

    # ── Model Summary ──────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Model Summary[/bold cyan]")
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Key", style="bold")
    summary_table.add_column("Value")

    rows: list[tuple[str, str]] = [
        ("Model Name", profile.model_name or "N/A"),
        ("Architecture", profile.architecture),
        ("Quantization", profile.quantization or "N/A"),
        ("Parameters", f"{profile.parameter_count:,}"),
        ("Layers", str(profile.layer_count)),
        ("Attention Heads", f"{profile.head_count} (KV: {profile.kv_head_count})"),
        ("Embedding Dim", str(profile.embedding_dim)),
        ("FFN Dim", str(profile.ffn_dim)),
        ("Context Length", f"{profile.context_length:,}"),
        ("File Size", _format_bytes(profile.file_size_bytes)),
        ("Total Weight Size", _format_bytes(profile.total_weight_bytes)),
        ("Embedding Size", _format_bytes(profile.embedding_bytes)),
        ("Output Head Size", _format_bytes(profile.output_head_bytes)),
    ]
    for key, value in rows:
        summary_table.add_row(key, value)
    console.print(summary_table)

    # ── Memory Breakdown ───────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Memory Breakdown by Component[/bold cyan]")
    mem_table = Table(show_header=True, header_style="bold magenta")
    mem_table.add_column("Component", style="bold")
    mem_table.add_column("Size", justify="right")
    mem_table.add_column("%", justify="right")
    mem_table.add_column("Tensors", justify="right")

    for comp_name in sorted(profile.component_summary.keys()):
        cm = profile.component_summary[comp_name]
        mem_table.add_row(
            cm.name,
            _format_bytes(cm.size_bytes),
            f"{cm.percentage:.1f}%",
            str(cm.tensor_count),
        )
    console.print(mem_table)

    # ── KV Cache Estimates ─────────────────────────────────────────────────
    if profile.kv_cache_estimates:
        console.print()
        console.rule("[bold cyan]KV Cache Estimates (batch=1, F16)[/bold cyan]")
        kv_table = Table(show_header=True, header_style="bold magenta")
        kv_table.add_column("Context Length", justify="right")
        kv_table.add_column("Estimated Size", justify="right")

        for est in profile.kv_cache_estimates:
            kv_table.add_row(
                f"{est.context_length:,}",
                _format_bytes(est.total_size_bytes),
            )
        console.print(kv_table)

    # ── Per-Layer Breakdown (verbose) ──────────────────────────────────────
    if verbose and profile.layer_profiles:
        console.print()
        console.rule("[bold cyan]Per-Layer Breakdown[/bold cyan]")
        layer_table = Table(show_header=True, header_style="bold magenta")
        layer_table.add_column("Layer", justify="right")
        layer_table.add_column("Size", justify="right")
        layer_table.add_column("% of Model", justify="right")
        layer_table.add_column("Components")

        for lp in profile.layer_profiles:
            comp_strs = ", ".join(
                f"{c.name}: {_format_bytes(c.size_bytes)}"
                for c in sorted(lp.components.values(), key=lambda x: -x.size_bytes)
            )
            layer_table.add_row(
                str(lp.layer_index),
                _format_bytes(lp.total_size_bytes),
                f"{lp.percentage_of_model:.1f}%",
                comp_strs,
            )
        console.print(layer_table)

    console.print()
    return buf.getvalue()
