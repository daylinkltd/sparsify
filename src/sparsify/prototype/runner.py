"""Runner to initialize, compile, serialize, and run the MoE research prototype."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

from rich.console import Console
from rich.table import Table

import mlx.core as mx
from sparsify.prototype.lru_cache import ExpertLRUCache
from sparsify.prototype.moe_model import MoeTransformer
from sparsify.utils.config import SPARSIFY_DIR

console = Console()


def compile_and_save_prototype(experts_dir: Path) -> Tuple[MoeTransformer, Dict[str, Any]]:
    """Initialize a 100M parameter MoE model and serialize experts to disk."""
    console.print("[bold yellow]Initializing 100M parameter MoE model structure...[/bold yellow]")
    
    # Bounded cache pointing to target experts directory
    cache = ExpertLRUCache(max_active_experts=8, experts_dir=experts_dir)
    
    # 1. Instantiate the MoE model
    model = MoeTransformer(
        vocab_size=4000,
        d_model=256,
        n_heads=8,
        n_layers=8,
        n_experts=16,
        cache=cache
    )
    
    # 2. Serialize experts to disk (mock random weights)
    console.print(f"Serializing 128 experts (8 layers * 16 experts) to disk under [bold]{experts_dir}[/bold]...")
    
    for l_idx in range(8):
        for e_idx in range(16):
            # Generate random FFN weight matrices
            gate_w = mx.random.normal((1024, 256), dtype=mx.float32)
            up_w = mx.random.normal((1024, 256), dtype=mx.float32)
            down_w = mx.random.normal((256, 1024), dtype=mx.float32)
            
            expert_path = experts_dir / f"layer_{l_idx}_expert_{e_idx}.npz"
            mx.savez(
                str(expert_path),
                **{
                    "gate_proj.weight": gate_w,
                    "up_proj.weight": up_w,
                    "down_proj.weight": down_w
                }
            )
            
    # Calculate parameter count sizes
    def count_params(d: Any) -> int:
        if isinstance(d, dict):
            return sum(count_params(v) for v in d.values())
        elif isinstance(d, list):
            return sum(count_params(v) for v in d)
        elif hasattr(d, "size"):
            return d.size
        return 0

    shared_params = count_params(model.parameters())
    expert_params_one = 3 * 256 * 1024
    total_expert_params = 128 * expert_params_one
    total_model_params = shared_params + total_expert_params
    
    stats = {
        "shared_parameters": shared_params,
        "parameters_per_expert": expert_params_one,
        "total_experts": 128,
        "total_expert_parameters": total_expert_params,
        "total_model_parameters": total_model_params,
    }
    
    console.print("[green]Successfully initialized and stored model weights.[/green]")
    return model, stats


def execute_prototype_inference(model: MoeTransformer, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Run causality-masked sequence completion and measure working active memory footprint."""
    console.print("\n[bold yellow]Running causally masked sequence completion inference...[/bold yellow]")
    
    # Clear cache before running
    model.cache.clear()
    
    # 5 tokens in input sequence (prefill)
    prompt_tokens = [15, 342, 999, 104, 2580]
    inputs = mx.array(prompt_tokens).reshape(1, -1)
    
    # Measure execution speed and active experts loaded
    start_time = time.perf_counter()
    logits = model(inputs)
    mx.eval(logits)
    duration = time.perf_counter() - start_time
    
    # Retrieve dynamic loading metrics
    hits = model.cache.hits
    misses = model.cache.misses
    evictions = model.cache.evictions
    active_expert_count = model.cache.active_count
    
    # Active parameters in memory (shared + bounded cache)
    active_expert_params = active_expert_count * stats["parameters_per_expert"]
    total_active_params = stats["shared_parameters"] + active_expert_params
    
    ratio = total_active_params / stats["total_model_parameters"]
    
    results = {
        "duration_seconds": duration,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_evictions": evictions,
        "active_experts_in_memory": active_expert_count,
        "active_parameters_bytes": total_active_params,
        "total_parameters_bytes": stats["total_model_parameters"],
        "active_to_total_ratio": ratio,
    }
    
    # Render rich results table
    render_prototype_results_table(stats, results)
    
    return results


def render_prototype_results_table(stats: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Print a summary table of the Mixture-of-Experts prototype metrics."""
    table = Table(title="Sparsify MoE Research Prototype Summary (SP-010)", title_style="bold yellow")
    table.add_column("Parameter Classification", style="bold")
    table.add_column("Value / Count", justify="right")
    table.add_column("Memory Footprint", justify="right")
    
    # Format helper
    def fmt(val: int) -> str:
        return f"{val:,}"
        
    table.add_row("Shared Parameters (Always Active)", fmt(stats["shared_parameters"]), f"{stats['shared_parameters']*4/1024/1024:.2f} MB")
    table.add_row("Total Expert Parameters (On-Disk)", fmt(stats["total_expert_parameters"]), f"{stats['total_expert_parameters']*4/1024/1024:.2f} MB")
    table.add_row("Total Model Parameters (Stored Intelligence)", fmt(stats["total_model_parameters"]), f"{stats['total_model_parameters']*4/1024/1024:.2f} MB")
    
    table.add_section()
    
    table.add_row("Active Experts in Cache", fmt(results["active_experts_in_memory"]), f"{results['active_experts_in_memory'] * stats['parameters_per_expert']*4/1024/1024:.2f} MB")
    table.add_row("Active Working Parameters (Active Intelligence)", fmt(results["active_parameters_bytes"]), f"{results['active_parameters_bytes']*4/1024/1024:.2f} MB")
    
    ratio_pct = results["active_to_total_ratio"] * 100.0
    status = "[green]PASS (<= 11%)[/green]" if ratio_pct <= 11.0 else "[red]FAIL[/red]"
    table.add_row("Sparsify Metric (Active / Total)", f"{ratio_pct:.3f}%", status)
    
    console.print(table)
    
    # Print cache statistics
    console.print(f"[bold]Expert Loading Telemetry:[/bold]")
    console.print(f"  Cache Hits: [green]{results['cache_hits']}[/green]")
    console.print(f"  Cache Misses (Disk Loads): [red]{results['cache_misses']}[/red]")
    console.print(f"  Cache Evictions (Memory Cleared): [cyan]{results['cache_evictions']}[/cyan]")
    console.print(f"  Execution Time: [bold]{results['duration_seconds']*1000.0:.2f} ms[/bold]\n")
