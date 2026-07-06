"""Experiment runner to orchestrate the 8 Sparsify validation experiments."""
from __future__ import annotations

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from rich.console import Console
from rich.table import Table

from sparsify.backends.mlx_backend import MLXBackend
from sparsify.experiments.hooks import patch_model_for_experimentation
from sparsify.experiments.evaluator import evaluate_perplexity
from sparsify.experiments.layer_skip import run_layer_importance_experiment, run_dynamic_layer_skip_experiment
from sparsify.experiments.head_pruning import run_head_importance_experiment, run_joint_pruning_experiment
from sparsify.experiments.sparsity_profile import run_static_sparsity_experiment, run_activation_sparsity_experiment
from sparsify.experiments.temporal_profile import run_temporal_activity_experiment, run_activation_predictability_experiment
from sparsify.utils.config import EXPORT_DIR, ensure_dirs

console = Console()


def execute_all_experiments(model_path: str, max_heads_to_test: int = 20) -> Dict[str, Any]:
    """Execute all 8 validation experiments and save a machine-readable JSON report."""
    ensure_dirs()
    
    console.print(f"[bold cyan]Starting Sparsify V1 Validation Experiments[/bold cyan]")
    console.print(f"Target model: [yellow]{model_path}[/yellow]\n")

    # 1. Initialize backend and load model
    backend = MLXBackend()
    if not backend.is_available():
        raise RuntimeError("MLX backend is not available. Apple Silicon with Metal is required.")
        
    console.print("Loading model into MLX...")
    model_info = backend.load_model(Path(model_path))
    
    model = backend._model
    tokenizer = backend._tokenizer

    # 2. Establish baseline quality metrics
    console.print("Calculating baseline perplexity (clean run)...")
    baseline_ppl = evaluate_perplexity(model, tokenizer)
    console.print(f"Baseline Perplexity: [green]{baseline_ppl:.4f}[/green]\n")

    # 3. Patch model with Sparsify hooks
    console.print("Patching model layers with Sparsify hooks...")
    bypass_hooks, head_hooks, sparsity_hooks = patch_model_for_experimentation(model)
    console.print(f"Patched {len(bypass_hooks)} layers with execution hooks.\n")

    # 4. Run experiments sequentially
    experiment_reports: Dict[str, Any] = {}

    # -- Exp 1: Layer Skipping
    console.print("[bold]Running Experiment 1: Layer Importance...[/bold]")
    exp1 = run_layer_importance_experiment(model, tokenizer, bypass_hooks, baseline_ppl)
    experiment_reports["exp_1_layer_importance"] = exp1
    
    # Extract skippable layers from Exp 1 for Exp 5
    skippable_layers = exp1["summary"]["skippable_layers"]

    # -- Exp 2: Head Masking
    console.print("[bold]Running Experiment 2: Head Masking...[/bold]")
    exp2 = run_head_importance_experiment(model, tokenizer, head_hooks, baseline_ppl, max_heads_to_test)
    experiment_reports["exp_2_head_importance"] = exp2
    
    # Extract maskable heads for Exp 5
    maskable_heads = exp2["summary"]["maskable_heads"]

    # -- Exp 3: Static Sparsity (Hot/Cold Components)
    console.print("[bold]Running Experiment 3: Static Sparsity...[/bold]")
    exp3 = run_static_sparsity_experiment(model, tokenizer, bypass_hooks)
    experiment_reports["exp_3_static_sparsity"] = exp3

    # -- Exp 4: Dynamic Layer Skip (Simulation)
    console.print("[bold]Running Experiment 4: Dynamic Layer Skipping...[/bold]")
    exp4 = run_dynamic_layer_skip_experiment(model, tokenizer, bypass_hooks, baseline_ppl)
    experiment_reports["exp_4_dynamic_layer_skip"] = exp4

    # -- Exp 5: Joint Multi-Component Pruning
    console.print("[bold]Running Experiment 5: Joint Pruning...[/bold]")
    exp5 = run_joint_pruning_experiment(
        model, tokenizer, bypass_hooks, head_hooks, skippable_layers, maskable_heads, baseline_ppl
    )
    experiment_reports["exp_5_joint_pruning"] = exp5

    # -- Exp 6: Activation Gating Sparsity
    console.print("[bold]Running Experiment 6: Activation Gating Sparsity...[/bold]")
    exp6 = run_activation_sparsity_experiment(model, tokenizer, sparsity_hooks, baseline_ppl)
    experiment_reports["exp_6_activation_gating_sparsity"] = exp6

    # -- Exp 7: Temporal Activity Profiling
    console.print("[bold]Running Experiment 7: Temporal Activity Profiling...[/bold]")
    exp7 = run_temporal_activity_experiment(model, tokenizer, bypass_hooks)
    experiment_reports["exp_7_temporal_profiling"] = exp7

    # -- Exp 8: Next-Token Activation Predictability
    console.print("[bold]Running Experiment 8: Activation Predictability...[/bold]")
    exp8 = run_activation_predictability_experiment(model, tokenizer, bypass_hooks)
    experiment_reports["exp_8_activation_predictability"] = exp8

    # Unload model to free VRAM
    backend.unload_model()

    # 5. Compile final validation report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "model_name": model_info.name,
        "architecture": model_info.architecture,
        "baseline_perplexity": baseline_ppl,
        "experiments": experiment_reports,
    }

    # Save to disk
    report_filename = f"experiments_report_{model_info.name}_{timestamp}.json"
    report_path = EXPORT_DIR / report_filename

    def numpy_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=numpy_serializer)

    console.print(f"\n[green]Completed all experiments.[/green] Saved report to [bold]{report_path}[/bold]\n")
    
    # Render final console summary table
    print_experiments_summary_table(report)

    return report


def print_experiments_summary_table(report: Dict[str, Any]) -> None:
    """Print a summary table of the validation experiments to the terminal."""
    table = Table(title="Sparsify V1 Research Validation Summary", title_style="bold cyan")
    table.add_column("Exp ID", style="bold")
    table.add_column("Experiment Name")
    table.add_column("Hypothesis Confirmed?", justify="center")
    table.add_column("Key Result / Metric")

    exps = report["experiments"]
    
    # Exp 1
    exp1 = exps["exp_1_layer_importance"]
    table.add_row(
        "1",
        "Layer Importance",
        "[green]YES[/green]" if exp1["summary"]["opportunity_confirmed"] else "[red]NO[/red]",
        f"Skippable layers: {exp1['summary']['skippable_layers']}",
    )
    
    # Exp 2
    exp2 = exps["exp_2_head_importance"]
    table.add_row(
        "2",
        "Head Importance",
        "[green]YES[/green]" if exp2["summary"]["opportunity_confirmed"] else "[red]NO[/red]",
        f"Maskable heads: {exp2['summary']['maskable_count']} tested",
    )
    
    # Exp 3
    exp3 = exps["exp_3_static_sparsity"]
    table.add_row(
        "3",
        "Static Sparsity",
        "[green]YES[/green]" if exp3["summary"]["opportunity_confirmed"] else "[red]NO[/red]",
        f"Cold components: {exp3['summary']['cold_components_percentage']:.1f}%",
    )
    
    # Exp 4
    exp4 = exps["exp_4_dynamic_layer_skip"]
    table.add_row(
        "4",
        "Dynamic Layer Skipping",
        "[green]YES[/green]" if exp4["passed_success_criteria"] else "[red]NO[/red]",
        f"Skipped layers: {exp4['layers_skipped_percentage']:.1f}% | PPL deg: {exp4['relative_perplexity_degradation']*100:.2f}%",
    )
    
    # Exp 5
    exp5 = exps["exp_5_joint_pruning"]
    if "error" in exp5:
        table.add_row("5", "Joint Pruning", "[red]N/A[/red]", "No components skippable")
    else:
        table.add_row(
            "5",
            "Joint Pruning",
            "[green]YES[/green]" if exp5["passed_success_criteria"] else "[red]NO[/red]",
            f"Compute saved: {exp5['estimated_compute_saved_percentage']:.1f}% | PPL deg: {exp5['relative_perplexity_degradation']*100:.2f}%",
        )
        
    # Exp 6
    exp6 = exps["exp_6_activation_gating_sparsity"]
    table.add_row(
        "6",
        "Activation Sparsity",
        "[green]YES[/green]" if exp6["passed_success_criteria"] else "[red]NO[/red]",
        f"FFN Sparsity: {exp6['average_natural_sparsity_percentage']:.1f}% | PPL deg: {exp6['relative_perplexity_degradation']*100:.2f}%",
    )
    
    # Exp 7
    exp7 = exps["exp_7_temporal_profiling"]
    table.add_row(
        "7",
        "Temporal Profiling",
        "[green]YES[/green]" if exp7["summary"]["passed_success_criteria"] else "[red]NO[/red]",
        f"Bursty components: {exp7['summary']['dynamic_bursty_count']} detected",
    )
    
    # Exp 8
    exp8 = exps["exp_8_activation_predictability"]
    table.add_row(
        "8",
        "Activation Predictability",
        "[green]YES[/green]" if exp8["passed_success_criteria"] else "[red]NO[/red]",
        f"ROC AUC: {exp8['prediction_roc_auc']:.3f}",
    )

    console.print(table)
