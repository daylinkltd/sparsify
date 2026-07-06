"""Research experiment SP-009: Domain Locality Analysis in LLMs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from rich.console import Console
from rich.table import Table

import mlx.core as mx
from sparsify.backends.mlx_backend import MLXBackend
from sparsify.experiments.hooks import patch_model_for_experimentation
from sparsify.experiments.prompts_bank import PROMPT_BANK
from sparsify.experiments.plots import (
    plot_layer_heatmap,
    plot_attention_head_heatmap,
    plot_similarity_matrix,
    plot_pca_scatter
)
from sparsify.utils.config import EXPORT_DIR, ensure_dirs

console = Console()


def run_domain_locality_experiment(
    model_path: str,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    """Execute Research Experiment SP-009: Domain Locality Analysis.

    Measures layer and head activation norms across 6 domains (50 prompts each),
    calculates Cosine Overlap, Jaccard Similarity, Silhouette Score, and PCA projections.
    """
    if output_dir is None:
        output_dir = EXPORT_DIR / "domain_locality"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Starting SP-009 Domain Locality Analysis[/bold cyan]")
    console.print(f"Target model: [bold]{model_path}[/bold]")

    # 1. Load model using MLXBackend
    backend = MLXBackend()
    console.print("Loading model and tokenizer...")
    backend.load_model(model_path)
    model = backend._model
    tokenizer = backend._tokenizer

    # 2. Patch model layers with execution hooks
    console.print("Patching model layers with Sparsify telemetry hooks...")
    bypass_hooks, head_hooks, sparsity_hooks = patch_model_for_experimentation(model)
    n_layers = len(bypass_hooks)
    n_heads = head_hooks[0].original_attention.n_heads if head_hooks else 32
    
    # Total dimension of feature vector:
    # 16 layers (Attn ratio) + 16 layers (FFN ratio) + (16 layers * 32 heads) = 544 dims
    feature_dim = n_layers + n_layers + (n_layers * n_heads)

    # 3. Collect activations across all domains and prompts
    domains = list(PROMPT_BANK.keys())
    all_vectors: List[np.ndarray] = []
    prompt_labels: List[str] = []
    
    console.print("Executing prompt bank evaluation...")
    
    for d_idx, domain in enumerate(domains):
        prompts = PROMPT_BANK[domain]
        console.print(f"  Domain [bold green]{domain}[/bold green]: processing {len(prompts)} prompts...")
        
        for p_idx, prompt in enumerate(prompts):
            tokens = tokenizer.encode(prompt)
            inputs = mx.array(tokens).reshape(1, -1)
            
            # Clear caches
            mx.clear_cache()
            
            # Run prefill forward pass
            _ = model(inputs)
            mx.eval(_)
            
            # Extract activation telemetry norms
            vector = []
            
            # Layer Attention ratios: ||r_attn|| / ||x||
            for bh, hh in zip(bypass_hooks, head_hooks):
                ratio = hh.last_output_norm / bh.last_input_norm if bh.last_input_norm > 0 else 0.0
                vector.append(ratio)
                
            # Layer FFN ratios: ||r_ffn|| / ||x||
            for bh, sh in zip(bypass_hooks, sparsity_hooks):
                ratio = sh.last_output_norm / bh.last_input_norm if bh.last_input_norm > 0 else 0.0
                vector.append(ratio)
                
            # Relative head importance in layer: head_norm / sum(all_head_norms)
            for hh in head_hooks:
                total_head_norm = sum(hh.last_head_norms)
                for hn in hh.last_head_norms:
                    ratio = hn / total_head_norm if total_head_norm > 0 else 0.0
                    vector.append(ratio)
                    
            all_vectors.append(np.array(vector, dtype=np.float32))
            prompt_labels.append(domain)

    # Convert to NumPy array
    X = np.stack(all_vectors, axis=0)  # Shape: (300, 544)
    
    # Unload model to release Metal VRAM
    backend.unload_model()

    # 4. Math Calculations: Average activation profiles per domain
    domain_averages: Dict[str, np.ndarray] = {}
    for domain in domains:
        mask = [lbl == domain for lbl in prompt_labels]
        domain_averages[domain] = np.mean(X[mask], axis=0)

    # 5. Extract Heatmap Coordinates
    # Layer Heatmaps
    attn_heatmap_data = np.zeros((n_layers, len(domains)), dtype=np.float32)
    ffn_heatmap_data = np.zeros((n_layers, len(domains)), dtype=np.float32)
    
    for d_idx, domain in enumerate(domains):
        avg_vec = domain_averages[domain]
        attn_heatmap_data[:, d_idx] = avg_vec[0:n_layers]
        ffn_heatmap_data[:, d_idx] = avg_vec[n_layers:2*n_layers]
        
    # Head Heatmap
    head_heatmap_data = np.zeros((n_layers * n_heads, len(domains)), dtype=np.float32)
    for d_idx, domain in enumerate(domains):
        avg_vec = domain_averages[domain]
        head_heatmap_data[:, d_idx] = avg_vec[2*n_layers:]

    # 6. Cosine Overlap Matrix
    overlap_matrix = np.zeros((len(domains), len(domains)), dtype=np.float32)
    for i, d_i in enumerate(domains):
        for j, d_j in enumerate(domains):
            v_i = domain_averages[d_i]
            v_j = domain_averages[d_j]
            norm_i = np.linalg.norm(v_i)
            norm_j = np.linalg.norm(v_j)
            overlap_matrix[i, j] = np.dot(v_i, v_j) / (norm_i * norm_j) if norm_i * norm_j > 0 else 0.0

    # 7. Jaccard Similarity Matrix
    jaccard_matrix = np.zeros((len(domains), len(domains)), dtype=np.float32)
    # Define top active components as the indices in the top 10% highest values
    top_k = int(feature_dim * 0.10)  # ~54 components
    
    active_sets: Dict[str, set] = {}
    for domain in domains:
        avg_vec = domain_averages[domain]
        top_indices = set(np.argsort(avg_vec)[-top_k:])
        active_sets[domain] = top_indices
        
    for i, d_i in enumerate(domains):
        for j, d_j in enumerate(domains):
            set_i = active_sets[d_i]
            set_j = active_sets[d_j]
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))
            jaccard_matrix[i, j] = intersection / union if union > 0 else 0.0

    # 8. Cluster Analysis: Silhouette Score
    # Distance matrix: Euclidean distance between all prompt vectors
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i in range(n_samples):
        for j in range(n_samples):
            dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])

    silhouette_scores = []
    for i in range(n_samples):
        curr_label = prompt_labels[i]
        
        # a_i: mean intra-cluster distance
        same_cluster_indices = [idx for idx, lbl in enumerate(prompt_labels) if lbl == curr_label and idx != i]
        a_i = np.mean(dist_matrix[i, same_cluster_indices]) if same_cluster_indices else 0.0
        
        # b_i: min mean inter-cluster distance to any other cluster
        b_i = float("inf")
        for other_label in domains:
            if other_label == curr_label:
                continue
            other_cluster_indices = [idx for idx, lbl in enumerate(prompt_labels) if lbl == other_label]
            mean_dist = np.mean(dist_matrix[i, other_cluster_indices])
            if mean_dist < b_i:
                b_i = mean_dist
                
        # s_i
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
        silhouette_scores.append(s_i)
        
    avg_silhouette_score = float(np.mean(silhouette_scores))

    # 9. Principal Component Analysis (PCA)
    X_centered = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_pca = np.dot(X_centered, Vt[:2, :].T)  # Shape: (300, 2)

    # 10. Generate Plot Artifacts
    console.print("Generating visualization charts...")
    plot_layer_heatmap(attn_heatmap_data, ffn_heatmap_data, domains, output_dir / "layer_activation_heatmap.png")
    plot_attention_head_heatmap(head_heatmap_data, domains, output_dir / "attention_head_heatmap.png")
    plot_similarity_matrix(overlap_matrix, domains, "Domain Activation Cosine Overlap Matrix", output_dir / "domain_overlap_matrix.png", cmap="Blues")
    plot_similarity_matrix(jaccard_matrix, domains, "Domain Activation Jaccard Similarity (Top 10% Components)", output_dir / "jaccard_similarity_matrix.png", cmap="Purples")
    plot_pca_scatter(X_pca, prompt_labels, domains, output_dir / "pca_clusters.png")

    # 11. Interpret Success & Failure Rules
    # Compute average overlap between unrelated domains (non-diagonal elements)
    diagonal_mask = np.eye(len(domains), dtype=bool)
    unrelated_overlap = float(np.mean(overlap_matrix[~diagonal_mask]))
    
    passed_success = unrelated_overlap < 0.70 and avg_silhouette_score > 0.50
    passed_failure = unrelated_overlap > 0.90 or avg_silhouette_score < 0.20
    
    # Formulate recommendation
    if passed_success:
        recommendation = (
            "STRONG EVIDENCE OF DOMAIN LOCALITY. Sparsify should proceed with storage-assisted "
            "selective layer loading, adaptive precision, and predictive memory scheduling. Different domains "
            "activate distinct sub-graphs of the model, confirming memory tiering opportunities."
        )
    elif passed_failure:
        recommendation = (
            "FALSIFIED LOCALITY ASSUMPTION. All domains activate essentially identical dense sub-graphs (>90% overlap). "
            "Dense transformers cannot support storage-assisted memory tiering or selective layer loading. "
            "Recommendation: Pivot Sparsify toward native sparse architectures (MoEs) or attention pruning."
        )
    else:
        recommendation = (
            "PARTIAL LOCALITY DETECTED. Moderate activation overlap. Some clustering exists but clusters are "
            "partially overlapping. Suggest running with larger context sequences or specialized fine-tuning "
            "to establish stronger routing boundaries."
        )

    # 12. Save JSON report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "metrics": {
            "unrelated_domain_average_overlap": unrelated_overlap,
            "average_silhouette_score": avg_silhouette_score,
            "passed_success_criteria": passed_success,
            "passed_failure_criteria": passed_failure,
        },
        "cosine_overlap_matrix": overlap_matrix.tolist(),
        "jaccard_similarity_matrix": jaccard_matrix.tolist(),
        "domains": domains,
        "recommendation": recommendation,
    }
    
    report_path = output_dir / "domain_locality_report.json"
    
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

    console.print(f"\n[green]Completed Domain Locality Analysis.[/green] Saved reports and plots to [bold]{output_dir}[/bold]\n")
    
    # Print summary table
    print_locality_summary_table(report)

    return report


def print_locality_summary_table(report: Dict[str, Any]) -> None:
    """Print the final locality summary metrics to the console."""
    table = Table(title="Domain Locality Analysis Summary (SP-009)", title_style="bold green")
    table.add_column("Metric Name", style="bold")
    table.add_column("Measured Value", justify="right")
    table.add_column("Target Status")
    
    metrics = report["metrics"]
    
    overlap_val = f"{metrics['unrelated_domain_average_overlap']*100.0:.2f}%"
    overlap_status = "[green]Pass (<70%)[/green]" if metrics["unrelated_domain_average_overlap"] < 0.70 else (
        "[red]Fail (>90%)[/red]" if metrics["unrelated_domain_average_overlap"] > 0.90 else "[yellow]Marginal[/yellow]"
    )
    table.add_row("Average Unrelated Domain Overlap", overlap_val, overlap_status)
    
    sil_val = f"{metrics['average_silhouette_score']:.3f}"
    sil_status = "[green]Pass (>0.50)[/green]" if metrics["average_silhouette_score"] > 0.50 else (
        "[red]Fail (<0.20)[/red]" if metrics["average_silhouette_score"] < 0.20 else "[yellow]Marginal[/yellow]"
    )
    table.add_row("Clustering Silhouette Score", sil_val, sil_status)
    
    success_status = "[green]YES[/green]" if metrics["passed_success_criteria"] else "[red]NO[/red]"
    table.add_row("Locality Hypothesis Confirmed", "", success_status)
    
    console.print(table)
    console.print(f"\n[bold]Strategic Recommendation:[/bold]\n[cyan]{report['recommendation']}[/cyan]\n")
