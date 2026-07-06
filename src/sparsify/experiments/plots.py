"""Plotting utilities for generating heatmaps and scatter plots (SP-009)."""
from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np

# Set matplotlib backend to Agg to allow headless generation (no GUI window popup)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_layer_heatmap(
    attn_data: np.ndarray,
    ffn_data: np.ndarray,
    domains: List[str],
    output_path: Path,
) -> None:
    """Generate and save the Layer Activation Heatmap (Attn & FFN).

    Data shape: (16, 6) where rows are layers and columns are domains.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    
    # 1. Attention Heatmap
    im1 = axes[0].imshow(attn_data, cmap="viridis", aspect="auto")
    axes[0].set_title("Attention Layer Contribution Norm Ratio", fontsize=12, pad=10)
    axes[0].set_xticks(range(len(domains)))
    axes[0].set_xticklabels(domains, rotation=45, ha="right")
    axes[0].set_ylabel("Layer Index", fontsize=11)
    axes[0].set_yticks(range(16))
    fig.colorbar(im1, ax=axes[0], label="Ratio (||r_attn|| / ||x||)")
    
    # 2. FFN Heatmap
    im2 = axes[1].imshow(ffn_data, cmap="plasma", aspect="auto")
    axes[1].set_title("FFN Layer Contribution Norm Ratio", fontsize=12, pad=10)
    axes[1].set_xticks(range(len(domains)))
    axes[1].set_xticklabels(domains, rotation=45, ha="right")
    fig.colorbar(im2, ax=axes[1], label="Ratio (||r_ffn|| / ||x||)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_attention_head_heatmap(
    head_data: np.ndarray,
    domains: List[str],
    output_path: Path,
) -> None:
    """Generate and save the high-resolution Attention Head Heatmap.

    Data shape: (512, 6) where rows are heads and columns are domains.
    """
    # Create a tall figure to resolve individual heads
    plt.figure(figsize=(8, 20))
    plt.imshow(head_data, cmap="inferno", aspect="auto", interpolation="nearest")
    plt.title("Attention Head Relative Importance Matrix", fontsize=14, pad=15)
    plt.xlabel("Domains", fontsize=12)
    plt.xticks(range(len(domains)), domains, rotation=45, ha="right")
    plt.ylabel("Attention Head (Layer * 32 + Head Index)", fontsize=12)
    
    # Label every 32nd head (layer boundaries) for readability
    plt.yticks(range(0, 512, 32), [f"L{l} H0" for l in range(16)])
    
    plt.colorbar(label="Relative Importance in Layer")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_similarity_matrix(
    data: np.ndarray,
    labels: List[str],
    title: str,
    output_path: Path,
    cmap: str = "Blues",
) -> None:
    """Generate a 2D similarity heatmap (Cosine Overlap or Jaccard)."""
    plt.figure(figsize=(8, 7))
    plt.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.title(title, fontsize=12, pad=12)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    
    # Add numerical labels inside the grid
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(
                j, i, f"{data[i, j]:.2f}",
                ha="center", va="center",
                color="black" if data[i, j] < 0.7 else "white",
                fontsize=10, weight="bold"
            )
            
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca_scatter(
    coords: np.ndarray,
    labels: List[str],
    unique_domains: List[str],
    output_path: Path,
) -> None:
    """Generate a 2D scatter plot showing PCA clusters of domain activations."""
    plt.figure(figsize=(10, 8))
    
    # Color map for domains
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    domain_to_color = {d: colors[idx % len(colors)] for idx, d in enumerate(unique_domains)}
    
    for d in unique_domains:
        mask = [lbl == d for lbl in labels]
        d_coords = coords[mask]
        plt.scatter(
            d_coords[:, 0], d_coords[:, 1],
            label=d.capitalize(),
            color=domain_to_color[d],
            alpha=0.8, edgecolors="none", s=50
        )
        
    plt.title("PCA Dimensionality Reduction of Domain Activations (SP-009)", fontsize=14, pad=12)
    plt.xlabel("Principal Component 1", fontsize=11)
    plt.ylabel("Principal Component 2", fontsize=11)
    plt.legend(title="Domains", frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
