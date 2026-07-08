"""Generate README/site charts from MEASURED data (docs/measurements/).

Brand palette on the slate ground so the PNGs read as intentional cards in
both GitHub light/dark themes. Every number here is measured on the dev
machine (16 GB M-series, models on internal NVMe unless noted).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

GROUND = "#0B0E14"; PANEL = "#121826"; INK = "#E7EBF2"
SOFT = "#9AA5B8"; FAINT = "#5B6577"; LINE = "#232B3C"
ACCENT = "#E8A33D"; GOOD = "#3FB27F"

plt.rcParams.update({
    "figure.facecolor": GROUND, "axes.facecolor": GROUND,
    "savefig.facecolor": GROUND, "text.color": INK,
    "axes.labelcolor": SOFT, "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.edgecolor": LINE, "font.size": 12,
    "font.family": "sans-serif",
})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=LINE, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def memory():
    models = ["Mixtral 8x7B", "Qwen3-30B-A3B", "OLMoE-1B-7B"]
    stored = [26.3, 16.3, 3.9]
    rss = [3.33, 4.15, 1.34]
    import numpy as np
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    b1 = ax.bar(x - w/2, stored, w, label="stored on SSD", color=FAINT)
    b2 = ax.bar(x + w/2, rss, w, label="RAM used (Sparsify)", color=ACCENT)
    for b, v in list(zip(b1, stored)) + list(zip(b2, rss)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:g} GB",
                ha="center", va="bottom", color=INK, fontsize=10.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("gigabytes"); ax.set_ylim(0, 30)
    ax.set_title("Model on disk vs RAM actually used", color=INK,
                 fontsize=14, fontweight="bold", pad=14, loc="left")
    ax.legend(frameon=False, labelcolor=SOFT, loc="upper right")
    _clean(ax)
    fig.tight_layout(); fig.savefig("site/assets/chart-memory.png"); plt.close(fig)


def storage_speed():
    labels = ["USB SSD\n3 GB budget", "internal NVMe\n3 GB budget", "internal NVMe\n4.5 GB budget"]
    tps = [1.8, 8.5, 11.0]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    bars = ax.bar(labels, tps, 0.55,
                  color=[FAINT, ACCENT, ACCENT])
    for b, v in zip(bars, tps):
        ax.text(b.get_x()+b.get_width()/2, v+0.2, f"{v:g} tok/s",
                ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel("decode tokens / sec"); ax.set_ylim(0, 13)
    ax.set_title("Qwen3-30B decode speed — same model, faster storage",
                 color=INK, fontsize=14, fontweight="bold", pad=14, loc="left")
    _clean(ax)
    fig.tight_layout(); fig.savefig("site/assets/chart-speed.png"); plt.close(fig)


def overhead():
    labels = ["vanilla mlx-lm\n(full RAM)", "Sparsify\n(resident)"]
    tps = [161.6, 168.0]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    bars = ax.bar(labels, tps, 0.5, color=[FAINT, GOOD])
    for b, v in zip(bars, tps):
        ax.text(b.get_x()+b.get_width()/2, v+2, f"{v:g} tok/s",
                ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel("decode tokens / sec"); ax.set_ylim(0, 190)
    ax.set_title("Zero paging overhead (OLMoE, fits in RAM)",
                 color=INK, fontsize=14, fontweight="bold", pad=14, loc="left")
    _clean(ax)
    fig.tight_layout(); fig.savefig("site/assets/chart-overhead.png"); plt.close(fig)


if __name__ == "__main__":
    memory(); storage_speed(); overhead()
    print("charts written to site/assets/")
