"""
Phase 5 Visualization

All plotting functions for the MoE emergence project. Each function returns
a matplotlib Figure object (no plt.show() calls). Optional save_path param
for direct file saving.

Style: dpi=150 display / 300 save, font 11pt, white background, no grid.
Domain colors: code=#2196F3, math=#FF9800, prose=#4CAF50.
Expert heatmap colormap: YlOrRd.
"""

from pathlib import Path
from typing import Optional

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

DOMAIN_COLORS = {"code": "#2196F3", "math": "#FF9800", "prose": "#4CAF50"}
DOMAIN_ORDER = ["code", "math", "prose"]
HEATMAP_CMAP = "YlOrRd"
FONT_SIZE = 11
DPI_DISPLAY = 150
DPI_SAVE = 300


def _apply_style():
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": DPI_DISPLAY,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=FONT_SIZE - 1)


def _save_if_needed(fig: plt.Figure, save_path: Optional[str | Path]):
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=DPI_SAVE, bbox_inches="tight")


def plot_expert_domain_heatmap_grid(
    fractions: dict[int, dict[str, torch.Tensor]],
    layers: list[int] = [8, 9, 10, 11],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    2x2 subplot grid of expert-domain heatmaps, one per MoE layer.

    Args:
        fractions: {layer_idx: {"code": Tensor[n_experts], ...}}
        layers: Which layers to plot
        save_path: Optional path to save figure
    """
    _apply_style()
    n_experts = len(next(iter(next(iter(fractions.values())).values())))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    norm = Normalize(vmin=0, vmax=0.30)

    for idx, (ax, layer) in enumerate(zip(axes_flat, layers)):
        data = np.zeros((len(DOMAIN_ORDER), n_experts))
        for row, domain in enumerate(DOMAIN_ORDER):
            data[row] = fractions[layer][domain].numpy()

        sns.heatmap(
            data,
            ax=ax,
            cmap=HEATMAP_CMAP,
            vmin=0,
            vmax=0.30,
            annot=True,
            fmt=".2f",
            xticklabels=[f"E{i}" for i in range(n_experts)],
            yticklabels=[d.capitalize() for d in DOMAIN_ORDER],
            cbar=False,
            linewidths=0.5,
            linecolor="#f2f2f2",
        )
        ax.set_title(f"Layer {layer}", fontsize=FONT_SIZE + 2)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Domain" if idx in [0, 2] else "")
        _style_axes(ax)

    sm = ScalarMappable(norm=norm, cmap=HEATMAP_CMAP)
    sm.set_array([])
    fig.colorbar(
        sm, ax=axes_flat.tolist(), fraction=0.02, pad=0.02, label="Expert fraction"
    )
    fig.text(0.01, 0.01, "Uniform routing baseline = 0.125", fontsize=FONT_SIZE - 1)

    fig.suptitle(
        "Expert Utilization by Domain (MoE Main, Final Model)",
        fontsize=FONT_SIZE + 4,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.02, 0.95, 0.98))
    _save_if_needed(fig, save_path)
    return fig


def plot_collapse_comparison(
    moe_fracs: dict[int, torch.Tensor],
    nolb_fracs: dict[int, torch.Tensor],
    layer_idx: int,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Side-by-side bar charts: with vs without load balancing for one layer.

    Args:
        moe_fracs: {layer_idx: Tensor[n_experts]} from MoE main
        nolb_fracs: {layer_idx: Tensor[n_experts]} from no-LB ablation
        layer_idx: Which layer to plot
        save_path: Optional path to save figure
    """
    _apply_style()

    moe_data = moe_fracs[layer_idx].numpy()
    nolb_data = nolb_fracs[layer_idx].numpy()
    n_experts = len(moe_data)
    x = np.arange(n_experts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    bars1 = ax1.bar(x, moe_data, color="#4CAF50", alpha=0.85)
    ax1.axhline(
        y=1 / n_experts, color="gray", linestyle="--", alpha=0.7, label="Uniform"
    )
    ax1.set_title("With Load Balancing")
    ax1.set_xlabel("Expert")
    ax1.set_ylabel("Token Fraction")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"E{i}" for i in range(n_experts)])
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.bar_label(bars1, fmt="%.2f", fontsize=FONT_SIZE - 3, padding=2)
    _style_axes(ax1)

    bars2 = ax2.bar(x, nolb_data, color="#F44336", alpha=0.85)
    ax2.axhline(
        y=1 / n_experts, color="gray", linestyle="--", alpha=0.7, label="Uniform"
    )
    ax2.set_title("Without Load Balancing")
    ax2.set_xlabel("Expert")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"E{i}" for i in range(n_experts)])
    ax2.legend()
    ax2.bar_label(bars2, fmt="%.2f", fontsize=FONT_SIZE - 3, padding=2)
    _style_axes(ax2)

    fig.suptitle(
        f"Expert Collapse Comparison — Layer {layer_idx}",
        fontsize=FONT_SIZE + 2,
    )
    fig.tight_layout()
    _save_if_needed(fig, save_path)
    return fig


def plot_collapse_trajectory(
    trajectory: list[dict],
    layer_idx: int,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Multi-line plot showing expert fracs at each snapshot step for one layer.

    Args:
        trajectory: [{"step": 100, layer_idx: Tensor[8], ...}, ...]
        layer_idx: Which layer to plot
        save_path: Optional path to save figure
    """
    _apply_style()

    steps = [t["step"] for t in trajectory]
    n_experts = len(trajectory[0][layer_idx])
    dominant_expert = int(torch.argmax(trajectory[-1][layer_idx]).item())

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_experts))
    for expert_id in range(n_experts):
        fracs = [t[layer_idx][expert_id].item() for t in trajectory]
        linewidth = 2.2 if expert_id == dominant_expert else 1.2
        ax.plot(
            steps,
            fracs,
            marker="o",
            markersize=4,
            linewidth=linewidth,
            color=colors[expert_id],
            label=f"Expert {expert_id}",
        )

    ax.axhline(
        y=1 / n_experts, color="gray", linestyle="--", alpha=0.5, label="Uniform"
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Token Fraction")
    ax.set_title(f"Expert Utilization Over Training (No-LB, Layer {layer_idx})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=FONT_SIZE - 2)
    ax.set_ylim(0, 1.0)
    _style_axes(ax)

    fig.tight_layout()
    _save_if_needed(fig, save_path)
    return fig


def plot_router_entropy_over_training(
    entropy_df: pd.DataFrame,
    layers: list[int] = [8, 9, 10, 11],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Line plot of per-layer router entropy over training steps.

    Data is sparse (logged every 100 steps), so we use scatter+line with
    markers at actual data points. No interpolation or smoothing.

    Args:
        entropy_df: DataFrame from W&B export CSV with _step and entropy columns
        layers: Which layers to plot
        save_path: Optional path to save figure
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    for i, layer in enumerate(layers):
        col = f"router/layer_{layer}_entropy_per_token"
        if col not in entropy_df.columns:
            continue
        subset = entropy_df[["_step", col]].dropna()
        ax.plot(
            subset["_step"],
            subset[col],
            marker="o",
            markersize=3,
            linewidth=1,
            color=colors[i % len(colors)],
            label=f"Layer {layer}",
        )
        ax.scatter(
            subset["_step"],
            subset[col],
            s=14,
            color=colors[i % len(colors)],
            alpha=0.9,
        )

    max_entropy = np.log(8)  # ln(n_experts) = theoretical max
    ax.axhline(
        y=max_entropy,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Max entropy (ln(8) = {max_entropy:.3f})",
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Entropy per Token")
    ax.set_title("Router Entropy Over Training (MoE Main)")
    ax.legend(fontsize=FONT_SIZE - 2)
    ax.text(
        0.01,
        0.01,
        "Router metrics logged every 100 steps",
        transform=ax.transAxes,
        fontsize=FONT_SIZE - 2,
        color="#555555",
        ha="left",
        va="bottom",
    )
    _style_axes(ax)

    fig.tight_layout()
    _save_if_needed(fig, save_path)
    return fig


def plot_token_type_routing(
    token_type_fracs: dict[int, dict[str, torch.Tensor]],
    layer_idx: int,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Heatmap of token type -> expert routing for one layer.

    Args:
        token_type_fracs: {layer_idx: {"KEYWORD": Tensor[8], ...}}
        layer_idx: Which layer to plot
        save_path: Optional path to save figure
    """
    _apply_style()

    type_order = ["KEYWORD", "NAME", "NUMBER", "STRING", "OP", "COMMENT", "OTHER"]
    layer_data = token_type_fracs[layer_idx]
    n_experts = len(next(iter(layer_data.values())))

    data = np.zeros((len(type_order), n_experts))
    for row, tt in enumerate(type_order):
        if tt in layer_data:
            data[row] = layer_data[tt].numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        data,
        ax=ax,
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=0.30,
        annot=True,
        fmt=".2f",
        xticklabels=[f"E{i}" for i in range(n_experts)],
        yticklabels=type_order,
        cbar_kws={"label": "Token fraction"},
        linewidths=0.5,
        linecolor="#f2f2f2",
    )
    ax.set_title(f"Python Token Type Routing — Layer {layer_idx}")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Token Type")
    _style_axes(ax)

    fig.tight_layout()
    _save_if_needed(fig, save_path)
    return fig


def plot_training_curves(
    metrics_dict: dict[str, pd.DataFrame],
    best_evals: Optional[dict[str, dict]] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Two-panel figure: (left) eval/loss over steps, (right) per-domain best-eval bars.

    Args:
        metrics_dict: {"moe-main": DataFrame, "dense": DataFrame, ...}
        best_evals: {"moe-main": {best eval row dict}, ...}
        save_path: Optional path to save figure
    """
    _apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    run_colors = {
        "dense": "#9E9E9E",
        "moe-main": "#2196F3",
        "no-lb": "#F44336",
        "top-2": "#FF9800",
    }
    run_labels = {
        "dense": "Dense Baseline",
        "moe-main": "MoE (top-1)",
        "no-lb": "No-LB Ablation",
        "top-2": "MoE (top-2)",
    }

    for run_name, df in metrics_dict.items():
        eval_rows = df.dropna(subset=["eval/loss"])
        if eval_rows.empty:
            continue
        ax1.plot(
            eval_rows["step"],
            eval_rows["eval/loss"],
            marker="o",
            markersize=2,
            linewidth=1.5,
            color=run_colors.get(run_name, "#000"),
            label=run_labels.get(run_name, run_name),
        )

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Eval Loss")
    ax1.set_title("Evaluation Loss Over Training")
    ax1.legend(fontsize=FONT_SIZE - 2)

    if best_evals:
        domains = ["code", "math", "prose"]
        domain_keys = [f"eval/loss_{d}" for d in domains]
        x = np.arange(len(domains))
        runs_to_plot = [
            r
            for r in ["dense", "moe-main", "no-lb", "top-2"]
            if r in best_evals and "eval/loss" in best_evals[r]
        ]
        width = 0.8 / max(1, len(runs_to_plot))
        offsets = (np.arange(len(runs_to_plot)) - (len(runs_to_plot) - 1) / 2) * width

        for i, run_name in enumerate(runs_to_plot):
            vals = [best_evals[run_name].get(dk, np.nan) for dk in domain_keys]
            ax2.bar(
                x + offsets[i],
                vals,
                width * 0.92,
                color=run_colors.get(run_name, "#000"),
                label=run_labels.get(run_name, run_name),
                alpha=0.8,
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels([d.capitalize() for d in domains])
        ax2.set_ylabel("Best Eval Loss")
        ax2.set_title("Per-Domain Best Evaluation Loss")
        ax2.legend(fontsize=FONT_SIZE - 2)
        _style_axes(ax2)

    _style_axes(ax1)

    fig.tight_layout()
    _save_if_needed(fig, save_path)
    return fig
