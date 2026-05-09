from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch


DEFAULT_CATEGORY_COLORS = {
    "Constraint-based": "#1f77b4",
    "Function-based": "#2ca02c",
    "Score-based": "#ff7f0e",
    "Gradient-based": "#d62728",
}

DEFAULT_METRIC_COLORS = {
    "F-score": "#003366",
    "Recall": "#ff4500",
    "Precision": "#800000",
    "TPR": "#bc5090",
    "SHD": "#4caf50",
    "SID": "#ffd700",
    "FDR": "#87cefa",
    "FPR": "purple",
    "nnz": "#ef5675",
}


def _to_dataframe(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(Path(data))


def _save_or_show(
    fig: Figure,
    save_path: str | Path | None,
    *,
    show: bool,
    dpi: int,
) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def circle_barplot(
    data: pd.DataFrame | str | Path,
    *,
    value_col: str = "SID",
    method_col: str = "Method",
    category_col: str = "Category",
    comparison_data: pd.DataFrame | str | Path | None = None,
    comparison_value_col: str | None = None,
    primary_label: str | None = None,
    comparison_label: str | None = None,
    exclude_methods: Sequence[str] | None = None,
    category_colors: Mapping[str, str] | None = None,
    comparison_color: str = "#9e9e9e",
    figsize: tuple[int, int] = (10, 10),
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """Create a circular bar plot from notebook-style summary metrics."""

    category_colors = dict(DEFAULT_CATEGORY_COLORS if category_colors is None else category_colors)
    df = _to_dataframe(data)

    if exclude_methods:
        df = df[~df[method_col].isin(exclude_methods)].reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows available for circle_barplot after filtering.")

    methods = df[method_col].astype(str).tolist()
    categories = df[category_col].astype(str).tolist()
    radii_primary = df[value_col].astype(float).to_numpy()

    merged = df.copy()
    if comparison_data is not None:
        comp = _to_dataframe(comparison_data)
        comp_value_col = comparison_value_col or value_col
        comp = comp[[method_col, category_col, comp_value_col]].rename(
            columns={comp_value_col: "__comparison__"}
        )
        merged = merged.merge(comp, on=[method_col, category_col], how="inner")
        methods = merged[method_col].astype(str).tolist()
        categories = merged[category_col].astype(str).tolist()
        radii_primary = merged[value_col].astype(float).to_numpy()
        radii_secondary = merged["__comparison__"].astype(float).to_numpy()
    else:
        radii_secondary = None

    if len(methods) == 0:
        raise ValueError("No overlapping rows found for circle_barplot comparison.")

    n_bars = len(methods)
    theta = np.linspace(0.0, 2 * np.pi, n_bars, endpoint=False)
    full_width = 2 * np.pi / n_bars
    bar_colors = [category_colors.get(cat, "#607d8b") for cat in categories]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.set_aspect("equal")

    if radii_secondary is None:
        width = full_width * 0.85
        ax.bar(theta, radii_primary, width=width, color=bar_colors, edgecolor="white", alpha=0.9)
        label_angles = theta
        label_heights = radii_primary
    else:
        width = full_width * 0.38
        ax.bar(
            theta,
            radii_primary,
            width=width,
            color=bar_colors,
            edgecolor="white",
            alpha=0.9,
            label=primary_label or value_col,
        )
        ax.bar(
            theta + width,
            radii_secondary,
            width=width,
            color=comparison_color,
            edgecolor="white",
            alpha=0.65,
            label=comparison_label or (comparison_value_col or value_col),
        )
        label_angles = theta + width / 2
        label_heights = np.maximum(radii_primary, radii_secondary)

    max_radius = float(np.nanmax(label_heights)) if len(label_heights) else 1.0
    label_offset = max(max_radius * 0.06, 1.0)

    for angle, radius, label in zip(label_angles, label_heights, methods):
        angle_deg = np.degrees(angle)
        text_radius = radius + label_offset
        if np.pi / 2 < angle < 3 * np.pi / 2:
            rotation = angle_deg + 180
            ha = "right"
        else:
            rotation = angle_deg
            ha = "left"

        ax.text(
            angle,
            text_radius,
            label,
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=10,
        )

    legend_items = [Patch(facecolor=color, label=cat) for cat, color in category_colors.items()]
    if radii_secondary is not None:
        legend_items.append(Patch(facecolor=comparison_color, label=comparison_label or "Comparison"))
    ax.legend(handles=legend_items, loc="upper right", bbox_to_anchor=(1.22, 1.12), title="Category")

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False)
    if title:
        ax.set_title(title, va="bottom", pad=30)

    fig.tight_layout()
    _save_or_show(fig, save_path, show=show, dpi=dpi)
    return fig, ax


def barplot(
    data: pd.DataFrame | str | Path,
    *,
    method_col: str = "Method",
    metrics: Sequence[str] | None = None,
    mode: str = "standardized",
    reverse_metrics: Sequence[str] = ("FDR", "SHD", "SID", "FPR"),
    true_values: Mapping[str, float] | None = None,
    colors: Mapping[str, str] | None = None,
    bar_width: float | None = None,
    figsize: tuple[int, int] = (14, 6),
    ylabel: str | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 300,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Create a grouped bar plot using the notebook's standardized/error logic."""

    df = _to_dataframe(data)
    if method_col not in df.columns:
        raise KeyError(f"Column '{method_col}' not found in input data.")

    if metrics is None:
        preferred = ["SHD", "SID", "FDR", "FPR", "Precision", "Recall", "F-score", "TPR", "nnz"]
        metrics = [metric for metric in preferred if metric in df.columns]
    else:
        missing = [metric for metric in metrics if metric not in df.columns]
        if missing:
            raise KeyError(f"Missing metric columns: {missing}")

    if not metrics:
        raise ValueError("No metric columns available for barplot.")

    plot_df = df[list(metrics)].astype(float).copy()
    reverse_metrics = set(reverse_metrics)

    if mode == "standardized":
        for col in metrics:
            if col in reverse_metrics:
                max_val = plot_df[col].max()
                min_val = plot_df[col].min()
                plot_df[col] = max_val - plot_df[col] + min_val
        denom = (plot_df.max() - plot_df.min()).replace(0, 1)
        plot_df = (plot_df - plot_df.min()) / denom + 0.01
        default_ylabel = "Standardized Score"
    elif mode == "percent_error":
        true_values = dict(true_values or {})
        missing_truth = [metric for metric in metrics if metric not in true_values]
        if missing_truth:
            raise KeyError(
                "true_values must include every plotted metric when mode='percent_error'. "
                f"Missing: {missing_truth}"
            )
        for col in metrics:
            true_val = true_values[col]
            if true_val == 0:
                plot_df[col] = plot_df[col].abs()
            else:
                plot_df[col] = ((plot_df[col] - true_val).abs() / abs(true_val)) * 100
        default_ylabel = "Error in Percentage"
    else:
        raise ValueError("mode must be either 'standardized' or 'percent_error'.")

    colors = dict(DEFAULT_METRIC_COLORS if colors is None else colors)
    methods = df[method_col].astype(str).tolist()
    n_metrics = len(metrics)
    bar_width = bar_width or min(0.8 / max(n_metrics, 1), 0.12)
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=figsize)
    for idx, metric in enumerate(metrics):
        ax.bar(
            x + idx * bar_width,
            plot_df[metric].to_numpy(),
            width=bar_width,
            label=metric,
            color=colors.get(metric),
        )

    ax.set_xticks(x + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(ylabel or default_ylabel, fontsize=13)
    ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, fontsize=14, pad=12)

    fig.tight_layout()
    _save_or_show(fig, save_path, show=show, dpi=dpi)
    plot_df.insert(0, method_col, methods)
    return fig, ax, plot_df


def heatmap(
    matrix: pd.DataFrame | np.ndarray | str | Path,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    cmap: str = "seismic",
    vmin: float | None = -5,
    vmax: float | None = 5,
    center: float | None = 0,
    linewidths: float = 0.5,
    linecolor: str = "gray",
    figsize: tuple[int, int] = (10, 9),
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """Plot an adjacency-style heatmap adapted from the notebook."""

    if isinstance(matrix, (str, Path)):
        matrix_df = pd.read_csv(Path(matrix), header=None)
    elif isinstance(matrix, pd.DataFrame):
        matrix_df = matrix.copy()
    else:
        matrix_df = pd.DataFrame(np.asarray(matrix))

    values = matrix_df.to_numpy()
    if values.ndim != 2:
        raise ValueError("heatmap expects a 2D matrix.")

    if labels is not None:
        if len(labels) != values.shape[0] or len(labels) != values.shape[1]:
            raise ValueError("labels length must match both heatmap dimensions.")
        matrix_df = pd.DataFrame(values, index=list(labels), columns=list(labels))

    sns.set(style="white")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix_df,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=linewidths,
        linecolor=linecolor,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    if title:
        ax.set_title(title, fontsize=14, pad=15)

    fig.tight_layout()
    _save_or_show(fig, save_path, show=show, dpi=dpi)
    return fig, ax
