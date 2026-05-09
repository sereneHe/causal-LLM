from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def _resolve_problem_path(problem: str) -> Path:
    candidate = Path(problem)
    if candidate.suffix == ".yaml" and candidate.exists():
        return candidate.resolve()

    named = SRC_ROOT / "conf" / "problem" / f"{problem}.yaml"
    if named.exists():
        return named

    raise FileNotFoundError(f"Unable to resolve problem config: {problem}")


def _load_problem_config(problem: str) -> tuple[Path, dict]:
    problem_path = _resolve_problem_path(problem)
    config = OmegaConf.to_container(OmegaConf.load(problem_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Unexpected problem config format: {problem_path}")
    return problem_path, config


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _load_labels(ts_dir: Path) -> list[str]:
    series_files = sorted(path for path in ts_dir.iterdir() if path.is_file())
    if not series_files:
        raise FileNotFoundError(f"No time-series files found in {ts_dir}")
    first = pd.read_csv(series_files[0], delimiter="\t", index_col=0, header=None)
    return [str(label) for label in first.index]


def _load_truth_matrix(data_root: Path, true_graph_file: str) -> np.ndarray:
    true_graph_path = _resolve_path(data_root, true_graph_file)
    if true_graph_path is None or not true_graph_path.exists():
        raise FileNotFoundError(f"Missing true graph file: {true_graph_path}")

    if true_graph_path.suffix == ".npz":
        return np.asarray(np.load(true_graph_path, allow_pickle=True)["arr_0"])
    if true_graph_path.suffix == ".csv":
        return pd.read_csv(true_graph_path, header=None).to_numpy()

    raise ValueError(f"Unsupported true graph format: {true_graph_path}")


def _normalise_node_name(node: str) -> str:
    node = node.strip()
    node = re.sub(r"_lag\d+$", "", node)
    node = re.sub(r"_t-?\d+$", "", node)
    return node


def _read_edge_list(graph_file: Path) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    with graph_file.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.add((parts[0], parts[1]))
    return edges


def _edge_list_to_adjacency(edges: set[tuple[str, str]], labels: list[str]) -> np.ndarray:
    adjacency = np.zeros((len(labels), len(labels)), dtype=int)
    index = {label: i for i, label in enumerate(labels)}
    for source, target in edges:
        src = _normalise_node_name(source)
        dst = _normalise_node_name(target)
        if src in index and dst in index:
            adjacency[index[src], index[dst]] = 1
    return adjacency


def _load_estimated_matrix(graph_file: Path, labels: list[str]) -> np.ndarray:
    suffix = graph_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(graph_file, header=None).to_numpy()
    if suffix == ".npy":
        return np.load(graph_file, allow_pickle=True)
    if suffix == ".npz":
        data = np.load(graph_file, allow_pickle=True)
        if "arr_0" in data:
            return np.asarray(data["arr_0"])
        if "x" in data:
            return np.asarray(data["x"])
        raise ValueError(f"Unsupported npz payload in {graph_file}")
    return _edge_list_to_adjacency(_read_edge_list(graph_file), labels)


def _plot_matrix(ax: plt.Axes, matrix: np.ndarray, labels: list[str], title: str) -> None:
    ax.imshow(matrix, cmap="binary", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=80, fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)


def save_heatmap_comparison(
    estimated: np.ndarray,
    truth: np.ndarray,
    labels: list[str],
    output_file: Path,
    title: str,
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    _plot_matrix(axes[0], truth, labels, "Ground Truth")
    _plot_matrix(axes[1], estimated, labels, "Estimated")
    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.3, wspace=0.25)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Krebs cycle heatmaps from Dynotears-style outputs.")
    parser.add_argument(
        "--problem",
        default="krebs_cycle_3",
        help="Problem config name under conf/problem or a direct YAML path.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Root containing *_TS directories and true_graph.npz.",
    )
    parser.add_argument(
        "--graph-file",
        default=None,
        help="Estimated graph file. Supports edge-list txt, csv, npy, or npz.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output image path. Defaults to the problem heatmap output_dir.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, problem_cfg = _load_problem_config(args.problem)

    data_root = Path(args.data_root).resolve()
    heatmap_cfg = problem_cfg.get("heatmap", {})
    ts_dir_name = heatmap_cfg.get("ts_dir") or f"{problem_cfg['dataset_name']}_TS"
    true_graph_file = heatmap_cfg.get("true_graph_file", "true_graph.npz")
    graph_file_value = args.graph_file or heatmap_cfg.get("graph_file")
    if not graph_file_value:
        raise ValueError("graph-file is required unless heatmap.graph_file is set in the problem YAML.")

    ts_dir = _resolve_path(data_root, ts_dir_name)
    if ts_dir is None or not ts_dir.exists():
        raise FileNotFoundError(f"Missing time-series directory: {ts_dir}")

    labels = _load_labels(ts_dir)
    truth = _load_truth_matrix(data_root, true_graph_file)
    graph_file = (_resolve_path(PROJECT_ROOT, graph_file_value) or Path(graph_file_value)).resolve()
    estimated = _load_estimated_matrix(graph_file, labels)

    if estimated.shape != truth.shape:
        raise ValueError(
            f"Estimated graph shape {estimated.shape} does not match truth shape {truth.shape}."
        )

    if args.output_file:
        output_file = Path(args.output_file).resolve()
    else:
        output_dir = _resolve_path(PROJECT_ROOT, heatmap_cfg.get("output_dir")) or (PROJECT_ROOT / "output")
        output_file = output_dir / f"{graph_file.stem}_heatmap.png"

    title = f"{problem_cfg['dataset_name']} Dynotears Heatmap"
    saved = save_heatmap_comparison(estimated, truth, labels, output_file, title)
    print(saved)


if __name__ == "__main__":
    main()
