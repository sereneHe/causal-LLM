#!/usr/bin/env python3
"""Utilities for loading, validating, and exporting DAG adjacency arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx


DEFAULT_METHODS = ["CaMML", "MINOBSx", "LLMCD", "LLM_CaMML", "LLM_MINOBSx"]
DEFAULT_CANDIDATES = [
    "dag.txt",
    "graph.txt",
    "adjacency.npy",
    "adjacency.csv",
    "adjacency.json",
    "final_graph_pruned.npy",
    "final_graph.npy",
    "pc_causal_matrix_llm_prior.npy",
    "pc_causal_matrix.npy",
    "adj_PC_g.csv",
    "junction_tree_artifact.json",
    "summary.json",
    "results_custom.npy",
]


def load_graph_npy(path: str | Path, transpose: bool = False) -> np.ndarray:
    """Load a graph adjacency array from .npy and optionally transpose it."""
    graph = np.load(path)
    if transpose:
        graph = np.transpose(graph)
    return graph


def load_adjacency_matrix(path: str | Path) -> np.ndarray:
    """Load an adjacency matrix from .npy, .csv, or whitespace-delimited text."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        payload = np.load(file_path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.dtype != object:
            return np.asarray(payload)
        if isinstance(payload, np.ndarray) and payload.shape == ():
            payload = payload.item()
        if isinstance(payload, dict):
            for key in ("adjacency", "graph", "final_dag", "best_adj", "matrix"):
                value = payload.get(key)
                if value is not None:
                    return np.asarray(value)
        raise ValueError(f"Unsupported numpy payload format: {file_path}")
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("adjacency", "graph", "final_dag", "best_adj", "matrix"):
                value = payload.get(key)
                if value is not None:
                    return np.asarray(value)
        raise ValueError(f"Unsupported JSON payload format: {file_path}")
    if suffix in {".csv", ".txt"}:
        try:
            return np.loadtxt(file_path, dtype=float, delimiter=",")
        except Exception:
            return np.loadtxt(file_path, dtype=float)
    raise ValueError(f"Unsupported adjacency file type: {file_path}")


def binarize_adjacency(matrix: np.ndarray) -> np.ndarray:
    """Convert any numeric adjacency matrix to a 0/1 matrix."""
    return (np.asarray(matrix) != 0).astype(int)


def adjacency_is_dag(matrix: np.ndarray) -> bool:
    """Return whether the adjacency matrix represents a DAG."""
    adj = binarize_adjacency(matrix)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    rows, cols = np.where(adj != 0)
    graph.add_edges_from((int(r), int(c)) for r, c in zip(rows, cols))
    return nx.is_directed_acyclic_graph(graph)


def save_dag_matrix(matrix: np.ndarray, path: str | Path) -> Path:
    """Save a DAG adjacency matrix as a whitespace-delimited integer text file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(file_path, binarize_adjacency(matrix).astype(int), fmt="%d")
    return file_path


def export_adjacency_to_dag(
    input_path: str | Path,
    output_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Load an adjacency matrix and export it as a dag.txt-style file.

    By default, this helper writes next to the source file as ``dag.txt``.
    If ``force`` is false, non-DAG inputs raise a ValueError before writing.
    """
    source = Path(input_path)
    target = Path(output_path) if output_path is not None else source.with_name("dag.txt")
    matrix = load_adjacency_matrix(source)
    if not force and not adjacency_is_dag(matrix):
        raise ValueError(f"Input is not a DAG: {source}")
    return save_dag_matrix(matrix, target)


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def find_source_file(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    for name in DEFAULT_CANDIDATES:
        direct = base_dir / name
        if direct.exists():
            return direct
    for candidate in sorted(base_dir.rglob("*")):
        if not candidate.is_file():
            continue
        stem = candidate.stem.lower()
        suffix = candidate.suffix.lower()
        if suffix in {".npy", ".csv", ".txt", ".json"} and ("adj" in stem or "graph" in stem or "dag" in stem or "result" in stem):
            return candidate
    return None


def convert_one(input_path: Path, output_path: Path, force: bool = False) -> Path:
    matrix = load_adjacency_matrix(input_path)
    if not adjacency_is_dag(matrix) and not force:
        raise ValueError(f"Input is not a DAG: {input_path}")
    return save_dag_matrix(matrix, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert adjacency matrices to DAG text files.")
    parser.add_argument("--input", type=Path, help="Adjacency matrix file to convert.")
    parser.add_argument("--output", type=Path, help="Output DAG text file path.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/xiaoyuhe/Causal-LLM/reports/outputs"),
        help="Root directory containing method/problem output folders.",
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated method folders to scan.")
    parser.add_argument("--problems", default="", help="Comma-separated problems to scan. If omitted, scan all subfolders.")
    parser.add_argument("--force", action="store_true", help="Write the DAG file even if the input matrix is not acyclic.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.input:
        output = args.output or args.input.with_suffix(".dag.txt")
        save_dag_matrix(load_adjacency_matrix(args.input), output)
        print(f"Wrote DAG file to {output}")
        return 0

    methods = parse_list(args.methods) or DEFAULT_METHODS
    problems = parse_list(args.problems)

    for method in methods:
        method_root = args.root / method
        if not method_root.exists():
            print(f"Skipping {method}: missing {method_root}")
            continue

        problem_dirs = [method_root / problem for problem in problems] if problems else [p for p in sorted(method_root.iterdir()) if p.is_dir()]
        for problem_dir in problem_dirs:
            source = find_source_file(problem_dir)
            if source is None:
                print(f"Skipping {method}/{problem_dir.name}: no adjacency source found")
                continue
            output = problem_dir / "dag.txt"
            try:
                convert_one(source, output, force=args.force)
            except Exception as exc:
                print(f"Skipping {method}/{problem_dir.name}: {exc}")
                continue
            print(f"Wrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
