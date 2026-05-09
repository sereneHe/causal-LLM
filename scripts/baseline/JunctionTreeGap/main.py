#!/usr/bin/env python3
"""Junction-tree separator-gap baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from junction_tree_gap import JunctionTreeGapConfig, JunctionTreeGapLearner, save_outputs

BACKEND_CHOICES = ["gaussian", "bn", "nn", "transformer", "mlp"]


def load_bundle(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if isinstance(payload, np.ndarray) and payload.dtype != object:
        return np.asarray(payload), None
    if not isinstance(payload, dict) or "x" not in payload:
        raise ValueError(f"Unsupported bundle format: {path}")
    x = np.asarray(payload["x"])
    y = payload.get("y")
    if y is not None:
        y = np.asarray(y)
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JunctionTreeGap baseline.")
    parser.add_argument("--data-npy", required=True, type=Path, help="Path to processed data.npy bundle.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write results.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--gap-weight", type=float, default=0.5)
    parser.add_argument("--sparsity-threshold", type=float, default=0.0)
    parser.add_argument("--edge-threshold", type=float, default=0.0)
    parser.add_argument("--covariance-jitter", type=float, default=1e-6)
    parser.add_argument("--gap-metric", type=str, default="sym_kl")
    parser.add_argument("--clique-potential-model", type=str, default="gaussian", choices=BACKEND_CHOICES)
    parser.add_argument("--separator-model", type=str, default="gaussian", choices=BACKEND_CHOICES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    X, y = load_bundle(args.data_npy)
    config = JunctionTreeGapConfig(
        top_k=args.top_k,
        gap_weight=args.gap_weight,
        sparsity_threshold=args.sparsity_threshold,
        edge_threshold=args.edge_threshold,
        covariance_jitter=args.covariance_jitter,
        gap_metric=args.gap_metric,
        clique_potential_model=args.clique_potential_model,
        separator_model=args.separator_model,
    )
    learner = JunctionTreeGapLearner(config=config).fit(X)
    adj, metrics = save_outputs(args.output_dir, learner, y)

    payload = {
        "adjacency_shape": list(adj.shape),
        "metrics": metrics,
        "config": config.__dict__,
        "iter_num": 1,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote JunctionTreeGap outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
