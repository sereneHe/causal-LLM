#!/usr/bin/env python3
"""Run a lightweight cloud-friendly baseline using built-in dataset assets."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    problem_name = os.getenv("PROBLEM_NAME", "asia")
    enabled_models = os.getenv("ENABLED_MODELS", "PC")
    keep_alive = _as_bool(os.getenv("KEEP_ALIVE"), False)

    output_dir = Path(os.getenv("OUTPUT_DIR", "/workspace/reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = (
        repo_root
        / "scripts"
        / "baseline"
        / "Causal-LLM_Unified_One-Shot_Framework"
        / "Data"
        / problem_name
    )
    samples_path = data_root / f"{problem_name}.csv"
    nodes_path = data_root / "nodes.npy"
    adj_path = data_root / "adj.npy"

    if not samples_path.exists() or not adj_path.exists():
        print(f"Missing dataset assets under: {data_root}", file=sys.stderr)
        return 2

    work_tmp = output_dir / "tmp_inputs"
    work_tmp.mkdir(parents=True, exist_ok=True)

    samples_df = pd.read_csv(samples_path)
    adj = np.load(adj_path, allow_pickle=True)

    if nodes_path.exists():
        nodes = np.load(nodes_path, allow_pickle=True)
        columns = [str(item) for item in nodes.tolist()]
    else:
        columns = [str(col) for col in samples_df.columns]

    pd.DataFrame(adj, columns=columns).to_csv(work_tmp / "adj.csv", index=False)

    runner_dir = (
        repo_root
        / "scripts"
        / "baseline"
        / "Causal-LLM_Unified_One-Shot_Framework"
        / "Dag_generation and model_evaluation"
    )
    command = [
        sys.executable,
        "main_runner.py",
        "--ground-truth-path",
        str(work_tmp / "adj.csv"),
        "--gaussian-path",
        str(samples_path),
        "--output-dir",
        str(output_dir),
        "--enabled-models",
        enabled_models,
    ]

    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=str(runner_dir), check=False)
    if result.returncode != 0:
        return result.returncode

    print(f"Experiment completed. Results written to: {output_dir}")

    if keep_alive:
        print("KEEP_ALIVE is enabled; container will stay running.")
        try:
            while True:
                # Keep one replica alive for log inspection and manual exec.
                subprocess.run(["sleep", "600"], check=False)
        except KeyboardInterrupt:
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
