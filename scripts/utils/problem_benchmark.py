#!/usr/bin/env python3
"""Run the four causal discovery baselines per problem and summarize results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/xiaoyuhe/Causal-LLM")
RAW_ROOT = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
REPORTS_ROOT = PROJECT_ROOT / "reports"
OUTPUT_ROOT = REPORTS_ROOT / "outputs"
PROBLEM_CONFIG_DIR = PROJECT_ROOT / "experiment-config" / "problems"


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        text=True,
        capture_output=True,
        check=True,
    )


def ensure_processed(problem: str) -> Path:
    processed_dir = PROCESSED_ROOT / problem
    data_path = processed_dir / "data.npy"
    if data_path.exists():
        return processed_dir

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "data" / "preprocessing.py"),
        "--dataset",
        problem,
        "--raw-root",
        str(RAW_ROOT),
        "--processed-root",
        str(PROCESSED_ROOT),
    ]
    run_command(cmd, cwd=PROJECT_ROOT)
    return processed_dir


def bundle_to_csv_inputs(processed_dir: Path, tmp_dir: Path) -> tuple[Path, Path]:
    payload = np.load(processed_dir / "data.npy", allow_pickle=True).item()
    x = np.asarray(payload["x"])
    y = payload.get("y")
    if y is None:
        raise ValueError(f"{processed_dir} has no ground-truth graph")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    samples_path = tmp_dir / "samples.csv"
    adj_path = tmp_dir / "adj.csv"
    pd.DataFrame(x).to_csv(samples_path, index=False)
    pd.DataFrame(np.asarray(y)).to_csv(adj_path, index=False, header=False)
    return samples_path, adj_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_counts(metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(metrics)
    tp = metrics.get("tp")
    fp = metrics.get("fp")
    tn = metrics.get("tn")
    fn = metrics.get("fn")
    if tp is not None and fp is not None:
        metrics.setdefault("precision", tp / max(tp + fp, 1))
        metrics.setdefault("fdr", fp / max(tp + fp, 1))
    if tp is not None and fn is not None:
        metrics.setdefault("recall", tp / max(tp + fn, 1))
        metrics.setdefault("tpr", tp / max(tp + fn, 1))
    if tp is not None and tn is not None and fp is not None and fn is not None:
        metrics.setdefault("accuracy", (tp + tn) / max(tp + tn + fp + fn, 1))
    if "recall" in metrics and "precision" in metrics:
        denom = metrics["precision"] + metrics["recall"]
        metrics.setdefault(
            "f1",
            0.0 if denom == 0 else 2 * metrics["precision"] * metrics["recall"] / denom,
        )
    return metrics


def row_from_metrics(problem: str, method: str, status: str, metrics: dict[str, Any] | None, artifact: str | None = None, note: str | None = None) -> dict[str, Any]:
    row = {
        "problem": problem,
        "method": method,
        "status": status,
        "artifact": artifact,
        "note": note,
    }
    for key in [
        "tpr",
        "fdr",
        "shd",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tp",
        "fp",
        "tn",
        "fn",
        "fpr",
        "tnr",
        "fnr",
        "nnz",
    ]:
        row[key] = None
    if metrics:
        metrics = normalize_counts(metrics)
        for key in row.keys():
            pass
        for key in [
            "tpr",
            "fdr",
            "shd",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "tp",
            "fp",
            "tn",
            "fn",
            "fpr",
            "tnr",
            "fnr",
            "nnz",
        ]:
            if key in metrics:
                row[key] = metrics[key]
    return row


def read_dataassist_result(output_dir: Path) -> tuple[str, dict[str, Any] | None, Path | None]:
    llm_eval = output_dir / "llm_prior" / "pc_causal_matrix_llm_prior_eval.json"
    base_eval = output_dir / "base" / "pc_causal_matrix_eval.json"
    flat_llm_eval = output_dir / "pc_causal_matrix_llm_prior_eval.json"
    flat_base_eval = output_dir / "pc_causal_matrix_eval.json"
    if llm_eval.exists():
        return "llm_prior", load_json(llm_eval), llm_eval
    if base_eval.exists():
        return "baseline", load_json(base_eval), base_eval
    if flat_llm_eval.exists():
        return "llm_prior", load_json(flat_llm_eval), flat_llm_eval
    if flat_base_eval.exists():
        return "baseline", load_json(flat_base_eval), flat_base_eval
    return "missing", None, None


def read_guide_result(output_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    bundle = output_dir / "results_custom.npy"
    if not bundle.exists():
        return None, None
    payload = np.load(bundle, allow_pickle=True).item()
    return payload.get("metrics"), bundle


def read_kcrl_result(output_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    summary = output_dir / "final_metrics.json"
    if summary.exists():
        payload = load_json(summary)
        pruned = payload.get("pruned")
        return pruned, summary
    acc = output_dir / "accuracy_res2.npy"
    if acc.exists():
        arr = np.load(acc, allow_pickle=True)
        if arr.size:
            fdr, tpr, fpr, shd, nnz = arr[-1]
            return {
                "fdr": float(fdr),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "shd": int(shd),
                "nnz": int(nnz),
            }, acc
    return None, None


def read_unified_result(output_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    csv_candidates = [
        output_dir / "evaluation_results.csv",
        output_dir / "adj_combined_results.csv",
    ]
    json_candidates = [
        output_dir / "evaluation_results.json",
        output_dir / "adj_combined_results.json",
    ]
    for csv_path in csv_candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if df.empty:
                return None, csv_path
            if "Causal Model" in df.columns:
                model_rows = df[df["Causal Model"].astype(str).str.startswith("PC")]
                if not model_rows.empty:
                    return model_rows.iloc[0].to_dict(), csv_path
            return df.iloc[0].to_dict(), csv_path
    for json_path in json_candidates:
        if json_path.exists():
            rows = load_json(json_path)
            if not rows:
                return None, json_path
            return rows[0], json_path
    return None, None


def run_dataassist(problem: str, processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    base_out = output_dir / "base"
    llm_out = output_dir / "llm_prior"
    base_out.mkdir(parents=True, exist_ok=True)
    llm_out.mkdir(parents=True, exist_ok=True)

    try:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "utils" / "metrics.py"),
                "--data-npy",
                str(processed_dir / "data.npy"),
                "--output-dir",
                str(base_out),
            ],
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        return row_from_metrics(problem, "DataAssist", "failed", None, artifact=str(base_out), note=exc.stderr[-500:] if exc.stderr else "baseline run failed")

    variant = "baseline"
    metrics, artifact = read_dataassist_result(output_dir)

    try:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "baseline" / "CausalDiscovery_from_Data_Assisted_llm" / "DataAssist_llm.py"),
                "--output-dir",
                str(llm_out),
            ],
            cwd=PROJECT_ROOT,
        )
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "utils" / "metrics.py"),
                "--data-npy",
                str(processed_dir / "data.npy"),
                "--prior-json",
                str(llm_out / "llm_prior_edges.json"),
                "--output-dir",
                str(llm_out),
            ],
            cwd=PROJECT_ROOT,
        )
        variant = "llm_prior"
        metrics, artifact = read_dataassist_result(output_dir)
    except subprocess.CalledProcessError as exc:
        note = exc.stderr[-500:] if exc.stderr else "LLM prior generation failed"
        if metrics is None:
            return row_from_metrics(problem, "DataAssist", "failed", None, artifact=str(base_out), note=note)
        row = row_from_metrics(problem, "DataAssist", "ok", metrics, artifact=str(artifact) if artifact else None, note=f"{variant}; {note}")
        row["variant"] = variant
        return row

    row = row_from_metrics(problem, "DataAssist", "ok", metrics, artifact=str(artifact) if artifact else None, note=variant)
    row["variant"] = variant
    return row


def run_guide(problem: str, processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "baseline" / "GUIDE" / "main.py"),
                "--data_path",
                str(processed_dir / "X.npy"),
                "--adj_path",
                str(processed_dir / "adj.npy"),
                "--output_dir",
                str(output_dir),
                "--num_epochs",
                "1",
                "--hidden_dim",
                "64",
                "--nheads",
                "8",
                "--batch_size",
                "64",
            ],
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        return row_from_metrics(problem, "GUIDE", "failed", None, artifact=str(output_dir), note=exc.stderr[-500:] if exc.stderr else "run failed")

    metrics, artifact = read_guide_result(output_dir)
    return row_from_metrics(problem, "GUIDE", "ok", metrics, artifact=str(artifact) if artifact else None)


def run_kcrl(problem: str, processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "KCRL_OUTPUT_DIR": str(output_dir),
        "KCRL_DATA_PATH": str(processed_dir),
        "KCRL_NB_EPOCH": "1",
        "KCRL_INPUT_DIMENSION": "64",
        "KCRL_LAMBDA_ITER_NUM": "10",
        "KCRL_READ_DATA": "1",
    }
    try:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "baseline" / "KCRL" / "kcrl_demo.py"),
            ],
            cwd=PROJECT_ROOT / "scripts" / "baseline" / "KCRL",
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        return row_from_metrics(problem, "KCRL", "failed", None, artifact=str(output_dir), note=exc.stderr[-500:] if exc.stderr else "run failed")

    metrics, artifact = read_kcrl_result(output_dir)
    return row_from_metrics(problem, "KCRL", "ok", metrics, artifact=str(artifact) if artifact else None)


def run_unified(problem: str, processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp_inputs"
    samples_path, adj_path = bundle_to_csv_inputs(processed_dir, tmp_dir)
    method_dir = (
        PROJECT_ROOT
        / "scripts"
        / "baseline"
        / "Causal-LLM_Unified_One-Shot_Framework"
        / "Dag_generation and model_evaluation"
    )
    try:
        run_command(
            [
                sys.executable,
                str(method_dir / "main_runner.py"),
                "--ground-truth-path",
                str(adj_path),
                "--gaussian-path",
                str(samples_path),
                "--output-dir",
                str(output_dir),
                "--enabled-models",
                "PC",
            ],
            cwd=method_dir,
        )
    except subprocess.CalledProcessError as exc:
        return row_from_metrics(problem, "UnifiedOneShot", "failed", None, artifact=str(output_dir), note=exc.stderr[-500:] if exc.stderr else "run failed")

    metrics, artifact = read_unified_result(output_dir)
    return row_from_metrics(problem, "UnifiedOneShot", "ok", metrics, artifact=str(artifact) if artifact else None)


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = df.columns.tolist()
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_problem(problem: str, results: list[dict[str, Any]], output_dir: Path) -> Path:
    df = pd.DataFrame(results)
    desired_cols = [
        "problem",
        "method",
        "status",
        "variant",
        "tpr",
        "fdr",
        "shd",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tp",
        "fp",
        "tn",
        "fn",
        "fpr",
        "tnr",
        "fnr",
        "nnz",
        "artifact",
        "note",
    ]
    for col in desired_cols:
        if col not in df.columns:
            df[col] = None
    df = df[desired_cols]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{problem}_summary.csv"
    md_path = output_dir / f"{problem}_summary.md"
    df.to_csv(csv_path, index=False)
    write_markdown_table(df, md_path)
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and summarize causal discovery baselines per problem.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", help="Single problem name to run.")
    group.add_argument("--problems", help="Comma-separated problem names to run.")
    group.add_argument("--all-problems", action="store_true", help="Run all problem YAMLs in experiment-config/problems.")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods to run: DataAssist,GUIDE,KCRL,UnifiedOneShot or all.",
    )
    parser.add_argument("--skip-run", action="store_true", help="Only summarize existing outputs.")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_ROOT, help="Directory for summary CSV/MD files.")
    return parser.parse_args()


def methods_to_run(selection: str) -> list[str]:
    if selection == "all":
        return ["DataAssist", "GUIDE", "KCRL", "UnifiedOneShot"]
    methods = [item.strip() for item in selection.split(",") if item.strip()]
    allowed = {"DataAssist", "GUIDE", "KCRL", "UnifiedOneShot"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def run_problem(problem: str, methods: list[str]) -> list[dict[str, Any]]:
    processed_dir = ensure_processed(problem)
    results: list[dict[str, Any]] = []
    for method in methods:
        output_dir = OUTPUT_ROOT / method / problem
        if method == "DataAssist":
            results.append(run_dataassist(problem, processed_dir, output_dir))
        elif method == "GUIDE":
            results.append(run_guide(problem, processed_dir, output_dir))
        elif method == "KCRL":
            results.append(run_kcrl(problem, processed_dir, output_dir))
        elif method == "UnifiedOneShot":
            results.append(run_unified(problem, processed_dir, output_dir))
    return results


def main() -> int:
    args = parse_args()
    selected_methods = methods_to_run(args.methods)
    problems = (
        sorted(p.stem for p in PROBLEM_CONFIG_DIR.glob("*.yaml"))
        if args.all_problems
        else ([item.strip() for item in args.problems.split(",") if item.strip()] if args.problems else [args.problem])
    )

    for problem in problems:
        if args.skip_run:
            processed_dir = PROCESSED_ROOT / problem
            if not (processed_dir / "data.npy").exists():
                raise FileNotFoundError(f"Processed bundle not found for {problem}: {processed_dir / 'data.npy'}")
            results = []
            for method in selected_methods:
                output_dir = OUTPUT_ROOT / method / problem
                if method == "DataAssist":
                    _, metrics, artifact = read_dataassist_result(output_dir)
                elif method == "GUIDE":
                    metrics, artifact = read_guide_result(output_dir)
                elif method == "KCRL":
                    metrics, artifact = read_kcrl_result(output_dir)
                elif method == "UnifiedOneShot":
                    metrics, artifact = read_unified_result(output_dir)
                else:
                    metrics, artifact = None, None
                status = "ok" if metrics else "missing"
                results.append(row_from_metrics(problem, method, status, metrics, artifact=str(artifact) if artifact else None))
        else:
            results = run_problem(problem, selected_methods)

        summarize_problem(problem, results, args.output_dir)
        print(f"Wrote summary for {problem} to {args.output_dir / f'{problem}_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
