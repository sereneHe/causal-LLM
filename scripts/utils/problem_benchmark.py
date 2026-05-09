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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.utils.metrics import compute_sid, evaluate_against_truth, normalize_metric_dict, summary_columns
except ImportError:  # pragma: no cover - direct script fallback
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from metrics import compute_sid, evaluate_against_truth, normalize_metric_dict, summary_columns

RAW_ROOT = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
REPORTS_ROOT = PROJECT_ROOT / "reports"
OUTPUT_ROOT = REPORTS_ROOT / "outputs"
HEATMAP_ROOT = REPORTS_ROOT / "heatmap"
PROBLEM_CONFIG_DIR = PROJECT_ROOT / "experiment-config" / "problems"
ILSCSL_ROOT = PROJECT_ROOT / "scripts" / "baseline" / "ILS-CSL"
FAMILY_PROBLEMS = {"BIAS", "LEGAL"}


def has_processed_bundle(processed_dir: Path) -> bool:
    if (processed_dir / "data.npy").exists():
        return True
    return any(
        child.is_dir() and (child / "data.npy").exists()
        for child in processed_dir.iterdir()
    ) if processed_dir.exists() else False


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
    if has_processed_bundle(processed_dir):
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


def family_variant_dirs(processed_dir: Path) -> list[Path]:
    return sorted(
        p for p in processed_dir.iterdir() if p.is_dir() and (p / "data.npy").exists()
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_truth_adjacency(processed_dir: Path, variant: str | None = None) -> np.ndarray | None:
    truth_dir = processed_dir / variant if variant else processed_dir
    data_path = truth_dir / "data.npy"
    if not data_path.exists() and variant is not None:
        data_path = processed_dir / "data.npy"
    if not data_path.exists():
        return None

    payload = np.load(data_path, allow_pickle=True).item()
    truth = payload.get("y")
    if truth is None:
        return None
    return np.asarray(truth)


def load_prediction_adjacency(method: str, output_dir: Path) -> np.ndarray | None:
    generic_candidates = [
        output_dir / "dag.txt",
        output_dir / "graph.txt",
        *sorted(output_dir.glob("graphs/*.txt")),
        *sorted(output_dir.glob("**/*dag*.txt")),
        *sorted(output_dir.glob("**/*graph*.txt")),
    ]
    for candidate in generic_candidates:
        if candidate.exists():
            try:
                if candidate.suffix.lower() == ".txt":
                    return np.loadtxt(candidate, dtype=float)
                return pd.read_csv(candidate, header=None).to_numpy()
            except Exception:
                try:
                    return np.loadtxt(candidate, dtype=float)
                except Exception:
                    continue

    if method == "DataAssist":
        candidates = [
            output_dir / "llm_prior" / "pc_causal_matrix_llm_prior.npy",
            output_dir / "llm_prior" / "pc_causal_matrix.npy",
            output_dir / "pc_causal_matrix_llm_prior.npy",
            output_dir / "base" / "pc_causal_matrix.npy",
            output_dir / "base" / "pc_causal_matrix_llm_prior.npy",
            output_dir / "pc_causal_matrix.npy",
        ]
        for candidate in candidates:
            if candidate.exists():
                return np.asarray(np.load(candidate, allow_pickle=True))
        return None

    if method == "GUIDE":
        bundle = output_dir / "results_custom.npy"
        if not bundle.exists():
            return None
        payload = np.load(bundle, allow_pickle=True).item()
        for key in ("final_dag", "best_adj", "adjacency"):
            value = payload.get(key)
            if value is not None:
                return np.asarray(value)
        return None

    if method == "KCRL":
        for candidate in (output_dir / "final_graph_pruned.npy", output_dir / "final_graph.npy"):
            if candidate.exists():
                return np.asarray(np.load(candidate, allow_pickle=True))
        return None

    if method == "UnifiedOneShot":
        candidate = output_dir / "adj_PC_g.csv"
        if candidate.exists():
            return pd.read_csv(candidate, header=None).to_numpy()
        return None

    if method == "JunctionTreeGap":
        artifact = output_dir / "junction_tree_artifact.json"
        summary = output_dir / "summary.json"
        payload: dict[str, Any] = {}
        if artifact.exists():
            payload.update(load_json(artifact))
        if summary.exists():
            payload.update(load_json(summary))
        adjacency = payload.get("adjacency")
        if adjacency is not None:
            return np.asarray(adjacency)
        candidate = output_dir / "adjacency.npy"
        if candidate.exists():
            return np.asarray(np.load(candidate, allow_pickle=True))
        candidate = output_dir / "adjacency.csv"
        if candidate.exists():
            return pd.read_csv(candidate, header=None).to_numpy()
        return None

    if method == "ILSCSL":
        for candidate in sorted(output_dir.glob("**/*.npy")):
            stem = candidate.stem.lower()
            if "adj" in stem or "graph" in stem:
                try:
                    return np.asarray(np.load(candidate, allow_pickle=True))
                except Exception:
                    continue
        for candidate in sorted(output_dir.glob("**/*.csv")):
            stem = candidate.stem.lower()
            if "adj" in stem or "graph" in stem:
                try:
                    return pd.read_csv(candidate, header=None).to_numpy()
                except Exception:
                    continue
        return None

    return None


def attach_sid(
    problem: str,
    method: str,
    processed_dir: Path,
    output_dir: Path,
    metrics: dict[str, Any] | None,
    *,
    variant: str | None = None,
) -> dict[str, Any] | None:
    truth = load_truth_adjacency(processed_dir, variant=variant)
    predicted = load_prediction_adjacency(method, output_dir)
    if truth is None or predicted is None:
        return metrics
    if truth.shape != predicted.shape:
        return metrics

    try:
        sid_result = compute_sid(truth, predicted)
    except Exception:
        return metrics

    enriched: dict[str, Any] = dict(metrics or {})
    enriched["sid"] = sid_result["sid"]
    return enriched


def enrich_metrics_from_graph(
    problem: str,
    method: str,
    processed_dir: Path,
    output_dir: Path,
    metrics: dict[str, Any] | None,
    *,
    variant: str | None = None,
) -> dict[str, Any] | None:
    predicted = load_prediction_adjacency(method, output_dir)
    truth = load_truth_adjacency(processed_dir, variant=variant)
    if predicted is None or truth is None or predicted.shape != truth.shape:
        return metrics

    computed = evaluate_against_truth(predicted, truth)
    merged: dict[str, Any] = dict(metrics or {})
    for key, value in computed.items():
        if key not in merged or merged[key] is None:
            merged[key] = value
            continue
        current = merged[key]
        if pd.isna(current):
            merged[key] = value
    return merged


def row_from_metrics(
    problem: str,
    method: str,
    status: str,
    metrics: dict[str, Any] | None,
    artifact: str | None = None,
    note: str | None = None,
    *,
    variant: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "problem": problem,
        "method": method,
        "status": status,
        "artifact": artifact,
        "note": note,
    }
    if variant is not None:
        row["variant"] = variant
    if metrics:
        row.update(normalize_metric_dict(metrics))
    return row


def _load_dataassist_truth_and_prediction(
    output_dir: Path,
    candidate: Path,
) -> dict[str, np.ndarray] | None:
    data_candidates = [
        output_dir / "data.npy",
        output_dir / "base" / "data.npy",
        output_dir / "llm_prior" / "data.npy",
    ]
    for data_path in data_candidates:
        if not data_path.exists():
            continue
        payload = np.load(data_path, allow_pickle=True).item()
        truth = payload.get("y")
        if truth is None:
            continue
        if not candidate.exists():
            continue
        prediction = np.asarray(np.load(candidate, allow_pickle=True))
        return {
            "truth": np.asarray(truth),
            "prediction": prediction,
        }
    return None


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

    llm_candidate = output_dir / "llm_prior" / "pc_causal_matrix_llm_prior.npy"
    base_candidate = output_dir / "base" / "pc_causal_matrix.npy"
    flat_llm_candidate = output_dir / "pc_causal_matrix_llm_prior.npy"
    flat_base_candidate = output_dir / "pc_causal_matrix.npy"
    for variant, candidate, eval_path in [
        ("llm_prior", llm_candidate, llm_eval),
        ("baseline", base_candidate, base_eval),
        ("llm_prior", flat_llm_candidate, flat_llm_eval),
        ("baseline", flat_base_candidate, flat_base_eval),
    ]:
        payload = _load_dataassist_truth_and_prediction(output_dir, candidate)
        if payload is None:
            continue
        metrics = evaluate_against_truth(payload["prediction"], payload["truth"])
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return variant, metrics, eval_path
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


def read_ilscsl_result(output_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    csv_candidates = [
        output_dir / "metrics.csv",
        *sorted(output_dir.glob("*/metrics.csv")),
    ]
    json_candidates = [
        output_dir / "summary.json",
        *sorted(output_dir.glob("*/summary.json")),
    ]
    for csv_path in csv_candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if df.empty:
                return None, csv_path
            return df.iloc[-1].to_dict(), csv_path
    for json_path in json_candidates:
        if json_path.exists():
            payload = load_json(json_path)
            if isinstance(payload, list) and payload:
                return payload[-1], json_path
            if isinstance(payload, dict):
                return payload, json_path
    return None, None


def read_jtgap_result(output_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    summary = output_dir / "summary.json"
    artifact = output_dir / "junction_tree_artifact.json"
    payload: dict[str, Any] = {}
    if summary.exists():
        payload.update(load_json(summary))
    if artifact.exists():
        payload["artifact_payload"] = load_json(artifact)
    if not payload:
        return None, None

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    metrics = dict(metrics)
    art = payload.get("artifact_payload")
    if isinstance(art, dict):
        if "dag_threshold" in art:
            metrics.setdefault("dag_threshold", art.get("dag_threshold"))
        if "iter_num" in art:
            metrics.setdefault("iter_num", art.get("iter_num"))
        adjacency = art.get("adjacency")
        if adjacency is not None and "nnz" not in metrics:
            metrics["nnz"] = int(np.asarray(adjacency).sum())
        if "weight_matrix" in art and art.get("weight_matrix") is not None:
            metrics.setdefault("weight_matrix_shape", list(np.asarray(art["weight_matrix"]).shape))
    if "iter_num" in payload and "iter_num" not in metrics:
        metrics["iter_num"] = payload.get("iter_num")
    if metrics:
        return metrics, artifact if artifact.exists() else summary
    return None, None


def read_llmcd_result(output_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    summary = output_dir / "results.txt"
    if not summary.exists():
        return None, None
    text = summary.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if "{" not in line:
            continue
        payload_text = line[line.find("{") :]
        try:
            payload = eval(
                payload_text,
                {"__builtins__": {}},
                {"nan": float("nan"), "inf": float("inf"), "Infinity": float("inf")},
            )
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload, summary
    return None, summary


def run_jtgap(problem: str, processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiment-config" / "run_experiment.py"),
                "--multirun",
                "--config-name=config-cluster",
                f"experiment=JunctionTreeGap_experiment",
                f"problem={problem}",
                "problem.generator=JunctionTreeGap",
                "jtgap.top_k=3",
                "jtgap.gap_weight=0.5",
                "jtgap.sparsity_threshold=0.0",
                "jtgap.edge_threshold=0.0",
                "jtgap.covariance_jitter=1e-6",
            ],
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        return row_from_metrics(problem, "JunctionTreeGap", "failed", None, artifact=str(output_dir), note=exc.stderr[-500:] if exc.stderr else "run failed")

    metrics, artifact = read_jtgap_result(output_dir)
    metrics = attach_sid(problem, "JunctionTreeGap", processed_dir, output_dir, metrics)
    return row_from_metrics(
        problem,
        "JunctionTreeGap",
        "ok" if metrics else "missing",
        metrics,
        artifact=str(artifact) if artifact else None,
    )


def run_llmcd(problem: str, processed_dir: Path, output_dir: Path, method: str, backend: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiment-config" / "run_experiment.py"),
                "--multirun",
                "--config-name=config-cluster",
                f"experiment={method}_experiment",
                f"problem={problem}",
                f"problem.generator={method}",
                f"llmcd.backend={backend}",
            ],
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        return row_from_metrics(problem, method, "failed", None, artifact=str(output_dir), note=exc.stderr[-500:] if exc.stderr else "run failed")

    metrics, artifact = read_llmcd_result(output_dir)
    metrics = enrich_metrics_from_graph(problem, method, processed_dir, output_dir, metrics)
    row = row_from_metrics(problem, method, "ok" if metrics else "missing", metrics, artifact=str(artifact) if artifact else None)
    return row


def run_ilscsl(problem: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    alg = os.environ.get("ILSCSL_ALG", "HC")
    score = os.environ.get("ILSCSL_SCORE", "bdeu")
    model = os.environ.get("ILSCSL_MODEL", "gpt-4")
    data_index = os.environ.get("ILSCSL_DATA_INDEX", "1")
    datasize_index = os.environ.get("ILSCSL_DATASIZE_INDEX", "0")
    llm_url = os.environ.get("ILS_CSL_LLM_URL")
    if not llm_url:
        return row_from_metrics(problem, "ILSCSL", "failed", None, artifact=str(output_dir), note="missing ILS_CSL_LLM_URL")

    try:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "experiment-config" / "run_experiment.py"),
                "--multirun",
                "--config-name=config-cluster",
                "experiment=ILSCSL_experiment",
                f"problem={problem}",
                "problem.generator=ILSCSL",
                f"ilscsl.alg={alg}",
                f"ilscsl.score={score}",
                f"ilscsl.model={model}",
                f"ilscsl.data_index={data_index}",
                f"ilscsl.datasize_index={datasize_index}",
                f"ilscsl.llm_url={llm_url}",
            ],
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        return row_from_metrics(problem, "ILSCSL", "failed", None, artifact=str(output_dir), note=exc.stderr[-500:] if exc.stderr else "run failed")

    metrics, artifact = read_ilscsl_result(output_dir)
    metrics = attach_sid(problem, "ILSCSL", PROCESSED_ROOT / problem, output_dir, metrics)
    row = row_from_metrics(problem, "ILSCSL", "ok" if metrics else "missing", metrics, artifact=str(artifact) if artifact else None)
    row["variant"] = alg
    return row


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
    metrics = attach_sid(problem, "DataAssist", processed_dir, output_dir, metrics)

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
        metrics = attach_sid(problem, "DataAssist", processed_dir, output_dir, metrics)
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
    metrics = attach_sid(problem, "GUIDE", processed_dir, output_dir, metrics)
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
    metrics = attach_sid(problem, "KCRL", processed_dir, output_dir, metrics)
    return row_from_metrics(problem, "KCRL", "ok", metrics, artifact=str(artifact) if artifact else None)


def run_unified(problem: str, processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    method_dir = (
        PROJECT_ROOT
        / "scripts"
        / "baseline"
        / "Causal-LLM_Unified_One-Shot_Framework"
        / "Dag_generation and model_evaluation"
    )
    if problem in {"BIAS", "LEGAL"}:
        rows: list[dict[str, Any]] = []
        for variant_dir in family_variant_dirs(processed_dir):
            variant_output_dir = output_dir / variant_dir.name
            tmp_dir = variant_output_dir / "tmp_inputs"
            samples_path, adj_path = bundle_to_csv_inputs(variant_dir, tmp_dir)
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
                        str(variant_output_dir),
                        "--enabled-models",
                        "PC",
                    ],
                    cwd=method_dir,
                )
            except subprocess.CalledProcessError as exc:
                rows.append(
                    row_from_metrics(
                        problem,
                        "UnifiedOneShot",
                        "failed",
                        None,
                        artifact=str(variant_output_dir),
                        note=exc.stderr[-500:] if exc.stderr else "run failed",
                    )
                )
                rows[-1]["variant"] = variant_dir.name
                continue

            metrics, artifact = read_unified_result(variant_output_dir)
            metrics = attach_sid(
                problem,
                "UnifiedOneShot",
                variant_dir,
                variant_output_dir,
                metrics,
                variant=variant_dir.name,
            )
            row = row_from_metrics(
                problem,
                "UnifiedOneShot",
                "ok",
                metrics,
                artifact=str(artifact) if artifact else None,
            )
            row["variant"] = variant_dir.name
            rows.append(row)
        return rows

    tmp_dir = output_dir / "tmp_inputs"
    samples_path, adj_path = bundle_to_csv_inputs(processed_dir, tmp_dir)
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
    metrics = attach_sid(problem, "UnifiedOneShot", processed_dir, output_dir, metrics)
    return row_from_metrics(problem, "UnifiedOneShot", "ok", metrics, artifact=str(artifact) if artifact else None)


def summarize_problem(problem: str, results: list[dict[str, Any]], output_dir: Path) -> Path:
    df = pd.DataFrame(results)
    desired_cols = summary_columns(results)
    if not desired_cols:
        desired_cols = ["problem", "method"]
    if not df.empty:
        payload_cols = [col for col in desired_cols if col not in {"problem", "method"}]
        if payload_cols:
            blank = (
                df[payload_cols]
                .fillna("")
                .astype(str)
                .apply(lambda col: col.map(lambda x: x.strip() == ""))
                .all(axis=1)
            )
            df = df.loc[~blank].copy()
        else:
            df = df.iloc[0:0].copy()
    for col in desired_cols:
        if col not in df.columns:
            df[col] = None
    df = df[desired_cols]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{problem}_summary.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _build_single_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {method} on a single problem.")
    parser.add_argument("--problem", required=True, help="Problem name, e.g. sachs or SmBFO.")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_ROOT)
    return parser


def _run_single_method(method: str, problem: str, output_dir: Path) -> int:
    processed_dir = ensure_processed(problem)
    method_output_dir = OUTPUT_ROOT / method / problem
    if method == "DataAssist":
        row = run_dataassist(problem, processed_dir, method_output_dir)
    elif method == "GUIDE":
        row = run_guide(problem, processed_dir, method_output_dir)
    elif method == "KCRL":
        row = run_kcrl(problem, processed_dir, method_output_dir)
    elif method == "UnifiedOneShot":
        row = run_unified(problem, processed_dir, method_output_dir)
    else:
        raise ValueError(f"Unknown method: {method}")
    summarize_problem(problem, [row], output_dir)
    print(f"Wrote summary for {problem} to {output_dir / f'{problem}_summary.csv'}")
    return 0


def run_dataassist_main() -> int:
    args = _build_single_parser("DataAssist").parse_args()
    return _run_single_method("DataAssist", args.problem, args.output_dir)


def run_guide_main() -> int:
    args = _build_single_parser("GUIDE").parse_args()
    return _run_single_method("GUIDE", args.problem, args.output_dir)


def run_kcrl_main() -> int:
    args = _build_single_parser("KCRL").parse_args()
    return _run_single_method("KCRL", args.problem, args.output_dir)


def run_unified_main() -> int:
    args = _build_single_parser("UnifiedOneShot").parse_args()
    return _run_single_method("UnifiedOneShot", args.problem, args.output_dir)


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
        return [
            "DataAssist",
            "GUIDE",
            "KCRL",
            "UnifiedOneShot",
            "JunctionTreeGap",
            "LLM_CaMML",
            "LLM_MINOBSx",
        ]
    methods = [item.strip() for item in selection.split(",") if item.strip()]
    allowed = {"DataAssist", "GUIDE", "KCRL", "UnifiedOneShot", "ILSCSL", "JunctionTreeGap", "LLM_CaMML", "LLM_MINOBSx"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def _graphdag_problem_list(args: argparse.Namespace) -> list[str]:
    if args.all_problems:
        return sorted(p.name for p in PROCESSED_ROOT.iterdir() if has_processed_bundle(p))
    if args.problems:
        return [item.strip() for item in args.problems.split(",") if item.strip()]
    return [args.problem]


def _graphdag_render_one(problem: str, method: str, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from castle.common import GraphDAG

    truth = load_truth_adjacency(PROCESSED_ROOT / problem)
    if truth is None:
        return []

    method_root = OUTPUT_ROOT / method / problem
    if not method_root.exists():
        return []

    rendered: list[Path] = []

    if problem in FAMILY_PROBLEMS and method == "UnifiedOneShot":
        for variant_dir in family_variant_dirs(PROCESSED_ROOT / problem):
            pred = load_prediction_adjacency(method, OUTPUT_ROOT / method / problem / variant_dir.name)
            if pred is None:
                continue
            pred = np.asarray(pred)
            truth_variant = np.asarray(load_truth_adjacency(variant_dir))
            if pred.ndim != 2 or truth_variant.ndim != 2:
                continue
            if pred.shape != truth_variant.shape or pred.shape[0] != pred.shape[1] or pred.shape[0] < 2:
                continue
            save_path = output_dir / method / problem / f"{variant_dir.name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            GraphDAG(est_dag=(np.asarray(pred) != 0).astype(int), true_dag=(np.asarray(truth_variant) != 0).astype(int), show=False, save_name=str(save_path))
            plt.close("all")
            rendered.append(save_path)
        return rendered

    pred = load_prediction_adjacency(method, method_root)
    if pred is None:
        return []
    pred = np.asarray(pred)
    if truth.ndim != 2 or pred.ndim != 2:
        return []
    if truth.shape != pred.shape or pred.shape[0] != pred.shape[1] or pred.shape[0] < 2:
        return []

    save_path = output_dir / method / f"{problem}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    GraphDAG(est_dag=(np.asarray(pred) != 0).astype(int), true_dag=(np.asarray(truth) != 0).astype(int), show=False, save_name=str(save_path))
    plt.close("all")
    rendered.append(save_path)
    return rendered


def graphdag_report_main() -> int:
    parser = argparse.ArgumentParser(description="Generate GraphDAG comparison plots for problem/method outputs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--problem", help="Single problem name to render.")
    group.add_argument("--problems", help="Comma-separated problem names to render.")
    group.add_argument("--all-problems", action="store_true", help="Render all processed problems.")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods to render: DataAssist,GUIDE,KCRL,UnifiedOneShot,JunctionTreeGap,ILSCSL or all.",
    )
    parser.add_argument("--output-dir", type=Path, default=HEATMAP_ROOT, help="Directory where images will be written.")
    args = parser.parse_args()

    methods = methods_to_run(args.methods)
    problems = _graphdag_problem_list(args)
    rendered_total = 0
    for problem in problems:
        for method in methods:
            rendered = _graphdag_render_one(problem, method, args.output_dir)
            rendered_total += len(rendered)
            for path in rendered:
                print(f"Wrote {path}")
    print(f"Rendered {rendered_total} comparison plots under {args.output_dir}")
    return 0


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
            unified_result = run_unified(problem, processed_dir, output_dir)
            if isinstance(unified_result, list):
                results.extend(unified_result)
            else:
                results.append(unified_result)
        elif method == "ILSCSL":
            results.append(run_ilscsl(problem, output_dir))
        elif method == "JunctionTreeGap":
            results.append(run_jtgap(problem, processed_dir, output_dir))
        elif method == "LLM_CaMML":
            results.append(run_llmcd(problem, processed_dir, output_dir, "LLM_CaMML", "CaMML"))
        elif method == "LLM_MINOBSx":
            results.append(run_llmcd(problem, processed_dir, output_dir, "LLM_MINOBSx", "MINOBSx"))
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
            if not has_processed_bundle(processed_dir):
                print(f"Skipping {problem}: processed bundle not found at {processed_dir / 'data.npy'}")
                continue
            results = []
            for method in selected_methods:
                output_dir = OUTPUT_ROOT / method / problem
                if method == "DataAssist":
                    _, metrics, artifact = read_dataassist_result(output_dir)
                    metrics = attach_sid(problem, method, processed_dir, output_dir, metrics)
                elif method == "GUIDE":
                    metrics, artifact = read_guide_result(output_dir)
                    metrics = attach_sid(problem, method, processed_dir, output_dir, metrics)
                elif method == "KCRL":
                    metrics, artifact = read_kcrl_result(output_dir)
                    metrics = attach_sid(problem, method, processed_dir, output_dir, metrics)
                elif method == "UnifiedOneShot":
                    if problem in {"BIAS", "LEGAL"}:
                        rows: list[dict[str, Any]] = []
                        if output_dir.exists():
                            for variant_dir in family_variant_dirs(output_dir):
                                metrics, artifact = read_unified_result(variant_dir)
                                metrics = attach_sid(
                                    problem,
                                    method,
                                    PROCESSED_ROOT / problem,
                                    variant_dir,
                                    metrics,
                                    variant=variant_dir.name,
                                )
                                row = row_from_metrics(
                                    problem,
                                    method,
                                    "ok" if metrics else "missing",
                                    metrics,
                                    artifact=str(artifact) if artifact else None,
                                    variant=variant_dir.name,
                                )
                                rows.append(row)
                        if rows:
                            results.extend(rows)
                            continue
                        metrics, artifact = None, None
                    else:
                        metrics, artifact = read_unified_result(output_dir)
                        metrics = attach_sid(problem, method, processed_dir, output_dir, metrics)
                elif method == "ILSCSL":
                    metrics, artifact = read_ilscsl_result(output_dir)
                    metrics = attach_sid(problem, method, processed_dir, output_dir, metrics)
                elif method == "JunctionTreeGap":
                    metrics, artifact = read_jtgap_result(output_dir)
                    metrics = attach_sid(problem, method, processed_dir, output_dir, metrics)
                elif method == "LLM_CaMML":
                    metrics, artifact = read_llmcd_result(output_dir)
                    metrics = enrich_metrics_from_graph(problem, method, processed_dir, output_dir, metrics)
                elif method == "LLM_MINOBSx":
                    metrics, artifact = read_llmcd_result(output_dir)
                    metrics = enrich_metrics_from_graph(problem, method, processed_dir, output_dir, metrics)
                else:
                    metrics, artifact = None, None
                status = "ok" if metrics else "missing"
                results.append(
                    row_from_metrics(
                        problem,
                        method,
                        status,
                        metrics,
                        artifact=str(artifact) if artifact else None,
                    )
                )
        else:
            try:
                results = run_problem(problem, selected_methods)
            except FileNotFoundError as exc:
                print(f"Skipping {problem}: {exc}")
                continue

        summarize_problem(problem, results, args.output_dir)
        print(f"Wrote summary for {problem} to {args.output_dir / f'{problem}_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
