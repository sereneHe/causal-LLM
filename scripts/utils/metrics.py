#!/usr/bin/env python3
"""Run PC causal discovery on SmBFO data.

This script extracts the data-preparation and PC-discovery parts from
`causal_llm_1.ipynb` and makes them reusable from the command line.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import networkx as nx
from castle.algorithms import PC
from castle.common.priori_knowledge import PrioriKnowledge
from sklearn.preprocessing import StandardScaler

from dag_io import adjacency_is_dag


COMPOSITION_TAGS = [0, 0, 0, 7, 7, 7, 7, 7, 10, 10, 13, 13, 20, 20]

IMG_FILENAMES = [
    "Sm_0_0_HAADF.h5",
    "Sm_0_1_HAADF.h5",
    "Sm_0_2_HAADF.h5",
    "Sm_7_0_HAADF.h5",
    "Sm_7_1_HAADF.h5",
    "Sm_7_2_HAADF.h5",
    "Sm_7_3_HAADF.h5",
    "Sm_7_4_HAADF.h5",
    "SM_10_0_HAADF.h5",
    "Sm_10_1_HAADF.h5",
    "Sm_13_0_HAADF.h5",
    "Sm_13_1_HAADF.h5",
    "Sm_20_0_HAADF.h5",
    "Sm_20_1_HAADF.h5",
]

UCPARAM_FILENAMES = [
    "Sm_0_0_UCParameterization.h5",
    "Sm_0_1_UCParameterization.h5",
    "Sm_0_2_UCParameterization.h5",
    "Sm_7_0_UCParameterization.h5",
    "Sm_7_1_UCParameterization.h5",
    "Sm_7_2_UCParameterization.h5",
    "Sm_7_3_UCParameterization.h5",
    "Sm_7_4_UCParameterization.h5",
    "Sm_10_0_UCParameterization.h5",
    "Sm_10_1_UCParameterization.h5",
    "Sm_13_0_UCParameterization.h5",
    "Sm_13_1_UCParameterization.h5",
    "Sm_20_0_UCParameterization.h5",
    "Sm_20_1_UCParameterization.h5",
]

ALL_VARS = {
    "Alkali_Cations": 0,
    "Transition_Metal_Cations": 1,
    "Lattice_Parameter": 2,
    "Composition": 3,
    "Unit_Cell_Angle": 4,
    "Volume": 5,
    "In_Plane_Polarization": 6,
}


def map2grid(inab: np.ndarray, in_val: np.ndarray) -> tuple[np.ndarray, list[int]]:
    default_val = np.nan
    abrng = [
        int(np.min(inab[:, 0])),
        int(np.max(inab[:, 0])),
        int(np.min(inab[:, 1])),
        int(np.max(inab[:, 1])),
    ]
    abind = inab.copy()
    abind[:, 0] -= abrng[0]
    abind[:, 1] -= abrng[2]
    valgrid = np.empty((abrng[1] - abrng[0] + 1, abrng[3] - abrng[2] + 1))
    valgrid[:] = default_val
    valgrid[abind[:, 0].astype(int), abind[:, 1].astype(int)] = in_val[:]
    return valgrid, abrng


def load_smbfo_entries(data_dir: Path) -> list[dict]:
    uc_params = []
    for filename in UCPARAM_FILENAMES:
        uc_params.append(h5py.File(data_dir / filename, "r"))

    img_data = []
    for filename in IMG_FILENAMES:
        img_data.append(h5py.File(data_dir / filename, "r")["MainImage"])

    sbfo_data = []
    for i in range(len(IMG_FILENAMES)):
        temp_dict = {
            "Index": i,
            "Composition": COMPOSITION_TAGS[i],
            "Image": img_data[i],
            "Filename": IMG_FILENAMES[i],
        }
        for key in uc_params[i].keys():
            temp_dict[key] = uc_params[i][key][()]

        temp_dict["ab_a"] = map2grid(uc_params[i]["ab"][()].T, uc_params[i]["ab"][()].T[:, 0])[0]
        temp_dict["ab_b"] = map2grid(uc_params[i]["ab"][()].T, uc_params[i]["ab"][()].T[:, 1])[0]
        temp_dict["ab_x"] = map2grid(uc_params[i]["ab"][()].T, uc_params[i]["xy_COM"][()].T[:, 0])[0]
        temp_dict["ab_y"] = map2grid(uc_params[i]["ab"][()].T, uc_params[i]["xy_COM"][()].T[:, 1])[0]
        temp_dict["ab_Px"] = map2grid(uc_params[i]["ab"][()].T, uc_params[i]["Pxy"][0])[0]
        temp_dict["ab_Py"] = map2grid(uc_params[i]["ab"][()].T, uc_params[i]["Pxy"][1])[0]
        sbfo_data.append(temp_dict)

    return sbfo_data


def extract_physical_values(entry: dict) -> dict:
    return {
        "Alkali_Cations": entry["I1"].flatten(),
        "Transition_Metal_Cations": entry["I5"].flatten(),
        "Lattice_Parameter": entry["a"].flatten(),
        "Composition": entry["Composition"],
        "Unit_Cell_Angle": entry["alpha"].flatten(),
        "Volume": entry["Vol"].flatten(),
        "In_Plane_Polarization": entry["Pxy"][0].flatten(),
    }


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    sbfo_data = load_smbfo_entries(data_dir)
    data_entries: list[dict] = []
    for entry in sbfo_data:
        extracted = extract_physical_values(entry)
        for i in range(len(extracted["Alkali_Cations"])):
            data_entries.append(
                {
                    "Alkali_Cations": extracted["Alkali_Cations"][i],
                    "Transition_Metal_Cations": extracted["Transition_Metal_Cations"][i],
                    "Lattice_Parameter": extracted["Lattice_Parameter"][i],
                    "Composition": extracted["Composition"],
                    "Unit_Cell_Angle": extracted["Unit_Cell_Angle"][i],
                    "Volume": extracted["Volume"][i],
                    "In_Plane_Polarization": extracted["In_Plane_Polarization"][i],
                }
            )

    df = pd.DataFrame(data_entries).replace([np.inf, -np.inf], np.nan).dropna()
    max_value = np.finfo(np.float64).max
    return df.clip(upper=max_value)


def scale_dataframe(df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df)


def load_prior(prior_path: Path) -> PrioriKnowledge:
    payload = json.loads(prior_path.read_text())
    prior = PrioriKnowledge(n_nodes=len(ALL_VARS))
    required = [tuple(edge) for edge in payload.get("required_edges", [])]
    forbidden = [tuple(edge) for edge in payload.get("forbidden_edges", [])]
    if required:
        prior.add_required_edges(required)
    if forbidden:
        prior.add_forbidden_edges(forbidden)
    return prior


def run_pc(scaled_data: np.ndarray, prior: PrioriKnowledge | None = None) -> np.ndarray:
    if prior is None:
        pc = PC(variant="stable")
    else:
        pc = PC(priori_knowledge=prior, variant="stable")
    pc.learn(scaled_data)
    return pc.causal_matrix


def named_matrix(matrix: np.ndarray) -> pd.DataFrame:
    inverse_var_map = {v: k for k, v in ALL_VARS.items()}
    labels = [inverse_var_map[i] for i in range(matrix.shape[0])]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def load_data_bundle(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype != object:
        return payload, None

    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()

    if not isinstance(payload, dict) or "x" not in payload:
        raise ValueError(f"Unsupported data bundle format: {path}")

    x = np.asarray(payload["x"])
    y = payload.get("y")
    if y is not None:
        y = np.asarray(y)
    return x, y


def resolve_graph_path(data_npy: Path | None, data_dir: Path) -> Path | None:
    candidates: list[Path] = []
    if data_npy is not None:
        candidates.extend(
            [
                data_npy.parent / "adj.npy",
                data_npy.parent / "DAG.npy",
            ]
        )
    candidates.extend(
        [
            data_dir / "adj.npy",
            data_dir / "DAG.npy",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def orient_adjacency(matrix: np.ndarray, edge_direction: str = "from row to column") -> np.ndarray:
    adj = binarize_matrix(matrix)
    if edge_direction == "from row to column":
        return adj
    if edge_direction == "from column to row":
        return adj.T
    raise ValueError(f"Unsupported edge_direction: {edge_direction}")


def adjacency_to_digraph(matrix: np.ndarray) -> nx.DiGraph:
    adj = binarize_matrix(matrix)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    rows, cols = np.where(adj != 0)
    graph.add_edges_from((int(r), int(c)) for r, c in zip(rows, cols))
    return graph


def _d_separated_after_removing_outgoing(
    true_adj: np.ndarray,
    x: int,
    y: int,
    conditioning: set[int],
) -> bool:
    graph = adjacency_to_digraph(true_adj)
    if graph.has_node(x):
        graph = graph.copy()
        graph.remove_edges_from(list(graph.out_edges(x)))

    relevant = set(conditioning) | {x, y}
    for node in list(relevant):
        relevant.update(nx.ancestors(graph, node))

    subgraph = graph.subgraph(relevant).copy()
    moral = nx.Graph()
    moral.add_nodes_from(subgraph.nodes)
    moral.add_edges_from(subgraph.to_undirected().edges)
    for child in subgraph.nodes:
        parents = list(subgraph.predecessors(child))
        for idx, left in enumerate(parents):
            for right in parents[idx + 1 :]:
                moral.add_edge(left, right)

    moral.remove_nodes_from(conditioning)
    if x not in moral or y not in moral:
        return True
    return not nx.has_path(moral, x, y)


def _sid_fallback(true_adj: np.ndarray, est_adj: np.ndarray) -> tuple[float, int]:
    graph_true = adjacency_to_digraph(true_adj)
    graph_est = adjacency_to_digraph(est_adj)
    if not nx.is_directed_acyclic_graph(graph_true):
        raise ValueError("true_adj must represent a DAG")
    if not nx.is_directed_acyclic_graph(graph_est):
        raise ValueError("est_adj must represent a DAG")

    p = true_adj.shape[0]
    descendants = {node: nx.descendants(graph_true, node) for node in range(p)}
    mistakes = 0
    for i in range(p):
        parents_est = set(np.flatnonzero(est_adj[:, i]).tolist())
        if i in parents_est:
            mistakes += p - 1
            continue
        invalid_due_to_descendants = parents_est & descendants[i]
        if invalid_due_to_descendants:
            mistakes += p - 1
            continue
        for j in range(p):
            if i == j:
                continue
            if not _d_separated_after_removing_outgoing(true_adj, i, j, parents_est):
                mistakes += 1
    normalized = mistakes / max(p * (p - 1), 1)
    return normalized, mistakes


def _sid_via_r_package(true_adj: np.ndarray, est_adj: np.ndarray) -> tuple[float, int]:
    """Compute SID through the R `SID` package."""
    if true_adj.shape != est_adj.shape:
        raise ValueError(
            f"SID requires matching shapes, got {true_adj.shape} and {est_adj.shape}"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        true_path = tmp_path / "true.csv"
        est_path = tmp_path / "est.csv"
        script_path = tmp_path / "sid_runner.R"

        np.savetxt(true_path, binarize_matrix(true_adj).astype(int), fmt="%d", delimiter=",")
        np.savetxt(est_path, binarize_matrix(est_adj).astype(int), fmt="%d", delimiter=",")
        script_path.write_text(
            """
suppressPackageStartupMessages(library(SID))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
true_graph <- as.matrix(read.csv(args[1], header = FALSE))
est_graph <- as.matrix(read.csv(args[2], header = FALSE))
res <- structIntervDist(true_graph, est_graph)
payload <- list(
  sid = as.integer(res$sid),
  sidUpperBound = as.integer(res$sidUpperBound),
  sidLowerBound = as.integer(res$sidLowerBound)
)
cat(jsonlite::toJSON(payload, auto_unbox = TRUE))
""".strip()
            + "\n",
            encoding="utf-8",
        )

        completed = subprocess.run(
            ["Rscript", "--vanilla", str(script_path), str(true_path), str(est_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout.strip() or "{}")
        mistakes = int(payload["sid"])
        normalized = mistakes / max(true_adj.shape[0] * (true_adj.shape[0] - 1), 1)
        return normalized, mistakes


def _project_to_dag(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Greedily project a weighted adjacency matrix onto a DAG."""
    weighted = np.asarray(matrix)
    adj = binarize_matrix(weighted)
    n = adj.shape[0]
    dag = np.zeros((n, n), dtype=int)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    removed = 0

    edges: list[tuple[int, int, float]] = []
    for row, col in zip(*np.where(adj != 0)):
        if row == col:
            removed += 1
            continue
        weight = float(np.asarray(weighted[row, col]).reshape(()))
        edges.append((int(row), int(col), weight))

    edges.sort(key=lambda item: (-abs(item[2]), item[0], item[1]))
    for row, col, _ in edges:
        graph.add_edge(row, col)
        if nx.is_directed_acyclic_graph(graph):
            dag[row, col] = 1
        else:
            graph.remove_edge(row, col)
            removed += 1

    return dag, removed


def _sid_score(true_adj: np.ndarray, est_adj: np.ndarray) -> tuple[float, int]:
    """Compute standard SID for DAG inputs, using the R package when possible."""
    try:
        return _sid_via_r_package(true_adj, est_adj)
    except Exception:  # pragma: no cover - fallback for backend quirks
        return _sid_fallback(true_adj, est_adj)


def compute_sid(
    true_graph: np.ndarray,
    est_graph: np.ndarray,
    *,
    edge_direction: str = "from row to column",
) -> dict[str, float | int]:
    true_adj = orient_adjacency(true_graph, edge_direction=edge_direction)
    est_adj = orient_adjacency(est_graph, edge_direction=edge_direction)

    true_is_dag = adjacency_is_dag(true_adj)
    est_is_dag = adjacency_is_dag(est_adj)
    sid_mode = "sid"
    cycle_penalty = 0

    if true_is_dag and est_is_dag:
        normalized, mistakes = _sid_score(true_adj, est_adj)
    else:
        projected_true, true_removed = _project_to_dag(true_adj)
        projected_est, est_removed = _project_to_dag(est_adj)
        normalized, mistakes = _sid_score(projected_true, projected_est)
        cycle_penalty = int(true_removed + est_removed)
        mistakes += cycle_penalty
        normalized = mistakes / max(true_adj.shape[0] * (true_adj.shape[0] - 1), 1)
        sid_mode = "sid_like"

    return {
        "sid": mistakes,
        "sid_normalized": normalized,
        "sid_mode": sid_mode,
        "sid_cycle_penalty": cycle_penalty,
    }


METRIC_FIELD_ORDER = [
    "shd",
    "f1",
    "sid",
    "dag_threshold",
    "iter_num",
    "precision",
    "recall",
    "fdr",
    "fpr",
    "tnr",
    "fnr",
    "accuracy",
    "tp",
    "fp",
    "tn",
    "fn",
    "nnz",
    "n_nodes",
    "extra",
    "missing",
    "reverse",
    "gscore",
    "bic",
    "cycness",
    "penalty",
    "reward",
    "kl",
    "sym_kl",
    "l2",
    "time",
    "exist_mode",
]

METRIC_ALIASES = {
    "F1": "f1",
    "FDR": "fdr",
    "FPR": "fpr",
    "TPR": "recall",
    "TNR": "tnr",
    "FNR": "fnr",
    "NNZ": "nnz",
    "SID": "sid",
    "BIC": "bic",
    "CYCNESS": "cycness",
    "KL": "kl",
    "SYM_KL": "sym_kl",
    "L2": "l2",
    "Pred Size": "nnz",
    "pred_size": "nnz",
}

SUMMARY_METADATA_FIELDS = ["problem", "method", "variant"]


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value


def _canonical_metric_key(key: str) -> str:
    stripped = key.strip()
    if stripped in METRIC_ALIASES:
        return METRIC_ALIASES[stripped]
    lowered = stripped.lower().replace(" ", "_").replace("-", "_")
    return METRIC_ALIASES.get(stripped.upper(), lowered)


def binarize_matrix(matrix: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix) != 0).astype(int)


def confusion_counts(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, int]:
    pred = binarize_matrix(predicted)
    truth = binarize_matrix(ground_truth)
    if pred.shape != truth.shape:
        raise ValueError(
            f"Predicted graph shape {pred.shape} does not match ground truth {truth.shape}"
        )

    pred_flat = pred.reshape(-1)
    truth_flat = truth.reshape(-1)
    tp = int(np.sum((pred_flat == 1) & (truth_flat == 1)))
    fp = int(np.sum((pred_flat == 1) & (truth_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (truth_flat == 1)))
    tn = int(np.sum((pred_flat == 0) & (truth_flat == 0)))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "nnz": int(pred.sum()),
        "n_nodes": int(pred.shape[0]),
        "shd": int(np.sum(pred != truth)),
    }


def metrics_from_confusion(counts: dict[str, int | float]) -> dict[str, float | int]:
    tp = int(counts.get("tp", 0) or 0)
    fp = int(counts.get("fp", 0) or 0)
    fn = int(counts.get("fn", 0) or 0)
    tn = int(counts.get("tn", 0) or 0)
    nnz = int(counts.get("nnz", tp + fp) or 0)
    total = tp + fp + fn + tn

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    fdr = safe_div(fp, tp + fp)
    fpr = safe_div(fp, fp + tn)
    tnr = safe_div(tn, tn + fp)
    fnr = safe_div(fn, fn + tp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, total)

    metrics: dict[str, float | int] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "nnz": nnz,
        "precision": precision,
        "recall": recall,
        "fdr": fdr,
        "fpr": fpr,
        "tnr": tnr,
        "fnr": fnr,
        "f1": f1,
        "accuracy": accuracy,
    }
    if "shd" in counts:
        metrics["shd"] = int(counts["shd"])
    if "n_nodes" in counts:
        metrics["n_nodes"] = int(counts["n_nodes"])
    return metrics


def normalize_metric_dict(metrics: dict[str, object] | None) -> dict[str, object]:
    if not metrics:
        return {}

    normalized: dict[str, object] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        canonical = _canonical_metric_key(key)
        if canonical == "pred_size":
            canonical = "nnz"
        normalized[canonical] = _to_builtin(value)

    if "tp" in normalized or "fp" in normalized or "tn" in normalized or "fn" in normalized:
        counts: dict[str, int | float] = {}
        for field in ("tp", "fp", "tn", "fn", "nnz", "shd", "n_nodes"):
            if field in normalized:
                counts[field] = normalized[field]
        normalized.update(metrics_from_confusion(counts))

    if "precision" not in normalized and "fdr" in normalized:
        normalized["precision"] = 1.0 - float(normalized["fdr"])
    if "fdr" not in normalized and "precision" in normalized:
        normalized["fdr"] = 1.0 - float(normalized["precision"])
    if "recall" not in normalized and "tpr" in normalized:
        normalized["recall"] = float(normalized["tpr"])
    if "tnr" not in normalized and "fpr" in normalized:
        normalized["tnr"] = 1.0 - float(normalized["fpr"])
    if "fnr" not in normalized and "recall" in normalized:
        normalized["fnr"] = 1.0 - float(normalized["recall"])
    if "f1" not in normalized and "precision" in normalized and "recall" in normalized:
        normalized["f1"] = safe_div(
            2 * float(normalized["precision"]) * float(normalized["recall"]),
            float(normalized["precision"]) + float(normalized["recall"]),
        )
    if "accuracy" not in normalized and {"tp", "fp", "tn", "fn"} <= set(normalized):
        normalized["accuracy"] = safe_div(
            float(normalized["tp"]) + float(normalized["tn"]),
            float(normalized["tp"]) + float(normalized["fp"]) + float(normalized["tn"]) + float(normalized["fn"]),
        )
    if "sid" not in normalized:
        normalized["sid"] = None
    return normalized


def summary_columns(rows: list[dict[str, object]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()

    def add(column: str) -> None:
        if column not in seen:
            seen.add(column)
            columns.append(column)

    for field in SUMMARY_METADATA_FIELDS:
        if any(field in row for row in rows):
            add(field)

    for field in METRIC_FIELD_ORDER:
        if any(field in row for row in rows):
            add(field)

    excluded = {"status", "artifact", "note", "causal_model"}
    for row in rows:
        for key, value in row.items():
            if key in seen or key in excluded:
                continue
            if key in SUMMARY_METADATA_FIELDS or key in METRIC_FIELD_ORDER:
                continue
            if value is None:
                continue
            if isinstance(value, (list, dict, tuple, np.ndarray)):
                continue
            add(key)

    return columns


def evaluate_against_truth(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
    counts = confusion_counts(predicted, ground_truth)
    metrics = metrics_from_confusion(counts)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PC causal discovery on SmBFO data.")
    parser.add_argument(
        "--data-npy",
        type=Path,
        default=None,
        help="Optional standardized X.npy input. If provided, raw h5 loading is skipped.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/xiaoyuhe/Causal-LLM/dataset/SmBFO"),
        help="Directory containing SmBFO .h5 files.",
    )
    parser.add_argument(
        "--prior-json",
        type=Path,
        default=None,
        help="Optional JSON file produced by DataAssist_llm.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where CSV outputs will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    truth_graph = None
    if args.data_npy is not None:
        scaled_data, truth_graph = load_data_bundle(args.data_npy)
    else:
        df = build_dataframe(args.data_dir)
        scaled_data = scale_dataframe(df)
        df.to_csv(args.output_dir / "SmBFO_features.csv", index=False)

    np.save(args.output_dir / "X.npy", scaled_data)
    graph_path = resolve_graph_path(args.data_npy, args.data_dir)
    if truth_graph is None and graph_path is not None:
        truth_graph = np.load(graph_path)
    if truth_graph is not None:
        np.save(args.output_dir / "adj.npy", truth_graph)
    np.save(
        args.output_dir / "data.npy",
        {"x": scaled_data, "y": truth_graph},
        allow_pickle=True,
    )

    baseline = run_pc(scaled_data)
    np.save(args.output_dir / "pc_causal_matrix.npy", baseline)
    named_matrix(baseline).to_csv(args.output_dir / "pc_causal_matrix.csv")
    if truth_graph is not None:
        baseline_eval = evaluate_against_truth(baseline, truth_graph)
        (args.output_dir / "pc_causal_matrix_eval.json").write_text(
            json.dumps(baseline_eval, indent=2),
            encoding="utf-8",
        )

    if args.prior_json is not None:
        prior = load_prior(args.prior_json)
        prior_matrix = np.clip(prior.matrix, 0, 1)
        np.save(args.output_dir / "llm_prior_matrix.npy", prior_matrix)
        named_matrix(prior_matrix).to_csv(args.output_dir / "llm_prior_matrix.csv")
        informed = run_pc(scaled_data, prior=prior)
        np.save(args.output_dir / "pc_causal_matrix_llm_prior.npy", informed)
        named_matrix(informed).to_csv(args.output_dir / "pc_causal_matrix_llm_prior.csv")
        if truth_graph is not None:
            informed_eval = evaluate_against_truth(informed, truth_graph)
            (args.output_dir / "pc_causal_matrix_llm_prior_eval.json").write_text(
                json.dumps(informed_eval, indent=2),
                encoding="utf-8",
            )

    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
