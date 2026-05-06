#!/usr/bin/env python3
"""Run PC causal discovery on SmBFO data.

This script extracts the data-preparation and PC-discovery parts from
`causal_llm_1.ipynb` and makes them reusable from the command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from castle.algorithms import PC
from castle.common.priori_knowledge import PrioriKnowledge
from sklearn.preprocessing import StandardScaler


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


def binarize_matrix(matrix: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix) != 0).astype(int)


def evaluate_against_truth(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
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

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / len(pred_flat) if len(pred_flat) else 0.0
    shd = int(np.sum(pred != truth))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "shd": shd,
        "n_nodes": int(pred.shape[0]),
    }


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
