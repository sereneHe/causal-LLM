#!/usr/bin/env python3
"""Unified raw-to-processed preprocessing for causal discovery datasets.

Each dataset is standardized to:

  /dataset/processed/<dataset>/data.npy

where `data.npy` contains a dictionary:

  {"x": observational_data, "y": adjacency_matrix_or_none}

Raw inputs are staged under:

  /dataset/raw/<dataset>/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/xiaoyuhe/Causal-LLM")
DEFAULT_RAW_ROOT = PROJECT_ROOT / "dataset" / "raw"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"

BASELINE_ROOT = PROJECT_ROOT / "scripts" / "baseline"
ONE_SHOT_DATA_ROOT = BASELINE_ROOT / "Causal-LLM_Unified_One-Shot_Framework" / "Data"
GUIDE_DATA_ROOT = BASELINE_ROOT / "GUIDE" / "Datasets"
LEGACY_SMBFO_ROOT = PROJECT_ROOT / "dataset" / "SmBFO"

SMBFO_BASE_URL = "https://zenodo.org/record/4555979/files"
SMBFO_FILES = [
    "Sm_0_0_HAADF.h5",
    "Sm_0_1_HAADF.h5",
    "Sm_0_2_HAADF.h5",
    "Sm_0_0_UCParameterization.h5",
    "Sm_0_1_UCParameterization.h5",
    "Sm_0_2_UCParameterization.h5",
    "Sm_7_0_HAADF.h5",
    "Sm_7_1_HAADF.h5",
    "Sm_7_2_HAADF.h5",
    "Sm_7_3_HAADF.h5",
    "Sm_7_4_HAADF.h5",
    "Sm_7_0_UCParameterization.h5",
    "Sm_7_1_UCParameterization.h5",
    "Sm_7_2_UCParameterization.h5",
    "Sm_7_3_UCParameterization.h5",
    "Sm_7_4_UCParameterization.h5",
    "SM_10_0_HAADF.h5",
    "Sm_10_1_HAADF.h5",
    "Sm_10_0_UCParameterization.h5",
    "Sm_10_1_UCParameterization.h5",
    "Sm_13_0_HAADF.h5",
    "Sm_13_1_HAADF.h5",
    "Sm_13_0_UCParameterization.h5",
    "Sm_13_1_UCParameterization.h5",
    "Sm_20_0_HAADF.h5",
    "Sm_20_1_HAADF.h5",
    "Sm_20_0_UCParameterization.h5",
    "Sm_20_1_UCParameterization.h5",
]


class CausalDiscoveryPreprocessing:
    GRAPH_NAME_HINTS = ("adj", "dag", "admg", "mag", "pag")
    GRAPH_SUFFIXES = (".npy", ".csv", ".txt", ".tsv")

    @staticmethod
    def _as_numpy(data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        return np.asarray(data)

    @classmethod
    def to_adjacency_matrix(cls, graph_like) -> np.ndarray | None:
        if graph_like is None:
            return None
        arr = cls._as_numpy(graph_like)
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Expected square graph matrix, got shape {arr.shape}")
        arr = (np.abs(arr) > 1e-12).astype(int)
        np.fill_diagonal(arr, 0)
        return arr

    @classmethod
    def load_graph_file(cls, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            graph = np.load(path, allow_pickle=True)
        elif suffix in {".csv", ".txt", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            graph = pd.read_csv(path, header=None, sep=sep).to_numpy()
            if graph.ndim == 2 and graph.shape[0] != graph.shape[1]:
                # Keep the initial header=None read for plain numeric matrices,
                # but fall back to a header-aware read when the CSV carries
                # column names for adjacency labels.
                header_graph = pd.read_csv(path, header=0, sep=sep).to_numpy()
                if (
                    header_graph.ndim == 2
                    and header_graph.shape[0] == header_graph.shape[1]
                ):
                    graph = header_graph
        else:
            raise ValueError(f"Unsupported graph file: {path}")
        return cls.to_adjacency_matrix(graph)

    @classmethod
    def search_ground_truth_graph(cls, directory: Path) -> Path | None:
        directory = Path(directory)
        if not directory.exists():
            return None

        exact_names = ("adj.npy", "DAG.npy", "dag.npy", "adj.csv", "DAG.csv", "dag.csv")
        for name in exact_names:
            candidate = directory / name
            if candidate.exists():
                return candidate

        for candidate in sorted(directory.iterdir()):
            if not candidate.is_file():
                continue
            lowered = candidate.name.lower()
            if candidate.suffix.lower() not in cls.GRAPH_SUFFIXES:
                continue
            if any(hint in lowered for hint in cls.GRAPH_NAME_HINTS):
                return candidate
        return None

    @classmethod
    def save_bundle(
        cls,
        output_dir: Path,
        x,
        y=None,
        nodes=None,
        csv_name: str | None = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        x_np = cls._as_numpy(x)
        y_np = cls.to_adjacency_matrix(y) if y is not None else None

        if csv_name is not None:
            pd.DataFrame(x_np).to_csv(output_dir / csv_name, index=False)
        np.save(output_dir / "X.npy", x_np)
        if y_np is not None:
            np.save(output_dir / "adj.npy", y_np)
            np.save(output_dir / "DAG.npy", y_np)
        if nodes is not None:
            np.save(output_dir / "nodes.npy", np.asarray(nodes))
        bundle_path = output_dir / "data.npy"
        np.save(bundle_path, {"x": x_np, "y": y_np}, allow_pickle=True)
        return bundle_path


def read_observational_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def stage_dir(source_dir: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.is_file():
            copy_if_missing(item, raw_dir / item.name)


def stage_smbfo_raw(raw_root: Path) -> Path:
    raw_dir = raw_root / "SmBFO"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if LEGACY_SMBFO_ROOT.exists():
        for item in LEGACY_SMBFO_ROOT.iterdir():
            if item.is_file():
                copy_if_missing(item, raw_dir / item.name)

    missing = [name for name in SMBFO_FILES if not (raw_dir / name).exists()]
    for name in missing:
        url = f"{SMBFO_BASE_URL}/{name}?download=1"
        try:
            urlretrieve(url, raw_dir / name)
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Failed to download {name}: {exc}") from exc

    return raw_dir


def package_tabular_dataset(
    dataset_name: str,
    raw_root: Path,
    processed_root: Path,
    source_dir: Path,
    csv_name: str,
    adj_name: str,
    nodes_name: str | None = None,
) -> Path:
    raw_dir = raw_root / dataset_name
    stage_dir(source_dir, raw_dir)
    data = read_observational_csv(raw_dir / csv_name)
    adj = CausalDiscoveryPreprocessing.load_graph_file(raw_dir / adj_name)
    nodes = None
    if nodes_name is not None and (raw_dir / nodes_name).exists():
        nodes = np.load(raw_dir / nodes_name, allow_pickle=True)
    return CausalDiscoveryPreprocessing.save_bundle(
        processed_root / dataset_name,
        x=data,
        y=adj,
        nodes=nodes,
        csv_name=csv_name,
    )


def package_smBFO(raw_root: Path, processed_root: Path) -> Path:
    import h5py
    from sklearn.preprocessing import StandardScaler

    composition_tags = [0, 0, 0, 7, 7, 7, 7, 7, 10, 10, 13, 13, 20, 20]
    img_filenames = [
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
    uc_filenames = [
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

    raw_dir = stage_smbfo_raw(raw_root)
    uc_params = [h5py.File(raw_dir / name, "r") for name in uc_filenames]
    entries = []
    for i in range(len(img_filenames)):
        comp = composition_tags[i]
        fields = uc_params[i]
        physical = {
            "Alkali_Cations": fields["I1"][()].flatten(),
            "Transition_Metal_Cations": fields["I5"][()].flatten(),
            "Lattice_Parameter": fields["a"][()].flatten(),
            "Composition": comp,
            "Unit_Cell_Angle": fields["alpha"][()].flatten(),
            "Volume": fields["Vol"][()].flatten(),
            "In_Plane_Polarization": fields["Pxy"][0].flatten(),
        }
        for j in range(len(physical["Alkali_Cations"])):
            entries.append(
                {
                    "Alkali_Cations": physical["Alkali_Cations"][j],
                    "Transition_Metal_Cations": physical["Transition_Metal_Cations"][j],
                    "Lattice_Parameter": physical["Lattice_Parameter"][j],
                    "Composition": physical["Composition"],
                    "Unit_Cell_Angle": physical["Unit_Cell_Angle"][j],
                    "Volume": physical["Volume"][j],
                    "In_Plane_Polarization": physical["In_Plane_Polarization"][j],
                }
            )
    df = pd.DataFrame(entries).replace([np.inf, -np.inf], np.nan).dropna()
    x = StandardScaler().fit_transform(df)
    graph_path = CausalDiscoveryPreprocessing.search_ground_truth_graph(raw_dir)
    y = (
        CausalDiscoveryPreprocessing.load_graph_file(graph_path)
        if graph_path is not None
        else None
    )
    return CausalDiscoveryPreprocessing.save_bundle(
        processed_root / "SmBFO",
        x=x,
        y=y,
        csv_name="SmBFO.csv",
    )


def package_bias_family(raw_root: Path, processed_root: Path) -> list[Path]:
    raw_dir = raw_root / "BIAS"
    stage_dir(ONE_SHOT_DATA_ROOT / "BIAS" / "DATASET", raw_dir / "DATASET")
    stage_dir(ONE_SHOT_DATA_ROOT / "BIAS" / "ADJ_MATRICES", raw_dir / "ADJ_MATRICES")
    outputs: list[Path] = []
    variants = [
        "gaussian_i2e",
        "gaussian_n2e",
        "gaussian_n2i",
        "non-gaussian_i2e",
        "non-gaussian_n2e",
        "non-gaussian_n2i",
    ]
    for variant in variants:
        suffix = variant.split("_", 1)[1]
        csv_path = raw_dir / "DATASET" / f"{variant}.csv"
        adj_path = raw_dir / "ADJ_MATRICES" / f"adj_matrix_{suffix}.csv"
        if not csv_path.exists() or not adj_path.exists():
            continue
        data = read_observational_csv(csv_path)
        adj = CausalDiscoveryPreprocessing.load_graph_file(adj_path)
        outputs.append(
            CausalDiscoveryPreprocessing.save_bundle(
                processed_root / "BIAS" / variant,
                x=data,
                y=adj,
                csv_name=f"{variant}.csv",
            )
        )
    return outputs


def package_legal_family(raw_root: Path, processed_root: Path) -> list[Path]:
    raw_dir = raw_root / "LEGAL"
    stage_dir(ONE_SHOT_DATA_ROOT / "LEGAL" / "DATASET", raw_dir / "DATASET")
    stage_dir(ONE_SHOT_DATA_ROOT / "LEGAL" / "ADJ_MATRICES", raw_dir / "ADJ_MATRICES")
    outputs: list[Path] = []
    adj_path = raw_dir / "ADJ_MATRICES" / "adj_matrix_legal.csv"
    if not adj_path.exists():
        return outputs
    adj = CausalDiscoveryPreprocessing.load_graph_file(adj_path)
    for variant in ("gaussian_legal", "non-gaussian_legal"):
        csv_path = raw_dir / "DATASET" / f"{variant}.csv"
        if not csv_path.exists():
            continue
        data = read_observational_csv(csv_path)
        outputs.append(
            CausalDiscoveryPreprocessing.save_bundle(
                processed_root / "LEGAL" / variant,
                x=data,
                y=adj,
                csv_name=f"{variant}.csv",
            )
        )
    return outputs


def package_guide_synthetic(
    dataset_name: str,
    raw_root: Path,
    processed_root: Path,
    data_filename: str,
    adj_filename: str,
) -> Path:
    source_dir = GUIDE_DATA_ROOT / "SYNTHETIC"
    raw_dir = raw_root / dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    copy_if_missing(source_dir / data_filename, raw_dir / data_filename)
    copy_if_missing(source_dir / "Ground truth" / adj_filename, raw_dir / adj_filename)
    data = read_observational_csv(raw_dir / data_filename)
    adj = CausalDiscoveryPreprocessing.load_graph_file(raw_dir / adj_filename)
    return CausalDiscoveryPreprocessing.save_bundle(
        processed_root / dataset_name,
        x=data,
        y=adj,
        csv_name=data_filename,
    )


def run_dataset(dataset: str, raw_root: Path, processed_root: Path) -> list[Path]:
    single_sources = {
        "asia": (ONE_SHOT_DATA_ROOT / "asia", "asia.csv", "adj.npy", "nodes.npy"),
        "alarm": (ONE_SHOT_DATA_ROOT / "alarm", "alarm.csv", "adj.npy", "nodes.npy"),
        "Hepar2": (ONE_SHOT_DATA_ROOT / "Hepar2", "hepar2.csv", "hepar2adj.npy", None),
        "Lucas": (ONE_SHOT_DATA_ROOT / "Lucas", "lucas.csv", "adj_matrix_lucas.npy", None),
        "sachs": (ONE_SHOT_DATA_ROOT / "sachs", "sachs.csv", "adj.npy", "nodes.npy"),
        "dream41": (ONE_SHOT_DATA_ROOT / "dream41", "dream41.csv", "adj.npy", "nodes.npy"),
        "dream42": (ONE_SHOT_DATA_ROOT / "dream42", "dream42.csv", "adj.npy", "nodes.npy"),
        "dream43": (ONE_SHOT_DATA_ROOT / "dream43", "dream43.csv", "adj.npy", "nodes.npy"),
        "dream44": (ONE_SHOT_DATA_ROOT / "dream44", "dream44.csv", "adj.npy", "nodes.npy"),
        "lucas": (GUIDE_DATA_ROOT / "PUBLIC" / "Lucas", "lucas.csv", "adj_matrix_lucas.npy", None),
        "hepar2": (GUIDE_DATA_ROOT / "PUBLIC" / "Hepar2", "hepar2.csv", "hepar2adj.npy", None),
    }

    if dataset == "SmBFO":
        return [package_smBFO(raw_root, processed_root)]
    if dataset == "BIAS":
        return package_bias_family(raw_root, processed_root)
    if dataset == "LEGAL":
        return package_legal_family(raw_root, processed_root)
    if dataset == "gaussian_30":
        return [package_guide_synthetic(dataset, raw_root, processed_root, "gaussian_30nodes.csv", "adj_matrix_30nodes_linear.csv")]
    if dataset == "gaussian_50":
        return [package_guide_synthetic(dataset, raw_root, processed_root, "gaussian_50nodes.csv", "adj_matrix_50nodes_linear.csv")]
    if dataset == "non_gaussian_30":
        return [package_guide_synthetic(dataset, raw_root, processed_root, "non-gaussian_30nodes.csv", "adj_matrix_30nodes_linear.csv")]
    if dataset == "non_gaussian_50":
        return [package_guide_synthetic(dataset, raw_root, processed_root, "non-gaussian_50nodes.csv", "adj_matrix_50nodes_linear.csv")]
    if dataset in single_sources:
        source_dir, csv_name, adj_name, nodes_name = single_sources[dataset]
        return [package_tabular_dataset(dataset, raw_root, processed_root, source_dir, csv_name, adj_name, nodes_name)]
    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare causal discovery datasets.")
    parser.add_argument("--dataset", required=True, help="Dataset name to prepare.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_dataset(args.dataset, args.raw_root, args.processed_root)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
