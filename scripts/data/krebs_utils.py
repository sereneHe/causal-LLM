from __future__ import annotations

"""Krebs-cycle dataset staging and standardization utilities.

This module mirrors the source benchmark's standardization path:

* stage the Krebs_Cycle_*_TS time-series directory
* load the companion ``true_graph.npz``
* emit a standardized ``data.npy`` bundle for preprocessing
"""

from pathlib import Path
import shutil
import tarfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/xiaoyuhe/Causal-LLM")
SOURCE_KREBS_ROOT = Path("/Users/xiaoyuhe/Causal-Methods/krebcycle/data")

KREBS_VARIANT_ALIASES = {
    "krebs": "Krebs_Cycle_3",
    "krebst": "Krebs_Cycle_3",
    "krebs_cycle_1": "Krebs_Cycle_1",
    "krebs_cycle_3": "Krebs_Cycle_3",
    "krebs_cycle_normalised_1": "Krebs_Cycle_Normalised_1",
    "krebs_cycle_normalised_3": "Krebs_Cycle_Normalised_3",
}


def _save_bundle(
    processed_dir: Path,
    x,
    y,
    csv_name: str | None = None,
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    if y_np.ndim != 2 or y_np.shape[0] != y_np.shape[1]:
        raise ValueError(f"Expected a square adjacency matrix, got {y_np.shape}")
    y_np = (np.abs(y_np) > 1e-12).astype(int)
    np.fill_diagonal(y_np, 0)
    if csv_name is not None:
        pd.DataFrame(x_np.reshape(x_np.shape[0], -1)).to_csv(processed_dir / csv_name, index=False)
    np.save(processed_dir / "X.npy", x_np)
    np.save(processed_dir / "adj.npy", y_np)
    np.save(processed_dir / "DAG.npy", y_np)
    bundle_path = processed_dir / "data.npy"
    np.save(bundle_path, {"x": x_np, "y": y_np}, allow_pickle=True)
    return bundle_path


class Real_Data_Standardization:
    def __init__(self, file_path: str | Path, filename: str, measurements: int | None = None):
        self.file_path = Path(file_path)
        self.filename = filename
        self.measurements = measurements

    def standardize_data(self) -> tuple[np.ndarray, pd.DataFrame | np.ndarray]:
        """Standardize a Krebs time-series bundle exactly once.

        The raw time-series are loaded from a ``*_TS`` directory and paired
        with the benchmark truth graph.  The result is saved as a normalized
        ``npz`` bundle under the same benchmark-style naming convention.
        """
        raw_data, true_dag = self._produce_raw_data()
        result_dir = self.file_path
        result_dir.mkdir(parents=True, exist_ok=True)

        dag_array = true_dag.values if isinstance(true_dag, pd.DataFrame) else np.asarray(true_dag)
        data_name = f"{self.filename}_{dag_array.shape[0]}Nodes_{int(np.count_nonzero(dag_array))}Edges_TS.npz"
        np.savez(result_dir / data_name, x=np.asarray(raw_data), y=dag_array)
        return np.asarray(raw_data), true_dag

    def _produce_raw_data(self) -> tuple[np.ndarray, pd.DataFrame | np.ndarray]:
        direct_npz = self.file_path / f"{self.filename}.npz"
        if direct_npz.exists():
            data = np.load(direct_npz, allow_pickle=True)
            return np.asarray(data["x"]), np.asarray(data["y"])

        direct_csv = self.file_path / f"{self.filename}.csv"
        if direct_csv.exists():
            true_graph_csv = self.file_path / "true_graph.csv"
            if not true_graph_csv.exists():
                raise FileNotFoundError(f"Missing ground truth file: {true_graph_csv}")
            return pd.read_csv(direct_csv), pd.read_csv(true_graph_csv, index_col=0)

        tar_path = self.file_path / f"{self.filename}.tar.gz"
        if tar_path.exists():
            return self._load_tar_gz(tar_path)

        ts_dir = self.file_path / f"{self.filename}_TS"
        if ts_dir.exists():
            return self._load_time_series(ts_dir)

        legacy_ts_dir = self.file_path / self.filename
        if legacy_ts_dir.is_dir() and legacy_ts_dir.name.endswith("_TS"):
            return self._load_time_series(legacy_ts_dir)

        raise FileNotFoundError(
            f"Unable to find dataset '{self.filename}' under {self.file_path}. "
            "Expected .npz, .csv, .tar.gz, or a *_TS directory."
        )

    def _load_tar_gz(self, tar_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
        with tarfile.open(tar_path) as archive:
            archive.extractall(self.file_path)
            names = archive.getnames()
        npy_candidates = [self.file_path / name for name in names if name.endswith(".npy")]
        csv_candidates = [self.file_path / name for name in names if name.endswith(".csv")]
        if not npy_candidates or not csv_candidates:
            raise FileNotFoundError(f"{tar_path} did not contain the expected .npy/.csv files.")
        raw_data = np.load(npy_candidates[0], allow_pickle=True)
        true_dag = pd.read_csv(csv_candidates[0])
        return np.asarray(raw_data), true_dag

    def _load_time_series(self, ts_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
        series_files = sorted(
            [
                path
                for path in ts_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".tsv"
            ]
        )
        if self.measurements is not None:
            series_files = series_files[: int(self.measurements)]
        if not series_files:
            raise FileNotFoundError(f"No readable files found under {ts_dir}")

        true_graph = self._find_true_graph(ts_dir)
        true_dag = pd.DataFrame(np.load(true_graph, allow_pickle=True)["arr_0"])

        first = pd.read_csv(series_files[0], delimiter="\t", index_col=0, header=None)
        feature_names = np.array(first.index)
        feature_num = len(feature_names)
        sample_num = len(series_files)
        time_num = first.shape[1]
        raw_data = np.zeros((feature_num, sample_num, time_num))

        for sample_index, file_path in enumerate(series_files):
            sample_frame = pd.read_csv(file_path, delimiter="\t", index_col=0, header=None).T
            for feature_index, feature_name in enumerate(feature_names):
                raw_data[feature_index, sample_index, :] = sample_frame[feature_name].to_numpy()
        return raw_data, true_dag

    def _find_true_graph(self, ts_dir: Path) -> Path:
        candidates = [
            ts_dir.parent / "true_graph.npz",
            ts_dir / "true_graph.npz",
            self.file_path / "true_graph.npz",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Missing true_graph.npz near {ts_dir}")


def _stage_variant(raw_variant_dir: Path, source_variant_dir: Path) -> None:
    raw_variant_dir.mkdir(parents=True, exist_ok=True)
    if not source_variant_dir.exists():
        raise FileNotFoundError(f"Missing source Krebs directory: {source_variant_dir}")
    target_ts_dir = raw_variant_dir / f"{source_variant_dir.name}"
    target_ts_dir.mkdir(parents=True, exist_ok=True)
    for item in source_variant_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, target_ts_dir / item.name)
    true_graph = source_variant_dir.parent / "true_graph.npz"
    if true_graph.exists():
        shutil.copy2(true_graph, raw_variant_dir / "true_graph.npz")


def build_krebs_bundle(raw_root: Path, processed_root: Path, cfg) -> Path:
    output_name = cfg.get("output_name", "krebs")
    variant = str(cfg.get("variant", "krebs")).strip()
    source_variant = KREBS_VARIANT_ALIASES.get(variant.lower(), variant)
    source_variant_dir = SOURCE_KREBS_ROOT / f"{source_variant}_TS"
    raw_variant_dir = raw_root / output_name / source_variant
    _stage_variant(raw_variant_dir, source_variant_dir)

    measurements = cfg.get("measurements")
    standardizer = Real_Data_Standardization(
        raw_variant_dir,
        source_variant,
        measurements=None if measurements is None else int(measurements),
    )
    raw_data, true_dag = standardizer.standardize_data()
    dag_array = true_dag.values if isinstance(true_dag, pd.DataFrame) else np.asarray(true_dag)
    return _save_bundle(
        processed_root / output_name,
        x=raw_data,
        y=dag_array,
        csv_name=f"{source_variant}.csv",
    )
