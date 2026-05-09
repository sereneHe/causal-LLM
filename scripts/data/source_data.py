"""Local copies of source-repo dataset loaders and synthetic generators.

This module hosts the data-generation / loading logic that was previously
spread across the source repository.  The goal is to keep
``scripts/data/preprocessing.py`` thin: it should decide *which* dataset to
prepare, while this module knows *how* to stage or synthesize the raw inputs.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pgmpy.readwrite import NETReader


PROJECT_ROOT = Path("/Users/xiaoyuhe/Causal-LLM")
SOURCE_REPO_ROOT = Path("/Users/xiaoyuhe/project-bestdagsolverintheworld")
BASELINE_ROOT = PROJECT_ROOT / "scripts" / "baseline"

SOURCE_ADMISSIONS_DIR = SOURCE_REPO_ROOT / "datasets" / "admissions"
SOURCE_CDS_DIR = SOURCE_REPO_ROOT / "datasets" / "CDS_Data"
SOURCE_CODIET_DIR = SOURCE_REPO_ROOT / "codiet_data"
SOURCE_KREBS_DIR = SOURCE_REPO_ROOT / "datasets" / "krebs"
SOURCE_MINOBSX_DATA_DIR = BASELINE_ROOT / "MINOBSx" / "minobsxx" / "data"


def copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def stage_dir(source_dir: Path, raw_dir: Path) -> None:
    if not source_dir.exists():
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.is_file():
            copy_if_missing(item, raw_dir / item.name)


def _stage_bn_benchmark_assets(raw_dir: Path, dataset_name: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    net_src = SOURCE_MINOBSX_DATA_DIR / f"{dataset_name}.net"
    mapping_src = SOURCE_MINOBSX_DATA_DIR / "mappings" / f"{dataset_name}.mapping"
    parsed_src = SOURCE_MINOBSX_DATA_DIR / f"{dataset_name}_results_parsed"
    for src in (net_src, mapping_src, parsed_src):
        if src.exists():
            copy_if_missing(src, raw_dir / src.name)


def _read_bn_nodes(raw_dir: Path, dataset_name: str, model) -> list[str]:
    mapping_path = raw_dir / f"{dataset_name}.mapping"
    if mapping_path.exists():
        nodes = [line.strip() for line in mapping_path.read_text().splitlines() if line.strip()]
        if nodes:
            return nodes
    return list(model.nodes())


def _encode_bn_dataframe(df: pd.DataFrame, nodes: list[str], state_names: dict[str, list[str]]) -> np.ndarray:
    encoded_columns = []
    for node in nodes:
        series = df[node]
        categories = state_names.get(node)
        if categories:
            encoded = pd.Categorical(series, categories=categories, ordered=True).codes
        else:
            encoded = pd.factorize(series, sort=False)[0]
        if (encoded < 0).any():
            missing = sorted(set(series.astype(str)) - set(map(str, categories or [])))
            raise ValueError(f"Could not encode BN states for {node}: {missing[:5]}")
        encoded_columns.append(encoded.astype(int))
    return np.column_stack(encoded_columns)


def _sample_bn_from_cpds(model, nodes: list[str], n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    node_codes: dict[str, np.ndarray] = {}
    node_frames: dict[str, pd.Categorical] = {}
    state_names: dict[str, list[str]] = {}
    for cpd in model.get_cpds():
        if cpd.variable in nodes and cpd.variable in cpd.state_names:
            state_names[cpd.variable] = list(cpd.state_names[cpd.variable])

    topo_order = list(nx.topological_sort(nx.DiGraph(model.edges())))
    for node in topo_order:
        cpd = model.get_cpds(node)
        if cpd is None:
            raise ValueError(f"Missing CPD for node {node}")
        node_states = state_names.get(node)
        if not node_states:
            node_states = [str(idx) for idx in range(cpd.get_values().shape[0])]
            state_names[node] = node_states

        values = np.asarray(cpd.get_values(), dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        evidence = list(cpd.get_evidence() or [])
        if evidence:
            evidence_cards = [len(state_names[parent]) for parent in evidence]
            parent_codes = [node_codes[parent] for parent in evidence]
            col_index = np.zeros(n_samples, dtype=int)
            stride = 1
            for parent_codes_arr, card in zip(reversed(parent_codes), reversed(evidence_cards)):
                col_index += parent_codes_arr * stride
                stride *= card
            probs = values[:, col_index]
        else:
            probs = values[:, np.zeros(n_samples, dtype=int)]

        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 0.0, 1.0)
        col_sums = probs.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        probs = probs / col_sums
        cumprobs = np.cumsum(probs, axis=0)
        draws = rng.random(n_samples)
        samples = (cumprobs < draws).sum(axis=0)
        node_codes[node] = samples.astype(int)

        node_frames[node] = pd.Categorical.from_codes(node_codes[node], categories=node_states, ordered=True)

    sampled_df = pd.DataFrame({node: node_frames[node] for node in nodes})
    return sampled_df


def build_bn_benchmark_bundle(raw_root: Path, processed_root: Path, dataset_name: str, n_samples: int = 1000, seed: int = 0) -> Path:
    raw_dir = raw_root / dataset_name
    _stage_bn_benchmark_assets(raw_dir, dataset_name)

    bundle_path = processed_root / dataset_name / "data.npy"
    if bundle_path.exists():
        return bundle_path

    net_path = raw_dir / f"{dataset_name}.net"
    if not net_path.exists():
        raise FileNotFoundError(f"Missing BN benchmark net file: {net_path}")

    reader = NETReader(str(net_path))
    model = reader.get_model()
    nodes = _read_bn_nodes(raw_dir, dataset_name, model)

    try:
        model.check_model()
    except Exception:
        pass

    sampled = _sample_bn_from_cpds(model, nodes, int(n_samples), int(seed))
    state_names: dict[str, list[str]] = {}
    for cpd in model.get_cpds():
        if cpd.variable in nodes and cpd.variable in cpd.state_names:
            state_names[cpd.variable] = list(cpd.state_names[cpd.variable])
    x = _encode_bn_dataframe(sampled, nodes, state_names)
    adj = nx.to_numpy_array(nx.DiGraph(model.edges()), nodelist=nodes, dtype=int).astype(int)
    np.save(raw_dir / "adj.npy", adj)
    np.save(raw_dir / "X.npy", x)
    np.save(raw_dir / "nodes.npy", np.asarray(nodes))
    return save_bundle(processed_root / dataset_name, x=x, y=adj, nodes=nodes, csv_name=f"{dataset_name}.csv")


def save_bundle(
    processed_dir: Path,
    x,
    y=None,
    nodes=None,
    csv_name: str | None = None,
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    x_np = np.asarray(x)
    y_np = None if y is None else _to_adjacency_matrix(y)

    if csv_name is not None:
        pd.DataFrame(x_np).to_csv(processed_dir / csv_name, index=False)
    np.save(processed_dir / "X.npy", x_np)
    if y_np is not None:
        np.save(processed_dir / "adj.npy", y_np)
        np.save(processed_dir / "DAG.npy", y_np)
    if nodes is not None:
        np.save(processed_dir / "nodes.npy", np.asarray(nodes))
    bundle_path = processed_dir / "data.npy"
    np.save(bundle_path, {"x": x_np, "y": y_np}, allow_pickle=True)
    return bundle_path


def _to_adjacency_matrix(graph_like) -> np.ndarray:
    arr = np.asarray(graph_like)
    if arr.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {arr.shape}")
    arr = (np.abs(arr) > 1e-12).astype(int)
    np.fill_diagonal(arr, 0)
    return arr


def read_csv_table(path: Path, sep: str = ",") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _simple_dag(num_nodes: int, degree: float, graph_type: str = "erdos-renyi") -> np.ndarray:
    if num_nodes < 2:
        raise ValueError("DAG must have at least 2 nodes")
    if graph_type == "erdos-renyi":
        p_edge = min(max(degree / max(num_nodes - 1, 1), 0.0), 1.0)
        lower = (np.random.rand(num_nodes, num_nodes) < p_edge).astype(float)
        lower = np.tril(lower, k=-1)
    elif graph_type == "barabasi-albert":
        m = max(1, int(round(degree / 2)))
        lower = np.zeros((num_nodes, num_nodes))
        bag = [0]
        for i in range(1, num_nodes):
            dest = np.random.choice(bag, size=min(m, len(bag)), replace=False)
            for j in np.atleast_1d(dest):
                lower[i, j] = 1
            bag.append(i)
            bag.extend(np.atleast_1d(dest).tolist())
    elif graph_type == "full":
        lower = np.tril(np.ones((num_nodes, num_nodes)), k=-1)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    perm = np.random.permutation(np.eye(num_nodes))
    dag = perm.T @ lower @ perm
    return (dag != 0).astype(int)


def simulate_dag(d: int, s0: float, graph_type: str) -> np.ndarray:
    if graph_type == "SF":
        graph_type = "barabasi-albert"
    elif graph_type == "ER":
        graph_type = "erdos-renyi"
    return _simple_dag(d, s0 * 2 / d if d else 0, graph_type)


def simulate_parameter(B_true: np.ndarray, w_ranges=((0.5, 2.0),)) -> np.ndarray:
    W = np.zeros_like(B_true, dtype=float)
    low, high = w_ranges[0]
    mask = B_true != 0
    W[mask] = np.random.uniform(low, high, size=int(mask.sum()))
    signs = np.random.choice([-1.0, 1.0], size=int(mask.sum()))
    W[mask] *= signs
    return W


def simulate_linear_sem(
    W_true: np.ndarray,
    n: int,
    sem_type: str,
    noise_scale=1.0,
    internal_normalization: bool = False,
) -> np.ndarray:
    del internal_normalization
    sem_type = sem_type.lower()
    if isinstance(noise_scale, (int, float)):
        noise_scale = [float(noise_scale)] * W_true.shape[0]
    G = nx.DiGraph(W_true)
    order = list(nx.topological_sort(G))
    X = np.zeros((n, W_true.shape[0]))
    for j in order:
        parents = list(G.predecessors(j))
        mean = X[:, parents] @ W_true[parents, j] if parents else 0.0
        if sem_type in {"gauss", "gaussian", "normal"}:
            eps = np.random.normal(scale=noise_scale[j], size=n)
        elif sem_type in {"exp", "exponential"}:
            eps = np.random.exponential(scale=noise_scale[j], size=n)
        elif sem_type in {"gumbel"}:
            eps = np.random.gumbel(scale=noise_scale[j], size=n)
        else:
            raise ValueError(f"Unsupported sem_type: {sem_type}")
        X[:, j] = mean + eps
    return X


def _read_graph_edges(graph_file: Path) -> set[tuple[str, str]]:
    with open(graph_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return {tuple(line.strip().split()) for line in lines if line.strip()}


def _graph_to_adj(vertices: list[str], edges: set[tuple[str, str]], p: int = 0):
    A = np.zeros((len(vertices), len(vertices)))
    if p <= 0:
        for row_idx, row in enumerate(vertices):
            for col_idx, col in enumerate(vertices):
                if (f"{row}_lag0", f"{col}_lag0") in edges:
                    A[row_idx, col_idx] = 1
        return A

    A_inter = [np.zeros((len(vertices), len(vertices))) for _ in range(p)]
    for row_idx, row in enumerate(vertices):
        for col_idx, col in enumerate(vertices):
            if (f"{row}_lag0", f"{col}_lag0") in edges:
                A[row_idx, col_idx] = 1
            for lag in range(p):
                if (f"{row}_lag{lag+1}", f"{col}_lag0") in edges:
                    A_inter[lag][row_idx, col_idx] = 1
    return A, A_inter


def build_admissions_bundle(raw_root: Path, processed_root: Path) -> Path:
    raw_dir = raw_root / "admissions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("UCBadmit.csv", "UCBadmit_long_samples.csv", "ExMAG_Berkeley_Admission_Example.ipynb"):
        src = SOURCE_ADMISSIONS_DIR / name
        if src.exists():
            copy_if_missing(src, raw_dir / name)

    csv_path = raw_dir / "UCBadmit_long_samples.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing admissions raw file: {csv_path}. "
            "Copy it from the source repo or stage the raw files first."
        )
    df = pd.read_csv(csv_path, sep=";")
    if "department" in df.columns:
        df = pd.get_dummies(df, columns=["department"], prefix="D", dtype=int)
    for col in ("gender", "admit"):
        if col in df.columns:
            df[col] = pd.factorize(df[col])[0]
    x = df.to_numpy()
    y = np.zeros((x.shape[1], x.shape[1]), dtype=int)
    return save_bundle(processed_root / "admissions", x=x, y=y, csv_name="UCBadmit_long_samples.csv")


def build_bowfree_admg_bundle(raw_root: Path, processed_root: Path, cfg) -> Path:
    raw_dir = raw_root / "bowfree_admg"
    raw_dir.mkdir(parents=True, exist_ok=True)
    n = int(cfg.get("number_of_variables", 10))
    pdir = float(cfg.get("pdir", 0.4))
    pbidir = float(cfg.get("pbidir", 0.3))
    max_in_arrows = cfg.get("max_in_arrows", None)
    if max_in_arrows is not None:
        max_in_arrows = int(max_in_arrows)
    n_samples = int(cfg.get("number_of_samples", 100))
    seed = int(cfg.get("seed", 1))
    enforce_mag = bool(cfg.get("enforce_mag", True))
    np.random.seed(seed)
    random.seed(seed)

    directed, bidirected = _generate_bowfree_graph(n, pdir, pbidir, max_in_arrows)
    x = _sample_admg_sem(directed, bidirected, n_samples)
    adj = directed
    np.save(raw_dir / "adj.npy", adj)
    np.save(raw_dir / "X.npy", x)
    return save_bundle(processed_root / "bowfree_admg", x=x, y=adj, csv_name="bowfree_admg.csv")


def _generate_bowfree_graph(nnodes, pdir, pbidir, max_in_arrows=None):
    D = np.zeros((nnodes, nnodes), dtype=int)
    B = np.zeros((nnodes, nnodes), dtype=int)
    for i in range(nnodes - 1):
        for j in range(i + 1, nnodes):
            r = random.random()
            if r < pdir:
                D[i, j] = 1
            elif r < pdir + pbidir:
                B[i, j] = 1
                B[j, i] = 1
    if max_in_arrows is not None:
        D, B = _cap_incoming_edges(D, B, max_in_arrows)
    return D, B


def _cap_incoming_edges(D, B, max_in_arrows):
    nnodes = len(D)
    for j in range(nnodes):
        parents = [i for i in range(nnodes) if D[i][j] == 1 or B[i][j] == 1]
        while len(parents) > max_in_arrows:
            idx = random.randrange(len(parents))
            i = parents.pop(idx)
            D[i][j] = 0
            B[i][j] = 0
            B[j][i] = 0
    return D, B


def _sample_admg_sem(D, B, n):
    G = nx.DiGraph(D)
    order = list(nx.topological_sort(G))
    d = len(D)
    X = np.zeros((n, d))
    noise = np.random.multivariate_normal(np.zeros(d), _admg_covariance(B), size=n)
    weights = np.random.uniform(0.5, 2.0, size=(d, d))
    weights[np.random.rand(d, d) < 0.5] *= -1
    weights *= D
    for j in order:
        parents = list(G.predecessors(j))
        mean = X[:, parents] @ weights[parents, j] if parents else 0.0
        X[:, j] = mean + noise[:, j]
    return X


def _admg_covariance(B):
    cov = np.eye(B.shape[0])
    cov += 0.2 * (B + B.T)
    np.fill_diagonal(cov, 1.0)
    return cov + 1e-6 * np.eye(B.shape[0])


def build_er_bundle(raw_root: Path, processed_root: Path, cfg, dataset_name: str = "er") -> Path:
    raw_dir = raw_root / dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    d = int(cfg.get("number_of_variables", 7))
    edge_ratio = float(cfg.get("edge_ratio", 2))
    n = int(cfg.get("number_of_samples", 100))
    sem_type = cfg.get("sem_type", "gauss")
    seed = int(cfg.get("seed", 1))
    noise_scale = cfg.get("noise_scale", 1.0)
    internal_normalization = bool(cfg.get("internal_normalization", False))
    np.random.seed(seed)
    random.seed(seed)

    B_true = simulate_dag(d, edge_ratio * d, "ER")
    W_true = simulate_parameter(B_true)
    x = simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale, internal_normalization=internal_normalization)
    np.save(raw_dir / "adj.npy", B_true)
    np.save(raw_dir / "X.npy", x)
    return save_bundle(processed_root / dataset_name, x=x, y=B_true, csv_name=f"{dataset_name}.csv")


def build_sf_bundle(raw_root: Path, processed_root: Path, cfg, dataset_name: str = "sf") -> Path:
    raw_dir = raw_root / dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    d = int(cfg.get("number_of_variables", 7))
    edge_ratio = float(cfg.get("edge_ratio", 3))
    n = int(cfg.get("number_of_samples", 100))
    sem_type = cfg.get("sem_type", "gauss")
    seed = int(cfg.get("seed", 3))
    noise_scale = cfg.get("noise_scale", 1.0)
    internal_normalization = bool(cfg.get("internal_normalization", False))
    np.random.seed(seed)
    random.seed(seed)

    B_true = simulate_dag(d, edge_ratio * d, "SF")
    W_true = simulate_parameter(B_true)
    x = simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale, internal_normalization=internal_normalization)
    np.save(raw_dir / "adj.npy", B_true)
    np.save(raw_dir / "X.npy", x)
    return save_bundle(processed_root / dataset_name, x=x, y=B_true, csv_name=f"{dataset_name}.csv")


def build_ermag_bundle(raw_root: Path, processed_root: Path, cfg) -> Path:
    raw_dir = raw_root / "ermag"
    raw_dir.mkdir(parents=True, exist_ok=True)
    d = int(cfg.get("number_of_variables", 7))
    edge_ratio = float(cfg.get("edge_ratio", 2))
    n = int(cfg.get("number_of_samples", 100))
    sem_type = cfg.get("sem_type", "gauss")
    seed = int(cfg.get("seed", 1))
    hidden_vertices_ratio = float(cfg.get("hidden_vertices_ratio", 0.2))
    np.random.seed(seed)
    random.seed(seed)

    B_true = simulate_dag(d, edge_ratio * d, "ER")
    W_true = simulate_parameter(B_true)
    x = simulate_linear_sem(W_true, n, sem_type, noise_scale=cfg.get("noise_scale", 1.0))
    keep = np.random.choice(range(d), size=max(1, int(d * (1 - hidden_vertices_ratio))), replace=False)
    keep = np.sort(keep)
    x = x[:, keep]
    B_true = B_true[np.ix_(keep, keep)]
    np.save(raw_dir / "adj.npy", B_true)
    np.save(raw_dir / "X.npy", x)
    return save_bundle(processed_root / "ermag", x=x, y=B_true, csv_name="ermag.csv")


def build_dynamic_bundle(raw_root: Path, processed_root: Path, cfg, dataset_name: str = "dynamic") -> Path:
    raw_dir = raw_root / dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    num_nodes = int(cfg.get("number_of_variables", 7))
    n_samples = int(cfg.get("number_of_samples", 100))
    p = int(cfg.get("p", 1))
    degree_intra = float(cfg.get("intra_edge_ratio", 2)) * 2
    degree_inter = float(cfg.get("inter_edge_ratio", 1)) * 2
    graph_type_intra = cfg.get("graph_type_intra", "er")
    graph_type_inter = cfg.get("graph_type_inter", "er")
    w_min_intra = float(cfg.get("w_min_intra", 0.5))
    w_max_intra = float(cfg.get("w_max_intra", 2.0))
    w_min_inter = float(cfg.get("w_min_inter", 0.5))
    w_max_inter = float(cfg.get("w_max_inter", 0.7))
    w_decay = float(cfg.get("w_decay", 1.1))
    sem_type = cfg.get("sem_type", "linear-gauss")
    noise_scale = cfg.get("noise_scale", 1.0)
    noise_scale_variance = cfg.get("noise_scale_variance", None)
    seed = int(cfg.get("seed", 1))
    np.random.seed(seed)
    random.seed(seed)

    if noise_scale_variance is not None:
        scale = [
            random.uniform(noise_scale - noise_scale_variance, noise_scale + noise_scale_variance)
            for _ in range(num_nodes)
        ]
    else:
        scale = [noise_scale] * num_nodes

    g, df, intra_nodes, inter_nodes = _generate_stationary_dyn_net_and_df(
        num_nodes=num_nodes,
        n_samples=n_samples,
        p=p,
        degree_intra=degree_intra,
        degree_inter=degree_inter,
        graph_type_intra=graph_type_intra,
        graph_type_inter=graph_type_inter,
        w_min_intra=w_min_intra,
        w_max_intra=w_max_intra,
        w_min_inter=w_min_inter,
        w_max_inter=w_max_inter,
        w_decay=w_decay,
        sem_type=sem_type,
        noise_scale=scale,
    )
    intra_nodes = sorted(intra_nodes)
    inter_nodes = sorted(inter_nodes)
    w_true = nx.to_numpy_array(g, nodelist=intra_nodes)
    b_true = (w_true != 0).astype(int)
    x = df[intra_nodes].to_numpy()
    y = []
    a_true = []
    b_lags_true = []
    a_mat = nx.to_numpy_array(g, nodelist=intra_nodes + inter_nodes)[len(intra_nodes):, : len(intra_nodes)]
    for lag in range(1, p + 1):
        lag_cols = [c for c in inter_nodes if f"_lag{lag}" in c]
        y.append(df[lag_cols].to_numpy())
        idxs = [f"_lag{lag}" in c for c in inter_nodes]
        a_lag = a_mat[idxs, :]
        a_true.append(a_lag)
        b_lags_true.append((a_lag != 0).astype(int))
    np.save(raw_dir / "adj.npy", b_true)
    np.save(raw_dir / "X.npy", x)
    return save_bundle(processed_root / dataset_name, x=x, y=b_true, csv_name=f"{dataset_name}.csv")


def _generate_structure_dynamic(
    num_nodes: int,
    p: int,
    degree_intra: float,
    degree_inter: float,
    graph_type_intra: str = "erdos-renyi",
    graph_type_inter: str = "erdos-renyi",
    w_min_intra: float = 0.5,
    w_max_intra: float = 0.5,
    w_min_inter: float = 0.5,
    w_max_inter: float = 0.5,
    w_decay: float = 1.0,
) -> nx.DiGraph:
    graph_type_intra = "barabasi-albert" if graph_type_intra == "sf" else "erdos-renyi"
    graph_type_inter = "erdos-renyi" if graph_type_inter == "er" else graph_type_inter
    intra = _simple_dag(num_nodes, degree_intra, graph_type_intra)
    inter = np.zeros((p * num_nodes, num_nodes))
    for lag in range(p):
        prob = min(max(degree_inter / max(num_nodes, 1) / (w_decay ** lag), 0.0), 1.0)
        block = (np.random.rand(num_nodes, num_nodes) < prob).astype(float)
        inter[lag * num_nodes : (lag + 1) * num_nodes, :] = block
    g = nx.DiGraph()
    for i in range(num_nodes):
        g.add_node(f"{i}_lag0")
    for lag in range(1, p + 1):
        for i in range(num_nodes):
            g.add_node(f"{i}_lag{lag}")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if intra[i, j]:
                g.add_edge(f"{i}_lag0", f"{j}_lag0", weight=float(np.random.uniform(w_min_intra, w_max_intra)))
    for lag in range(1, p + 1):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if inter[(lag - 1) * num_nodes + i, j]:
                    w = float(np.random.uniform(w_min_inter, w_max_inter) / (w_decay ** (lag - 1)))
                    g.add_edge(f"{i}_lag{lag}", f"{j}_lag0", weight=w)
    return g


def _generate_dataframe_dynamic(
    g: nx.DiGraph,
    n_samples: int = 1000,
    burn_in: int = 100,
    sem_type: str = "linear-gauss",
    noise_scale: Iterable[float] | None = None,
    drift: np.ndarray | None = None,
) -> pd.DataFrame:
    intra_nodes = sorted(el for el in g.nodes if "_lag0" in el)
    inter_nodes = sorted(el for el in g.nodes if "_lag0" not in el)
    w_mat = nx.to_numpy_array(g, nodelist=intra_nodes)
    a_mat = nx.to_numpy_array(g, nodelist=intra_nodes + inter_nodes)[len(intra_nodes):, : len(intra_nodes)]
    d = w_mat.shape[0]
    p = 0 if d == 0 else a_mat.shape[0] // d
    if noise_scale is None:
        noise_scale = [1.0] * d
    noise_scale = list(noise_scale)
    total_length = n_samples + burn_in
    X = np.zeros((total_length, d))
    Xlags = np.zeros((total_length, p * d))
    order = list(nx.topological_sort(nx.DiGraph(w_mat)))
    if drift is None:
        drift = np.zeros(d)
    for t in range(total_length):
        for j in order:
            parents = np.where(w_mat[:, j] != 0)[0].tolist()
            parents_prev = np.where(a_mat[:, j] != 0)[0].tolist()
            mean = drift[j]
            if parents:
                mean += X[t, parents] @ w_mat[parents, j]
            if parents_prev:
                mean += Xlags[t, parents_prev] @ a_mat[parents_prev, j]
            if sem_type in {"linear-gauss", "gauss", "gaussian"}:
                X[t, j] = mean + np.random.normal(scale=noise_scale[j])
            elif sem_type == "linear-exp":
                X[t, j] = mean + np.random.exponential(scale=noise_scale[j])
            elif sem_type == "linear-gumbel":
                X[t, j] = mean + np.random.gumbel(scale=noise_scale[j])
            else:
                raise ValueError(f"unknown sem type {sem_type}")
        if (t + 1) < total_length:
            Xlags[t + 1, :] = np.concatenate([X[t, :], Xlags[t, :]])[: d * p]
    return pd.concat(
        [
            pd.DataFrame(X[-n_samples:], columns=intra_nodes),
            pd.DataFrame(Xlags[-n_samples:], columns=inter_nodes),
        ],
        axis=1,
    )


def _generate_stationary_dyn_net_and_df(
    num_nodes: int = 10,
    n_samples: int = 100,
    p: int = 1,
    degree_intra: float = 3,
    degree_inter: float = 3,
    graph_type_intra: str = "erdos-renyi",
    graph_type_inter: str = "erdos-renyi",
    w_min_intra: float = 0.5,
    w_max_intra: float = 0.5,
    w_min_inter: float = 0.5,
    w_max_inter: float = 0.5,
    w_decay: float = 1.0,
    sem_type: str = "linear-gauss",
    noise_scale: Iterable[float] | None = None,
) -> tuple[nx.DiGraph, pd.DataFrame, list[str], list[str]]:
    g = _generate_structure_dynamic(
        num_nodes=num_nodes,
        p=p,
        degree_intra=degree_intra,
        degree_inter=degree_inter,
        graph_type_intra=graph_type_intra,
        graph_type_inter=graph_type_inter,
        w_min_intra=w_min_intra,
        w_max_intra=w_max_intra,
        w_min_inter=w_min_inter,
        w_max_inter=w_max_inter,
        w_decay=w_decay,
    )
    df = _generate_dataframe_dynamic(
        g,
        n_samples=n_samples,
        sem_type=sem_type,
        noise_scale=noise_scale,
    )
    intra_nodes = [el for el in g.nodes if "_lag0" in el]
    inter_nodes = [el for el in g.nodes if "_lag0" not in el]
    return g, df, intra_nodes, inter_nodes


def build_cds_bundle(raw_root: Path, processed_root: Path, cfg) -> Path:
    raw_dir = raw_root / "CDS_Data"
    stage_dir(SOURCE_CDS_DIR, raw_dir)
    if not raw_dir.exists() or not any(raw_dir.glob("*.csv")):
        raise FileNotFoundError(
            f"Missing CDS raw data under {raw_dir} and source repo CDS_Data is unavailable."
        )

    n = int(cfg.get("n", 1000))
    p = int(cfg.get("p", 0))
    granularity = int(cfg.get("granularity", 1))
    data_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".csv")])
    frames = [pd.read_csv(raw_dir / file, index_col=0) for file in data_files]
    data_names = [file.split("_")[3] if len(file.split("_")) > 3 else Path(file).stem for file in data_files]
    result_df = None
    for i, name in enumerate(data_names):
        df = frames[i].iloc[: (n + p) * granularity : granularity, [0]].copy()
        df.rename(columns={df.columns[0]: name}, inplace=True)
        result_df = df if result_df is None else pd.merge(result_df, df, left_index=True, right_index=True, how="inner")

    x = result_df.iloc[:n, :].to_numpy(copy=True) * 1000
    y = np.zeros((len(data_names), len(data_names)), dtype=int)
    np.save(raw_dir / "adj.npy", y)
    np.save(raw_dir / "X.npy", x)
    return save_bundle(processed_root / "cds", x=x, y=y, csv_name="cds.csv")


def build_codiet_bundle(
    raw_root: Path,
    processed_root: Path,
    cfg,
    output_name: str = "codiet",
) -> Path:
    from scripts.data.codiet_utils import build_codiet_bundle as _build_codiet_bundle

    return _build_codiet_bundle(raw_root, processed_root, cfg, output_name=output_name)


def build_krebs_bundle(raw_root: Path, processed_root: Path, cfg) -> Path:
    from scripts.data.krebs_utils import build_krebs_bundle as _build_krebs_bundle

    return _build_krebs_bundle(raw_root, processed_root, cfg)


def build_dynamic_er_bundle(raw_root: Path, processed_root: Path, cfg, dataset_name: str = "dynamic-er") -> Path:
    return build_dynamic_bundle(raw_root, processed_root, cfg, dataset_name=dataset_name)
