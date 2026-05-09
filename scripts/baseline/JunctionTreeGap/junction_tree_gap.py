from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as torch_nn
import torch.optim as torch_optim


EPS = 1e-8


def to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


@dataclass
class JunctionTreeGapConfig:
    top_k: int = 3
    gap_weight: float = 0.5
    sparsity_threshold: float = 0.0
    edge_threshold: float = 0.0
    covariance_jitter: float = 1e-6
    gap_metric: str = "sym_kl"
    clique_potential_model: str = "gaussian"
    separator_model: str = "gaussian"
    score_bins: int = 16
    hidden_dim: int = 16
    num_heads: int = 2
    num_layers: int = 2


class ResidualMLPBlock(torch_nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = torch_nn.LayerNorm(hidden_dim)
        self.fc1 = torch_nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = torch_nn.GELU()
        self.fc2 = torch_nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = torch_nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + self.dropout(x)


class DeepResidualMLPEncoder(torch_nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = torch_nn.Sequential(
            torch_nn.Linear(input_dim, hidden_dim),
            torch_nn.LayerNorm(hidden_dim),
            torch_nn.GELU(),
        )
        self.blocks = torch_nn.ModuleList([ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(max(2, depth))])
        self.head = torch_nn.Sequential(
            torch_nn.LayerNorm(hidden_dim),
            torch_nn.Linear(hidden_dim, hidden_dim),
            torch_nn.GELU(),
            torch_nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class TokenTransformerEncoder(torch_nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_proj = torch_nn.Linear(1, self.hidden_dim)
        self.cls_token = torch_nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.feature_pos = torch_nn.Parameter(torch.zeros(1, max(self.input_dim, 1), self.hidden_dim))
        encoder_layer = torch_nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=max(1, min(int(num_heads), self.hidden_dim)),
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = torch_nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.head = torch_nn.Sequential(
            torch_nn.LayerNorm(self.hidden_dim),
            torch_nn.Linear(self.hidden_dim, self.hidden_dim),
            torch_nn.GELU(),
            torch_nn.Linear(self.hidden_dim, 1),
        )
        torch_nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        torch_nn.init.normal_(self.feature_pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.feature_proj(x)
        if x.shape[1] <= self.feature_pos.shape[1]:
            x = x + self.feature_pos[:, : x.shape[1], :]
        else:
            pos = self.feature_pos
            repeat = int(np.ceil(x.shape[1] / pos.shape[1]))
            x = x + pos.repeat(1, repeat, 1)[:, : x.shape[1], :]
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        seq = torch.cat([cls, x], dim=1)
        seq = self.encoder(seq)
        return self.head(seq[:, 0, :])


def standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return (X - mean) / std


def model_seed(indices: tuple[int, ...], salt: str) -> int:
    seed = 1469598103934665603
    for value in indices:
        seed ^= int(value) + 0x9E3779B97F4A7C15
        seed *= 1099511628211
        seed &= (1 << 64) - 1
    for char in salt:
        seed ^= ord(char)
        seed *= 1099511628211
        seed &= (1 << 64) - 1
    return int(seed % (2**32 - 1))


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= EPS:
        return np.full(values.shape, 1.0 / max(values.size, 1), dtype=float)
    return values / total


def histogram_distribution(scores: np.ndarray, bins: int = 16) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size == 0:
        return np.array([1.0], dtype=float)
    if np.allclose(scores.min(), scores.max()):
        return np.array([1.0], dtype=float)
    counts, _ = np.histogram(scores, bins=bins)
    counts = counts.astype(float) + 1e-6
    return normalize_probabilities(counts)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return x - np.log(np.sum(exp_x, axis=-1, keepdims=True) + EPS)


def one_hot_bins(X: np.ndarray, bins: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    edges = []
    encoded = []
    for col in range(X.shape[1]):
        if np.allclose(X[:, col].min(), X[:, col].max()):
            idx = np.zeros(X.shape[0], dtype=int)
        else:
            quantiles = np.linspace(0, 1, bins + 1)[1:-1]
            thresholds = np.unique(np.quantile(X[:, col], quantiles))
            idx = np.digitize(X[:, col], thresholds, right=False)
        edges.append(idx)
        encoded.append(idx)
    return np.vstack(encoded).T


def joint_bin_scores(X: np.ndarray, bins: int = 4) -> np.ndarray:
    encoded = one_hot_bins(X, bins)
    if encoded.ndim == 1:
        encoded = encoded[:, None]
    base = bins
    codes = np.zeros(encoded.shape[0], dtype=np.int64)
    for col in range(encoded.shape[1]):
        codes = codes * base + encoded[:, col]
    _, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    probs = counts[inverse].astype(float)
    return np.log(probs / probs.sum() + EPS)


def attention_features(X: np.ndarray, heads: int, hidden_dim: int, seed: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    head_dim = max(hidden_dim // max(heads, 1), 1)
    q = rng.normal(scale=1.0 / np.sqrt(max(d, 1)), size=(d, heads, head_dim))
    k = rng.normal(scale=1.0 / np.sqrt(max(d, 1)), size=(d, heads, head_dim))
    v = rng.normal(scale=1.0 / np.sqrt(max(d, 1)), size=(d, heads, head_dim))
    outputs = []
    for h in range(heads):
        qh = X @ q[:, h, :]
        kh = X @ k[:, h, :]
        vh = X @ v[:, h, :]
        scores = qh @ kh.T / np.sqrt(max(head_dim, 1))
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), EPS)
        outputs.append((weights @ vh).mean(axis=1))
    return np.column_stack(outputs)


def ridge_fit_predict(features: np.ndarray, target: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1, 1)
    X = np.column_stack([np.ones(features.shape[0]), features])
    reg = alpha * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(X.T @ X + reg) @ X.T @ target
    return (X @ beta).reshape(-1)


class TrainableScoreEncoder(torch_nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kind: str, num_heads: int, num_layers: int):
        super().__init__()
        self.kind = kind.lower().strip()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(max(hidden_dim, 4))
        self.num_heads = int(max(num_heads, 1))
        self.num_layers = int(max(num_layers, 1))

        if self.kind == "nn":
            self.net = torch_nn.Sequential(
                torch_nn.Linear(self.input_dim, self.hidden_dim),
                torch_nn.Tanh(),
                torch_nn.Linear(self.hidden_dim, self.hidden_dim),
                torch_nn.Tanh(),
                torch_nn.Linear(self.hidden_dim, 1),
            )
        elif self.kind == "mlp":
            self.net = DeepResidualMLPEncoder(self.input_dim, self.hidden_dim, depth=max(4, self.num_layers + 2), dropout=0.1)
        elif self.kind == "transformer":
            self.net = TokenTransformerEncoder(self.input_dim, self.hidden_dim, self.num_heads, self.num_layers)
        else:
            raise ValueError(f"Unsupported trainable encoder kind: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind in {"nn", "mlp", "transformer"}:
            return self.net(x)
        raise ValueError(f"Unsupported trainable encoder kind: {self.kind}")


def train_score_encoder(
    features: np.ndarray,
    target: np.ndarray,
    kind: str,
    seed: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
) -> tuple[np.ndarray, dict]:
    kind = kind.lower().strip()
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1, 1)
    if features.ndim == 1:
        features = features[:, None]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TrainableScoreEncoder(features.shape[1], hidden_dim, kind, num_heads, num_layers)
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(target, dtype=torch.float32)
    optimizer = torch_optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = torch_nn.MSELoss()

    model.train()
    epochs = 220 if kind == "transformer" else (180 if kind == "mlp" else 120)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = model(x_tensor).squeeze(-1).cpu().numpy()
    params = {
        "kind": kind,
        "input_dim": int(features.shape[1]),
        "hidden_dim": int(hidden_dim),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "trainable": True,
    }
    return scores, params


def build_score_backend(
    X: np.ndarray,
    model: str,
    indices: tuple[int, ...],
    bins: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
) -> tuple[np.ndarray, dict]:
    model = model.lower().strip()
    seed = model_seed(indices, model)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]

    if model in {"gaussian", "bn", "bayesian_network"}:
        if model == "gaussian":
            mu, cov = fit_gaussian(X, jitter=EPS)
            centered = X - mu.reshape(1, -1)
            inv_cov = np.linalg.pinv(cov)
            scores = -0.5 * np.sum(centered * (centered @ inv_cov), axis=1)
            params = {"mean": mu.tolist(), "cov": cov.tolist()}
        else:
            scores = joint_bin_scores(X, bins=max(2, bins))
            params = {"bins": int(bins), "table": histogram_distribution(scores, bins=max(2, bins)).tolist()}
        return scores, params

    if model in {"mlp", "nn", "transformer"}:
        base_scores, base_params = build_score_backend(X, "bn", indices, bins, hidden_dim, num_heads, num_layers)
        seed_local = seed % (2**32 - 1)
        features = [X, X**2, np.sin(X), np.cos(X)]
        if model in {"mlp", "nn"}:
            features.append(X * X)
            features.append(np.abs(X))
        else:
            features.append(attention_features(X, heads=max(1, num_heads), hidden_dim=max(hidden_dim, 4), seed=seed_local))
            features.append(np.roll(X, shift=1, axis=1))
        feats = np.column_stack([np.asarray(f, dtype=float).reshape(X.shape[0], -1) for f in features])
        scores, encoder_params = train_score_encoder(
            feats,
            base_scores,
            model,
            seed_local,
            hidden_dim,
            num_heads,
            num_layers,
        )
        params = {
            "base_model": "bn",
            "base": base_params,
            "feature_dim": int(feats.shape[1]),
            "hidden_dim": int(hidden_dim),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "encoder": encoder_params,
        }
        return scores, params

    raise ValueError(f"Unsupported potential model: {model}")


def fit_gaussian(X: np.ndarray, jitter: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov)
    cov = cov + np.eye(cov.shape[0]) * jitter
    return mu, cov


def gaussian_kl(
    mu0: np.ndarray,
    cov0: np.ndarray,
    mu1: np.ndarray,
    cov1: np.ndarray,
) -> float:
    mu0 = np.atleast_1d(np.asarray(mu0, dtype=float))
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=float))
    cov0 = np.atleast_2d(np.asarray(cov0, dtype=float))
    cov1 = np.atleast_2d(np.asarray(cov1, dtype=float))
    d = mu0.shape[0]
    inv_cov1 = np.linalg.pinv(cov1)
    diff = (mu1 - mu0).reshape(-1, 1)
    sign0, logdet0 = np.linalg.slogdet(cov0)
    sign1, logdet1 = np.linalg.slogdet(cov1)
    if sign0 <= 0 or sign1 <= 0:
        return 0.0
    trace_term = np.trace(inv_cov1 @ cov0)
    quad_term = float(np.asarray(diff.T @ inv_cov1 @ diff).squeeze())
    return 0.5 * (trace_term + quad_term - d + (logdet1 - logdet0))


def symmetric_gaussian_kl(
    mu0: np.ndarray,
    cov0: np.ndarray,
    mu1: np.ndarray,
    cov1: np.ndarray,
) -> float:
    return gaussian_kl(mu0, cov0, mu1, cov1) + gaussian_kl(mu1, cov1, mu0, cov0)


def separator_gap_value(info: dict, metric: str) -> float:
    metric = metric.lower().strip()
    if metric in {"sym_kl", "symmetric_kl", "symkl"}:
        return float(info["sym_kl"])
    if metric == "kl":
        return float(info["kl"])
    if metric == "l2":
        return float(info["l2"])
    raise ValueError(f"Unsupported gap metric: {metric}")


def build_association_matrix(X: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return np.abs(corr)


def build_skeleton(assoc: np.ndarray, top_k: int) -> nx.Graph:
    d = assoc.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(d))

    complete = nx.Graph()
    complete.add_nodes_from(range(d))
    for i in range(d):
        for j in range(i + 1, d):
            complete.add_edge(i, j, weight=float(assoc[i, j]))

    mst = nx.maximum_spanning_tree(complete, weight="weight")
    graph.add_edges_from(mst.edges(data=True))

    for i in range(d):
        neighbors = [
            j
            for j in np.argsort(assoc[i])[::-1]
            if j != i and assoc[i, j] > 0
        ]
        for j in neighbors[: max(top_k, 0)]:
            graph.add_edge(i, j, weight=float(assoc[i, j]))

    return graph


def maximal_cliques(skeleton: nx.Graph) -> list[tuple[int, ...]]:
    cliques = [tuple(sorted(clique)) for clique in nx.find_cliques(skeleton)]
    if not cliques:
        cliques = [(node,) for node in skeleton.nodes]
    cliques = sorted(set(cliques), key=lambda c: (-len(c), c))
    return cliques


def build_clique_tree(cliques: list[tuple[int, ...]]) -> nx.Graph:
    tree = nx.Graph()
    for idx, clique in enumerate(cliques):
        tree.add_node(idx, clique=clique)

    if len(cliques) <= 1:
        return tree

    complete = nx.Graph()
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            overlap = len(set(cliques[i]).intersection(cliques[j]))
            if overlap > 0:
                complete.add_edge(i, j, weight=overlap)

    if complete.number_of_edges() == 0:
        for i in range(len(cliques) - 1):
            tree.add_edge(i, i + 1, weight=0)
        return tree

    mst = nx.maximum_spanning_tree(complete, weight="weight")
    tree.add_edges_from(mst.edges(data=True))
    return tree


def clique_potentials(
    X: np.ndarray,
    cliques: list[tuple[int, ...]],
    config: JunctionTreeGapConfig,
) -> list[dict]:
    stats: list[dict] = []
    for clique in cliques:
        subset = X[:, clique]
        scores, params = build_score_backend(
            subset,
            config.clique_potential_model,
            tuple(clique),
            config.score_bins,
            config.hidden_dim,
            config.num_heads,
            config.num_layers,
        )
        mu, cov = fit_gaussian(subset, jitter=config.covariance_jitter)
        stats.append(
            {
                "clique": list(clique),
                "model": config.clique_potential_model,
                "mean": mu.tolist(),
                "cov": cov.tolist(),
                "scores": scores.tolist(),
                "score_mean": float(np.mean(scores)) if scores.size else 0.0,
                "score_std": float(np.std(scores)) if scores.size else 0.0,
                "params": params,
            }
        )
    return stats


def separator_gap(
    X: np.ndarray,
    cliques: list[tuple[int, ...]],
    tree: nx.Graph,
    config: JunctionTreeGapConfig,
) -> dict[tuple[int, int], dict]:
    gaps: dict[tuple[int, int], dict] = {}
    for i, j in tree.edges():
        clique_i = cliques[i]
        clique_j = cliques[j]
        separator = tuple(sorted(set(clique_i).intersection(clique_j)))
        if not separator:
            continue

        excl_i = [v for v in clique_i if v not in separator]
        excl_j = [v for v in clique_j if v not in separator]

        view_i = X[:, list(separator)]
        view_j = X[:, list(separator)]
        if excl_i:
            view_i = view_i - X[:, excl_i].mean(axis=1, keepdims=True)
        if excl_j:
            view_j = view_j - X[:, excl_j].mean(axis=1, keepdims=True)

        model_i = config.separator_model.lower().strip()
        model_j = config.separator_model.lower().strip()
        scores_i, params_i = build_score_backend(
            view_i,
            model_i,
            tuple(clique_i),
            config.score_bins,
            config.hidden_dim,
            config.num_heads,
            config.num_layers,
        )
        scores_j, params_j = build_score_backend(
            view_j,
            model_j,
            tuple(clique_j),
            config.score_bins,
            config.hidden_dim,
            config.num_heads,
            config.num_layers,
        )

        if config.separator_model.lower().strip() == "gaussian":
            mu_i, cov_i = fit_gaussian(view_i, jitter=config.covariance_jitter)
            mu_j, cov_j = fit_gaussian(view_j, jitter=config.covariance_jitter)
        else:
            mu_i = np.array([float(np.mean(scores_i))], dtype=float)
            mu_j = np.array([float(np.mean(scores_j))], dtype=float)
            cov_i = np.array([[float(np.var(scores_i) + config.covariance_jitter)]], dtype=float)
            cov_j = np.array([[float(np.var(scores_j) + config.covariance_jitter)]], dtype=float)

        gaps[(i, j)] = {
            "separator": list(separator),
            "clique_potential_model": config.clique_potential_model,
            "separator_model": config.separator_model,
            "up_message": {
                "mean": mu_i.tolist(),
                "cov": cov_i.tolist(),
                "scores": scores_i.tolist(),
                "params": params_i,
            },
            "down_message": {
                "mean": mu_j.tolist(),
                "cov": cov_j.tolist(),
                "scores": scores_j.tolist(),
                "params": params_j,
            },
            "kl": float(gaussian_kl(mu_i, cov_i, mu_j, cov_j)),
            "sym_kl": float(symmetric_gaussian_kl(mu_i, cov_i, mu_j, cov_j)),
            "l2": float(np.sum((mu_i - mu_j) ** 2)),
            "up_hist": histogram_distribution(scores_i, bins=config.score_bins).tolist(),
            "down_hist": histogram_distribution(scores_j, bins=config.score_bins).tolist(),
        }
    return gaps


def node_gap_penalty(
    d: int,
    tree: nx.Graph,
    gaps: dict[tuple[int, int], dict],
    metric: str,
) -> np.ndarray:
    penalty = np.zeros(d, dtype=float)
    counts = np.zeros(d, dtype=float)
    for info in gaps.values():
        separator = info["separator"]
        gap = separator_gap_value(info, metric)
        for node in separator:
            penalty[node] += gap
            counts[node] += 1.0
    counts = np.where(counts == 0, 1.0, counts)
    return penalty / counts


def orient_by_score(assoc: np.ndarray, node_score: np.ndarray) -> np.ndarray:
    d = assoc.shape[0]
    order = np.argsort(node_score)
    rank = {node: idx for idx, node in enumerate(order)}
    adj = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if assoc[i, j] <= 0:
                continue
            if rank[i] < rank[j]:
                adj[i, j] = 1
    return adj


def build_weight_matrix(assoc: np.ndarray, node_score: np.ndarray) -> np.ndarray:
    d = assoc.shape[0]
    weights = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            if i == j or assoc[i, j] <= 0:
                continue
            margin = node_score[j] - node_score[i]
            if margin > 0:
                weights[i, j] = float(assoc[i, j] * margin)
    return weights


def find_minimal_dag_threshold(W: np.ndarray) -> tuple[float, np.ndarray]:
    W = np.asarray(W, dtype=float).copy()
    if nx.is_directed_acyclic_graph(nx.DiGraph(W)):
        return 0.0, W
    possible_thresholds = sorted({abs(t) for t in W.flatten() if abs(t) > 0})
    for threshold in possible_thresholds:
        W[np.abs(W) < threshold] = 0.0
        if nx.is_directed_acyclic_graph(nx.DiGraph(W)):
            return float(threshold), W
    raise AssertionError("Should always find a DAG threshold.")


def evaluate(pred: np.ndarray, truth: np.ndarray | None) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    if truth is None:
        return metrics
    pred = (np.asarray(pred) != 0).astype(int)
    truth = (np.asarray(truth) != 0).astype(int)
    if pred.shape != truth.shape:
        raise ValueError(f"Predicted shape {pred.shape} does not match truth {truth.shape}")
    p = pred.reshape(-1)
    t = truth.reshape(-1)
    tp = int(np.sum((p == 1) & (t == 1)))
    fp = int(np.sum((p == 1) & (t == 0)))
    fn = int(np.sum((p == 0) & (t == 1)))
    tn = int(np.sum((p == 0) & (t == 0)))
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tn"] = tn
    metrics["tpr"] = tp / max(tp + fn, 1)
    metrics["fdr"] = fp / max(tp + fp, 1)
    metrics["precision"] = tp / max(tp + fp, 1)
    metrics["recall"] = tp / max(tp + fn, 1)
    metrics["f1"] = 0.0 if metrics["precision"] + metrics["recall"] == 0 else 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    metrics["accuracy"] = (tp + tn) / max(tp + tn + fp + fn, 1)
    metrics["fpr"] = fp / max(fp + tn, 1)
    metrics["tnr"] = tn / max(fp + tn, 1)
    metrics["fnr"] = fn / max(tp + fn, 1)
    metrics["nnz"] = int(pred.sum())
    metrics["shd"] = int(fp + fn)
    return metrics


class JunctionTreeGapLearner:
    def __init__(self, config: JunctionTreeGapConfig | None = None):
        self.config = config or JunctionTreeGapConfig()
        self.association_: np.ndarray | None = None
        self.skeleton_: nx.Graph | None = None
        self.cliques_: list[tuple[int, ...]] | None = None
        self.clique_tree_: nx.Graph | None = None
        self.clique_stats_: list[dict] | None = None
        self.separator_gaps_: dict[tuple[int, int], dict] | None = None
        self.node_scores_: np.ndarray | None = None
        self.weight_matrix_: np.ndarray | None = None
        self.dag_threshold_: float | None = None
        self.adjacency_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "JunctionTreeGapLearner":
        X = standardize(X)
        assoc = build_association_matrix(X)
        skeleton = build_skeleton(assoc, self.config.top_k)
        cliques = maximal_cliques(skeleton)
        clique_tree = build_clique_tree(cliques)
        stats = clique_potentials(X, cliques, self.config)
        gaps = separator_gap(X, cliques, clique_tree, self.config)
        penalties = node_gap_penalty(X.shape[1], clique_tree, gaps, self.config.gap_metric)
        strengths = assoc.sum(axis=1)
        clique_bonus = np.zeros(X.shape[1], dtype=float)
        clique_counts = np.zeros(X.shape[1], dtype=float)
        for stat in stats:
            score_mean = float(stat.get("score_mean", 0.0))
            for node in stat["clique"]:
                clique_bonus[node] += score_mean
                clique_counts[node] += 1.0
        clique_counts = np.where(clique_counts == 0, 1.0, clique_counts)
        clique_bonus = clique_bonus / clique_counts
        node_scores = strengths + 0.5 * clique_bonus - self.config.gap_weight * penalties
        node_scores = node_scores - self.config.sparsity_threshold * np.arange(X.shape[1])
        effective_assoc = np.where(assoc >= self.config.edge_threshold, assoc, 0.0)
        weight_matrix = build_weight_matrix(effective_assoc, node_scores)
        dag_threshold, thresholded_weights = find_minimal_dag_threshold(weight_matrix)
        adjacency = (thresholded_weights != 0).astype(int)

        self.association_ = assoc
        self.skeleton_ = skeleton
        self.cliques_ = cliques
        self.clique_tree_ = clique_tree
        self.clique_stats_ = stats
        self.separator_gaps_ = gaps
        self.node_scores_ = node_scores
        self.weight_matrix_ = thresholded_weights
        self.dag_threshold_ = dag_threshold
        self.adjacency_ = adjacency
        return self

    def to_artifact(self) -> dict:
        if self.adjacency_ is None:
            raise RuntimeError("Model has not been fitted.")
        payload = {
            "config": asdict(self.config),
            "node_scores": self.node_scores_.tolist() if self.node_scores_ is not None else None,
            "association": self.association_.tolist() if self.association_ is not None else None,
            "weight_matrix": self.weight_matrix_.tolist() if self.weight_matrix_ is not None else None,
            "dag_threshold": self.dag_threshold_,
            "iter_num": 1,
            "cliques": self.cliques_,
            "clique_stats": self.clique_stats_,
            "separator_gaps": {
                f"{i}-{j}": value for (i, j), value in (self.separator_gaps_ or {}).items()
            },
            "gap_metric": self.config.gap_metric,
            "separator_model": self.config.separator_model,
            "adjacency": self.adjacency_.tolist(),
        }
        return to_builtin(payload)


def save_outputs(
    output_dir: Path,
    learner: JunctionTreeGapLearner,
    truth: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adj = np.asarray(learner.adjacency_, dtype=int)
    np.save(output_dir / "adjacency.npy", adj)
    if learner.weight_matrix_ is not None:
        np.save(output_dir / "weight_matrix.npy", np.asarray(learner.weight_matrix_, dtype=float))
    pd.DataFrame(adj).to_csv(output_dir / "adjacency.csv", index=False, header=False)
    artifact = learner.to_artifact()
    (output_dir / "junction_tree_artifact.json").write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )
    metrics = evaluate(adj, truth)
    if metrics:
        (output_dir / "final_metrics.json").write_text(
            json.dumps({"metrics": metrics, "artifact": "junction_tree_artifact.json", "iter_num": 1}, indent=2),
            encoding="utf-8",
        )
    return adj, metrics
