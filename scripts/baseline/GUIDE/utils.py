"""
Utility functions for GUIDE framework.
Contains helper functions for cycle removal, graph pruning, partial prior creation, and metrics.
"""

import numpy as np
import torch
import networkx as nx
import random
from scipy.linalg import expm as matrix_exponential
from sklearn.linear_model import LinearRegression


def ensure_acyclic(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Remove cycles from adjacency matrix by removing lowest-weight edges.
    
    Args:
        adj_matrix: 2D adjacency matrix tensor
        
    Returns:
        acyclic_adj_matrix: acyclic adjacency matrix
    """
    if adj_matrix.dim() != 2:
        raise ValueError("ensure_acyclic expects a 2D adjacency matrix.")

    device = adj_matrix.device
    A = adj_matrix.detach().cpu().numpy()
    N = A.shape[0]

    G = nx.DiGraph()
    for i in range(N):
        for j in range(N):
            if i != j and A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])

    while True:
        try:
            cycle_edges = nx.find_cycle(G, orientation='original')
            # Find all edges with the minimum weight
            min_edges = []
            min_w = float('inf')
            for (u, v, direction) in cycle_edges:
                w_ = G[u][v]['weight']
                if w_ < min_w:
                    min_edges = [(u, v)]
                    min_w = w_
                elif w_ == min_w:
                    min_edges.append((u, v))
            # Randomly pick one edge from the minimum-weight edges
            min_edge = random.choice(min_edges)
            G.remove_edge(min_edge[0], min_edge[1])
            A[min_edge[0], min_edge[1]] = 0.0
        except nx.NetworkXNoCycle:
            break

    return torch.from_numpy(A).float().to(device)


def calculate_threshold(W, d):
    """
    Calculate threshold for graph pruning based on coefficients.
    
    Args:
        W: weight matrix
        d: dimension parameter
        
    Returns:
        threshold: calculated threshold value
    """
    flattened = W.flatten()
    sorted_weights = np.sort(flattened)[::-1]
    d_idx = min(d-1, len(sorted_weights)-1)
    return sorted_weights[d_idx]


def graph_prunned_by_coef(graph_batch, X):
    """
    Prune graph based on regression coefficients.
    
    Args:
        graph_batch: adjacency matrix to prune
        X: input data matrix
        
    Returns:
        pruned_graph: pruned adjacency matrix
    """
    d = len(graph_batch)
    reg = LinearRegression()
    W = []

    for i in range(d):
        col = np.abs(graph_batch[i]) > 0.5
        if np.sum(col) == 0:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]
        y = X[:, i]
        reg.fit(X_train, y)
        reg_coeff = reg.coef_

        new_reg_coeff = np.zeros(d, dtype=float)
        parent_indices = np.where(col)[0]
        for idx_parent, coef_val in zip(parent_indices, reg_coeff):
            new_reg_coeff[idx_parent] = coef_val

        W.append(new_reg_coeff)

    W = np.array(W)
    th = calculate_threshold(W, X.shape[1])
    pruned = (np.abs(W) >= th).astype(np.float32)
    return pruned


def create_partial_prior(actual_dag: np.ndarray, fraction=0.25) -> np.ndarray:
    """
    Create partial prior knowledge from ground truth DAG.
    
    Args:
        actual_dag: ground truth adjacency matrix
        fraction: fraction of edges to keep as known
        
    Returns:
        prior_adj: partial prior adjacency matrix
                  -1: unknown, 0: known no-edge, 1: known edge
    """
    d = actual_dag.shape[0]
    prior_adj = -1 * np.ones((d, d), dtype=int)

    edges = np.argwhere(actual_dag > 0.5)
    np.random.shuffle(edges)
    num_edges = len(edges)
    keep_count = int(fraction * num_edges)

    known_edges = edges[:keep_count]
    for (i, j) in known_edges:
        prior_adj[i, j] = 1  # we are sure there's an edge i->j

    return prior_adj


def count_accuracy(B_true, B_est):
    """
    Calculate accuracy metrics for DAG comparison.
    
    Args:
        B_true: true adjacency matrix
        B_est: estimated adjacency matrix
        
    Returns:
        metrics: dictionary containing various accuracy metrics
    """
    d = B_true.shape[0]

    # Predicted edges and undirected edges
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # True positives (TP)
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])

    # False positives (FP)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])

    # Reverse edges
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

    # Predicted size and condition negatives
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)

    # False discovery rate (FDR), true positive rate (TPR), false positive rate (FPR)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)

    # True negatives (TN) and false negatives (FN)
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)

    # SHD: structural hamming distance
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    # FN and TN calculations
    fn = len(cond) - len(true_pos)
    tn = cond_neg_size - len(false_pos)

    # True negative rate (TNR) and false negative rate (FNR)
    tnr = float(tn) / max(cond_neg_size, 1)
    fnr = float(fn) / max(len(cond), 1)

    return {
        'fdr': fdr,
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr,
        'tp': len(true_pos),
        'tn': tn,
        'fp': len(false_pos),
        'fn': fn,
        'shd': shd,
        'nnz': pred_size
    }


def load_generative_prior(file_path):
    """
    Load generative prior from file.
    
    Args:
        file_path: path to the generative prior file
        
    Returns:
        prior: loaded generative prior matrix
    """
    try:
        return np.load(file_path)
    except FileNotFoundError:
        print(f"Warning: Generative prior file {file_path} not found. Using zeros.")
        return None


def save_results(results, file_path):
    """
    Save results to file.
    
    Args:
        results: results dictionary to save
        file_path: path to save the results
    """
    np.save(file_path, results)
    print(f"Results saved to {file_path}")


def plot_training_progress(rewards, losses, edges, cycles, save_path="training_progress.png"):
    """
    Plot training progress metrics.
    
    Args:
        rewards: list of episode rewards
        losses: list of actor losses
        edges: list of edge counts
        cycles: list of cycle counts
        save_path: path to save the plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title("Episode Reward")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    
    plt.subplot(1, 3, 2)
    plt.plot(losses)
    plt.title("Actor Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(1, 3, 3)
    plt.plot(edges, label="Edges (Best so far)")
    plt.plot(cycles, label="Cycles (Best so far)")
    plt.legend()
    plt.title("Edges & Cycles")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training progress plot saved to {save_path}")
