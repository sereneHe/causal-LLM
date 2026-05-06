#!/usr/bin/env python3
"""Utilities for loading DAG adjacency arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_graph_npy(path: str | Path, transpose: bool = False) -> np.ndarray:
    """Load a graph adjacency array from .npy and optionally transpose it."""
    graph = np.load(path)
    if transpose:
        graph = np.transpose(graph)
    return graph
