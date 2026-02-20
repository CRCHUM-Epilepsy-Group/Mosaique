"""Graph-theoretic features computed on connectivity matrices."""

from typing import Any

import networkx as nx
import numpy as np


def _validate_matrix(mat: np.ndarray) -> np.ndarray:
    """Validate a connectivity matrix."""
    mat = np.asarray(mat, dtype=float)
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        raise ValueError("Connectivity matrix must be at least 2x2")
    if np.any(np.isnan(mat)):
        raise ValueError("Connectivity matrix contains NaN values")
    if np.any(np.isinf(mat)):
        raise ValueError("Connectivity matrix contains Inf values")
    return mat


def connected_threshold(mat: np.ndarray) -> np.ndarray:
    """Threshold a connectivity matrix using the minimum spanning tree.

    Edges below the minimum weight in the maximum spanning tree are set
    to zero, guaranteeing the resulting graph stays connected.

    Parameters
    ----------
    mat : np.ndarray
        Symmetric connectivity matrix of shape ``(n_channels, n_channels)``.

    Returns
    -------
    np.ndarray
        Thresholded matrix (same shape), with sub-threshold entries zeroed.
    """
    G = nx.from_numpy_array(mat)
    st = nx.minimum_spanning_tree(G)
    edges = [w for _, _, w in st.edges(data="weight")]
    threshold = min(edges)

    thresholded_mat = np.where(mat >= threshold, mat, 0)

    return thresholded_mat


def binary_threshold(
    mat: np.ndarray, max_threshold: float = 0.4
) -> np.ndarray:
    """Binary-threshold a connectivity matrix while keeping it connected.

    Uses binary search to find the highest threshold at which the resulting
    binary graph remains connected.

    Parameters
    ----------
    mat : np.ndarray
        Symmetric connectivity matrix of shape ``(n_channels, n_channels)``.
    max_threshold : float
        Upper bound for the threshold search (default 0.4).

    Returns
    -------
    np.ndarray
        Binary matrix (same shape), with 1 where the original value meets
        the threshold and 0 otherwise.
    """
    lo, hi = 0.0, max_threshold
    # Binary search for the highest connected threshold (precision ~0.005)
    for _ in range(20):
        mid = (lo + hi) / 2
        binary_mat = np.where(mat >= mid, 1, 0)
        if nx.is_connected(nx.from_numpy_array(binary_mat)):
            lo = mid
        else:
            hi = mid

    return np.where(mat >= lo, 1, 0)


def average_clustering(mat: np.ndarray, **kwargs: Any) -> float:
    """Average clustering coefficient of the graph.

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    G = nx.from_numpy_array(mat)
    return nx.average_clustering(G)


def average_node_connectivity(mat: np.ndarray, **kwargs: Any) -> float:
    """Average node connectivity (expected number of node-independent paths).

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    G = nx.from_numpy_array(mat)
    return nx.average_node_connectivity(G)


def average_degree(mat: np.ndarray, **kwargs: Any) -> float:
    """Average node degree of the graph.

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    G = nx.from_numpy_array(mat)
    return np.mean([d for _, d in nx.degree(G)])


def global_efficiency(mat: np.ndarray, **kwargs: Any) -> float:
    """Global efficiency of the graph.

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    G = nx.from_numpy_array(mat)
    return nx.global_efficiency(G)


def average_shortest_path_length(mat: np.ndarray, **kwargs: Any) -> float:
    """Average shortest path length of the graph.

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    G = nx.from_numpy_array(mat)
    if not nx.is_connected(G):
        total = 0.0
        total_pairs = 0
        for comp in nx.connected_components(G):
            sg = G.subgraph(comp)
            n = len(sg)
            if n > 1:
                total += nx.average_shortest_path_length(sg) * n * (n - 1)
                total_pairs += n * (n - 1)
        return total / total_pairs if total_pairs > 0 else 0.0
    return nx.average_shortest_path_length(G)
