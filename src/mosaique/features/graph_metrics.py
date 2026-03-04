"""Graph-theoretic features computed on connectivity matrices."""

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    connected_components,
    maximum_flow,
    minimum_spanning_tree,
    shortest_path,
)

from mosaique.features.registry import register_feature


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


def _binary_adjacency(mat: np.ndarray) -> np.ndarray:
    """Return binary adjacency matrix with zero diagonal."""
    A = (mat != 0).astype(float)
    np.fill_diagonal(A, 0)
    return A


def connected_threshold(mat: np.ndarray) -> np.ndarray:
    """Threshold a connectivity matrix using the minimum spanning tree.

    Edges below the minimum weight in the minimum spanning tree are set
    to zero.

    Parameters
    ----------
    mat : np.ndarray
        Symmetric connectivity matrix of shape ``(n_channels, n_channels)``.

    Returns
    -------
    np.ndarray
        Thresholded matrix (same shape), with sub-threshold entries zeroed.
    """
    sparse = csr_matrix(mat)
    mst = minimum_spanning_tree(sparse)
    threshold = mst.data.min() if mst.nnz > 0 else 0.0
    return np.where(mat >= threshold, mat, 0)


def binary_threshold(mat: np.ndarray, max_threshold: float = 0.4) -> np.ndarray:
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
    for _ in range(20):
        mid = (lo + hi) / 2
        binary_mat = np.where(mat >= mid, 1, 0)
        n_components, _ = connected_components(csr_matrix(binary_mat), directed=False)
        if n_components == 1:
            lo = mid
        else:
            hi = mid
    return np.where(mat >= lo, 1, 0)


@register_feature(transform="connectivity")
def average_clustering(mat: np.ndarray, **kwargs: Any) -> float:
    """Average clustering coefficient of the graph (unweighted).

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    A = _binary_adjacency(mat)
    degree = A.sum(axis=1)
    # Triangles per node: diag(A^3) counts each triangle twice
    A3_diag = np.diag(A @ A @ A)
    denom = degree * (degree - 1)
    mask = denom > 0
    clustering = np.zeros(len(degree))
    clustering[mask] = A3_diag[mask] / denom[mask]
    return float(clustering.mean())


@register_feature(transform="connectivity")
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
    A = _binary_adjacency(mat)
    n = len(A)
    if n < 2 or A.sum() == 0:
        return 0.0

    # Build auxiliary directed graph: split each node v into v_in=2v, v_out=2v+1
    # Edge v_in → v_out with capacity 1 (node capacity constraint)
    # For each edge (u,v): u_out → v_in and v_out → u_in with capacity n
    size = 2 * n
    rows, cols, data = [], [], []

    for v in range(n):
        # v_in → v_out, capacity 1
        rows.append(2 * v)
        cols.append(2 * v + 1)
        data.append(1)

    for u in range(n):
        for v in range(n):
            if u != v and A[u, v] > 0:
                rows.append(2 * u + 1)
                cols.append(2 * v)
                data.append(1)

    aux = csr_matrix((data, (rows, cols)), shape=(size, size))

    total = 0.0
    for s in range(n):
        for t in range(n):
            if s != t:
                result = maximum_flow(aux, 2 * s + 1, 2 * t)
                total += result.flow_value
    return total / (n * (n - 1))


@register_feature(transform="connectivity")
def average_degree(mat: np.ndarray, **kwargs: Any) -> float:
    """Average node degree of the graph (unweighted).

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    A = _binary_adjacency(mat)
    return float(A.sum(axis=1).mean())


@register_feature(transform="connectivity")
def global_efficiency(mat: np.ndarray, **kwargs: Any) -> float:
    """Global efficiency of the graph (unweighted shortest paths).

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    A = _binary_adjacency(mat)
    n = len(A)
    if n < 2:
        return 0.0

    dist = shortest_path(csr_matrix(A), directed=False, unweighted=True)
    np.fill_diagonal(dist, np.inf)
    mask = (dist > 0) & np.isfinite(dist)
    if not mask.any():
        return 0.0
    return float(np.sum(1.0 / dist[mask]) / (n * (n - 1)))


@register_feature(transform="connectivity")
def average_shortest_path_length(mat: np.ndarray, **kwargs: Any) -> float:
    """Average shortest path length of the graph (unweighted).

    For disconnected graphs, averages over connected components
    weighted by the number of node pairs in each component.

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    A = _binary_adjacency(mat)
    n = len(A)
    sparse_A = csr_matrix(A)

    n_comp, labels = connected_components(sparse_A, directed=False)
    dist = shortest_path(sparse_A, directed=False, unweighted=True)

    if n_comp == 1:
        np.fill_diagonal(dist, 0)
        return float(dist.sum() / (n * (n - 1)))

    # Disconnected: weighted average over components
    total = 0.0
    total_pairs = 0
    for c in range(n_comp):
        nodes = np.where(labels == c)[0]
        nc = len(nodes)
        if nc > 1:
            comp_dist = dist[np.ix_(nodes, nodes)]
            np.fill_diagonal(comp_dist, 0)
            avg = comp_dist.sum() / (nc * (nc - 1))
            total += avg * nc * (nc - 1)
            total_pairs += nc * (nc - 1)
    return total / total_pairs if total_pairs > 0 else 0.0
