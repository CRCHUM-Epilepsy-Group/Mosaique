# Remove networkx Dependency — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Replace all networkx usage with numpy + scipy.sparse.csgraph, removing the networkx dependency entirely.

**Architecture:** All graph metrics operate on the **unweighted** graph structure (edges exist where matrix entries are nonzero). Only the threshold functions use actual edge weights. Replace networkx graph construction with binary adjacency matrices and scipy sparse graph algorithms.

**Tech Stack:** numpy, scipy.sparse.csgraph (shortest_path, minimum_spanning_tree, connected_components, maximum_flow)

**Key insight:** The current code calls `nx.average_clustering(G)`, `nx.global_efficiency(G)`, etc. without passing `weight='weight'`, so all metrics use hop-count / binary adjacency. No weighted clustering formula needed.

---

### Task 1: Add numerical regression tests

Capture exact networkx outputs for specific matrices so we can verify the replacement matches.

**Files:**
- Modify: `tests/test_features.py:316-349` (extend `TestConnectivityCorrectness`)

**Step 1: Add regression tests**

Add these tests to `TestConnectivityCorrectness` in `tests/test_features.py`:

```python
def test_average_degree_sparse(self):
    # 4-node cycle: each node has degree 2
    mat = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ], dtype=float)
    assert average_degree(mat) == pytest.approx(2.0)

def test_average_clustering_with_triangle(self):
    # Node 0: neighbors {1,2}, edge 1-2 exists → c=1.0
    # Node 1: neighbors {0,2,3}, triangles 0-1-2 and 1-2-3 → c=2/3
    # Node 2: neighbors {1,3,0}, triangles 0-1-2 and 1-2-3 → c=2/3 (edge 0-3 missing)
    # Node 3: neighbors {2,1}, edge 1-2 exists → c=1.0
    # Average = (1 + 2/3 + 2/3 + 1) / 4 = 5/6
    mat = np.array([
        [0.0, 0.9, 0.1, 0.0],
        [0.9, 0.0, 0.5, 0.8],
        [0.1, 0.5, 0.0, 0.7],
        [0.0, 0.8, 0.7, 0.0],
    ], dtype=float)
    # Verified against networkx 3.6.1
    assert average_clustering(mat) == pytest.approx(5 / 6)

def test_global_efficiency_chain(self):
    # 3-node chain: pairs (0,1)=1, (0,2)=2, (1,2)=1
    # efficiency = (1/1 + 1/2 + 1/1 + 1/2 + 1/1 + 1/1) / (3*2)
    #            = (1 + 0.5 + 1 + 0.5 + 1 + 1) / 6 = 5/6
    # Wait — (0,1)=1, (1,0)=1, (0,2)=2, (2,0)=2, (1,2)=1, (2,1)=1
    # = (1+1+0.5+0.5+1+1)/6 = 5/6
    mat = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=float)
    assert global_efficiency(mat) == pytest.approx(5 / 6)

def test_average_node_connectivity_complete(self):
    # Complete graph of 4 nodes: node connectivity = n-1 = 3
    n = 4
    mat = np.ones((n, n)) - np.eye(n)
    assert average_node_connectivity(mat) == pytest.approx(n - 1)

def test_average_node_connectivity_chain(self):
    # 3-node chain: node connectivity between any pair = 1
    mat = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=float)
    assert average_node_connectivity(mat) == pytest.approx(1.0)

def test_connected_threshold_values(self):
    # 3x3 matrix: MST uses edges 0.3 and 0.5
    # min MST edge = 0.3, so threshold at 0.3
    mat = np.array([
        [0.0, 0.8, 0.3],
        [0.8, 0.0, 0.5],
        [0.3, 0.5, 0.0],
    ])
    result = connected_threshold(mat)
    # All edges >= 0.3, so all kept
    np.testing.assert_array_equal(result, mat)
```

**Step 2: Run tests to verify they pass with networkx**

Run: `pytest tests/test_features.py::TestConnectivityCorrectness -v`
Expected: all PASS (these capture current networkx behavior)

**Step 3: Commit**

```bash
git add tests/test_features.py
git commit -m "test: add regression tests for graph metrics before networkx removal"
```

---

### Task 2: Rewrite graph_metrics.py with numpy + scipy

Replace the entire implementation. Same public API, no networkx.

**Files:**
- Modify: `src/mosaique/features/graph_metrics.py`

**Step 1: Rewrite the file**

Replace the full contents of `graph_metrics.py` with:

```python
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
    for _ in range(20):
        mid = (lo + hi) / 2
        binary_mat = np.where(mat >= mid, 1, 0)
        n_components, _ = connected_components(
            csr_matrix(binary_mat), directed=False
        )
        if n_components == 1:
            lo = mid
        else:
            hi = mid
    return np.where(mat >= lo, 1, 0)


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
                # u_out → v_in, capacity n (effectively infinite)
                rows.append(2 * u + 1)
                cols.append(2 * v)
                data.append(n)

    aux = csr_matrix((data, (rows, cols)), shape=(size, size))

    total = 0.0
    for s in range(n):
        for t in range(n):
            if s != t:
                result = maximum_flow(aux, 2 * s + 1, 2 * t)
                total += result.flow_value
    return total / (n * (n - 1))


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
```

**Step 2: Run tests**

Run: `pytest tests/test_features.py -v -k connectivity or Connectivity`
Expected: all PASS

**Step 3: Commit**

```bash
git add src/mosaique/features/graph_metrics.py
git commit -m "feat: replace networkx with numpy+scipy in graph_metrics"
```

---

### Task 3: Remove networkx from tests

**Files:**
- Modify: `tests/test_features.py:20` and `tests/test_features.py:312`

**Step 1: Remove networkx import and usage**

Remove `import networkx as nx` (line 20). Replace `nx.NetworkXError` with `ValueError` (line 312):

```python
# Line 312: change
except (ValueError, nx.NetworkXError):
# to
except ValueError:
```

**Step 2: Run full test suite**

Run: `pytest tests/test_features.py -v`
Expected: all PASS

**Step 3: Commit**

```bash
git add tests/test_features.py
git commit -m "test: remove networkx dependency from tests"
```

---

### Task 4: Remove networkx from project dependencies

**Files:**
- Modify: `pyproject.toml:17` (remove `"networkx"` line)

**Step 1: Remove from pyproject.toml**

Remove the `"networkx",` line from the `dependencies` list.

**Step 2: Regenerate lockfile**

Run: `uv lock`

**Step 3: Sync environment**

Run: `uv sync`

**Step 4: Run full test suite to verify**

Run: `pytest tests/ -v`
Expected: all PASS (networkx no longer importable, but not needed)

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: remove networkx dependency"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md:38`

**Step 1: Update the module description**

Change line 38 from:
```
| `features/connectivity.py` | Graph metrics (clustering, efficiency, degree, path length) via networkx |
```
to:
```
| `features/connectivity.py` | Graph metrics (clustering, efficiency, degree, path length) via numpy/scipy |
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md after networkx removal"
```
