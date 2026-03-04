# Remove networkx dependency

Replace all networkx usage in `graph_metrics.py` with numpy + scipy (`scipy.sparse.csgraph`). scipy is already a direct dependency.

## Scope

All graph metric functions in `src/mosaique/features/graph_metrics.py` currently convert numpy adjacency matrices to networkx graphs, compute a metric, and return a scalar. This round-trip is unnecessary — numpy and scipy can do it directly.

## Per-function replacement

| Function | Current (networkx) | Replacement |
|----------|-------------------|-------------|
| `average_degree` | `nx.degree(G)` | `np.mean(np.sum(mat != 0, axis=1))` |
| `average_clustering` | `nx.average_clustering(G)` | Onnela weighted formula via numpy (cube-root of normalized weight products) |
| `global_efficiency` | `nx.global_efficiency(G)` | `scipy.sparse.csgraph.shortest_path` + `mean(1/d)` |
| `average_shortest_path_length` | `nx.average_shortest_path_length(G)` | `shortest_path` + `connected_components` for disconnected handling |
| `connected_threshold` | `nx.minimum_spanning_tree(G)` | `scipy.sparse.csgraph.minimum_spanning_tree` on negated weights |
| `binary_threshold` | `nx.is_connected(G)` | `scipy.sparse.csgraph.connected_components` — check n_components == 1 |
| `average_node_connectivity` | `nx.average_node_connectivity(G)` | `scipy.sparse.csgraph.maximum_flow` between all node pairs |

## Behavioral notes

- **Weighted clustering:** networkx uses the Onnela et al. (2005) formula by default on weighted graphs. The replacement must match this exactly.
- **Shortest paths:** networkx uses edge weights as distances. scipy.sparse.csgraph does the same with `shortest_path`. For connectivity matrices where higher = stronger connection, we need to invert weights (1/w) before computing path lengths — matching networkx's `from_numpy_array` behavior.
- **Disconnected graphs:** `average_shortest_path_length` already handles disconnected graphs by averaging over connected components. The scipy version uses `connected_components` for the same logic.

## Files changed

1. `src/mosaique/features/graph_metrics.py` — rewrite (same public API)
2. `tests/test_features.py` — remove `import networkx`, replace `nx.NetworkXError` with `ValueError`
3. `pyproject.toml` — remove `"networkx"` from dependencies
4. `uv.lock` — regenerated via `uv lock`
5. `CLAUDE.md` — update reference to networkx
