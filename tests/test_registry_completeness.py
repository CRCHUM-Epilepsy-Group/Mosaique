"""Ensure every public feature function in mosaique/features/ is registered."""

import inspect

import mosaique.features.univariate as univariate_mod
import mosaique.features.graph_metrics as graph_metrics_mod
from mosaique.features.registry import FEATURE_REGISTRY

_UNIVARIATE_HELPERS = {
    "_validate_signal",
    "skip_nones",
    "logarithmic_r",
    "shannon_entropy",
    "ordinal_distribution",
    "rescaled_range",
    "_multitaper_psd",
}

_GRAPH_METRICS_HELPERS = {
    "_validate_matrix",
    "_binary_adjacency",
    "connected_threshold",
    "binary_threshold",
}


def _public_functions(module, exclude):
    """Get all public callable names from a module, minus exclusions."""
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_")
        and name not in exclude
        and obj.__module__ == module.__name__
    }


def test_all_univariate_functions_registered():
    public = _public_functions(univariate_mod, _UNIVARIATE_HELPERS)
    registered = set(FEATURE_REGISTRY.keys())
    missing = public - registered
    assert not missing, (
        f"Univariate functions not registered: {missing}. "
        f"Add @register_feature decorator."
    )


def test_all_graph_metrics_functions_registered():
    public = _public_functions(graph_metrics_mod, _GRAPH_METRICS_HELPERS)
    registered = set(FEATURE_REGISTRY.keys())
    missing = public - registered
    assert not missing, (
        f"Graph metric functions not registered: {missing}. "
        f"Add @register_feature decorator."
    )
