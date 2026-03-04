"""Tests for the feature registry."""

import numpy as np
import pytest

from mosaique.features.registry import FEATURE_REGISTRY, register_feature


def test_register_feature_adds_to_registry():
    """A decorated function is added to the registry."""

    @register_feature(transform="simple")
    def dummy_feature(x, **kwargs):
        return float(np.mean(x))

    assert "dummy_feature" in FEATURE_REGISTRY
    entry = FEATURE_REGISTRY["dummy_feature"]
    assert entry.func is dummy_feature
    assert entry.transforms == frozenset({"simple"})
    del FEATURE_REGISTRY["dummy_feature"]


def test_register_feature_multiple_transforms():
    """A feature can be registered for multiple transforms."""

    @register_feature(transform=["simple", "tf_decomposition"])
    def multi_feature(x, **kwargs):
        return 0.0

    entry = FEATURE_REGISTRY["multi_feature"]
    assert entry.transforms == frozenset({"simple", "tf_decomposition"})
    del FEATURE_REGISTRY["multi_feature"]


def test_register_feature_rejects_missing_kwargs():
    """Functions without **kwargs are rejected at registration time."""
    with pytest.raises(TypeError, match="kwargs"):

        @register_feature(transform="simple")
        def bad_feature(x):
            return 0.0


def test_register_feature_preserves_function():
    """The decorator returns the original function unchanged."""

    @register_feature(transform="simple")
    def my_feat(x, **kwargs):
        return 1.0

    assert my_feat(np.array([1.0])) == 1.0
    del FEATURE_REGISTRY["my_feat"]


def test_all_univariate_features_registered():
    """All univariate features are in the registry."""
    import mosaique.features.univariate  # noqa: F401

    expected = {
        "approximate_entropy",
        "sample_entropy",
        "spectral_entropy",
        "permutation_entropy",
        "fuzzy_entropy",
        "corr_dim",
        "line_length",
        "peak_alpha",
        "hurst_exp",
        "band_power",
    }
    registered = {
        name for name, entry in FEATURE_REGISTRY.items() if "simple" in entry.transforms
    }
    assert expected <= registered


def test_all_connectivity_features_registered():
    """All connectivity features are in the registry."""
    import mosaique.features.graph_metrics  # noqa: F401

    expected = {
        "average_clustering",
        "average_node_connectivity",
        "average_degree",
        "global_efficiency",
        "average_shortest_path_length",
    }
    registered = {
        name
        for name, entry in FEATURE_REGISTRY.items()
        if "connectivity" in entry.transforms
    }
    assert expected <= registered
