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
    assert entry.transforms == {"simple"}
    del FEATURE_REGISTRY["dummy_feature"]


def test_register_feature_multiple_transforms():
    """A feature can be registered for multiple transforms."""

    @register_feature(transform=["simple", "tf_decomposition"])
    def multi_feature(x, **kwargs):
        return 0.0

    entry = FEATURE_REGISTRY["multi_feature"]
    assert entry.transforms == {"simple", "tf_decomposition"}
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
