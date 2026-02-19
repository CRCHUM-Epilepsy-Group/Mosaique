"""Edge-case tests for univariate feature functions.

Focus: inputs that break things (flat signals, inf/nan, wrong dimensions,
very short signals), NOT mathematical correctness.
"""

import numpy as np
import pytest

from mosaique.features.univariate import (
    approximate_entropy,
    sample_entropy,
    spectral_entropy,
    permutation_entropy,
    fuzzy_entropy,
    corr_dim,
    hurst_exp,
    line_length,
    logarithmic_r,
    rescaled_range,
)

# All feature functions to test in bulk
FEATURE_FUNCTIONS = [
    approximate_entropy,
    sample_entropy,
    spectral_entropy,
    permutation_entropy,
    fuzzy_entropy,
    line_length,
]


class TestFlatSignal:
    """Constant/flat signal — zero std triggers division issues."""

    def test_approximate_entropy(self, flat_signal):
        result = approximate_entropy(flat_signal)
        assert np.isfinite(result)

    def test_sample_entropy(self, flat_signal):
        # tolerance = 0 when std = 0, all distances match
        result = sample_entropy(flat_signal)
        assert not np.isnan(result)

    def test_spectral_entropy(self, flat_signal):
        result = spectral_entropy(flat_signal, sfreq=200)
        assert np.isfinite(result)

    def test_fuzzy_entropy(self, flat_signal):
        result = fuzzy_entropy(flat_signal)
        assert result == pytest.approx(0.0)

    def test_corr_dim(self, flat_signal):
        result = corr_dim(flat_signal)
        assert result == pytest.approx(0.0)

    def test_corr_dim_zeros(self):
        result = corr_dim(np.zeros(500))
        assert result == pytest.approx(0.0)

    def test_hurst_exp(self, flat_signal):
        result = hurst_exp(flat_signal)
        assert result == pytest.approx(0.0)

    def test_line_length(self, flat_signal):
        assert line_length(flat_signal) == pytest.approx(0.0)

    def test_rescaled_range(self, flat_signal):
        assert rescaled_range(flat_signal) == 0


class TestNaNInput:
    """Signals containing NaN — should not silently produce wrong results."""

    @pytest.mark.parametrize("func", FEATURE_FUNCTIONS)
    def test_nan_propagates_or_finite(self, signal_with_nan, func):
        result = func(signal_with_nan)
        # Either NaN propagates or the function handles it — no exception
        assert isinstance(result, (float, np.floating))


class TestInfInput:
    """Signals containing +/- inf."""

    @pytest.mark.parametrize("func", FEATURE_FUNCTIONS)
    def test_inf_does_not_crash(self, signal_with_inf, func):
        # Should not raise an unhandled exception
        result = func(signal_with_inf)
        assert isinstance(result, (float, np.floating))


class TestShortSignal:
    """Very short signals — may break embedding or windowing logic."""

    def test_approximate_entropy(self, short_signal):
        result = approximate_entropy(short_signal, m=2)
        assert isinstance(result, (float, np.floating))

    def test_sample_entropy(self, short_signal):
        result = sample_entropy(short_signal, m=2)
        assert isinstance(result, (float, np.floating))

    def test_permutation_entropy(self, short_signal):
        result = permutation_entropy(short_signal, k=3)
        assert isinstance(result, (float, np.floating))

    def test_hurst_exp(self, short_signal):
        # 10 samples with default min_window=10 — tight fit
        result = hurst_exp(short_signal, min_window=5)
        assert isinstance(result, (float, np.floating))


class TestWrongDimensions:
    """2D arrays passed to functions expecting 1D."""

    def test_line_length_2d(self, signal_2d):
        # np.diff + np.mean with default axis collapses to scalar
        result = line_length(signal_2d)
        assert np.isscalar(result)

    def test_approximate_entropy_2d(self, signal_2d):
        with pytest.raises(Exception):
            approximate_entropy(signal_2d)

    def test_sample_entropy_2d(self, signal_2d):
        with pytest.raises(Exception):
            sample_entropy(signal_2d)


class TestLogarithmicR:
    def test_rejects_inverted_bounds(self):
        with pytest.raises(AssertionError):
            logarithmic_r(100, 1, 2)

    def test_rejects_factor_below_one(self):
        with pytest.raises(AssertionError):
            logarithmic_r(1, 100, 0.5)
