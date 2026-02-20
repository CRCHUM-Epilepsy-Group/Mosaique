"""Edge-case tests for all feature functions (univariate + connectivity).

Focus: inputs that break things (flat signals, inf/nan, wrong dimensions,
very short signals, empty signals), NOT mathematical correctness.

Tests assert the *desired* behavior. Many will fail against the current code
(e.g. NaN/inf not rejected, some features returning NaN on flat signals).
This is intentional — the test suite serves as a specification.

To add a new univariate feature:
  1. Implement the function
  2. Add it to UNIVARIATE_FEATURES below
  3. Run pytest — automatic coverage for valid input, flat signal, NaN/inf
     rejection, short signals, wrong dimensions, large values

If the feature has special parameters or return types (like band_power),
add a dedicated test class instead.
"""

import networkx as nx
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
    peak_alpha,
    band_power,
)
from mosaique.features.connectivity import (
    average_clustering,
    average_node_connectivity,
    average_degree,
    global_efficiency,
    average_shortest_path_length,
)

# ---------------------------------------------------------------------------
# Feature registries — add new features here for automatic edge-case coverage
# ---------------------------------------------------------------------------

UNIVARIATE_FEATURES = [
    approximate_entropy,
    sample_entropy,
    spectral_entropy,
    permutation_entropy,
    fuzzy_entropy,
    corr_dim,
    hurst_exp,
    line_length,
]

CONNECTIVITY_FEATURES = [
    average_clustering,
    average_node_connectivity,
    average_degree,
    global_efficiency,
    average_shortest_path_length,
]


def _fname(func):
    """Helper for nicer pytest IDs."""
    return func.__name__


# ===========================================================================
# Univariate features
# ===========================================================================


class TestUnivariateValidInput:
    """Happy-path: random signal → returns a finite float."""

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_returns_float(self, random_signal, func):
        result = func(random_signal)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_accepts_list_input(self, list_signal, func):
        result = func(list_signal)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)


class TestUnivariateFlatSignal:
    """Constant signal — must return a finite value, never NaN or inf."""

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_returns_finite(self, flat_signal, func):
        result = func(flat_signal)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)


class TestSampleEntropyUndefined:
    """sample_entropy returns np.nan when no template matches (B==0)."""

    def test_returns_nan_when_no_matches(self):
        # Monotonically increasing: all m-dim templates are far apart with small r
        x = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
        result = sample_entropy(x, m=2, r=0.001)
        assert np.isnan(result)


class TestUnivariateRejectsNaN:
    """NaN in input must raise ValueError (not silently return garbage)."""

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_signal_with_nan(self, signal_with_nan, func):
        with pytest.raises(ValueError):
            func(signal_with_nan)

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_all_nan_signal(self, all_nan_signal, func):
        with pytest.raises(ValueError):
            func(all_nan_signal)


class TestUnivariateRejectsInf:
    """Inf in input must raise ValueError."""

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_signal_with_inf(self, signal_with_inf, func):
        with pytest.raises(ValueError):
            func(signal_with_inf)

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_all_inf_signal(self, all_inf_signal, func):
        with pytest.raises(ValueError):
            func(all_inf_signal)


class TestUnivariateShortSignals:
    """Very short / empty / single-sample signals."""

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_short_signal_returns_float_or_raises(self, short_signal, func):
        """10-sample signal: either returns a finite float or raises clearly."""
        try:
            result = func(short_signal)
            assert isinstance(result, (float, np.floating))
        except (ValueError, ArithmeticError):
            pass  # acceptable — clear error on too-short input

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_single_sample_raises(self, single_sample_signal, func):
        with pytest.raises((ValueError, ArithmeticError)):
            func(single_sample_signal)

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_empty_signal_raises(self, empty_signal, func):
        with pytest.raises((ValueError, ArithmeticError)):
            func(empty_signal)


class TestUnivariateDimensions:
    """2D input must raise an exception (not silently collapse)."""

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_2d_input_raises(self, signal_2d, func):
        with pytest.raises(Exception):
            func(signal_2d)


class TestUnivariateLargeValues:
    """Very large values (~1e15) must not overflow to inf."""

    @pytest.mark.parametrize("func", UNIVARIATE_FEATURES, ids=_fname)
    def test_large_values_finite(self, large_value_signal, func):
        result = func(large_value_signal)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)


# ===========================================================================
# band_power (returns dict, needs freqs param)
# ===========================================================================

STANDARD_BANDS = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]


class TestBandPower:
    def test_valid_input(self, random_signal):
        result = band_power(random_signal, freqs=STANDARD_BANDS, sfreq=200)
        assert isinstance(result, dict)
        assert len(result) == len(STANDARD_BANDS)
        for value in result.values():
            assert np.isfinite(value)

    def test_flat_signal(self, flat_signal):
        result = band_power(flat_signal, freqs=STANDARD_BANDS, sfreq=200)
        assert isinstance(result, dict)
        for value in result.values():
            assert np.isfinite(value)

    def test_rejects_nan(self, signal_with_nan):
        with pytest.raises(ValueError):
            band_power(signal_with_nan, freqs=STANDARD_BANDS, sfreq=200)

    def test_rejects_inf(self, signal_with_inf):
        with pytest.raises(ValueError):
            band_power(signal_with_inf, freqs=STANDARD_BANDS, sfreq=200)


# ===========================================================================
# peak_alpha (needs sfreq, MNE dependency)
# ===========================================================================


class TestPeakAlpha:
    def test_valid_input(self, random_signal):
        result = peak_alpha(random_signal, sfreq=200)
        assert isinstance(result, (float, np.floating))
        assert 8.0 <= result <= 13.0

    def test_flat_signal(self, flat_signal):
        result = peak_alpha(flat_signal, sfreq=200)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_rejects_nan(self, signal_with_nan):
        with pytest.raises(ValueError):
            peak_alpha(signal_with_nan, sfreq=200)

    def test_rejects_inf(self, signal_with_inf):
        with pytest.raises(ValueError):
            peak_alpha(signal_with_inf, sfreq=200)


# ===========================================================================
# Connectivity features
# ===========================================================================


class TestConnectivityValidInput:
    """Happy-path: valid connectivity matrix → returns a finite float."""

    @pytest.mark.parametrize("func", CONNECTIVITY_FEATURES, ids=_fname)
    def test_returns_float(self, connectivity_matrix, func):
        result = func(connectivity_matrix)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)


class TestConnectivityEdgeCases:
    @pytest.mark.parametrize("func", CONNECTIVITY_FEATURES, ids=_fname)
    def test_zero_matrix(self, zero_connectivity_matrix, func):
        result = func(zero_connectivity_matrix)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    @pytest.mark.parametrize("func", CONNECTIVITY_FEATURES, ids=_fname)
    def test_identity_matrix(self, identity_connectivity_matrix, func):
        result = func(identity_connectivity_matrix)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    @pytest.mark.parametrize("func", CONNECTIVITY_FEATURES, ids=_fname)
    def test_nan_matrix_raises(self, func):
        mat = np.full((3, 3), np.nan)
        with pytest.raises((ValueError, TypeError)):
            func(mat)

    @pytest.mark.parametrize("func", CONNECTIVITY_FEATURES, ids=_fname)
    def test_1x1_matrix(self, func):
        mat = np.array([[1.0]])
        try:
            result = func(mat)
            assert isinstance(result, (float, np.floating))
        except (ValueError, nx.NetworkXError):
            pass  # acceptable — clear error on degenerate graph

