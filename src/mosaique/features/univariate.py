"""
Univariate features for EEG

Applicable to 1-D signal in the time domain

Must return single values or dictionaries if multiple values are generated
"""

import collections
import math
import numpy as np
from scipy.signal import periodogram
from scipy.spatial.distance import pdist
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simpson

from mosaique.features.timefrequency import FrequencyBand


# Helper functions


def _validate_signal(x, min_samples=2):
    """Validate 1-D signal input for univariate features."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D signal, got {x.ndim}-D input")
    if x.size < min_samples:
        raise ValueError(f"Signal too short: {x.size} samples, need >= {min_samples}")
    if np.any(np.isnan(x)):
        raise ValueError("Signal contains NaN values")
    if np.any(np.isinf(x)):
        raise ValueError("Signal contains Inf values")
    return x


def skip_nones(fun):
    def _(*args, **kwargs):
        for a, v in zip(fun.__code__.co_varnames, args):
            if v is not None:
                kwargs[a] = v
        return fun(**kwargs)

    return _


def logarithmic_r(min_n, max_n, factor):
    """
    Creates a list of values by successively multiplying a minimum value min_n by
    a factor > 1 until a maximum value max_n is reached.
    Args:
    min_n (float):
      minimum value (must be < max_n)
    max_n (float):
      maximum value (must be > min_n)
    factor (float):
      factor used to increase min_n (must be > 1)
    Returns:
    list of floats:
      min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
    """
    assert max_n > min_n
    assert factor > 1
    max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
    return np.array([min_n * (factor**i) for i in range(max_i + 1)])


def shannon_entropy(x, **kwargs):
    probabilities = [n_x / len(x) for _, n_x in collections.Counter(x).items()]
    e_x = [-p_x * math.log(p_x, 2) for p_x in probabilities]
    entropy = sum(e_x)
    return entropy


def ordinal_distribution(
    data, dx=3, dy=1, taux=1, tauy=1, return_missing=False, tie_precision=None
):
    """
    Applies the Bandt and Pompe\\ [#bandt_pompe]_ symbolization approach to obtain
    a probability distribution of ordinal patterns (permutations) from data.

    Source: A. A. B. Pessa, H. V. Ribeiro, ordpy: A Python package for data analysis with permutation entropy and ordinal network methods, Chaos 31, 063110 (2021).

    Parameters
    ----------
    data : array
    dx : int. Embedding dimension (horizontal axis) (default: 3).
    dy : int. Embedding dimension (vertical axis); it must be 1 for time X
         (default: 1).
    taux : int. Embedding delay (horizontal axis) (default: 1).
    tauy : int. Embedding delay (vertical axis) (default: 1).
    return_missing: boolean. If `True`, it returns ordinal patterns not appearing in the
                             symbolic sequence obtained from **data** are shown. If `False`,
                             these missing patterns (permutations) are omitted
                             (default: `False`).
    tie_precision : int. If not `None`, **data** is rounded with `tie_precision`
                    number of decimals (default: `None`).
    Returns
    -------
     : tuple containing two arrays, one with the ordinal patterns occurring in data
       and another with their corresponding probabilities.

    """
    try:
        ny, nx = np.shape(data)
        data = np.array(data)
    except:
        nx = np.shape(data)[0]
        ny = 1
        data = np.array([data])

    if tie_precision is not None:
        data = np.round(data, tie_precision)

    partitions = np.concatenate(
        [
            [
                np.concatenate(data[j : j + dy * tauy : tauy, i : i + dx * taux : taux])
                for i in range(nx - (dx - 1) * taux)
            ]
            for j in range(ny - (dy - 1) * tauy)
        ]
    )

    symbols = np.apply_along_axis(np.argsort, 1, partitions)
    symbols, symbols_count = np.unique(symbols, return_counts=True, axis=0)

    probabilities = symbols_count / len(partitions)

    return symbols, probabilities


def rescaled_range(x):
    mean = np.mean(x)
    deviations = x - mean
    Z = np.cumsum(deviations)
    R = max(Z) - min(Z)
    S = np.std(x, ddof=1)

    if R == 0 or S == 0:
        return 0

    return R / S


####################################################################
# Feature functions
####################################################################


# Entropies


def approximate_entropy(x, m=3, r=0.2, **kwargs):
    """Compute approximate entropy (ApEn) of a 1-D signal.

    Parameters
    ----------
    x : array
        1-D input signal.
    m : int
        Embedding dimension (default 3).
    r : float
        Tolerance factor, multiplied by std(x) (default 0.2).

    Returns
    -------
    float
        Approximate entropy value.
    """
    x = _validate_signal(x)
    N = x.shape[0]
    r *= np.std(x, axis=0)

    def _phi(m):
        z = N - m + 1.0
        emb = np.array([x[i : i + m] for i in range(int(z))])
        X = np.repeat(emb[:, np.newaxis], 1, axis=2)
        C = np.sum(np.absolute(emb - X).max(axis=2) <= r, axis=0) / z
        return np.log(C).sum() / z

    phi1 = _phi(m)
    phi0 = _phi(m + 1)
    apen = abs(phi0 - phi1)

    return apen


def sample_entropy(x, m=2, r=0.2, **kwargs):
    """Compute sample entropy (SampEn) of a 1-D signal.

    Parameters
    ----------
    x : array
        1-D input signal.
    m : int
        Embedding dimension (default 2).
    r : float
        Tolerance factor, multiplied by std(x) (default 0.2).

    Returns
    -------
    float
        Sample entropy value.  Returns ``np.inf`` when no template matches.
    """
    x = _validate_signal(x)
    N = len(x)
    sigma = np.std(x)
    tolerance = sigma * r

    # Create embedding vectors
    matches = np.zeros((N, m + 1))
    for i in range(m + 1):
        matches[0 : (N - i), i] = x[i:]

    # Remove empty templates
    matches = matches[:-m, :]

    dist_m = pdist(matches[:, :m], metric="chebyshev")
    B = (dist_m <= tolerance).sum()

    if B == 0:
        return np.inf

    dist_m1 = pdist(matches, metric="chebyshev")
    A = (dist_m1 <= tolerance).sum()

    with np.errstate(divide="ignore"):
        sampen = -np.log(A / B)

    return sampen


def spectral_entropy(x, sfreq=200, normalize=True, **kwargs):
    """
    Parameters
    ----------
    x : array. 1D or N-D data.
    sfreq : float. Sampling frequency, in Hz.
    normalize : bool. If True, divide by log2(psd.size) to normalize the spectral entropy
                between 0 and 1. Otherwise, return the spectral entropy in bit.

    Returns
    -------
    se : float. Spectral Entropy

    -------
    Source: https://github.com/raphaelvallat/antropy/tree/master/antropy
    """
    x = _validate_signal(x)
    # Compute and normalize power spectrum
    _, psd = periodogram(x, sfreq, nfft=None, axis=-1)
    total = psd.sum()
    if total == 0:
        return 0.0
    psd_norm = psd / total
    with np.errstate(divide="ignore", invalid="ignore"):
        se = -(psd_norm * np.log2(psd_norm)).sum()
    if normalize:
        se /= np.log2(psd_norm.size)
    return se


def permutation_entropy(x, k=3, **kwargs):
    """Compute permutation entropy of a 1-D signal.

    Parameters
    ----------
    x : array
        1-D input signal.
    k : int
        Embedding dimension for ordinal patterns (default 3).

    Returns
    -------
    float
        Permutation entropy in bits.
    """
    x = _validate_signal(x)
    _, perm_probabilities = ordinal_distribution(x, dx=k)
    permen = shannon_entropy(perm_probabilities)

    return permen


def fuzzy_entropy(x, m=2, r=0.2, n=2, **kwargs):
    """Compute fuzzy entropy (FuzzyEn) of a 1-D signal.

    Parameters
    ----------
    x : array
        1-D input signal.
    m : int
        Embedding dimension (default 2).
    r : float
        Tolerance factor, multiplied by std(x) (default 0.2).
    n : int
        Fuzzy function gradient exponent (default 2).

    Returns
    -------
    float
        Fuzzy entropy value.
    """

    def fuzzy_fun(dist, gradient, width):
        arg = (dist**gradient) / width
        arg = np.minimum(arg, 700)  # prevent exp overflow
        return np.exp(-arg)

    x = _validate_signal(x)
    N = len(x)
    sigma = np.std(x)
    if sigma == 0:
        return 0.0
    tolerance = sigma * r

    # Create embedding vectors
    matches = np.zeros((N, m + 1))
    for i in range(m + 1):
        matches[0 : (N - i), i] = x[i:]

    # Remove empty templates
    matches = matches[:-m, :]

    dist_m = pdist(matches[:, :m], metric="chebyshev")
    ps_m = fuzzy_fun(dist_m, n, tolerance).sum()

    dist_m1 = pdist(matches, metric="chebyshev")
    ps_m1 = fuzzy_fun(dist_m1, gradient=n, width=tolerance).sum()

    fuzzen = np.log(ps_m) - np.log(ps_m1)

    return fuzzen


# Non linear markers


def corr_dim(x, embed_dim=2, rvals=None, **kwargs):
    """Estimate the correlation dimension of a 1-D signal.

    Parameters
    ----------
    x : array
        1-D input signal.
    embed_dim : int
        Embedding dimension for delay embedding (default 2).
    rvals : array-like or None
        Radii at which to evaluate the correlation sum.  If ``None``,
        a logarithmic range based on the standard deviation is used.

    Returns
    -------
    float
        Estimated correlation dimension.  Returns ``np.nan`` when the
        signal is constant or the fit is degenerate.
    """
    x = _validate_signal(x)
    N = len(x)
    sd = np.std(x, ddof=1)

    if sd == 0:
        return 0.0

    if rvals is None:
        rvals = logarithmic_r(0.1 * sd, 0.5 * sd, 1.03)

    delay_embed = np.zeros((N, embed_dim))
    for i in range(embed_dim):
        delay_embed[0 : (N - i), i] = x[i:]
    # Remove empty templates
    delay_embed = delay_embed[: -(embed_dim - 1), :]

    dists = pdist(delay_embed, metric="euclidean")
    csums = np.zeros(len(rvals))
    for i, r in enumerate(rvals):
        A = (dists < r).sum()
        s = 1 / (N * (N - 1)) * A
        csums[i] = s

    nonzero = np.where(csums != 0)
    rvals = np.array(rvals)[nonzero]
    csums = csums[nonzero]

    if len(csums) == 0:
        # all sums are zero => we cannot fit a line
        return np.nan
    else:
        A = np.vstack([np.log(csums), np.ones(len(rvals))]).T

        poly = np.linalg.lstsq(A, np.log(rvals), rcond=None)[0]

        return poly[0]


def line_length(x, **kwargs):
    """Compute the line length of a 1-D signal.

    Line length is the mean absolute first-order difference of the signal,
    a proxy for signal complexity / high-frequency content.

    Parameters
    ----------
    x : array
        1-D input signal.

    Returns
    -------
    float
        Mean absolute difference.

    References
    ----------
    .. [1] Esteller, R. et al. (2001). Line length: an efficient feature for
           seizure onset detection. *Proc. 23rd Annual International
           Conference of the IEEE EMBS*, Vol. 2, pp. 1707-1710.
    """

    x = _validate_signal(x)
    diff = np.diff(x)
    length = np.mean(np.abs(diff))

    return length


# Linear markers


def peak_alpha(x, sfreq=200, **kwargs):
    """Find the peak frequency in the alpha band (8–13 Hz).

    Parameters
    ----------
    x : array
        1-D input signal.
    sfreq : float
        Sampling frequency in Hz (default 200).

    Returns
    -------
    float
        Peak alpha frequency in Hz.
    """
    x = _validate_signal(x)
    psd, freqs = psd_array_multitaper(x, sfreq, normalization="length")  # type: ignore
    alpha_band = np.where((freqs >= 8) & (freqs <= 13))[0]
    peak_alpha_ind = np.argmax(psd[alpha_band])
    peak_alpha = freqs[alpha_band][peak_alpha_ind]

    return peak_alpha


def hurst_exp(x, min_window=10, max_window=None, **kwargs):
    """Estimate the Hurst exponent via rescaled range (R/S) analysis.

    Parameters
    ----------
    x : array
        1-D input signal.
    min_window : int
        Minimum window size for R/S calculation (default 10).
    max_window : int or None
        Maximum window size.  If ``None``, uses ``len(x) - 1``.

    Returns
    -------
    float
        Estimated Hurst exponent (H ≈ 0.5 for random walk, H > 0.5 for
        persistent series, H < 0.5 for anti-persistent series).
    """

    x = _validate_signal(x)
    # Get windows sizes in log scale
    max_window = max_window or len(x) - 1
    window_sizes = list(
        map(
            lambda v: int(10**v),
            np.arange(math.log10(min_window), math.log10(max_window), 0.25),
        )
    )
    window_sizes.append(len(x))

    RS = []
    for w in window_sizes:
        rs = []
        for start in range(0, len(x), w):
            if (start + w) > len(x):
                break
            _ = rescaled_range(x[start : start + w])
            if _ != 0:
                rs.append(_)
        RS.append(np.mean(rs) if rs else 0.0)

    RS = np.array(RS)
    window_sizes = np.array(window_sizes)

    # Need at least one non-zero RS to fit a line in log-log space
    nonzero = RS > 0
    if nonzero.sum() < 2:
        return 0.0

    RS = RS[nonzero]
    window_sizes = window_sizes[nonzero]

    A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
    H, c = np.linalg.lstsq(A, np.log10(RS), rcond=None)[0]
    c = 10**c

    return H


def band_power(x, freqs: list[FrequencyBand], sfreq=200, **kwargs):
    """Get average band power for frequency bands.

    Parameters
    ----------
    x : array
        1D or N-D data
    freqs : list of tuples
        List of (low_freq, high_freq) tuples defining frequency bands
    sfreq : float
        Sampling frequency in Hz

    Returns
    -------
    dict
        Band powers for each frequency band
    """
    x = _validate_signal(x)
    # Calculate PSD
    psd, freq_bins = psd_array_multitaper(
        x,
        sfreq,
        low_bias=False,
        n_jobs=1,
        normalization="length",
    )

    # Find indices for each band
    def get_band_indices(freq_bins, band):
        low_idx = np.searchsorted(freq_bins, band[0])
        high_idx = np.searchsorted(freq_bins, band[1])
        return low_idx, high_idx

    # Calculate power for each band
    df = freq_bins[1] - freq_bins[0]
    total_power = simpson(psd, dx=df)

    if total_power == 0:
        return {band: 0.0 for band in freqs}

    powers = {}
    for band in freqs:
        low_idx, high_idx = get_band_indices(freq_bins, band)
        band_power = simpson(psd[low_idx:high_idx], dx=df)
        powers[band] = band_power / total_power

    return powers
