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


def shannon_entropy(U, **kwargs):
    probabilities = [n_x / len(U) for x, n_x in collections.Counter(U).items()]
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


def rescaled_range(X):
    mean = np.mean(X)
    deviations = X - mean
    Z = np.cumsum(deviations)
    R = max(Z) - min(Z)
    S = np.std(X, ddof=1)

    if R == 0 or S == 0:
        return 0

    return R / S


####################################################################
# Feature functions
####################################################################


# Entropies


def approximate_entropy(U, m=3, r=0.2, **kwargs):
    """
    https://gist.github.com/DustinAlandzes/a835909ffd15b9927820d175a48dee41
    """
    N = U.shape[0]
    r *= np.std(U, axis=0)

    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i : i + m] for i in range(int(z))])
        X = np.repeat(x[:, np.newaxis], 1, axis=2)
        C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
        return np.log(C).sum() / z

    phi1 = _phi(m)
    phi0 = _phi(m + 1)
    apen = abs(phi0 - phi1)

    return apen


def sample_entropy(X, m=2, r=0.2, **kwargs):
    N = len(X)
    sigma = np.std(X)
    tolerance = sigma * r

    # Create embedding vectors
    matches = np.zeros((N, m + 1))
    for i in range(m + 1):
        matches[0 : (N - i), i] = X[i:]

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


def spectral_entropy(U, sfreq=200, normalize=True, **kwargs):
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
    # Compute and normalize power spectrum
    _, psd = periodogram(U, sfreq, nfft=None, axis=-1)
    psd_norm = psd / psd.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        se = -(psd_norm * np.log2(psd_norm)).sum()
    if normalize:
        se /= np.log2(psd_norm.size)
    return se


def permutation_entropy(data, k=3, **kwargs):
    _, perm_probabilities = ordinal_distribution(data, dx=k)
    permen = shannon_entropy(perm_probabilities)

    return permen


def fuzzy_entropy(X, m=2, r=0.2, n=2, **kwargs):
    """
    Adapted from https://github.com/MattWillFlood/EntropyHub.jl/blob/master/src/_FuzzEn.jl
    """

    def fuzzy_fun(dist, gradient, width):
        y = np.exp(-(dist**gradient) / width)
        return y

    N = len(X)
    sigma = np.std(X)
    tolerance = sigma * r

    # Create embedding vectors
    matches = np.zeros((N, m + 1))
    for i in range(m + 1):
        matches[0 : (N - i), i] = X[i:]

    # Remove empty templates
    matches = matches[:-m, :]

    dist_m = pdist(matches[:, :m], metric="chebyshev")
    ps_m = fuzzy_fun(dist_m, n, tolerance).sum()

    dist_m1 = pdist(matches, metric="chebyshev")
    ps_m1 = fuzzy_fun(dist_m1, gradient=n, width=tolerance).sum()

    fuzzen = np.log(ps_m) - np.log(ps_m1)

    return fuzzen


# Non linear markers


def corr_dim(X, embed_dim=2, rvals=None, **kwargs):
    """
    https://github.com/CSchoel/nolds/blob/2fd45ecd8d36382de455e6b662a6b6a6e6ed32e7/nolds/measures.py#L932
    """
    if sum(X) == 0:
        return np.nan

    N = len(X)

    if rvals is None:
        sd = np.std(X, ddof=1)
        rvals = logarithmic_r(0.1 * sd, 0.5 * sd, 1.03)

    delay_embed = np.zeros((N, embed_dim))
    for i in range(embed_dim):
        delay_embed[0 : (N - i), i] = X[i:]
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


def line_length(X, **kwargs):
    """
    References
    ----------
    .. [1] Esteller, R. et al. (2001). Line length: an efficient feature for
           seizure onset detection. In Engineering in Medicine and Biology
           Society, 2001. Proceedings of the 23rd Annual International
           Conference of the IEEE (Vol. 2, pp. 1707-1710). IEEE.
    """

    diff = np.diff(X)
    length = np.mean(np.abs(diff))

    return length


# Linear markers


def peak_alpha(X, sfreq=200, **kwargs):
    psd, freqs = psd_array_multitaper(X, sfreq, normalization="length")  # type: ignore
    alpha_band = np.where((freqs >= 8) & (freqs <= 13))[0]
    peak_alpha_ind = np.argmax(psd[alpha_band])
    peak_alpha = freqs[alpha_band][peak_alpha_ind]

    return peak_alpha


def hurst_exp(X, min_window=10, max_window=None, **kwargs):
    """
    https://github.com/Mottl/hurst/blob/5ca5005485a679e6ce11a2769c948915ae27b2da/hurst/__init__.py#L22
    """

    # Get windows sizes in log scale
    max_window = max_window or len(X) - 1
    window_sizes = list(
        map(
            lambda x: int(10**x),
            np.arange(math.log10(min_window), math.log10(max_window), 0.25),
        )
    )
    window_sizes.append(len(X))

    RS = []
    for w in window_sizes:
        rs = []
        for start in range(0, len(X), w):
            if (start + w) > len(X):
                break
            _ = rescaled_range(X[start : start + w])
            if _ != 0:
                rs.append(_)
        RS.append(np.mean(rs))

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

    powers = {}
    for band in freqs:
        low_idx, high_idx = get_band_indices(freq_bins, band)
        band_power = simpson(psd[low_idx:high_idx], dx=df)
        powers[band] = band_power / total_power

    return powers
