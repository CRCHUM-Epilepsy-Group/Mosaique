"""Spectral connectivity computation from EEG or wavelet coefficients."""

import numpy as np
import pywt
from mosaique.features.timefrequency import (
    WaveletCoefficients,
    get_wavelet_scales,
)

# Re-export graph metrics for backward compatibility
from mosaique.features.graph_metrics import (  # noqa: F401
    _validate_matrix,
    average_clustering,
    average_degree,
    average_node_connectivity,
    average_shortest_path_length,
    binary_threshold,
    connected_threshold,
    global_efficiency,
)


def _compute_pli(band_data_t: np.ndarray) -> np.ndarray:
    """Compute Phase Lag Index from transposed band data.

    Parameters
    ----------
    band_data_t : np.ndarray
        Complex wavelet coefficients of shape
        ``(n_epochs, n_times, n_channels, n_freqs)``.

    Returns
    -------
    np.ndarray
        PLI connectivity matrices of shape ``(n_epochs, n_channels, n_channels)``.
    """
    csd = np.matmul(band_data_t, band_data_t.conj().transpose(0, 1, 3, 2))
    return np.abs(np.mean(np.sign(np.imag(csd)), axis=1))


def _compute_correlation(band_data_t: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation from transposed band data.

    Parameters
    ----------
    band_data_t : np.ndarray
        Complex wavelet coefficients of shape
        ``(n_epochs, n_times, n_channels, n_freqs)``.

    Returns
    -------
    np.ndarray
        Correlation matrices of shape ``(n_epochs, n_channels, n_channels)``.
    """
    n_epochs = band_data_t.shape[0]
    n_channels = band_data_t.shape[2]
    band_data_flat = np.abs(band_data_t).reshape(n_epochs, -1, n_channels)
    return np.array([np.corrcoef(x.T) for x in band_data_flat])


_CONNECTIVITY_METHODS = {
    "pli": _compute_pli,
    "corr": _compute_correlation,
}


def _compute_connectivity(
    band_data_t: np.ndarray, method: str
) -> np.ndarray:
    """Dispatch connectivity computation to the appropriate method."""
    if method not in _CONNECTIVITY_METHODS:
        raise ValueError(
            f"Method {method!r} not valid; choose from {list(_CONNECTIVITY_METHODS)}"
        )
    return _CONNECTIVITY_METHODS[method](band_data_t)


def cwt_spectral_connectivity(
    eeg: np.ndarray,
    *,  # arguments are keywords only
    freqs: list[tuple[float, float]] | tuple[float, float],
    method: str = "pli",
    wavelet: str = "cmor1.5-1.0",
    sfreq: float = 200,
) -> dict[tuple[float, float], np.ndarray]:
    """Calculate spectral connectivity matrices for EEG data.

    Uses continuous wavelet transform (CWT) to estimate cross-spectral
    density, then derives a connectivity measure per frequency band.

    Parameters
    ----------
    eeg : np.ndarray
        Array of shape ``(epochs, channels, times)``.
    freqs : list[tuple[float, float]] | tuple[float, float]
        Frequency bands as ``[(low, high), ...]`` or a single ``(low, high)``.
    method : str
        Connectivity method: ``"pli"`` (phase lag index) or ``"corr"``
        (Pearson correlation of wavelet magnitude).
    wavelet : str
        Wavelet name for CWT (default ``"cmor1.5-1.0"``).
    sfreq : float
        Sampling frequency in Hz (default 200).

    Returns
    -------
    dict[tuple[float, float], np.ndarray]
        Mapping from frequency band ``(low, high)`` to connectivity matrices
        of shape ``(n_epochs, n_channels, n_channels)``.
    """
    freqs = [freqs] if isinstance(freqs, tuple) else freqs

    # Calculate wavelet scales
    f_min, f_max = (
        min(f for band in freqs for f in band),
        max(f for band in freqs for f in band),
    )
    scales = get_wavelet_scales(f_min, f_max, wavelet, sfreq)

    # Reshape to (epochs * channels, times)
    n_epochs, n_channels, n_times = eeg.shape
    eeg_flat = eeg.reshape(-1, n_times)

    # Compute wavelet transform
    coef, frequencies = pywt.cwt(eeg_flat, scales, wavelet, 1 / sfreq)

    # Reshape back to (epochs, channels, freqs, times)
    cwt_eeg = coef.reshape(len(scales), n_epochs, n_channels, n_times)
    cwt_eeg = cwt_eeg.transpose(1, 2, 0, 3)  # (epochs, channels, freqs, times)

    # Calculate connectivity for each band
    connectivities = {}
    for band in freqs:
        band_mask = (frequencies >= band[0]) & (frequencies <= band[1])
        band_data = cwt_eeg[..., band_mask, :]
        # Reshape to (epochs, times, channels, freqs)
        band_data_t = band_data.transpose(0, 3, 1, 2)
        connectivities[band] = _compute_connectivity(band_data_t, method)

    return connectivities


def connectivity_from_coeff(
    coefficients: WaveletCoefficients,
    method: str = "pli",
    **kwargs,
) -> dict[tuple[float, float], np.ndarray]:
    """Calculate spectral connectivity from pre-computed wavelet coefficients.

    This avoids recomputing the CWT when coefficients are already cached by
    the extraction pipeline.

    Parameters
    ----------
    coefficients : WaveletCoefficients
        Dict mapping frequency bands to complex coefficient arrays of shape
        ``(epochs, channels, freqs, times)``.
    method : str
        Connectivity method: ``"pli"`` or ``"corr"`` (see
        :func:`cwt_spectral_connectivity` for details).

    Returns
    -------
    dict[tuple[float, float], np.ndarray]
        Connectivity matrices of shape ``(n_epochs, n_channels, n_channels)``
        per frequency band.
    """
    connectivities = {}

    for band, band_data in coefficients.items():
        band_data_t = band_data.transpose(0, 3, 1, 2)
        connectivities[band] = _compute_connectivity(band_data_t, method)

    return connectivities
