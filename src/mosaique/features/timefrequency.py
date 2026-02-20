"""Wavelet-based time-frequency decomposition functions.

Provides DWT, WPD, and CWT decomposition of EEG signals into frequency
bands.  These functions are typically used as *transform functions* in
the extraction pipeline (referenced in YAML configs under ``frameworks``).

Types
-----
``FrequencyBand``
    Alias for ``tuple[float, float]`` — a ``(low_hz, high_hz)`` pair.

``WaveletCoefficients``
    Alias for ``dict[FrequencyBand, np.ndarray]`` — wavelet coefficients
    keyed by frequency band.
"""

from rich.console import Console
from mosaique.utils.toolkit import calculate_over_pool
import pywt
import numpy as np

FrequencyBand = tuple[float, float]
"""A ``(low_hz, high_hz)`` frequency band specification."""

WaveletCoefficients = dict[FrequencyBand, np.ndarray]
"""Wavelet coefficients keyed by frequency band."""


def get_wavelet_scales(
    f_min: float, f_max: float, wavelet: str, sfreq: float, n_scales: int = 100
) -> np.ndarray:
    """Calculate appropriate wavelet scales for frequency range."""
    scales = np.logspace(1, 8, num=n_scales, base=2)
    frequencies = pywt.scale2frequency(wavelet, scales) * sfreq
    return scales[(f_max >= frequencies) & (frequencies >= f_min)]


def get_wpd_freqs(max_level: int, sfreq: float) -> list[FrequencyBand]:
    """Get frequency bands of Wavelet Packet decomposition"""
    wpd_freqs = [
        (
            n * sfreq / 2**max_level,
            (n + 1) * sfreq / 2**max_level,
        )
        for n in range(2 ** (max_level - 1))
    ]

    return wpd_freqs


def get_dwt_freqs(max_level: int, sfreq: float) -> list[FrequencyBand]:
    """Get frequency bands of Discrete Wavelet Transform decomposition."""
    dwt_freqs: list[FrequencyBand] = [
        (
            (sfreq / 2 ** (n + 2) if n != max_level else 0),
            sfreq / 2 ** (n + 1),
        )
        for n in reversed(range(max_level + 1))
    ]

    return dwt_freqs


def wpd_eeg(
    eeg: np.ndarray,
    wavelet: str = "sym5",
    max_level: int = 6,
    sfreq: float = 200,
    **kwargs,
) -> tuple[WaveletCoefficients, np.ndarray]:
    """Wavelet Packet Decomposition (WPD) of EEG signals.

    Decomposes the signal along the last axis into frequency sub-bands using
    a full wavelet packet tree.  Coefficients are downsampled at each level;
    the reconstructed signal retains the original length.

    Parameters
    ----------
    eeg : np.ndarray
        EEG array with the last dimension as time (e.g. ``(epochs, channels,
        times)``).
    wavelet : str
        PyWavelets wavelet name (default ``"sym5"``).
    max_level : int
        Maximum decomposition level (default 6).
    sfreq : float
        Sampling frequency in Hz (default 200).

    Returns
    -------
    tuple[WaveletCoefficients, np.ndarray]
        ``(coefficients, reconstructed)`` where *coefficients* maps each
        frequency band to its (downsampled) coefficient array, and
        *reconstructed* has shape ``(..., n_bands, times)`` with the
        band-reconstructed signals.
    """

    wp = pywt.WaveletPacket(eeg, wavelet, mode="smooth", maxlevel=max_level, axis=-1)
    all_nodes = [node.path for node in wp.get_level(max_level, "freq")]

    # Remove all frequencies above nyquist
    nyquist_node = int(len(all_nodes) / 2)
    nodes = all_nodes[:nyquist_node]

    reconstr = np.zeros(shape=(eeg.shape + (len(nodes),)))
    # Iterate through decompositions at last level
    for i, node in enumerate(nodes):
        new_wp = pywt.WaveletPacket(
            np.zeros(eeg.shape), wavelet, mode="smooth", maxlevel=max_level, axis=-1
        )
        new_wp[node] = wp[node].data
        reconstr[..., i] = new_wp.reconstruct()

    # Dictionary as each node has different shape (downsampled)
    freqs = get_wpd_freqs(max_level, sfreq)
    coeff = {f: wp[node].data for f, node in zip(freqs, nodes)}

    # Put timepoints on last axis
    return coeff, reconstr.swapaxes(-1, -2)


def dwt_eeg(
    eeg: np.ndarray,
    wavelet: str = "sym5",
    max_level: int = 5,
    sfreq: float = 200,
    **kwargs,
) -> tuple[WaveletCoefficients, np.ndarray]:
    """Discrete Wavelet Transform (DWT) of EEG signals.

    Decomposes the signal along the last axis into approximation and detail
    coefficients at each level.  Coefficients are downsampled; the
    reconstructed signal retains the original length.

    Parameters
    ----------
    eeg : np.ndarray
        EEG array with the last dimension as time.
    wavelet : str
        PyWavelets wavelet name (default ``"sym5"``).
    max_level : int
        Number of decomposition levels (default 5).
    sfreq : float
        Sampling frequency in Hz (default 200).

    Returns
    -------
    tuple[WaveletCoefficients, np.ndarray]
        ``(coefficients, reconstructed)`` — see :func:`wpd_eeg`.
    """

    dwt = pywt.wavedec(eeg, wavelet, mode="smooth", level=max_level, axis=-1)
    coeff_names = [f"cA{max_level}"] + [
        f"cD{n + 1}" for n in reversed(range(max_level))
    ]

    # Dictionary as each node has different shape (downsampled)
    coeff = {name: data for (name, data) in zip(coeff_names, dwt)}

    reconstr = np.zeros(shape=(eeg.shape + (len(coeff_names),)))
    # Iterate through every node of the decomposition
    for i, node in enumerate(coeff_names):
        reconstr_coeffs = {
            c: (np.zeros_like(data) if c != node else data) for c, data in coeff.items()
        }
        reconstr[..., i] = pywt.waverec(
            list(reconstr_coeffs.values()),
            wavelet,
            mode="smooth",
            axis=-1,
        )[..., : eeg.shape[-1]]  # Sometimes, waverec needs trimming

    freqs = get_dwt_freqs(max_level, sfreq)
    coeff = {f: coeff for f, coeff in zip(freqs, coeff.values())}

    # Put timepoints on last axis
    return coeff, reconstr.swapaxes(-1, -2)


def cwt_eeg(
    eeg: np.ndarray,
    freqs: list[FrequencyBand] | FrequencyBand,
    wavelet: str = "cmor1.5-1.0",
    sfreq: float = 200,
    method: str = "fft",
    num_workers: int = 1,
    skip_reconstr: bool = False,
    skip_complex: bool = False,
    debug: bool = False,
    disable_progress: bool = False,
    console: Console = Console(),
    **kwargs,
) -> tuple[WaveletCoefficients, np.ndarray]:
    """Continuous Wavelet Transform (CWT) of EEG signals.

    Computes CWT for each epoch in parallel, then groups the resulting
    wavelet scales into the requested frequency bands.

    Parameters
    ----------
    eeg : np.ndarray
        EEG array of shape ``(epochs, channels, times)``.
    freqs : list[FrequencyBand] | FrequencyBand
        Target frequency bands as ``[(low, high), ...]`` or ``(low, high)``.
    wavelet : str
        Complex wavelet name (default ``"cmor1.5-1.0"``).
    sfreq : float
        Sampling frequency in Hz (default 200).
    method : str
        ``"fft"`` (fast, default) or ``"conv"`` (direct convolution).
    num_workers : int
        Number of parallel processes for epoch-level computation.
    skip_reconstr : bool
        If ``True``, skip the band-reconstructed signal (saves memory).
    skip_complex : bool
        If ``True``, store magnitudes instead of complex coefficients.
    debug : bool
        Run in single-process mode when ``True``.
    disable_progress : bool
        Suppress the progress bar.
    console : rich.console.Console
        Console for progress output.

    Returns
    -------
    tuple[WaveletCoefficients, np.ndarray]
        ``(coefficients, reconstructed)`` where *coefficients* maps each
        band to an array of shape ``(epochs, channels, n_scales, times)``
        and *reconstructed* has shape ``(epochs, channels, n_bands, times)``.
    """
    freqs = [freqs] if isinstance(freqs, tuple) else freqs

    # Get appropriate scales for all frequency bands
    f_min = min(f for band in freqs for f in band)
    f_max = max(f for band in freqs for f in band)
    scales = get_wavelet_scales(f_min, f_max, wavelet, sfreq)

    # Compute CWT
    n_epochs, n_channels, n_times = eeg.shape

    if skip_complex:
        dtype = np.float32
    else:
        dtype = np.complex128

    def get_cwt(epoch_idx):
        epoch = eeg[epoch_idx].copy()
        coef, wt_frequencies = pywt.cwt(
            epoch, scales, wavelet, 1 / sfreq, method=method
        )
        if skip_complex:
            coef = np.abs(coef)
        # Process bands immediately for this epoch
        epoch_results = {}
        for band in freqs:
            band_mask = (wt_frequencies >= band[0]) & (wt_frequencies <= band[1])
            band_data = coef[band_mask, ...].astype(dtype=dtype)
            # Shape (channel, freqs, time)
            epoch_results[band] = band_data.transpose(1, 0, 2)

        return epoch_results

    # Parallel CWT computation
    results = calculate_over_pool(
        get_cwt,
        range(n_epochs),
        n_jobs=n_epochs,
        chunksize=100,
        num_workers=num_workers,
        debug=debug,
        task_name="Computing CWT",
        disable_progress=disable_progress,
        console=console,
    )

    # Get first epoch result to determine shapes
    first_result = results[0]
    coeff = {
        band: np.zeros((n_epochs,) + first_result[band].shape, dtype=dtype)
        for band in freqs
    }
    reconstr = np.zeros(shape=(eeg.shape + (len(freqs),)))

    # Fill pre-allocated arrays
    for epoch_idx, epoch_results in enumerate(results):
        for i, band in enumerate(freqs):
            coeff[band][epoch_idx] = epoch_results[band]
            if not skip_reconstr:
                reconstr[epoch_idx, ..., i] = np.abs(epoch_results[band]).mean(axis=1)

    return coeff, reconstr.swapaxes(-1, -2)
