import numpy as np
import pywt
from mosaique.features.timefrequency import (
    WaveletCoefficients,
    get_wavelet_scales,
)


import networkx as nx


def _validate_matrix(mat):
    """Validate a connectivity matrix."""
    mat = np.asarray(mat, dtype=float)
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        raise ValueError("Connectivity matrix must be at least 2x2")
    if np.any(np.isnan(mat)):
        raise ValueError("Connectivity matrix contains NaN values")
    if np.any(np.isinf(mat)):
        raise ValueError("Connectivity matrix contains Inf values")
    return mat


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

        if method == "pli":
            # Complex multiplication for cross-spectral density
            csd = np.matmul(band_data_t, band_data_t.conj().transpose(0, 1, 3, 2))
            # Final shape of con: (n_epochs, n_channels, n_channels)
            con = np.abs(np.mean(np.sign(np.imag(csd)), axis=1))

        elif method == "corr":
            # Get magnitude and flatten to array to (epochs, times * freqs, channels)
            band_data_flat = np.abs(band_data_t).reshape(n_epochs, -1, n_channels)

            # Calculate correlation for all epochs at once for
            # final shape of con: (n_epochs, n_channels, n_channels)
            con = np.array([np.corrcoef(x.T) for x in band_data_flat])
        else:
            raise ValueError(f"Method {method} not valid.")

        connectivities[band] = con

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
        # Reshape to (epochs, times, channels, freqs)
        band_data_t = band_data.transpose(0, 3, 1, 2)

        if method == "pli":
            # Complex multiplication for cross-spectral density
            csd = np.matmul(band_data_t, band_data_t.conj().transpose(0, 1, 3, 2))
            # Final shape of con: (n_epochs, n_channels, n_channels)
            con = np.abs(np.mean(np.sign(np.imag(csd)), axis=1))

        elif method == "corr":
            # Get magnitude and flatten array to (epochs, times * freqs, channels)
            n_epochs = band_data_t.shape[0]
            band_data_flat = np.abs(band_data_t).reshape(
                n_epochs, -1, band_data_t.shape[2]
            )

            # Calculate correlation for all epochs at once
            con = np.array([np.corrcoef(x.T) for x in band_data_flat])

        else:
            raise ValueError(f"Method {method} not valid.")

        connectivities[band] = con

    return connectivities


def connected_threshold(con_mat):
    """Threshold a connectivity matrix using the minimum spanning tree.

    Edges below the minimum weight in the maximum spanning tree are set
    to zero, guaranteeing the resulting graph stays connected.

    Parameters
    ----------
    con_mat : np.ndarray
        Symmetric connectivity matrix of shape ``(n_channels, n_channels)``.

    Returns
    -------
    np.ndarray
        Thresholded matrix (same shape), with sub-threshold entries zeroed.
    """
    G = nx.from_numpy_array(con_mat)
    st = nx.minimum_spanning_tree(G)
    edges = [w for _, _, w in st.edges(data="weight")]
    threshold = min(edges)

    thresholded_mat = np.where(con_mat >= threshold, con_mat, 0)

    return thresholded_mat


def threshold_2(con_matrix):
    """Binary-threshold a connectivity matrix while keeping it connected.

    Starts at threshold 0.4 and decreases by 0.01 until the resulting
    binary graph is connected.

    Parameters
    ----------
    con_matrix : np.ndarray
        Symmetric connectivity matrix of shape ``(n_channels, n_channels)``.

    Returns
    -------
    np.ndarray
        Binary matrix (same shape), with 1 where the original value meets
        the threshold and 0 otherwise.
    """
    n_channels, _ = con_matrix.shape
    thresholded_net = np.zeros((n_channels, n_channels))

    # take the highest threshold without getting deconnected network for each 19x19 matrix
    threshold = 0.4
    thresholded_net = np.where(con_matrix >= threshold, 1, 0)
    Graph = nx.from_numpy_array(thresholded_net)
    verify_connected = nx.is_connected(Graph)
    while verify_connected == False:
        # when thresholded_net is disconnected, decrease threshold until connected
        threshold = threshold - 0.01
        thresholded_net = np.where(con_matrix >= threshold, 1, 0)
        Graph = nx.from_numpy_array(thresholded_net)
        verify_connected = nx.is_connected(Graph)

    return thresholded_net


def average_clustering(mat, **kwargs):
    """Average clustering coefficient of the graph.

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    G = nx.from_numpy_array(mat)
    return nx.average_clustering(G)


def average_node_connectivity(mat, **kwargs):
    """Average node connectivity (expected number of node-independent paths).

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    G = nx.from_numpy_array(mat)
    return nx.average_node_connectivity(G)


def average_degree(mat, **kwargs):
    """Average node degree of the graph.

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    G = nx.from_numpy_array(mat)
    degree = nx.degree(G)
    return np.average(degree)


def global_efficiency(mat, **kwargs):
    """Global efficiency of the graph.

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    G = nx.from_numpy_array(mat)
    return nx.global_efficiency(G)


def average_shortest_path_length(mat, **kwargs):
    """Average shortest path length of the graph.

    Parameters
    ----------
    mat : np.ndarray
        Connectivity matrix ``(n_channels, n_channels)``.

    Returns
    -------
    float
    """
    mat = _validate_matrix(mat)
    G = nx.from_numpy_array(mat)
    if not nx.is_connected(G):
        total = 0.0
        total_pairs = 0
        for comp in nx.connected_components(G):
            sg = G.subgraph(comp)
            n = len(sg)
            if n > 1:
                total += nx.average_shortest_path_length(sg) * n * (n - 1)
                total_pairs += n * (n - 1)
        return total / total_pairs if total_pairs > 0 else 0.0
    return nx.average_shortest_path_length(G)
