import numpy as np
import pywt
from mosaique.features.timefrequency import (
    WaveletCoefficients,
    get_wavelet_scales,
)


import networkx as nx


def cwt_spectral_connectivity(
    eeg: np.ndarray,
    *,  # arguments are keywords only
    freqs: list[tuple[float, float]] | tuple[float, float],
    method: str = "pli",
    wavelet: str = "cmor1.5-1.0",
    sfreq: float = 200,
) -> dict[tuple[float, float], np.ndarray]:
    """Calculate spectral connectivity matrices for EEG data.

    Args:
        eeg: Array of shape (epochs, channels, times)
        freqs: Frequency bands as [(low, high), ...] or (low, high)
        method: Connectivity method ('pli' or 'corr')
        wavelet: Wavelet to use for transform
        sfreq: Recording sampling rate in Hz

    Returns:
        Dict mapping frequency bands to connectivity matrices
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
    """Calculate spectral connectivity matrices from wavelet coefficients.

    Args:
        coefficients: Dict mapping frequency bands to coefficient arrays of shape
                     (epochs, channels, times)
        method: Connectivity method ('pli' or 'corr')

    Returns:
        Dict mapping frequency bands to connectivity matrices
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
    G = nx.from_numpy_array(con_mat)
    st = nx.minimum_spanning_tree(G)
    edges = [w for _, _, w in st.edges(data="weight")]
    threshold = min(edges)

    thresholded_mat = np.where(con_mat >= threshold, con_mat, 0)

    return thresholded_mat


def threshold_2(con_matrix):
    """con_matrix input should be 4D array containing connectivity matrices
    input shape of n_bands,n_epochs, n_channels, n_channels
    method should be either "binary" or "weighted" (string)
    threshold should be a float between 0 and 1, default is "auto" which is computed over each epoch
    return a thresholded connectivity matric of same shape as input
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


def average_clustering(mat):
    G = nx.from_numpy_array(mat)
    average_clustering = nx.average_clustering(G)
    return average_clustering


def average_node_connectivity(mat):
    G = nx.from_numpy_array(mat)
    average_node_connectivity = nx.average_node_connectivity(G)
    return average_node_connectivity


def average_degree(mat):
    G = nx.from_numpy_array(mat)
    degree = nx.degree(G)
    average_degree = np.average(degree)
    return average_degree


def global_efficiency(mat):
    G = nx.from_numpy_array(mat)
    global_efficiency = nx.global_efficiency(G)
    return global_efficiency


def average_shortest_path_length(mat):
    G = nx.from_numpy_array(mat)
    average_shortest_path_length = nx.average_shortest_path_length(G)
    return average_shortest_path_length
