"""Connectivity pre-extraction transform."""

import numpy as np
import polars as pl

from mosaique.extraction.eegdata import EegData
from mosaique.extraction.transforms.base import PreExtractionTransform
from mosaique.features.timefrequency import FrequencyBand, cwt_eeg
from mosaique.utils.toolkit import calculate_over_pool


class ConnectivityTransform(PreExtractionTransform):
    """Spectral connectivity transform.

    First filters the EEG into frequency bands using CWT (reusing cached
    coefficients when available), then computes connectivity matrices
    (e.g. PLI or correlation) between channels for each band.

    Feature functions receive a connectivity matrix of shape
    ``(n_epochs, n_channels, n_channels)`` per frequency band.  They should
    return either a single scalar per epoch (graph-level metric) or a
    1-D array of length ``n_channels`` (node-level metric).

    The output DataFrame includes a ``freqs`` column identifying the
    ``(low, high)`` frequency band.
    """

    key = "freqs"
    events: list[str]
    ch_names: list[str]

    def _make_cache_tag(self) -> tuple:
        """Build a hashable tag from the CWT computation parameters."""
        return (
            self._params.get("wavelet", "cmor1.5-1.0"),
            self._params.get("sfreq", 200),
            self._params.get("n_scales", 100),
        )

    def transform(self, eeg: EegData) -> dict[FrequencyBand, np.ndarray]:
        eeg_data = eeg.data
        self.sfreq = eeg.sfreq

        tag = self._make_cache_tag()
        # Reuse cached coefficients only when the computation parameters match
        # Filter out params that are for the connectivity function, not CWT.
        # "method" conflicts: cwt_eeg expects "fft"/"conv", connectivity uses "pli"/"corr".
        cwt_params = {k: v for k, v in self._params.items() if k != "method"}
        try:
            bands_cached = all(
                f in self._cached_coeffs for f in self._params["freqs"]
            )
            if bands_cached and self._cache_tag == tag:
                coeffs = self._cached_coeffs
            else:
                coeffs, _ = cwt_eeg(eeg_data, **cwt_params)
        except KeyError:
            coeffs, _ = cwt_eeg(eeg_data, **cwt_params)

        self._cached_coeffs = coeffs
        self._cache_tag = tag

        con_matrices = self._function(
            coeffs, num_workers=self.num_workers, debug=self.debug, **self._params
        )
        self.events = eeg.event_labels
        self.ch_names = eeg.ch_names
        self.times = eeg.timestamps
        return con_matrices

    def extract_feature(
        self,
        transformed_eeg: dict[FrequencyBand, np.ndarray],
        feature_function,
        **feature_params,
    ) -> pl.DataFrame:
        """Extract features for transforms that return a dictionary"""

        features_at_each_level = []

        # Each key is a level, each value is an eeg as a numpy array
        for level, eeg in transformed_eeg.items():
            features = self._extract_for_single_level(
                eeg, feature_function, **feature_params
            )
            features = features.with_columns(
                pl.lit(str((level[0], level[1]))).alias(self.key)
            )
            features_at_each_level.append(features)

        df = pl.concat(features_at_each_level)
        return df

    def _extract_for_single_level(
        self, con_mat, feature_function, **feature_params
    ) -> pl.DataFrame:
        def process_con_mat(con_mat):
            return feature_function(con_mat, **feature_params)

        con_mat_list = [con_mat[i] for i in range(con_mat.shape[0])]
        values = np.array(
            calculate_over_pool(
                process_con_mat,
                con_mat_list,
                num_workers=self.num_workers,
                debug=self.debug,
                chunksize=8,
                disable_progress=True,
                **feature_params,
            )
        )
        match values.ndim:
            case 1:
                # Only one value per con_mat
                df = pl.DataFrame(
                    {
                        "epoch": self.events,
                        "timestamp": self.times,
                        "value": values.tolist(),
                    }
                )
            case 2:
                # One value per channel
                df = pl.DataFrame(
                    {
                        "epoch": np.repeat(self.events, len(self.ch_names)),
                        "timestamp": np.repeat(self.times, len(self.ch_names)),
                        "channel": np.tile(self.ch_names, len(self.events)),
                        "value": values.flatten().tolist(),
                    }
                )
            case _:
                raise ValueError(
                    f"Features from connectivity matrix has an unrecognized shape of {values.shape}"
                )
        return df
