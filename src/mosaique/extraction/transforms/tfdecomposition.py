"""Time-frequency decomposition pre-extraction transform."""

import numpy as np
import polars as pl
from mne import Epochs

from mosaique.extraction.transforms.base import PreExtractionTransform
from mosaique.features.timefrequency import FrequencyBand, WaveletCoefficients
from mosaique.utils.eeg_helpers import get_event_list
from mosaique.utils.toolkit import parallelize_over_axis


class TFDecompositionTransform(PreExtractionTransform):
    """Pre-extraction transform with DWT or WPD"""

    key = "freqs"
    events: list[str]
    ch_names: list[str]

    def transform(self, eeg: Epochs) -> dict[FrequencyBand, np.ndarray]:
        self.sfreq = eeg.info["sfreq"]
        coeffs, _ = self._function(  # type: ignore
            eeg.get_data(),
            sfreq=self.sfreq,
            num_workers=self.num_workers,
            debug=self.debug,
            disable_progress=True,
            **self._params,
        )
        self.events = get_event_list(eeg)
        self.ch_names = eeg.ch_names
        self.times = self._get_times(eeg)

        self._cached_coeffs.update(coeffs)  # type: ignore

        # Check for complex numbers and keep real part if needed
        # + downsample signal according to band (mimicking WPD)
        if self._name == "cwt":
            coeffs = self.simplify_cwt_coeffs(coeffs)
        return coeffs

    def simplify_cwt_coeffs(self, coeffs: WaveletCoefficients) -> WaveletCoefficients:
        for band, coeff in coeffs.items():
            real_coeff = np.real(coeff).mean(axis=2)

            # Calculate downsampling factor
            level = int(np.floor(np.log2(self.sfreq / (2 * band[1]))))
            downsample_factor = 2 * level

            if downsample_factor > 1:
                downsampled_coeff = real_coeff[..., ::downsample_factor]

                coeffs[band] = downsampled_coeff
        return coeffs

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
        self, eeg, feature_function, **feature_params
    ) -> pl.DataFrame:
        values = parallelize_over_axis(
            feature_function,
            eeg,
            axis=-1,
            num_workers=self.num_workers,
            debug=self.debug,
            sfreq=self.sfreq,
            disable_progress=True,
            **feature_params,  # type: ignore
        ).flatten()
        df = pl.DataFrame(
            {
                "epoch": np.repeat(self.events, len(self.ch_names)),
                "timestamp": np.repeat(self.times, len(self.ch_names)),
                "channel": np.tile(self.ch_names, len(self.events)),
                "value": values.tolist(),
            }
        )
        return df
