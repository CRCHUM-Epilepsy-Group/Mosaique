"""Simple (identity) pre-extraction transform."""

import numpy as np
import polars as pl
from mne import Epochs

from mosaique.extraction.transforms.base import PreExtractionTransform
from mosaique.utils.eeg_helpers import get_event_list
from mosaique.utils.toolkit import parallelize_over_axis


class SimpleTransform(PreExtractionTransform):
    """Identity transform — passes raw EEG data to feature functions.

    No signal decomposition is applied.  Feature functions receive the raw
    ``(epochs, channels, times)`` array sliced along the last axis.

    If the feature function returns a ``dict`` (e.g. :func:`band_power`
    returning one value per frequency band), each key becomes a separate
    row in the output DataFrame with a ``freqs`` label.  Otherwise a single
    scalar per (epoch, channel) is expected.
    """

    key = "freqs"
    events: list[str]
    ch_names: list[str]

    def transform(self, eeg: Epochs) -> np.ndarray:
        eeg_data = eeg.get_data()
        self.events = get_event_list(eeg)
        self.ch_names = eeg.ch_names
        self.times = self._get_times(eeg)
        self.sfreq = eeg.info["sfreq"]
        return eeg_data

    def extract_feature(
        self,
        transformed_eeg: np.ndarray,
        feature_function,
        **feature_params,
    ) -> pl.DataFrame:
        """Extract features for transforms that return a dictionary"""

        values = parallelize_over_axis(
            feature_function,
            transformed_eeg,
            axis=-1,
            num_workers=self.num_workers,
            debug=self.debug,
            sfreq=self.sfreq,
            disable_progress=True,
            **feature_params,  # type: ignore
        ).flatten()

        if any(isinstance(x, dict) for x in values):
            # If function returns dict (e.g.: band power)
            df = pl.DataFrame([{str(k): v for k, v in d.items()} for d in values]).melt(
                variable_name=self.key, value_name="value"
            )
            n_var = len(values[0].keys())
            # Ordering is freq:epoch:channel
            index_df = pl.DataFrame(
                {
                    "epoch": np.tile(np.repeat(self.events, len(self.ch_names)), n_var),
                    "timestamp": np.tile(
                        np.repeat(self.times, len(self.ch_names)), n_var
                    ),
                    "channel": np.tile(self.ch_names, len(self.events) * n_var),
                }
            )
            df = pl.concat([index_df, df], how="horizontal")
        else:
            df = pl.DataFrame(
                {
                    "epoch": np.repeat(self.events, len(self.ch_names)),
                    "timestamp": np.repeat(self.times, len(self.ch_names)),
                    "channel": np.tile(self.ch_names, len(self.events)),
                    "value": values.tolist(),
                }
            )

        return df
