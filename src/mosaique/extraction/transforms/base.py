"""Abstract base class for pre-extraction transforms."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import datetime

import numpy as np
import polars as pl
from mne import Epochs
from rich.console import Console

from mosaique.config.types import (
    FeatureFunction,
    TransformParams,
)
from mosaique.features.timefrequency import WaveletCoefficients

T = TypeVar("T")


class PreExtractionTransform(ABC, Generic[T]):
    """Defines a pre-extraction transform.

    These transforms take as input an EEG in the form of a numpy array with
    shape (channel, epoch, timepoints)
    """

    # For transforms that return a dict: meaning of the dict key (will be
    # stored in the final feature dataframe)
    key: str

    def __init__(
        self,
        transform: TransformParams,
        num_workers: int = 1,
        debug=False,
        console=Console(),
    ) -> None:
        """See the definition for TransformParams. These are parsed from a
        YAML config file specifying the pre-extraction transform name, function,
        and parameters"""
        self._name = transform.name
        self._function = (
            transform.function
        )  # Must conform to TransformFunction protocol
        self._params = transform.params
        self._cached_coeffs: WaveletCoefficients = {}
        self.num_workers = num_workers
        self.debug = debug
        self.console = console

    def complete_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Complete the dataframe with informations from _params.params"""
        df = df.with_columns(pre_transform=pl.lit(self._name))
        for k, v in self._params.items():
            if k == "freqs":  # freqs is already stored
                continue
            df = df.with_columns(pl.lit(v).alias(k))
        return df

    def _get_times(self, eeg: Epochs) -> np.ndarray:
        epoch_start_times = eeg.events[:, 0] / eeg.info["sfreq"]
        eeg_start_time = eeg.info["meas_date"]
        epoch_times = np.array(
            [
                np.datetime64(datetime.timedelta(seconds=t) + eeg_start_time)
                for t in epoch_start_times
            ]
        )
        return epoch_times

    @abstractmethod
    def transform(self, eeg: Epochs) -> T:
        """Apply pre-extraction transformation to EEG"""

    @abstractmethod
    def extract_feature(
        self,
        transformed_eeg: T,
        feature_function: FeatureFunction,
        **feature_params,
    ) -> pl.DataFrame:
        """Apply feature extraction function to transformed EEG"""
