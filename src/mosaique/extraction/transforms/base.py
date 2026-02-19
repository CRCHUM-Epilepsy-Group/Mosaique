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
    """Abstract base class for pre-extraction transforms.

    A pre-extraction transform sits between the raw MNE ``Epochs`` and the
    scalar feature functions.  It converts the EEG into an intermediate
    representation (e.g. wavelet coefficients, connectivity matrices) that
    feature functions can then operate on.

    Subclasses must implement:

    * :meth:`transform` – convert ``mne.Epochs`` into a transformed
      representation of type ``T``.
    * :meth:`extract_feature` – apply a single feature function to the
      transformed data and return a :class:`polars.DataFrame`.

    They should also set the class attribute :attr:`key` to the column name
    used to label sub-bands or decomposition levels in the output DataFrame
    (e.g. ``"freqs"``).

    Creating a custom transform
    ---------------------------
    1. Subclass ``PreExtractionTransform`` and implement ``transform`` and
       ``extract_feature``.
    2. Register the new class in
       :data:`~mosaique.extraction.transforms.TRANSFORM_REGISTRY` so that
       :class:`~mosaique.extraction.extractor.FeatureExtractor` can look it
       up by name.

    Example::

        from mosaique.extraction.transforms.base import PreExtractionTransform
        from mosaique.extraction.transforms import TRANSFORM_REGISTRY

        class MyCustomTransform(PreExtractionTransform[np.ndarray]):
            key = "custom_key"

            def transform(self, eeg):
                # Return the transformed representation
                ...

            def extract_feature(self, transformed_eeg, feature_function, **kw):
                # Apply feature_function and return a polars DataFrame
                ...

        # Register so the YAML config can reference it by name
        TRANSFORM_REGISTRY["my_custom"] = MyCustomTransform
    """

    # Column name used in the output DataFrame to label decomposition
    # levels or frequency bands (e.g. "freqs").
    key: str

    def __init__(
        self,
        transform: TransformParams,
        num_workers: int = 1,
        debug=False,
        console=Console(),
    ) -> None:
        """Initialise the transform from parsed configuration.

        Parameters
        ----------
        transform : TransformParams
            Parsed transform configuration (name, function, parameters).
        num_workers : int
            Number of parallel workers.
        debug : bool
            If ``True``, run in single-process mode for easier debugging.
        console : rich.console.Console
            Console used for progress output.
        """
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
        """Convert MNE Epochs into the transformed representation.

        Parameters
        ----------
        eeg : mne.Epochs
            Epoched EEG data.

        Returns
        -------
        T
            Transformed data whose type depends on the concrete subclass
            (e.g. ``dict[FrequencyBand, np.ndarray]`` for wavelet transforms,
            ``np.ndarray`` for the identity transform).
        """

    @abstractmethod
    def extract_feature(
        self,
        transformed_eeg: T,
        feature_function: FeatureFunction,
        **feature_params,
    ) -> pl.DataFrame:
        """Apply a feature function to the transformed EEG.

        Parameters
        ----------
        transformed_eeg : T
            Output of :meth:`transform`.
        feature_function : FeatureFunction
            Callable that computes a scalar (or dict of scalars) from a 1-D
            signal or a connectivity matrix.
        **feature_params
            Extra keyword arguments forwarded to *feature_function*.

        Returns
        -------
        polars.DataFrame
            Long-format table with at least columns ``epoch``, ``channel``
            (when applicable), and ``value``.
        """
