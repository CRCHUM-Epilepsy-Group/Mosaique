"""EegData container and EpochsLike protocol for MNE-independent extraction."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EpochsLike(Protocol):
    """Structural protocol matching the mne.Epochs subset used by this library.

    Any object that provides these attributes and methods can be passed to
    :meth:`~mosaique.extraction.extractor.FeatureExtractor.extract_feature`
    without importing MNE.
    """

    ch_names: list[str]
    events: np.ndarray  # shape (n_epochs, 3); column 2 holds event codes
    event_id: dict[str, int]

    def get_data(self) -> np.ndarray:
        """Return data array of shape (n_epochs, n_channels, n_times)."""
        ...

    @property
    def info(self) -> dict:
        """Mapping that at minimum contains ``sfreq`` and ``meas_date``."""
        ...


@dataclass
class EegData:
    """Normalized EEG epoch container used throughout the extraction pipeline.

    All transforms and feature functions operate on this object instead of
    directly on ``mne.Epochs``, keeping MNE an optional dependency.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n_epochs, n_channels, n_times)``.
    sfreq : float
        Sampling frequency in Hz.
    ch_names : list[str]
        Channel names, length ``n_channels``.
    event_labels : list[str]
        Event label for each epoch, length ``n_epochs``.
    timestamps : np.ndarray
        Start time of each epoch.  Shape ``(n_epochs,)``.  Values may be
        ``np.datetime64`` objects (when constructed from MNE Epochs) or
        plain floats (when constructed from a raw array).
    """

    data: np.ndarray
    sfreq: float
    ch_names: list[str]
    event_labels: list[str]
    timestamps: np.ndarray

    @classmethod
    def from_epochs(cls, eeg: EpochsLike) -> EegData:
        """Build an :class:`EegData` from any :class:`EpochsLike` object.

        Parameters
        ----------
        eeg : EpochsLike
            Epoched EEG object (e.g. ``mne.Epochs`` or ``mne.EpochsArray``).

        Returns
        -------
        EegData
        """
        data = eeg.get_data()
        sfreq = eeg.info["sfreq"]
        ch_names = list(eeg.ch_names)

        inv_event_id = {v: k for k, v in eeg.event_id.items()}
        event_labels = [inv_event_id[eid] for eid in eeg.events[:, 2]]

        epoch_start_times = eeg.events[:, 0] / sfreq
        eeg_start_time = eeg.info["meas_date"]
        timestamps = np.array(
            [
                np.datetime64(
                    (datetime.timedelta(seconds=t) + eeg_start_time).replace(
                        tzinfo=None
                    )
                )
                for t in epoch_start_times
            ]
        )

        return cls(
            data=data,
            sfreq=sfreq,
            ch_names=ch_names,
            event_labels=event_labels,
            timestamps=timestamps,
        )

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        sfreq: float,
        ch_names: list[str] | None = None,
        event_labels: list[str] | None = None,
        timestamps: np.ndarray | None = None,
    ) -> EegData:
        """Build an :class:`EegData` from a raw numpy array.

        Parameters
        ----------
        data : np.ndarray
            Shape ``(n_epochs, n_channels, n_times)``.
        sfreq : float
            Sampling frequency in Hz.
        ch_names : list[str] or None
            Channel names.  Defaults to ``["ch_0", "ch_1", ...]``.
        event_labels : list[str] or None
            Event labels.  Defaults to ``["0", "1", ...]``.
        timestamps : np.ndarray or None
            Epoch start times.  Defaults to ``np.arange(n_epochs, dtype=float)``.

        Returns
        -------
        EegData
        """
        n_epochs, n_channels, _ = data.shape
        if ch_names is None:
            ch_names = [f"ch_{i}" for i in range(n_channels)]
        if event_labels is None:
            event_labels = [str(i) for i in range(n_epochs)]
        if timestamps is None:
            timestamps = np.arange(n_epochs, dtype=float)

        return cls(
            data=data,
            sfreq=sfreq,
            ch_names=ch_names,
            event_labels=event_labels,
            timestamps=timestamps,
        )

    def slice(self, start: int, end: int) -> "EegData":
        """Return a new EegData containing epochs [start, end).

        Parameters
        ----------
        start : int
            First epoch index (inclusive).
        end : int
            Last epoch index (exclusive).

        Returns
        -------
        EegData
        """
        return EegData(
            data=self.data[start:end],
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            event_labels=self.event_labels[start:end],
            timestamps=self.timestamps[start:end],
        )
