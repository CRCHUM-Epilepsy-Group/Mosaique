"""EEG helper utilities for feature extraction."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from mne import Epochs


def get_event_list(eeg: Epochs) -> list[str]:
    """Extract event labels from MNE Epochs via event_id inversion.

    Parameters
    ----------
    eeg : Epochs
        MNE Epochs object with event_id mapping.

    Returns
    -------
    list[str]
        Ordered list of event labels, one per epoch.
    """
    inv_event_id = {v: k for k, v in eeg.event_id.items()}
    return [inv_event_id[eid] for eid in eeg.events[:, 2]]


# Standard 10-20 channel -> region/side mapping
_CHANNEL_REGION_MAP: dict[str, str] = {
    "Fp1": "frontal_left",
    "Fp2": "frontal_right",
    "F3": "frontal_left",
    "F4": "frontal_right",
    "F7": "frontal_left",
    "F8": "frontal_right",
    "Fz": "frontal_midline",
    "C3": "central_left",
    "C4": "central_right",
    "Cz": "central_midline",
    "T3": "temporal_left",
    "T4": "temporal_right",
    "T5": "temporal_left",
    "T6": "temporal_right",
    "T7": "temporal_left",
    "T8": "temporal_right",
    "P3": "parietal_left",
    "P4": "parietal_right",
    "Pz": "parietal_midline",
    "O1": "occipital_left",
    "O2": "occipital_right",
    "A1": "ear_left",
    "A2": "ear_right",
}


def get_region_side(channel_name: str) -> str | None:
    """Map a 10-20 system channel name to its region/side.

    Parameters
    ----------
    channel_name : str
        EEG channel name (e.g. "Fp1", "Cz").

    Returns
    -------
    str | None
        Region and side string, or None if channel is not recognized.
    """
    if channel_name is None:
        return None
    return _CHANNEL_REGION_MAP.get(channel_name)


def load_and_epoch_edf(
    edf_path: Path,
    epoch_duration: float = 5.0,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    tmax: float = 120.0,
) -> mne.Epochs:
    """Load an EDF file and segment it into fixed-length epochs.

    Parameters
    ----------
    edf_path : Path
        Path to the ``.edf`` file.
    epoch_duration : float
        Duration of each epoch in seconds.
    l_freq : float
        Lower bandpass frequency in Hz.
    h_freq : float
        Upper bandpass frequency in Hz.
    tmax : float
        Only keep the first ``tmax`` seconds of the recording.

    Returns
    -------
    mne.Epochs
        Epoched EEG data.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.crop(tmax=min(tmax, raw.times[-1]))
    raw.filter(l_freq, h_freq, verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, verbose=False)
    epochs.load_data()
    return epochs
