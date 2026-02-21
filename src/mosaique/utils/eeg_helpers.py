"""EEG helper utilities for feature extraction."""

from __future__ import annotations


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
