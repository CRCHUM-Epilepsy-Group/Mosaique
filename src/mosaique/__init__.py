"""mosaique - Parallel CPU-based EEG feature extraction.

Usage::

    from mosaique import FeatureExtractor, parse_featureextraction_config
    from mosaique.config import resolve_pipeline

    pipeline = parse_featureextraction_config("features_config.yaml")
    features, transforms = resolve_pipeline(pipeline)
    extractor = FeatureExtractor(features, transforms, num_workers=4)
    df = extractor.extract_feature(eeg_epochs, eeg_id="subject_01")

See :mod:`mosaique.extraction.transforms` for how to add custom transforms,
and :mod:`mosaique.features` for how to write new feature functions.
"""

from pathlib import Path

import numpy as np
import polars as pl

from mosaique.config import (
    get_config_schema,
    parse_featureextraction_config,
    resolve_pipeline,
)
from mosaique.extraction import FeatureExtractor
from mosaique.extraction.eegdata import EpochsLike


def extract(
    config: str | Path | dict,
    eeg: "EpochsLike | np.ndarray",
    eeg_id: str,
    *,
    num_workers: int = 1,
    batch_size: int = 128,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Extract features from one EEG recording in a single call.

    Parameters
    ----------
    config : str | Path | dict
        Pipeline configuration — a YAML file path, a raw YAML string, or a dict.
    eeg : EpochsLike or np.ndarray
        Epoched EEG data (MNE Epochs or ``(n_epochs, n_channels, n_times)`` array).
    eeg_id : str
        Identifier for the recording, used in logging.
    num_workers : int
        Number of parallel worker processes.
    batch_size : int
        Epochs per processing batch.
    sfreq : float, optional
        Sampling frequency in Hz — required when *eeg* is a numpy array.
    ch_names : list[str], optional
        Channel names — only used when *eeg* is a numpy array.
    **kwargs
        Forwarded to :meth:`~mosaique.extraction.extractor.FeatureExtractor.extract_feature`.

    Returns
    -------
    polars.DataFrame
        Long-format feature table.

    Example
    -------
    ::

        import mosaique

        df = mosaique.extract("config.yaml", epochs, eeg_id="subject_01")
    """
    extractor = FeatureExtractor(config, num_workers=num_workers, batch_size=batch_size)
    return extractor.extract_feature(
        eeg, eeg_id, sfreq=sfreq, ch_names=ch_names, **kwargs
    )


__all__ = [
    "FeatureExtractor",
    "extract",
    "get_config_schema",
    "parse_featureextraction_config",
    "resolve_pipeline",
]
