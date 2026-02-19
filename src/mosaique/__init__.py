"""mosaique - Parallel CPU-based EEG feature extraction.

Usage::

    from mosaique import FeatureExtractor, parse_featureextraction_config

    features, frameworks = parse_featureextraction_config("features_config.yaml")
    extractor = FeatureExtractor(features, frameworks, num_workers=4)
    df = extractor.extract_feature(eeg_epochs, eeg_id="subject_01")

See :mod:`mosaique.extraction.transforms` for how to add custom transforms,
and :mod:`mosaique.features` for how to write new feature functions.
"""

from mosaique.config import parse_featureextraction_config
from mosaique.extraction import FeatureExtractor

__all__ = ["FeatureExtractor", "parse_featureextraction_config"]
