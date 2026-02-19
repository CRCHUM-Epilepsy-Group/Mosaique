"""mosaique - Parallel CPU-based EEG feature extraction."""

from mosaique.config import parse_featureextraction_config
from mosaique.extraction import FeatureExtractor

__all__ = ["FeatureExtractor", "parse_featureextraction_config"]
