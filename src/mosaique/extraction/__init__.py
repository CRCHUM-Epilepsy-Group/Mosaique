"""EEG feature extraction orchestration."""

from mosaique.extraction.eegdata import EegData, EpochsLike
from mosaique.extraction.extractor import FeatureExtractor
from mosaique.extraction.transforms import (
    TRANSFORM_REGISTRY,
    ConnectivityTransform,
    PreExtractionTransform,
    SimpleTransform,
    TFDecompositionTransform,
)

__all__ = [
    "EegData",
    "EpochsLike",
    "FeatureExtractor",
    "PreExtractionTransform",
    "TFDecompositionTransform",
    "SimpleTransform",
    "ConnectivityTransform",
    "TRANSFORM_REGISTRY",
]
