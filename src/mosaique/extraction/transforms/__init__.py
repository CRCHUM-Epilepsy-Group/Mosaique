"""Pre-extraction transforms for EEG feature extraction."""

from mosaique.extraction.transforms.base import PreExtractionTransform
from mosaique.extraction.transforms.connectivity import ConnectivityTransform
from mosaique.extraction.transforms.simple import SimpleTransform
from mosaique.extraction.transforms.tfdecomposition import TFDecompositionTransform

TRANSFORM_REGISTRY: dict[str, type[PreExtractionTransform]] = {
    "tf_decomposition": TFDecompositionTransform,
    "simple": SimpleTransform,
    "connectivity": ConnectivityTransform,
}

__all__ = [
    "PreExtractionTransform",
    "TFDecompositionTransform",
    "SimpleTransform",
    "ConnectivityTransform",
    "TRANSFORM_REGISTRY",
]
