"""Pre-extraction transforms for EEG feature extraction.

Built-in transforms
-------------------
``"simple"``
    Identity transform — passes the raw ``(epochs, channels, times)`` array
    directly to feature functions.  Use this for features that operate on
    the raw time-domain signal (e.g. entropy, line length).

``"tf_decomposition"``
    Applies a wavelet decomposition (DWT, WPD, or CWT) to the signal,
    then extracts features separately for each frequency band.

``"connectivity"``
    Computes spectral connectivity matrices (e.g. PLI) between channels
    for each frequency band, then extracts graph-theoretic features from
    the resulting matrices.

Adding a custom transform
-------------------------
1. Create a new module under ``mosaique/extraction/transforms/`` and subclass
   :class:`PreExtractionTransform`.  You must implement ``transform()``
   (converts ``mne.Epochs`` → your intermediate representation) and
   ``extract_feature()`` (applies a scalar feature function and returns a
   ``polars.DataFrame``).

2. Register it in :data:`TRANSFORM_REGISTRY` so the YAML config can
   reference it by name::

       from mosaique.extraction.transforms import TRANSFORM_REGISTRY
       from my_module import MyTransform

       TRANSFORM_REGISTRY["my_transform"] = MyTransform

3. In your YAML config, use the registered name as a transform key::

       transforms:
         my_transform:
           - name: my_transform
             function: null
             params:
               some_param: [1, 2, 3]

       features:
         my_transform:
           - name: my_feature
             function: my_package.my_feature_func
             params: null
"""

from mosaique.extraction.transforms.base import PreExtractionTransform
from mosaique.extraction.transforms.connectivity import ConnectivityTransform
from mosaique.extraction.transforms.simple import SimpleTransform
from mosaique.extraction.transforms.tfdecomposition import TFDecompositionTransform

TRANSFORM_REGISTRY: dict[str, type[PreExtractionTransform]] = {
    "tf_decomposition": TFDecompositionTransform,
    "simple": SimpleTransform,
    "connectivity": ConnectivityTransform,
}
"""Mapping from transform name (as used in YAML configs) to transform class.

Add custom transforms here so that :class:`FeatureExtractor` can look them
up by name.
"""

__all__ = [
    "PreExtractionTransform",
    "TFDecompositionTransform",
    "SimpleTransform",
    "ConnectivityTransform",
    "TRANSFORM_REGISTRY",
]
