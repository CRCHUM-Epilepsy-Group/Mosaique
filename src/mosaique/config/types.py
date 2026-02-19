"""Type definitions for feature extraction configuration.

This module defines the protocols that transform and feature functions must
satisfy, as well as the dataclasses used to carry parsed configuration through
the extraction pipeline.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

from mosaique.features.timefrequency import FrequencyBand, WaveletCoefficients
from mosaique.utils.toolkit import deep_list


class TransformFunction(Protocol):
    """Protocol for pre-extraction transform callables.

    A transform function takes a NumPy array (typically ``(epochs, channels,
    times)``) and returns wavelet coefficients, reconstructed signals, or a
    connectivity matrix — depending on the transform type.

    Any function that matches this signature can be used as a transform
    function in a YAML config.
    """

    def __call__(
        self, eeg: np.ndarray, *args, **kwargs
    ) -> (
        dict[FrequencyBand, np.ndarray]
        | tuple[dict[str, np.ndarray], np.ndarray]
        | dict[str, np.ndarray]
        | WaveletCoefficients
        | np.ndarray
    ): ...


class FeatureFunction(Protocol):
    """Protocol for feature extraction callables.

    A feature function receives a 1-D signal (or a connectivity matrix) and
    returns a scalar, a dict of scalars, or an array.  It **must** accept
    ``**kwargs`` so that extra parameters from the config are forwarded
    without error.

    Writing a new feature function
    ------------------------------
    1. Place your function in a module under ``mosaique/features/`` (or any
       importable module).
    2. The first positional argument is the input signal.  Accept ``**kwargs``
       to ignore unused parameters.
    3. Return a ``float``, a ``dict[str, float]`` (for multi-valued features
       like band power), or a 1-D ``np.ndarray``.

    Example::

        def my_feature(X, sfreq=200, my_param=0.5, **kwargs):
            '''Compute my custom feature on a 1-D signal.'''
            return float(np.mean(X) * my_param)

    4. Reference it in the YAML config::

        features:
          simple:
            - name: my_feature
              function: my_module.my_feature
              params:
                my_param: [0.3, 0.5, 0.7]
    """

    def __call__(
        self, eeg: np.ndarray | WaveletCoefficients, *args, **kwargs
    ) -> float | dict[str, float] | np.ndarray: ...


T = TypeVar("T", bound=TransformFunction | FeatureFunction)


@dataclass
class ExtractionParams(ABC, Generic[T]):
    """Single extraction step (transform or feature) with resolved parameters.

    Attributes
    ----------
    name : str
        Human-readable identifier (appears in the output DataFrame).
    function : callable
        The transform or feature function.
    params : dict
        Keyword arguments to forward to *function*.
    """

    name: str
    function: T
    params: dict


@dataclass
class PreGridParams(ExtractionParams):
    """Extraction parameters before grid expansion.

    When a parameter has multiple values (e.g. ``m: [2, 3]``), they are
    expanded into a Cartesian product of single-valued
    :class:`ExtractionParams` by :class:`FeatureExtractor`.
    """

    def __post_init__(self):
        if self.params is None:
            self.params_for_grid = {}
        else:
            self.params_for_grid: dict[str, list[Any]] = deep_list(self.params)


@dataclass
class TransformParams(ExtractionParams[TransformFunction]):
    """Resolved (single-valued) parameters for a pre-extraction transform."""

    function: TransformFunction
    params: dict[str, Any]


@dataclass
class FeatureParams(ExtractionParams[FeatureFunction]):
    """Resolved (single-valued) parameters for a feature function."""

    function: FeatureFunction
    params: dict[str, str | float]
