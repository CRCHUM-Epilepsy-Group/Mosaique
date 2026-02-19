"""Type definitions for feature extraction configuration."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

from mosaique.features.timefrequency import FrequencyBand, WaveletCoefficients
from mosaique.utils.toolkit import deep_list


class TransformFunction(Protocol):
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
    def __call__(
        self, eeg: np.ndarray | WaveletCoefficients, *args, **kwargs
    ) -> float | dict[str, float] | np.ndarray: ...


T = TypeVar("T", bound=TransformFunction | FeatureFunction)


@dataclass
class ExtractionParams(ABC, Generic[T]):
    """Holds function and parameters necessary for feature
    pre-extraction and feature extraction."""

    name: str
    function: T
    params: dict


@dataclass
class PreGridParams(ExtractionParams):
    def __post_init__(self):
        if self.params is None:
            self.params_for_grid = {}
        else:
            self.params_for_grid: dict[str, list[Any]] = deep_list(self.params)


@dataclass
class TransformParams(ExtractionParams[TransformFunction]):
    function: TransformFunction
    params: dict[str, Any]


@dataclass
class FeatureParams(ExtractionParams[FeatureFunction]):
    function: FeatureFunction
    params: dict[str, str | float]
