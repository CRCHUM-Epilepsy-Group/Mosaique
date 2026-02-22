"""Type definitions for feature extraction configuration.

This module defines the protocols that transform and feature functions must
satisfy, as well as the Pydantic models used to carry parsed configuration
through the extraction pipeline.
"""

from collections.abc import Callable
from typing import Any, Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from mosaique.features.timefrequency import FrequencyBand, WaveletCoefficients


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


class ExtractionStepConfig(BaseModel):
    """A single feature or transform entry from user config (unresolved).

    Attributes
    ----------
    name : str
        Human-readable identifier.
    function : str | None
        Dotted import path, resolved later by the loader.
    params : dict[str, Any] | None
        Raw parameter values from the config.
    """

    name: str
    function: str | None = None
    params: dict[str, Any] | None = None


def _normalize_params(params: dict[str, Any] | None) -> dict[str, list[Any]]:
    """Wrap scalar param values in lists for grid expansion."""
    if params is None:
        return {}
    return {k: v if isinstance(v, list) else [v] for k, v in params.items()}


class ExtractionStep(BaseModel):
    """Resolved step with callable function and grid-ready params.

    Attributes
    ----------
    name : str
        Human-readable identifier (appears in the output DataFrame).
    function : callable | None
        The resolved transform or feature function.
    params : dict[str, list[Any]]
        All param values normalized to lists for Cartesian grid expansion.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    function: Callable | None = None
    params: dict[str, list[Any]]

    @property
    def params_for_grid(self) -> dict[str, list[Any]]:
        """Alias for backwards compatibility with param grid expansion."""
        return self.params


class PipelineConfig(BaseModel):
    """Top-level validated pipeline configuration.

    Attributes
    ----------
    features : dict[str, list[ExtractionStepConfig]]
        Feature functions grouped by transform name.
    transforms : dict[str, list[ExtractionStepConfig]]
        Pre-extraction transforms grouped by transform name.
    """

    features: dict[str, list[ExtractionStepConfig]]
    transforms: dict[str, list[ExtractionStepConfig]]

    @model_validator(mode="before")
    @classmethod
    def auto_generate_transforms(cls, data: Any) -> Any:
        """Auto-generate a simple transform when the ``transforms`` key is omitted.

        Also accepts ``features`` as a flat list, wrapping it as
        ``{"simple": <list>}`` with an auto-generated simple transform.
        """
        if not isinstance(data, dict):
            return data

        features = data.get("features")

        # If features is a list, wrap it as {"simple": <list>}
        if isinstance(features, list):
            data = {**data, "features": {"simple": features}}
            features = data["features"]

        if "transforms" not in data and features is not None:
            # Auto-generate simple transform entries for each feature group key.
            # Only valid for the "simple" transform type; other keys require an
            # explicit transforms section.
            non_simple = [k for k in features if k != "simple"]
            if non_simple:
                raise ValueError(
                    f"'transforms' section is required when features contain "
                    f"non-simple keys: {non_simple}"
                )
            data = {
                **data,
                "transforms": {k: [{"name": k}] for k in features},
            }

        return data

    @model_validator(mode="after")
    def keys_must_match(self) -> Self:
        feat_keys = set(self.features.keys())
        transform_keys = set(self.transforms.keys())
        if feat_keys != transform_keys:
            only_features = feat_keys - transform_keys
            only_transforms = transform_keys - feat_keys
            parts = []
            if only_features:
                parts.append(f"only in features: {only_features}")
            if only_transforms:
                parts.append(f"only in transforms: {only_transforms}")
            raise ValueError(
                f"features and transforms keys must match; {'; '.join(parts)}"
            )
        return self

    @model_validator(mode="after")
    def transform_keys_in_registry(self) -> Self:
        from mosaique.extraction.transforms import TRANSFORM_REGISTRY

        unknown = set(self.transforms.keys()) - set(TRANSFORM_REGISTRY.keys())
        if unknown:
            raise ValueError(
                f"unknown transform keys: {unknown}; "
                f"available: {set(TRANSFORM_REGISTRY.keys())}"
            )
        return self
