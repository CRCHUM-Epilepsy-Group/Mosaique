"""Configuration parsing for mosaique feature extraction."""

from mosaique.config.loader import parse_featureextraction_config, resolve_pipeline
from mosaique.config.types import (
    ExtractionStep,
    ExtractionStepConfig,
    FeatureFunction,
    PipelineConfig,
    TransformFunction,
)

def get_config_schema() -> dict:
    """Return the JSON schema for PipelineConfig.

    Useful for validating YAML configs or generating documentation.

    Example
    -------
    >>> import json
    >>> schema = get_config_schema()
    >>> print(json.dumps(schema, indent=2))
    """
    return PipelineConfig.model_json_schema()


__all__ = [
    "get_config_schema",
    "parse_featureextraction_config",
    "resolve_pipeline",
    "ExtractionStep",
    "ExtractionStepConfig",
    "FeatureFunction",
    "PipelineConfig",
    "TransformFunction",
]
