"""YAML configuration loading for feature extraction."""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from mosaique.config.types import (
    ExtractionStep,
    PipelineConfig,
    _normalize_params,
)


class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


def get_loader():
    """Add constructors to PyYAML loader."""
    loader = PrettySafeLoader
    loader.add_constructor(
        "tag:yaml.org,2002:python/tuple", PrettySafeLoader.construct_python_tuple
    )
    return loader


def parse_config(config_file: str | Path) -> dict[str, Any]:
    """Read a YAML config file.

    Raises
    ------
    ValueError
        If the file contains invalid YAML.
    """
    f = Path(config_file)

    with open(f, "r") as stream:
        try:
            config = yaml.load(stream, Loader=get_loader())
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {f}: {exc}") from exc

    return config


def load_feature_extraction_func(dotpath: str | None) -> Callable | None:
    """Load a function from a dotted module path.

    Resolution order:

    1. ``mosaique.features.<module>.<func>`` — looks inside the built-in
       feature modules first.
    2. ``<module>.<func>`` — falls back to an absolute import, allowing
       external packages to provide custom functions.

    Parameters
    ----------
    dotpath : str | None
        Dotted path such as ``"univariate.sample_entropy"`` or
        ``"my_package.my_module.my_func"``.  If ``None``, returns ``None``.

    Returns
    -------
    Callable | None
        The resolved function object, or ``None`` if *dotpath* is ``None``.
    """
    if dotpath is None:
        return None
    module_, func = dotpath.rsplit(".", maxsplit=1)
    try:
        m = import_module("mosaique.features." + module_)
    except Exception:
        m = import_module(module_)
    return getattr(m, func)


def parse_featureextraction_config(
    config: str | Path | dict,
) -> PipelineConfig:
    """Parse a feature extraction config from a YAML file or dict.

    The input must contain two top-level keys:

    ``transforms``
        Pre-extraction transforms keyed by transform name.  Each entry is a
        list of dicts with ``name``, ``function`` (dotted path or ``null``),
        and ``params``.

    ``features``
        Feature functions keyed by the same transform names.  Each entry is
        a list of dicts with ``name``, ``function`` (dotted path), and
        ``params``.  Parameter values can be lists — they will be expanded
        into a Cartesian grid at extraction time.

    Parameters
    ----------
    config : str | Path | dict
        Path to a YAML configuration file, or a dict with the same structure.

    Returns
    -------
    PipelineConfig
        Validated configuration ready to pass to
        :class:`~mosaique.extraction.extractor.FeatureExtractor`.

    Example
    -------
    ::

        pipeline = parse_featureextraction_config("config.yaml")
        extractor = FeatureExtractor(pipeline)
    """
    if isinstance(config, dict):
        raw = config
    else:
        raw = parse_config(config)

    pipeline = PipelineConfig.model_validate(raw)
    return pipeline


def resolve_pipeline(pipeline: PipelineConfig) -> tuple[
    dict[str, list[ExtractionStep]],
    dict[str, list[ExtractionStep]],
]:
    """Resolve a PipelineConfig into dicts of ExtractionStep with loaded functions.

    Returns
    -------
    tuple[dict[str, list[ExtractionStep]], dict[str, list[ExtractionStep]]]
        ``(features, transforms)`` with resolved callables and normalized params.
    """
    features: dict[str, list[ExtractionStep]] = {}
    transforms: dict[str, list[ExtractionStep]] = {}

    for group_name, step_configs in pipeline.features.items():
        features[group_name] = [
            ExtractionStep(
                name=sc.name,
                function=load_feature_extraction_func(sc.function),
                params=_normalize_params(sc.params),
            )
            for sc in step_configs
        ]

    for group_name, step_configs in pipeline.transforms.items():
        transforms[group_name] = [
            ExtractionStep(
                name=sc.name,
                function=load_feature_extraction_func(sc.function),
                params=_normalize_params(sc.params),
            )
            for sc in step_configs
        ]

    return features, transforms
