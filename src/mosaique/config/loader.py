"""YAML configuration loading for feature extraction."""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import yaml

from mosaique.config.types import PreGridParams


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


def parse_config(config_file):
    """Read a YAML config file."""
    f = Path(config_file)

    with open(f, "r") as stream:
        try:
            config = yaml.load(stream, Loader=get_loader())
        except yaml.YAMLError as exc:
            print(exc)
            return dict()

    return config


def load_feature_extraction_func(dotpath: str | None) -> Callable:
    """Load a function from a dotted module path.

    Tries ``mosaique.features.<module>.<func>`` first, then falls back to
    importing ``<module>`` directly.
    """
    if dotpath is None:
        return None
    module_, func = dotpath.rsplit(".", maxsplit=1)
    try:
        m = import_module("mosaique.features." + module_)
    except Exception:
        m = import_module(module_)
    return getattr(m, func)


def parse_featureextraction_config(config_file):
    """Parse a feature extraction YAML config file.

    Returns
    -------
    tuple[dict, dict]
        ``(features, frameworks)`` dicts with ``PreGridParams`` values.
    """
    conf = parse_config(config_file)

    features = conf["features"]
    frameworks = conf["frameworks"]

    for outer_dict in (features, frameworks):
        for framework, method_list in outer_dict.items():
            for i, extraction_params in enumerate(method_list):
                extraction_params["function"] = load_feature_extraction_func(
                    extraction_params["function"]
                )
                method_list[i] = PreGridParams(**extraction_params)

    return features, frameworks
