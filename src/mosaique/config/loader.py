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
    Callable
        The resolved function object.
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

    The YAML file must contain two top-level keys:

    ``frameworks``
        Pre-extraction transforms keyed by framework name.  Each entry is a
        list of dicts with ``name``, ``function`` (dotted path or ``null``),
        and ``params``.

    ``features``
        Feature functions keyed by the same framework names.  Each entry is
        a list of dicts with ``name``, ``function`` (dotted path), and
        ``params``.  Parameter values can be lists — they will be expanded
        into a Cartesian grid at extraction time.

    Parameters
    ----------
    config_file : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    tuple[dict[str, list[PreGridParams]], dict[str, list[PreGridParams]]]
        ``(features, frameworks)`` ready to pass to
        :class:`~mosaique.extraction.extractor.FeatureExtractor`.

    Example
    -------
    ::

        features, frameworks = parse_featureextraction_config("config.yaml")
        extractor = FeatureExtractor(features, frameworks)
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
