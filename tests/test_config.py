"""Regression tests for config parsing and function resolution."""

import pytest

from mosaique.config.loader import (
    load_feature_extraction_func,
    parse_config,
    parse_featureextraction_config,
)
from mosaique.config.types import PreGridParams
from mosaique.features.univariate import line_length, sample_entropy


class TestLoadFeatureExtractionFunc:
    def test_resolves_builtin(self):
        func = load_feature_extraction_func("univariate.line_length")
        assert func is line_length

    def test_resolves_none(self):
        assert load_feature_extraction_func(None) is None

    def test_invalid_path_raises(self):
        with pytest.raises(Exception):
            load_feature_extraction_func("nonexistent_module.no_func")


class TestParseConfig:
    def test_valid_yaml(self, minimal_config_file):
        config = parse_config(minimal_config_file)
        assert "features" in config
        assert "frameworks" in config

    def test_missing_file(self, tmp_path):
        with pytest.raises(Exception):
            parse_config(tmp_path / "does_not_exist.yaml")


class TestParseFeatureExtractionConfig:
    def test_returns_features_and_frameworks(self, minimal_config_file):
        features, frameworks = parse_featureextraction_config(minimal_config_file)
        assert "simple" in features
        assert "simple" in frameworks

    def test_functions_are_resolved(self, minimal_config_file):
        features, frameworks = parse_featureextraction_config(minimal_config_file)
        # line_length function should be resolved
        ll_entry = features["simple"][0]
        assert isinstance(ll_entry, PreGridParams)
        assert ll_entry.function is line_length

    def test_null_function_resolved_to_none(self, minimal_config_file):
        _, frameworks = parse_featureextraction_config(minimal_config_file)
        simple_fw = frameworks["simple"][0]
        assert simple_fw.function is None

    def test_param_grid_expanded(self, minimal_config_file):
        features, _ = parse_featureextraction_config(minimal_config_file)
        sampen = features["simple"][1]
        # m: [2, 3] should become a list in params_for_grid
        assert sampen.params_for_grid["m"] == [2, 3]
        assert sampen.params_for_grid["r"] == [0.2]
