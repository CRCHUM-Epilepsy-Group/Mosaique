"""Regression tests for config parsing and function resolution."""

import pytest

from mosaique.config.loader import (
    load_feature_extraction_func,
    parse_config,
    parse_featureextraction_config,
    resolve_pipeline,
)
from mosaique.config.types import ExtractionStep, PipelineConfig
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
        assert "transforms" in config

    def test_missing_file(self, tmp_path):
        with pytest.raises(Exception):
            parse_config(tmp_path / "does_not_exist.yaml")


class TestParseFeatureExtractionConfig:
    def test_returns_pipeline_config(self, minimal_config_file):
        pipeline = parse_featureextraction_config(minimal_config_file)
        assert isinstance(pipeline, PipelineConfig)
        assert "simple" in pipeline.features
        assert "simple" in pipeline.transforms

    def test_accepts_dict_input(self):
        raw = {
            "features": {
                "simple": [
                    {"name": "linelength", "function": "univariate.line_length"},
                ]
            },
            "transforms": {
                "simple": [
                    {"name": "simple", "function": None, "params": None},
                ]
            },
        }
        pipeline = parse_featureextraction_config(raw)
        assert isinstance(pipeline, PipelineConfig)
        assert "simple" in pipeline.features

    def test_resolve_pipeline(self, minimal_config_file):
        pipeline = parse_featureextraction_config(minimal_config_file)
        features, transforms = resolve_pipeline(pipeline)
        assert "simple" in features
        assert "simple" in transforms

    def test_functions_are_resolved(self, minimal_config_file):
        pipeline = parse_featureextraction_config(minimal_config_file)
        features, transforms = resolve_pipeline(pipeline)
        ll_entry = features["simple"][0]
        assert isinstance(ll_entry, ExtractionStep)
        assert ll_entry.function is line_length

    def test_null_function_resolved_to_none(self, minimal_config_file):
        pipeline = parse_featureextraction_config(minimal_config_file)
        _, transforms = resolve_pipeline(pipeline)
        simple_tf = transforms["simple"][0]
        assert simple_tf.function is None

    def test_param_grid_expanded(self, minimal_config_file):
        pipeline = parse_featureextraction_config(minimal_config_file)
        features, _ = resolve_pipeline(pipeline)
        sampen = features["simple"][1]
        assert sampen.params_for_grid["m"] == [2, 3]
        assert sampen.params_for_grid["r"] == [0.2]

    def test_scalar_params_normalized_to_lists(self):
        raw = {
            "features": {
                "simple": [
                    {
                        "name": "sampen",
                        "function": "univariate.sample_entropy",
                        "params": {"m": 2, "r": 0.2},
                    },
                ]
            },
            "transforms": {
                "simple": [
                    {"name": "simple", "function": None, "params": None},
                ]
            },
        }
        pipeline = parse_featureextraction_config(raw)
        features, _ = resolve_pipeline(pipeline)
        sampen = features["simple"][0]
        assert sampen.params["m"] == [2]
        assert sampen.params["r"] == [0.2]

    def test_keys_mismatch_raises(self):
        raw = {
            "features": {
                "simple": [{"name": "ll", "function": "univariate.line_length"}]
            },
            "transforms": {
                "tf_decomposition": [{"name": "cwt", "function": None}]
            },
        }
        with pytest.raises(Exception, match="keys must match"):
            parse_featureextraction_config(raw)

    def test_unknown_transform_key_raises(self):
        raw = {
            "features": {
                "nonexistent": [{"name": "ll", "function": "univariate.line_length"}]
            },
            "transforms": {
                "nonexistent": [{"name": "x", "function": None}]
            },
        }
        with pytest.raises(Exception, match="unknown transform keys"):
            parse_featureextraction_config(raw)
