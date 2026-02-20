"""Regression tests for FeatureExtractor and param grid expansion."""

import polars as pl
import pytest
from rich.console import Console

from mosaique import FeatureExtractor, parse_featureextraction_config
from mosaique.config.loader import resolve_pipeline
from mosaique.config.types import ExtractionStep
from mosaique.features.univariate import line_length, sample_entropy


@pytest.fixture
def simple_features():
    """Manually built features dict (no YAML dependency)."""
    return {
        "simple": [
            ExtractionStep(name="linelength", function=line_length, params={}),
            ExtractionStep(
                name="sampen",
                function=sample_entropy,
                params={"m": [2, 3], "r": [0.2]},
            ),
        ]
    }


@pytest.fixture
def simple_transforms():
    """Manually built transforms dict with null transform."""
    return {
        "simple": [
            ExtractionStep(name="simple", function=None, params={}),
        ]
    }


class TestParamGrid:
    def test_grid_expansion(self, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            simple_features, simple_transforms, debug=True,
            console=Console(quiet=True),
        )
        extractor._curr_features = simple_features["simple"]
        grid = extractor._feature_param_grid

        # linelength has no params → 1 entry
        # sampen has m=[2,3] x r=[0.2] → 2 entries
        assert len(grid) == 3

    def test_grid_names(self, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            simple_features, simple_transforms, debug=True,
            console=Console(quiet=True),
        )
        extractor._curr_features = simple_features["simple"]
        grid = extractor._feature_param_grid
        names = [g.name for g in grid]
        assert names == ["linelength", "sampen", "sampen"]

    def test_grid_params(self, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            simple_features, simple_transforms, debug=True,
            console=Console(quiet=True),
        )
        extractor._curr_features = simple_features["simple"]
        grid = extractor._feature_param_grid
        # Grid entries have single-element lists; unwrap for comparison
        sampen_params = [
            {k: v[0] for k, v in g.params.items()} for g in grid if g.name == "sampen"
        ]
        assert {"m": 2, "r": 0.2} in sampen_params
        assert {"m": 3, "r": 0.2} in sampen_params


class TestExtractFeature:
    def test_end_to_end(self, synthetic_epochs, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            simple_features,
            simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test")
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_output_columns(self, synthetic_epochs, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            simple_features,
            simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test")
        for col in ["epoch", "channel", "value", "feature", "region_side", "params"]:
            assert col in df.columns

    def test_features_present(self, synthetic_epochs, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            simple_features,
            simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test")
        feature_names = df["feature"].unique().to_list()
        assert "linelength" in feature_names
        assert "sampen" in feature_names

    def test_from_yaml(self, synthetic_epochs, minimal_config_file):
        pipeline = parse_featureextraction_config(minimal_config_file)
        features, transforms = resolve_pipeline(pipeline)
        extractor = FeatureExtractor(
            features, transforms, debug=True, console=Console(quiet=True)
        )
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test_yaml")
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
