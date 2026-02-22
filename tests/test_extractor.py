"""Regression tests for FeatureExtractor and param grid expansion."""

import numpy as np
import polars as pl
import pytest
from rich.console import Console

import mosaique
from mosaique import FeatureExtractor, extract, parse_featureextraction_config
from mosaique.config.loader import resolve_pipeline
from mosaique.config.types import ExtractionStep
from mosaique.extraction.eegdata import EegData
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
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        extractor._curr_features = simple_features["simple"]
        grid = extractor._feature_param_grid

        # linelength has no params → 1 entry
        # sampen has m=[2,3] x r=[0.2] → 2 entries
        assert len(grid) == 3

    def test_grid_names(self, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        extractor._curr_features = simple_features["simple"]
        grid = extractor._feature_param_grid
        names = [g.name for g in grid]
        assert names == ["linelength", "sampen", "sampen"]

    def test_grid_params(self, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
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
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test")
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_output_columns(self, synthetic_epochs, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test")
        for col in ["epoch", "channel", "value", "feature", "region_side", "params"]:
            assert col in df.columns

    def test_features_present(
        self, synthetic_epochs, simple_features, simple_transforms
    ):
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
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
            features=features,
            transforms=transforms,
            debug=True,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test_yaml")
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0


class TestFeatureExtractorConfigInit:
    """FeatureExtractor accepts config directly (file, string, dict, or kwargs)."""

    def test_from_yaml_file(self, synthetic_epochs, minimal_config_file):
        extractor = FeatureExtractor(
            minimal_config_file, debug=True, console=Console(quiet=True)
        )
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test_file")
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_from_yaml_string(self, synthetic_epochs):
        yaml_str = """\
features:
  simple:
    - name: linelength
      function: univariate.line_length
transforms:
  simple:
    - name: simple
      function: null
      params: null
"""
        extractor = FeatureExtractor(yaml_str, debug=True, console=Console(quiet=True))
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test_str")
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_from_dict(self, synthetic_epochs):
        config = {
            "features": {
                "simple": [{"name": "linelength", "function": "univariate.line_length"}]
            },
            "transforms": {
                "simple": [{"name": "simple", "function": None, "params": None}]
            },
        }
        extractor = FeatureExtractor(config, debug=True, console=Console(quiet=True))
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test_dict")
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_backwards_compat_kwargs(
        self, synthetic_epochs, simple_features, simple_transforms
    ):
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test_kwargs")
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_config_and_features_raises(self, simple_features, simple_transforms):
        with pytest.raises(ValueError, match="Cannot specify both"):
            FeatureExtractor(
                "some_config.yaml",
                features=simple_features,
                transforms=simple_transforms,
            )

    def test_neither_config_nor_features_raises(self):
        with pytest.raises(ValueError, match="Must specify either"):
            FeatureExtractor()


class TestExtractFeatureFromArray:
    def test_numpy_input(self, synthetic_array, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(
            synthetic_array, eeg_id="test_array", sfreq=200.0
        )
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_numpy_with_ch_names(
        self, synthetic_array, simple_features, simple_transforms
    ):
        ch_names = ["Fp1", "C3", "O1"]
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(
            synthetic_array, eeg_id="test_ch", sfreq=200.0, ch_names=ch_names
        )
        assert set(df["channel"].unique().to_list()) == set(ch_names)

    def test_numpy_missing_sfreq_raises(
        self, synthetic_array, simple_features, simple_transforms
    ):
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        with pytest.raises(ValueError, match="sfreq"):
            extractor.extract_feature(synthetic_array, eeg_id="test_no_sfreq")

    def test_numpy_default_channels(
        self, synthetic_array, simple_features, simple_transforms
    ):
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(
            synthetic_array, eeg_id="test_defaults", sfreq=200.0
        )
        channels = df["channel"].unique().to_list()
        assert "ch_0" in channels
        assert "ch_1" in channels
        assert "ch_2" in channels


class TestEegDataSlice:
    def test_slice_returns_correct_epochs(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 3, 400))
        eeg = EegData.from_array(data, sfreq=200.0)
        sliced = eeg.slice(2, 5)

        assert sliced.data.shape == (3, 3, 400)
        np.testing.assert_array_equal(sliced.data, data[2:5])

    def test_slice_preserves_metadata(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 3, 400))
        ch_names = ["Fp1", "C3", "O1"]
        eeg = EegData.from_array(data, sfreq=200.0, ch_names=ch_names)
        sliced = eeg.slice(0, 4)

        assert sliced.sfreq == 200.0
        assert sliced.ch_names == ch_names
        assert len(sliced.event_labels) == 4
        assert len(sliced.timestamps) == 4

    def test_slice_event_labels_and_timestamps(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 2, 100))
        labels = ["a", "b", "c", "d", "e"]
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        eeg = EegData.from_array(
            data, sfreq=200.0, event_labels=labels, timestamps=timestamps
        )
        sliced = eeg.slice(1, 4)

        assert sliced.event_labels == ["b", "c", "d"]
        np.testing.assert_array_equal(sliced.timestamps, [1.0, 2.0, 3.0])


class TestBatchSize:
    def test_default_batch_size(self, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            console=Console(quiet=True),
        )
        assert extractor.batch_size == 128

    def test_custom_batch_size(self, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            batch_size=64,
            console=Console(quiet=True),
        )
        assert extractor.batch_size == 64


class TestBatchingIntegration:
    def test_many_epochs_batched(self, simple_features, simple_transforms):
        """10 epochs with batch_size=3 → 4 batches (3+3+3+1)."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((10, 3, 400)) * 1e-6

        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            batch_size=3,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(data, eeg_id="many_epochs", sfreq=200.0)

        # 10 epochs × 3 channels × 3 features (linelength + sampen×2 params) = 90 rows
        assert len(df) == 90
        assert df["epoch"].n_unique() == 10

    def test_batch_size_one(self, simple_features, simple_transforms):
        """Extreme case: batch_size=1, one epoch per batch."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((4, 2, 200)) * 1e-6

        extractor = FeatureExtractor(
            features=simple_features,
            transforms=simple_transforms,
            debug=True,
            batch_size=1,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(data, eeg_id="single_batch", sfreq=200.0)

        # 4 epochs × 2 channels × 3 features = 24 rows
        assert len(df) == 24


class TestExtractConvenienceFunction:
    def test_extract_matches_explicit_workflow(
        self, synthetic_epochs, minimal_config_file
    ):
        df_explicit = FeatureExtractor(
            minimal_config_file, debug=True, console=Console(quiet=True)
        ).extract_feature(synthetic_epochs, eeg_id="ref")

        df_convenience = extract(minimal_config_file, synthetic_epochs, eeg_id="ref")

        assert isinstance(df_convenience, pl.DataFrame)
        assert len(df_convenience) == len(df_explicit)
        assert set(df_convenience.columns) == set(df_explicit.columns)
