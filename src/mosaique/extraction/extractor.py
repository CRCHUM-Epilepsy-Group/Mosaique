"""FeatureExtractor orchestrator for EEG feature extraction."""

import datetime
import logging
import time
from collections.abc import Iterable, Mapping
from functools import cached_property
from itertools import product
from pathlib import Path
from traceback import format_exc

import numpy as np
import polars as pl
from mne import Epochs
from rich.console import Console
from rich.progress import Progress

from mosaique.config.types import (
    ExtractionParams,
    FeatureParams,
    PreGridParams,
    TransformParams,
)
from mosaique.extraction.transforms import TRANSFORM_REGISTRY, PreExtractionTransform
from mosaique.features.timefrequency import FrequencyBand
from mosaique.utils.eeg_helpers import get_region_side


class FeatureExtractor:
    """Define and execute feature extraction pipelines for EEG data"""

    def __init__(
        self,
        features: Mapping[str, list[PreGridParams]],
        frameworks: Mapping[str, list[PreGridParams]],
        log_dir: str | Path | None = None,
        num_workers: int = 1,
        debug=False,
        console=Console(),
    ):
        # Feature extraction params for each feature
        self._features = features
        self._required_transforms = list(features.keys())
        self._frameworks = frameworks

        # List of param names to construct final df
        self._param_names = []

        self._eeg: Epochs
        self._transformed_eeg: np.ndarray

        # Cache wavelet coefficients
        self._cached_coeffs: dict[FrequencyBand, np.ndarray] = {}

        # Initial params with multiple values per params
        self._curr_features: list[PreGridParams]
        self._curr_framework: list[PreGridParams]

        # Params with single value, ready for function
        self._curr_feature: FeatureParams
        self._curr_transform_params: TransformParams

        # Final list of extracted features
        self._extracted_features: list[pl.DataFrame] = []

        self.log_dir = log_dir
        self.num_workers = num_workers
        self.debug = debug
        if self.num_workers <= 1:
            self.debug = True
        self.console = console

    @cached_property
    def _eeg_array(self):
        return self._eeg.get_data()

    @property
    def _feature_param_grid(self) -> list[FeatureParams]:
        return self._make_param_grid(self._curr_features)

    @property
    def _transform_grid(self) -> list[TransformParams]:
        """Make a list of all kwargs given a dict of possible values for kwargs"""
        return self._make_param_grid(self._curr_framework)

    def _init_logger(self, eeg_id):
        self.logger = logging.getLogger("rich")
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        if self.log_dir is not None:
            log_path = Path(self.log_dir)
            log_path.mkdir(exist_ok=True, parents=True)
            onset_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            log_file = f"{eeg_id}_{onset_time}.log"

            file_handler = logging.FileHandler(log_path / log_file)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.addHandler(logging.NullHandler())

    def _make_param_grid(self, params: list[PreGridParams]) -> list[ExtractionParams]:
        """Make a list of all kwargs given a dict of possible values for kwargs"""
        grid = []
        for element in params:
            single_param_grid = [
                dict(zip(element.params_for_grid.keys(), a))
                for a in product(*element.params_for_grid.values())
            ]
            for configuration in single_param_grid:
                grid.append(
                    ExtractionParams(
                        name=element.name,
                        function=element.function,
                        params=configuration,
                    )
                )
        return grid

    def _extract_for_single_transform(
        self,
        eeg: np.ndarray,
        progress,
        task_id,
    ) -> pl.DataFrame:
        """Extract all features from a single, transformed eeg array"""

        features = []

        # Feature param grid is based on curr_features (for a single framework)
        for self._curr_feature in self._feature_param_grid:
            func = self._curr_feature.function
            params = self._curr_feature.params
            name = self._curr_feature.name
            feature_start = time.perf_counter()

            self._param_names.extend(params.keys())

            ##########################################
            # Extract feature from pre-transformed EEG
            ##########################################
            try:
                df = self._curr_transform.extract_feature(eeg, func, **params)
            except Exception as e:
                self.logger.error(f"Error extracting {name}:\n{format_exc()}")
                # Skip this feature altogether
                continue

            feature_time = time.perf_counter() - feature_start
            self.logger.info(
                f"Feature {name}: {params} completed in {feature_time:.2f}s"
            )
            # Add the parameters to the dataframe
            df = df.with_columns(
                feature=pl.lit(name), computation_time=pl.lit(feature_time)
            )
            if name != "band_power":
                for k, v in params.items():
                    df = df.with_columns(pl.lit(v).alias(k))
            features.append(df)
            progress.advance(task_id)

        return pl.concat(features, how="diagonal_relaxed")

    def _concat_params(self, param_values: Iterable) -> str:
        """Concatenate parameter values into a single string."""
        params = [str(p) for p in param_values]
        return "_".join(params)

    def _clean_features_df(self, feature_df: pl.DataFrame) -> pl.DataFrame:
        param_cols = set(self._param_names)
        for removed in ["band_ranges", "freqs"]:
            try:
                param_cols.remove(removed)
            except (ValueError, KeyError):
                continue
        self._param_names = list(param_cols)

        feature_df = feature_df.with_columns(
            pl.col("channel")
            .map_elements(
                get_region_side, skip_nulls=False, return_dtype=pl.String
            )
            .alias("region_side"),
            pl.struct(list(param_cols))
            .map_elements(
                lambda x: self._concat_params(x.values()), return_dtype=pl.String
            )
            .alias("params"),
        )
        return feature_df

    def extract_feature(self, eeg: Epochs, eeg_id: str) -> pl.DataFrame:
        """Extract all features from one EEG"""

        self._eeg = eeg
        self._cached_coeffs = {}
        self._init_logger(eeg_id)
        total_start = time.perf_counter()
        log_str = (
            f"Starting extraction for {eeg_id} (number of epochs: {len(eeg.events)})"
        )
        self.logger.info(log_str)
        self.logger.info("=" * 50)  # Add separator

        # _curr_framework is a list[InitialParams]
        for framework_name, self._curr_framework in self._frameworks.items():
            # transform grid is a list of Extraction params based on curr_framework
            framework_start = time.perf_counter()
            log_str = f"Starting framework: {framework_name}"
            self.logger.info("-" * 50)  # Add separator
            self.logger.info(log_str)
            self.logger.info("-" * 50)

            with Progress(console=self.console, transient=True) as progress:
                progress_string = f"Extracting {framework_name}..."
                task_id = progress.add_task(
                    progress_string, total=len(self._transform_grid)
                )

                for self._curr_transform_params in self._transform_grid:
                    transform_start = time.perf_counter()

                    # 1. Construct a pre-extraction transform pipeline
                    self._curr_transform = TRANSFORM_REGISTRY[framework_name](
                        self._curr_transform_params,
                        num_workers=self.num_workers,
                        debug=self.debug,
                        console=self.console,
                    )
                    self._param_names.extend(self._curr_transform._params.keys())
                    # Transfer cached coeffs to Transform
                    self._curr_transform._cached_coeffs = self._cached_coeffs

                    # 2. Apply pre-extraction transform
                    self._transformed_eeg = self._curr_transform.transform(eeg)
                    transform_time = time.perf_counter() - transform_start

                    self.logger.info(
                        f"Transform {framework_name}: {self._curr_transform_params} completed in {transform_time:.2f}s"
                    )

                    # 3. Extract all features for this transform
                    self._curr_features = self._features[framework_name]
                    features = self._extract_for_single_transform(
                        self._transformed_eeg, progress=progress, task_id=task_id
                    )
                    # 4. Add hyperparameters to dataframe
                    self._transformed_df = self._curr_transform.complete_df(features)

                    # 5. Append to list of dataframes
                    self._extracted_features.append(self._transformed_df)

                    # Transfer cached coeffs back
                    self._cached_coeffs = self._curr_transform._cached_coeffs

                framework_time = (time.perf_counter() - framework_start) / 60
                self.logger.info(
                    f"All features for {framework_name} extracted in {(framework_time):.2f}m"
                )

        final_features = pl.concat(self._extracted_features, how="diagonal_relaxed")
        final_features_cleaned = self._clean_features_df(final_features)

        total_time = time.perf_counter() - total_start
        self.logger.info(f"Total extraction time: {total_time / 60:.2f}m")

        return final_features_cleaned
