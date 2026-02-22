"""FeatureExtractor orchestrator for EEG feature extraction."""

import datetime
import logging
import time
from collections.abc import Iterable, Mapping
from itertools import product
from pathlib import Path
from traceback import format_exc

import numpy as np
import polars as pl
from rich.console import Console
from rich.progress import Progress

from mosaique.config.types import ExtractionStep
from mosaique.extraction.eegdata import EegData, EpochsLike
from mosaique.extraction.transforms import TRANSFORM_REGISTRY, PreExtractionTransform
from mosaique.features.timefrequency import FrequencyBand
from mosaique.utils.eeg_helpers import get_region_side


class FeatureExtractor:
    """Orchestrates feature extraction pipelines for EEG data.

    ``FeatureExtractor`` ties together *transforms* (pre-extraction transforms
    such as wavelet decompositions or connectivity matrices) with *features*
    (scalar functions applied to each transformed signal). For every
    combination of transform parameters and feature parameters the extractor
    builds a parameter grid, runs the computation in parallel, and returns a
    single :class:`polars.DataFrame` with one row per (epoch, channel, feature,
    parameter-combination).

    Parameters
    ----------
    config : str | Path | dict | None
        Pipeline configuration.  Accepts:

        - a path to a YAML file (``str`` ending in ``.yaml``/``.yml`` or
          :class:`pathlib.Path`),
        - a raw YAML string,
        - a :class:`dict` with the same structure as a YAML config.

        When *config* is given, ``features`` and ``transforms`` must not be
        provided.  Pass ``None`` to use the legacy ``features``/``transforms``
        kwargs instead.
    features : Mapping[str, list[ExtractionStep]], optional
        Feature functions grouped by transform name.  Used for the legacy
        calling convention; ignored when *config* is provided.
    transforms : Mapping[str, list[ExtractionStep]], optional
        Pre-extraction transform definitions grouped by transform name.
        Used for the legacy calling convention; ignored when *config* is
        provided.
    log_dir : str | Path | None
        If set, one log file per EEG recording is written here.
    num_workers : int
        Number of parallel worker processes.  ``1`` forces single-process
        (debug) mode.
    batch_size : int
        Number of epochs to process per batch.  Limits peak memory by
        running the full transform→feature pipeline on a subset of epochs
        at a time.  Default 128.
    debug : bool
        When ``True``, disables multiprocessing for easier debugging.
    console : rich.console.Console
        Console instance used for progress bars.

    Example
    -------
    ::

        from mosaique import FeatureExtractor

        # Simplified: pass config directly
        extractor = FeatureExtractor("config.yaml", num_workers=4)
        df = extractor.extract_feature(epochs, eeg_id="subject_01")

        # Or pass a numpy array directly
        df = extractor.extract_feature(
            data,           # np.ndarray (n_epochs, n_channels, n_times)
            eeg_id="sub01",
            sfreq=200.0,
            ch_names=["Fp1", "C3"],
        )
    """

    def __init__(
        self,
        config: str | Path | dict | None = None,
        features: Mapping[str, list[ExtractionStep]] | None = None,
        transforms: Mapping[str, list[ExtractionStep]] | None = None,
        log_dir: str | Path | None = None,
        num_workers: int = 1,
        batch_size: int = 128,
        debug=False,
        console=Console(),
    ):
        if config is not None:
            if features is not None or transforms is not None:
                raise ValueError(
                    "Cannot specify both 'config' and 'features'/'transforms'"
                )
            from mosaique.config.loader import (
                parse_featureextraction_config,
                resolve_pipeline,
            )

            pipeline = parse_featureextraction_config(config)
            features, transforms = resolve_pipeline(pipeline)
        else:
            if features is None or transforms is None:
                raise ValueError(
                    "Must specify either 'config' or both 'features' and 'transforms'"
                )

        # Feature extraction params for each feature
        self._features = features
        self._required_transforms = list(features.keys())
        self._transforms = transforms

        # List of param names to construct final df
        self._param_names = []

        self._transformed_eeg: np.ndarray

        # Cache wavelet coefficients
        self._cached_coeffs: dict[FrequencyBand, np.ndarray] = {}

        # Initial params with multiple values per params
        self._curr_features: list[ExtractionStep]
        self._curr_transform_group: list[ExtractionStep]

        # Params with single value, ready for function
        self._curr_feature: ExtractionStep
        self._curr_transform_params: ExtractionStep

        # Final list of extracted features
        self._extracted_features: list[pl.DataFrame] = []

        self.log_dir = log_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.debug = debug
        if self.num_workers <= 1:
            self.debug = True
        self.console = console

    @property
    def _feature_param_grid(self) -> list[ExtractionStep]:
        return self._make_param_grid(self._curr_features)

    @property
    def _transform_grid(self) -> list[ExtractionStep]:
        """Make a list of all kwargs given a dict of possible values for kwargs"""
        return self._make_param_grid(self._curr_transform_group)

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

    def _make_param_grid(self, params: list[ExtractionStep]) -> list[ExtractionStep]:
        """Make a list of all kwargs given a dict of possible values for kwargs"""
        grid = []
        for element in params:
            single_param_grid = [
                dict(zip(element.params_for_grid.keys(), a))
                for a in product(*element.params_for_grid.values())
            ]
            for configuration in single_param_grid:
                grid.append(
                    ExtractionStep(
                        name=element.name,
                        function=element.function,
                        params={k: [v] for k, v in configuration.items()},
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

        # Feature param grid is based on curr_features (for a single transform)
        for self._curr_feature in self._feature_param_grid:
            func = self._curr_feature.function
            # For expanded grid entries, each param value is a single-element list;
            # unwrap to get the scalar value for the function call.
            params = {k: v[0] for k, v in self._curr_feature.params.items()}
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

        if param_cols:
            params_expr = (
                pl.struct(list(param_cols))
                .map_elements(
                    lambda x: self._concat_params(x.values()), return_dtype=pl.String
                )
                .alias("params")
            )
        else:
            params_expr = pl.lit("").alias("params")

        if "channel" in feature_df.columns:
            region_side_expr = (
                pl.col("channel")
                .map_elements(get_region_side, skip_nulls=False, return_dtype=pl.String)
                .alias("region_side")
            )
        else:
            region_side_expr = pl.lit(None).cast(pl.String).alias("region_side")

        feature_df = feature_df.with_columns(region_side_expr, params_expr)
        return feature_df

    def extract_feature(
        self,
        eeg: EpochsLike | np.ndarray,
        eeg_id: str,
        sfreq: float | None = None,
        ch_names: list[str] | None = None,
        event_labels: list[str] | None = None,
        timestamps: np.ndarray | None = None,
    ) -> pl.DataFrame:
        """Extract all features from one EEG recording.

        Iterates over every transform / transform-parameter / feature-parameter
        combination, applies the pre-extraction transform, extracts features,
        and concatenates the results into a single DataFrame.

        When the recording contains more epochs than ``batch_size``, the data
        is processed in batches to keep memory bounded.  Results are
        concatenated into a single DataFrame.

        Parameters
        ----------
        eeg : EpochsLike or np.ndarray
            Epoched EEG data.  Either an MNE ``Epochs`` object (or any object
            matching the :class:`~mosaique.extraction.eegdata.EpochsLike`
            protocol) or a raw ``numpy`` array of shape
            ``(n_epochs, n_channels, n_times)``.
        eeg_id : str
            Identifier for the recording (used in logging).
        sfreq : float, optional
            Sampling frequency in Hz.  Required when *eeg* is a numpy array.
        ch_names : list[str], optional
            Channel names.  Only used when *eeg* is a numpy array.  Defaults
            to ``["ch_0", "ch_1", ...]``.
        event_labels : list[str], optional
            Event label per epoch.  Only used when *eeg* is a numpy array.
            Defaults to ``["0", "1", ...]``.
        timestamps : np.ndarray, optional
            Epoch start times.  Only used when *eeg* is a numpy array.
            Defaults to ``np.arange(n_epochs, dtype=float)``.

        Returns
        -------
        polars.DataFrame
            Long-format table with columns ``epoch``, ``timestamp``,
            ``channel``, ``value``, ``feature``, ``region_side``, ``params``,
            plus any transform / feature parameter columns.
        """
        # Normalise input to EegData
        if isinstance(eeg, np.ndarray):
            if sfreq is None:
                raise ValueError(
                    "sfreq is required when passing a numpy array to extract_feature()"
                )
            eeg_data = EegData.from_array(
                eeg,
                sfreq=sfreq,
                ch_names=ch_names,
                event_labels=event_labels,
                timestamps=timestamps,
            )
        else:
            eeg_data = EegData.from_epochs(eeg)

        self._cached_coeffs = {}
        self._cache_tag: tuple = ()
        self._init_logger(eeg_id)
        total_start = time.perf_counter()
        n_epochs = eeg_data.data.shape[0]
        log_str = f"Starting extraction for {eeg_id} (number of epochs: {n_epochs})"
        self.logger.info(log_str)
        self.logger.info("=" * 50)

        n_batches = (n_epochs + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, n_epochs)
            batch_eeg = eeg_data.slice(batch_start, batch_end)

            self.logger.info(
                f"Batch {batch_idx + 1}/{n_batches} (epochs {batch_start}-{batch_end - 1})"
            )

            # _curr_transform_group is a list[ExtractionStep]
            for transform_name, self._curr_transform_group in self._transforms.items():
                transform_group_start = time.perf_counter()
                self.logger.info("-" * 50)
                self.logger.info(f"Starting transform: {transform_name}")
                self.logger.info("-" * 50)

                with Progress(console=self.console, transient=True) as progress:
                    n_features = len(
                        self._make_param_grid(self._features[transform_name])
                    )
                    task_id = progress.add_task(
                        f"[batch {batch_idx + 1}/{n_batches}] Extracting ({transform_name})...",
                        total=len(self._transform_grid) * n_features,
                    )

                    for self._curr_transform_params in self._transform_grid:
                        transform_start = time.perf_counter()

                        # 1. Construct a pre-extraction transform pipeline
                        self._curr_transform = TRANSFORM_REGISTRY[transform_name](
                            self._curr_transform_params,
                            num_workers=self.num_workers,
                            debug=self.debug,
                            console=self.console,
                        )
                        self._param_names.extend(self._curr_transform._params.keys())
                        # Transfer cached coeffs and tag to Transform
                        self._curr_transform._cached_coeffs = self._cached_coeffs
                        self._curr_transform._cache_tag = self._cache_tag

                        # 2. Apply pre-extraction transform
                        self._transformed_eeg = self._curr_transform.transform(
                            batch_eeg
                        )
                        transform_time = time.perf_counter() - transform_start
                        progress.update(
                            task_id,
                            description=f"[batch {batch_idx + 1}/{n_batches}] Extracting ({transform_name})...",
                        )

                        self.logger.info(
                            f"Transform {transform_name}: {self._curr_transform_params} completed in {transform_time:.2f}s"
                        )

                        # 3. Extract all features for this transform
                        self._curr_features = self._features[transform_name]
                        features = self._extract_for_single_transform(
                            self._transformed_eeg, progress=progress, task_id=task_id
                        )
                        # 4. Add hyperparameters to dataframe
                        self._transformed_df = self._curr_transform.complete_df(
                            features
                        )

                        # 5. Append to list of dataframes
                        self._extracted_features.append(self._transformed_df)

                        # Transfer cached coeffs and tag back
                        self._cached_coeffs = self._curr_transform._cached_coeffs
                        self._cache_tag = self._curr_transform._cache_tag

                    transform_group_time = (
                        time.perf_counter() - transform_group_start
                    ) / 60
                    self.logger.info(
                        f"All features for {transform_name} extracted in {(transform_group_time):.2f}m"
                    )

            # Free CWT cache between batches
            self._cached_coeffs = {}
            self._cache_tag = ()

        final_features = pl.concat(self._extracted_features, how="diagonal_relaxed")
        final_features_cleaned = self._clean_features_df(final_features)

        total_time = time.perf_counter() - total_start
        self.logger.info(f"Total extraction time: {total_time / 60:.2f}m")

        return final_features_cleaned
