#!/usr/bin/env python
"""MNE-only feature extraction script for benchmarking against mosaique.

Replicates the same features as feature_extraction.py but replaces the
mosaique extraction pipeline with direct MNE and numpy calls:

  simple        — raw epoch data, features applied per (epoch, channel)
  tf_decomp     — band-limited amplitude via mne.time_frequency.tfr_array_morlet
  connectivity  — PLI matrices from complex Morlet TFR, then graph features

Feature functions are imported directly from mosaique.features to ensure
identical computation — only the orchestration and transforms differ.
"""

import datetime
import time
from pathlib import Path
from typing import Any, Callable

import mne
import psutil
import mne.time_frequency
import numpy as np
import polars as pl

from mosaique.features import STANDARD_BANDS, TF_BANDS, connectivity as conn_feats
from mosaique.features import univariate as univ
from mosaique.utils.eeg_helpers import get_region_side

_PROC = psutil.Process()


def _snap() -> dict:
    """Capture a process resource snapshot."""
    cpu = _PROC.cpu_times()
    return {
        "wall": time.perf_counter(),
        "cpu_user": cpu.user,
        "cpu_sys": cpu.system,
        "rss_mb": _PROC.memory_info().rss / 1024**2,
    }


def _report(start: dict, end: dict, label: str) -> None:
    wall = end["wall"] - start["wall"]
    cpu = (end["cpu_user"] - start["cpu_user"]) + (end["cpu_sys"] - start["cpu_sys"])
    print(
        f"  [{label}] "
        f"wall={wall:.2f}s | cpu={cpu:.2f}s | "
        f"RAM start={start['rss_mb']:.1f} MB → end={end['rss_mb']:.1f} MB"
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# ---------------------------------------------------------------------------
# Feature configuration (mirrors features_config.yaml)
# ---------------------------------------------------------------------------
BAND_POWER_FREQS = STANDARD_BANDS
CONN_BANDS = TF_BANDS

# Morlet TFR parameters
N_FREQS_PER_BAND = 20   # log-spaced frequencies sampled within each band
N_CYCLES = 7.0          # Morlet wavelet cycles (used for both TF and connectivity)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_epoch_metadata(epochs: mne.Epochs) -> tuple[list[str], np.ndarray]:
    """Return event labels and start timestamps for each epoch."""
    inv_event_id = {v: k for k, v in epochs.event_id.items()}
    event_labels = [inv_event_id[eid] for eid in epochs.events[:, 2]]

    sfreq = epochs.info["sfreq"]
    meas_date = epochs.info["meas_date"]
    timestamps = np.array([
        np.datetime64(
            (datetime.timedelta(seconds=s / sfreq) + meas_date).replace(tzinfo=None)
        )
        for s in epochs.events[:, 0]
    ])
    return event_labels, timestamps


def _build_per_channel_df(
    event_labels: list[str],
    timestamps: np.ndarray,
    ch_names: list[str],
    values: np.ndarray,          # shape (n_epochs, n_channels)
    feature_name: str,
    pre_transform: str,
    params: dict,
    computation_time: float,
    freqs_label: str | None = None,
) -> pl.DataFrame:
    """Build a long-format DataFrame for per-(epoch, channel) feature values."""
    n_epochs = len(event_labels)
    n_ch = len(ch_names)

    df = pl.DataFrame({
        "epoch": np.repeat(event_labels, n_ch),
        "timestamp": np.repeat(timestamps, n_ch),
        "channel": np.tile(ch_names, n_epochs),
        "value": values.flatten().tolist(),
    })
    return _finalize_df(df, feature_name, pre_transform, params, computation_time, freqs_label)


def _build_per_epoch_df(
    event_labels: list[str],
    timestamps: np.ndarray,
    values: np.ndarray,          # shape (n_epochs,)
    feature_name: str,
    pre_transform: str,
    params: dict,
    computation_time: float,
    freqs_label: str | None = None,
) -> pl.DataFrame:
    """Build a long-format DataFrame for per-epoch feature values."""
    df = pl.DataFrame({
        "epoch": event_labels,
        "timestamp": timestamps,
        "value": values.tolist(),
    })
    return _finalize_df(df, feature_name, pre_transform, params, computation_time, freqs_label)


def _finalize_df(
    df: pl.DataFrame,
    feature_name: str,
    pre_transform: str,
    params: dict,
    computation_time: float,
    freqs_label: str | None,
) -> pl.DataFrame:
    """Attach metadata columns to match the mosaique output schema."""
    df = df.with_columns(
        feature=pl.lit(feature_name),
        pre_transform=pl.lit(pre_transform),
        computation_time=pl.lit(computation_time),
    )
    if freqs_label is not None:
        df = df.with_columns(freqs=pl.lit(freqs_label))

    param_strs: list[str] = []
    for k, v in params.items():
        df = df.with_columns(pl.lit(v).alias(k))
        param_strs.append(str(v))
    df = df.with_columns(params=pl.lit("_".join(param_strs)))

    if "channel" in df.columns:
        df = df.with_columns(
            pl.col("channel")
            .map_elements(get_region_side, skip_nulls=False, return_dtype=pl.String)
            .alias("region_side")
        )

    return df


def _band_freqs(band: tuple[float, float]) -> np.ndarray:
    """Log-spaced frequency grid within a band for Morlet TFR."""
    return np.logspace(np.log10(band[0]), np.log10(band[1]), N_FREQS_PER_BAND)


# ---------------------------------------------------------------------------
# Simple features (raw EEG, no transform)
# ---------------------------------------------------------------------------

def extract_simple_features(epochs: mne.Epochs) -> pl.DataFrame:
    """Extract features from raw epoch data using MNE-provided arrays.

    Equivalent to the ``simple`` framework in the mosaique pipeline.
    """
    data = epochs.get_data()          # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    event_labels, timestamps = get_epoch_metadata(epochs)
    ch_names = list(epochs.ch_names)
    n_epochs, n_channels, _ = data.shape

    all_dfs: list[pl.DataFrame] = []

    # ------------------------------------------------------------------
    # Scalar features (one value per epoch × channel)
    # ------------------------------------------------------------------
    scalar_specs: list[tuple[str, Callable[..., Any], dict]] = [
        ("line_length",         univ.line_length,         {}),
        ("peak_alpha",          univ.peak_alpha,           {}),
        ("spectral_entropy",    univ.spectral_entropy,     {}),
        ("permutation_entropy", univ.permutation_entropy,  {"k": 3}),
        ("hurst_exp",           univ.hurst_exp,            {}),
    ]

    for feat_name, func, params in scalar_specs:
        t0 = time.perf_counter()
        values = np.zeros((n_epochs, n_channels))
        for i in range(n_epochs):
            for j in range(n_channels):
                values[i, j] = func(data[i, j], sfreq=sfreq, **params)  # type: ignore[operator]
        elapsed = time.perf_counter() - t0

        df = _build_per_channel_df(
            event_labels, timestamps, ch_names, values,
            feat_name, "simple", params, elapsed,
        )
        all_dfs.append(df)
        print(f"    simple/{feat_name}: {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Band power (one value per epoch × channel × band)
    # Compute once per epoch/channel, then split by band.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    band_values: dict[tuple, np.ndarray] = {b: np.zeros((n_epochs, n_channels)) for b in BAND_POWER_FREQS}
    for i in range(n_epochs):
        for j in range(n_channels):
            result = univ.band_power(data[i, j], freqs=BAND_POWER_FREQS, sfreq=sfreq)
            for band, power in result.items():
                band_values[band][i, j] = power
    elapsed = time.perf_counter() - t0

    for band, values in band_values.items():
        df = _build_per_channel_df(
            event_labels, timestamps, ch_names, values,
            "band_power", "simple", {}, elapsed, freqs_label=str(band),
        )
        all_dfs.append(df)
    print(f"    simple/band_power: {elapsed:.2f}s")

    return pl.concat(all_dfs, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# TF-decomposition features (Morlet TFR → band-limited amplitude)
# ---------------------------------------------------------------------------

def extract_tf_features(epochs: mne.Epochs) -> pl.DataFrame:
    """Extract features from band-limited signals via MNE Morlet TFR.

    Replaces the mosaique ``cwt_eeg`` (pywt cmor1.5-1.0) transform with
    ``mne.time_frequency.tfr_array_morlet``.  Within each band the complex
    TFR is collapsed to a single amplitude time series by taking the
    magnitude and averaging across the frequency axis, then downsampled
    by the same factor as mosaique's ``simplify_cwt_coeffs``.
    """
    data = epochs.get_data()          # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    event_labels, timestamps = get_epoch_metadata(epochs)
    ch_names = list(epochs.ch_names)
    n_epochs, n_channels, _ = data.shape

    tf_specs: list[tuple[str, Callable[..., Any], dict]] = [
        ("sample_entropy",      univ.sample_entropy,      {"m": 2, "r": 0.2}),
        ("approximate_entropy", univ.approximate_entropy, {"m": 3, "r": 0.2}),
        ("fuzzy_entropy",       univ.fuzzy_entropy,        {"m": 2, "r": 0.2}),
        ("line_length",         univ.line_length,          {}),
        ("corr_dim",            univ.corr_dim,             {"embed_dim": 2}),
    ]

    all_dfs: list[pl.DataFrame] = []

    for band in TF_BANDS:
        freqs_in_band = _band_freqs(band)
        band_label = str((band[0], band[1]))

        # Complex Morlet TFR: (n_epochs, n_channels, n_freqs, n_times)
        # mne stubs have an imprecise return type; it is always ndarray here.
        tfr: np.ndarray = mne.time_frequency.tfr_array_morlet(  # type: ignore[assignment]
            data, sfreq, freqs_in_band,
            n_cycles=N_CYCLES, output="complex", verbose=False,
        )

        # Collapse to band amplitude: average |TFR| across frequency axis
        # → (n_epochs, n_channels, n_times)
        band_amplitude = np.abs(tfr).mean(axis=2)

        # Downsample to match mosaique's simplify_cwt_coeffs
        level = int(np.floor(np.log2(sfreq / (2 * band[1]))))
        downsample_factor = max(1, 2 * level)
        if downsample_factor > 1:
            band_amplitude = band_amplitude[..., ::downsample_factor]

        for feat_name, func, params in tf_specs:
            t0 = time.perf_counter()
            values = np.zeros((n_epochs, n_channels))
            for i in range(n_epochs):
                for j in range(n_channels):
                    values[i, j] = func(band_amplitude[i, j], sfreq=sfreq, **params)  # type: ignore[operator]
            elapsed = time.perf_counter() - t0

            df = _build_per_channel_df(
                event_labels, timestamps, ch_names, values,
                feat_name, "cwt", params, elapsed, freqs_label=band_label,
            )
            all_dfs.append(df)
            print(f"    tf/{band_label}/{feat_name}: {elapsed:.2f}s")

    return pl.concat(all_dfs, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# Connectivity features (PLI from complex Morlet TFR → graph metrics)
# ---------------------------------------------------------------------------

def extract_connectivity_features(epochs: mne.Epochs) -> pl.DataFrame:
    """Extract graph features from PLI connectivity matrices.

    Replaces mosaique's pywt-based ``cwt_spectral_connectivity`` with
    ``mne.time_frequency.tfr_array_morlet`` (complex output).  The PLI
    formula is identical to the mosaique implementation:

        PLI_ij = |mean_t [ sign( Im( sum_f X_i(t,f) · X_j*(t,f) ) ) ]|
    """
    data = epochs.get_data()          # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    event_labels, timestamps = get_epoch_metadata(epochs)
    n_epochs, _, _ = data.shape

    graph_specs: list[tuple[str, Callable[..., Any], dict]] = [
        ("average_clustering",         conn_feats.average_clustering,         {}),
        ("average_degree",             conn_feats.average_degree,             {}),
        ("global_efficiency",          conn_feats.global_efficiency,          {}),
        ("average_shortest_path_length", conn_feats.average_shortest_path_length, {}),
    ]

    all_dfs: list[pl.DataFrame] = []

    for band in CONN_BANDS:
        freqs_in_band = _band_freqs(band)
        band_label = str((band[0], band[1]))

        # Complex Morlet TFR: (n_epochs, n_channels, n_freqs, n_times)
        # mne stubs have an imprecise return type; it is always ndarray here.
        tfr: np.ndarray = mne.time_frequency.tfr_array_morlet(  # type: ignore[assignment]
            data, sfreq, freqs_in_band,
            n_cycles=N_CYCLES, output="complex", verbose=False,
        )

        # PLI: same formula as mosaique's connectivity_from_coeff
        # Reshape to (n_epochs, n_times, n_channels, n_freqs) for batched matmul
        tfr_t = tfr.transpose(0, 3, 1, 2)

        # CSD summed over frequencies: (n_epochs, n_times, n_channels, n_channels)
        csd = np.matmul(tfr_t, tfr_t.conj().transpose(0, 1, 3, 2))

        # PLI averaged over time: (n_epochs, n_channels, n_channels)
        pli_matrices = np.abs(np.mean(np.sign(np.imag(csd)), axis=1))

        for feat_name, func, params in graph_specs:
            t0 = time.perf_counter()
            values = np.array([func(pli_matrices[ep], **params) for ep in range(n_epochs)])  # type: ignore[operator]
            elapsed = time.perf_counter() - t0

            df = _build_per_epoch_df(
                event_labels, timestamps, values,
                feat_name, "pli", params, elapsed, freqs_label=band_label,
            )
            all_dfs.append(df)
            print(f"    conn/{band_label}/{feat_name}: {elapsed:.2f}s")

    return pl.concat(all_dfs, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    edf_files = sorted(DATA_DIR.rglob("*.edf"))[:2]
    if not edf_files:
        print(f"No EDF files found in {DATA_DIR}")
        return

    print(f"Processing {len(edf_files)} EDF file(s) from {DATA_DIR}\n")

    all_features: list[pl.DataFrame] = []
    overall_start = _snap()

    for edf_path in edf_files:
        print(f"Processing: {edf_path.name}")
        file_start = _snap()

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.crop(tmax=min(120.0, raw.times[-1]))
        raw.filter(1.0, 50.0, verbose=False)
        epochs = mne.make_fixed_length_epochs(raw, duration=5.0, verbose=False)
        epochs.load_data()
        print(
            f"  {len(epochs)} epochs, {len(epochs.ch_names)} channels, "
            f"sfreq={epochs.info['sfreq']} Hz"
        )

        print("  [simple features]")
        df_simple = extract_simple_features(epochs)

        print("  [tf-decomposition features]")
        df_tf = extract_tf_features(epochs)

        print("  [connectivity features]")
        df_conn = extract_connectivity_features(epochs)

        df = pl.concat([df_simple, df_tf, df_conn], how="diagonal_relaxed")
        df = df.with_columns(file=pl.lit(edf_path.name))
        all_features.append(df)

        file_end = _snap()
        print(f"  Extracted {len(df)} feature rows")
        _report(file_start, file_end, edf_path.name)
        print()

    overall_end = _snap()

    result = pl.concat(all_features, how="diagonal_relaxed")
    print("=" * 60)
    print(f"Total feature rows: {len(result)}")
    print(f"Columns: {result.columns}")
    print("=" * 60)
    print(result.head(20))
    print()
    _report(overall_start, overall_end, "TOTAL")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "features_mne.parquet"
    result.write_parquet(output_path)
    print(f"\nFeatures saved to {output_path}")


if __name__ == "__main__":
    main()
