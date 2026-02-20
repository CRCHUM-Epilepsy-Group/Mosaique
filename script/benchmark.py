#!/usr/bin/env python
"""Systematic benchmark: mosaique vs MNE-native feature extraction.

Runs both backends across a parameter grid (N files, feature groups,
N features, N workers), collects structured timing/memory data, and
outputs a Parquet file ready for analysis and plotting.

Usage
-----
    uv run script/benchmark.py              # full benchmark
    uv run script/benchmark.py --quick      # smoke test (1 file, 1 worker, 1 rep)
    uv run script/benchmark.py --groups simple connectivity
    uv run script/benchmark.py --workers 1 4
    uv run script/benchmark.py --max-files 3
    uv run script/benchmark.py --repetitions 5
    uv run script/benchmark.py --no-incremental
"""

from __future__ import annotations

import argparse
import gc
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mne
import numpy as np
import polars as pl
import psutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from mosaique import FeatureExtractor
from mosaique.config.types import PreGridParams
from mosaique.features import connectivity as conn_feats
from mosaique.features import univariate as univ
from mosaique.features.timefrequency import cwt_eeg
from mosaique.features.connectivity import connectivity_from_coeff as _connectivity_from_coeff


def connectivity_from_coeff(coefficients_or_tuple: Any, **kwargs: Any) -> Any:
    """Wrapper around connectivity_from_coeff that unwraps the cwt_eeg tuple.

    ConnectivityTransform passes the raw cwt_eeg output to the framework
    function. cwt_eeg returns (coeff_dict, reconstr_array); this wrapper
    extracts just the coeff dict before forwarding to connectivity_from_coeff.
    """
    if isinstance(coefficients_or_tuple, tuple):
        coefficients = coefficients_or_tuple[0]
    else:
        coefficients = coefficients_or_tuple
    return _connectivity_from_coeff(coefficients, **kwargs)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# ---------------------------------------------------------------------------
# Canonical feature lists (shared by both backends)
# ---------------------------------------------------------------------------
SIMPLE_FEATURES: list[str] = [
    "line_length",
    "band_power",
    "peak_alpha",
    "spectral_entropy",
    "permutation_entropy",
    "hurst_exp",
]

TF_FEATURES: list[str] = [
    "sample_entropy",
    "approximate_entropy",
    "fuzzy_entropy",
    "line_length",
    "corr_dim",
]

CONNECTIVITY_FEATURES: list[str] = [
    "average_clustering",
    "average_degree",
    "global_efficiency",
    "average_shortest_path_length",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "simple": SIMPLE_FEATURES,
    "tf_decomposition": TF_FEATURES,
    "connectivity": CONNECTIVITY_FEATURES,
}

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BAND_POWER_FREQS: list[tuple[float, float]] = [
    (1, 4), (4, 8), (8, 13), (13, 30), (30, 50),
]
TF_BANDS: list[tuple[float, float]] = [(4, 8), (8, 13), (13, 30), (30, 50)]
CONN_BANDS: list[tuple[float, float]] = [(4, 8), (8, 13), (13, 30), (30, 50)]
N_FREQS_PER_BAND = 20
N_CYCLES = 7.0

# ---------------------------------------------------------------------------
# MNE feature specs: (name, function, params)
# ---------------------------------------------------------------------------
MNE_SIMPLE_SPECS: list[tuple[str, Callable[..., Any], dict[str, Any]]] = [
    ("line_length", univ.line_length, {}),
    ("band_power", univ.band_power, {"freqs": BAND_POWER_FREQS}),
    ("peak_alpha", univ.peak_alpha, {}),
    ("spectral_entropy", univ.spectral_entropy, {}),
    ("permutation_entropy", univ.permutation_entropy, {"k": 3}),
    ("hurst_exp", univ.hurst_exp, {}),
]

MNE_TF_SPECS: list[tuple[str, Callable[..., Any], dict[str, Any]]] = [
    ("sample_entropy", univ.sample_entropy, {"m": 2, "r": 0.2}),
    ("approximate_entropy", univ.approximate_entropy, {"m": 3, "r": 0.2}),
    ("fuzzy_entropy", univ.fuzzy_entropy, {"m": 2, "r": 0.2}),
    ("line_length", univ.line_length, {}),
    ("corr_dim", univ.corr_dim, {"embed_dim": 2}),
]

MNE_CONN_SPECS: list[tuple[str, Callable[..., Any], dict[str, Any]]] = [
    ("average_clustering", conn_feats.average_clustering, {}),
    ("average_degree", conn_feats.average_degree, {}),
    ("global_efficiency", conn_feats.global_efficiency, {}),
    ("average_shortest_path_length", conn_feats.average_shortest_path_length, {}),
]

# ---------------------------------------------------------------------------
# Mosaique feature/framework param builders
# ---------------------------------------------------------------------------
MOSAIQUE_SIMPLE_PARAMS: list[tuple[str, str, dict[str, Any] | None]] = [
    ("line_length", "univariate.line_length", None),
    ("band_power", "univariate.band_power", {"freqs": [BAND_POWER_FREQS]}),
    ("peak_alpha", "univariate.peak_alpha", None),
    ("spectral_entropy", "univariate.spectral_entropy", None),
    ("permutation_entropy", "univariate.permutation_entropy", {"k": [3]}),
    ("hurst_exp", "univariate.hurst_exp", None),
]

MOSAIQUE_TF_PARAMS: list[tuple[str, str, dict[str, Any] | None]] = [
    ("sample_entropy", "univariate.sample_entropy", {"m": [2], "r": [0.2]}),
    ("approximate_entropy", "univariate.approximate_entropy", {"m": [3], "r": [0.2]}),
    ("fuzzy_entropy", "univariate.fuzzy_entropy", {"m": [2], "r": [0.2]}),
    ("line_length", "univariate.line_length", None),
    ("corr_dim", "univariate.corr_dim", {"embed_dim": [2]}),
]

MOSAIQUE_CONN_PARAMS: list[tuple[str, str, dict[str, Any] | None]] = [
    ("average_clustering", "connectivity.average_clustering", None),
    ("average_degree", "connectivity.average_degree", None),
    ("global_efficiency", "connectivity.global_efficiency", None),
    ("average_shortest_path_length", "connectivity.average_shortest_path_length", None),
]


def _resolve_function(dotted_path: str) -> Callable[..., Any]:
    """Resolve a dotted function path relative to mosaique.features."""
    module_name, func_name = dotted_path.rsplit(".", 1)
    mod = __import__(f"mosaique.features.{module_name}", fromlist=[func_name])
    return getattr(mod, func_name)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_epoch_edf(
    edf_path: Path,
    epoch_duration: float = 5.0,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    tmax: float = 120.0,
) -> mne.Epochs:
    """Load an EDF file and segment it into fixed-length epochs."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.crop(tmax=min(tmax, raw.times[-1]))
    raw.filter(l_freq, h_freq, verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, verbose=False)
    epochs.load_data()
    return epochs


def discover_edf_files(max_files: int) -> list[Path]:
    """Return up to *max_files* EDF paths from the test data directory."""
    files = sorted(DATA_DIR.rglob("*.edf"))[:max_files]
    if not files:
        raise FileNotFoundError(f"No EDF files found in {DATA_DIR}")
    return files


# ---------------------------------------------------------------------------
# Measurement harness
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Timing and memory metrics for a single benchmark run."""
    wall_s: float
    cpu_s: float
    peak_rss_mb: float


def measure_run(func: Callable[[], None]) -> RunMetrics:
    """Execute *func* and return wall time, CPU time, and peak RSS.

    Peak RSS is sampled via a background thread at ~100 ms intervals,
    including child processes (to capture multiprocessing workers).
    """
    proc = psutil.Process()
    peak_rss = 0.0
    stop_event = threading.Event()

    def _monitor() -> None:
        nonlocal peak_rss
        while not stop_event.is_set():
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak_rss = max(peak_rss, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            stop_event.wait(0.1)

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

    cpu_before = proc.cpu_times()
    wall_start = time.perf_counter()

    func()

    wall_end = time.perf_counter()
    cpu_after = proc.cpu_times()
    stop_event.set()
    monitor_thread.join(timeout=2.0)

    wall_s = wall_end - wall_start
    cpu_s = (
        (cpu_after.user - cpu_before.user)
        + (cpu_after.system - cpu_before.system)
    )
    return RunMetrics(
        wall_s=wall_s,
        cpu_s=cpu_s,
        peak_rss_mb=peak_rss / 1024**2,
    )


# ---------------------------------------------------------------------------
# MNE backend runners
# ---------------------------------------------------------------------------

def _band_freqs(band: tuple[float, float]) -> np.ndarray:
    """Log-spaced frequency grid within a band for Morlet TFR."""
    return np.logspace(np.log10(band[0]), np.log10(band[1]), N_FREQS_PER_BAND)


def _mne_extract_simple(
    epochs_list: list[mne.Epochs],
    specs: list[tuple[str, Callable[..., Any], dict[str, Any]]],
) -> None:
    """Run simple MNE feature extraction (discard results)."""
    for epochs in epochs_list:
        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]
        n_epochs, n_channels, _ = data.shape

        for feat_name, func, params in specs:
            if feat_name == "band_power":
                for i in range(n_epochs):
                    for j in range(n_channels):
                        func(data[i, j], sfreq=sfreq, **params)
            else:
                for i in range(n_epochs):
                    for j in range(n_channels):
                        func(data[i, j], sfreq=sfreq, **params)


def _mne_extract_tf(
    epochs_list: list[mne.Epochs],
    specs: list[tuple[str, Callable[..., Any], dict[str, Any]]],
) -> None:
    """Run TF-decomposition MNE feature extraction (discard results)."""
    for epochs in epochs_list:
        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]
        n_epochs, n_channels, _ = data.shape

        for band in TF_BANDS:
            freqs_in_band = _band_freqs(band)
            tfr: np.ndarray = mne.time_frequency.tfr_array_morlet(
                data, sfreq, freqs_in_band,
                n_cycles=N_CYCLES, output="complex", verbose=False,
            )  # type: ignore[assignment]
            band_amplitude = np.abs(tfr).mean(axis=2)

            level = int(np.floor(np.log2(sfreq / (2 * band[1]))))
            downsample_factor = max(1, 2 * level)
            if downsample_factor > 1:
                band_amplitude = band_amplitude[..., ::downsample_factor]

            for _, func, params in specs:
                for i in range(n_epochs):
                    for j in range(n_channels):
                        func(band_amplitude[i, j], sfreq=sfreq, **params)


def _mne_extract_conn(
    epochs_list: list[mne.Epochs],
    specs: list[tuple[str, Callable[..., Any], dict[str, Any]]],
) -> None:
    """Run connectivity MNE feature extraction (discard results)."""
    for epochs in epochs_list:
        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]
        n_epochs = data.shape[0]

        for band in CONN_BANDS:
            freqs_in_band = _band_freqs(band)
            tfr: np.ndarray = mne.time_frequency.tfr_array_morlet(
                data, sfreq, freqs_in_band,
                n_cycles=N_CYCLES, output="complex", verbose=False,
            )  # type: ignore[assignment]

            tfr_t = tfr.transpose(0, 3, 1, 2)
            csd = np.matmul(tfr_t, tfr_t.conj().transpose(0, 1, 3, 2))
            pli_matrices = np.abs(np.mean(np.sign(np.imag(csd)), axis=1))

            for _, func, params in specs:
                for ep in range(n_epochs):
                    func(pli_matrices[ep], **params)


# ---------------------------------------------------------------------------
# Mosaique backend runner
# ---------------------------------------------------------------------------

def _build_mosaique_config(
    group: str, n_features: int
) -> tuple[dict[str, list[PreGridParams]], dict[str, list[PreGridParams]]]:
    """Build features/frameworks dicts of PreGridParams for a feature subset."""
    if group == "simple":
        param_defs = MOSAIQUE_SIMPLE_PARAMS[:n_features]
    elif group == "tf_decomposition":
        param_defs = MOSAIQUE_TF_PARAMS[:n_features]
    elif group == "connectivity":
        param_defs = MOSAIQUE_CONN_PARAMS[:n_features]
    else:
        raise ValueError(f"Unknown group: {group}")

    feature_params: list[PreGridParams] = []
    for name, dotted_path, params in param_defs:
        func = _resolve_function(dotted_path)
        feature_params.append(PreGridParams(
            name=name,
            function=func,
            params=params if params is not None else {},
        ))

    # Build the corresponding framework
    fw_params: list[PreGridParams]
    if group == "simple":
        fw_params = [PreGridParams(name="simple", function=None, params={})]  # type: ignore[arg-type]
    elif group == "tf_decomposition":
        fw_params = [PreGridParams(
            name="cwt",
            function=cwt_eeg,
            params={
                "freqs": [TF_BANDS],
                "skip_reconstr": [True],
                "skip_complex": [True],
            },
        )]
    else:  # connectivity
        # Notes:
        # - skip_reconstr=True: PLI doesn't need the reconstructed signal
        # - skip_complex=False: PLI requires complex wavelet coefficients
        # - don't pass method="pli": that would override cwt_eeg's fft/conv method
        fw_params = [PreGridParams(
            name="pli",
            function=connectivity_from_coeff,
            params={
                "freqs": [CONN_BANDS],
                "wavelet": "cmor1.5-1.0",
                "skip_reconstr": True,
            },
        )]

    features = {group: feature_params}
    frameworks = {group: fw_params}
    return features, frameworks


def _run_mosaique(
    epochs_list: list[mne.Epochs],
    group: str,
    n_features: int,
    n_workers: int,
) -> None:
    """Run mosaique feature extraction (discard results)."""
    features, frameworks = _build_mosaique_config(group, n_features)
    for epochs in epochs_list:
        extractor = FeatureExtractor(
            features,
            frameworks,
            num_workers=n_workers,
            console=Console(quiet=True),
        )
        extractor.extract_feature(epochs, eeg_id="bench")


# ---------------------------------------------------------------------------
# Benchmark grid
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """A single benchmark configuration to run."""
    backend: str
    n_files: int
    feature_group: str
    n_features: int
    feature_names: str
    n_workers: int


def build_benchmark_grid(
    groups: list[str],
    max_files: int,
    workers: list[int],
    incremental: bool,
    max_n_features: int | None = None,
) -> list[BenchmarkConfig]:
    """Build the Cartesian product of benchmark configurations."""
    configs: list[BenchmarkConfig] = []
    n_files_range = range(1, max_files + 1)

    for group in groups:
        all_features = FEATURE_GROUPS[group]
        max_feat = len(all_features)
        if max_n_features is not None:
            max_feat = min(max_feat, max_n_features)

        if incremental:
            n_features_range = range(1, max_feat + 1)
        else:
            n_features_range = range(max_feat, max_feat + 1)

        for n_files in n_files_range:
            for n_feat in n_features_range:
                feat_names = ",".join(all_features[:n_feat])

                # MNE: always 1 worker
                configs.append(BenchmarkConfig(
                    backend="mne",
                    n_files=n_files,
                    feature_group=group,
                    n_features=n_feat,
                    feature_names=feat_names,
                    n_workers=1,
                ))

                # Mosaique: each worker count
                for w in workers:
                    configs.append(BenchmarkConfig(
                        backend="mosaique",
                        n_files=n_files,
                        feature_group=group,
                        n_features=n_feat,
                        feature_names=feat_names,
                        n_workers=w,
                    ))

    return configs


# ---------------------------------------------------------------------------
# System metadata
# ---------------------------------------------------------------------------

def collect_system_metadata() -> dict[str, str]:
    """Collect system info for reproducibility."""
    cpu_model = platform.processor() or platform.machine()

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        git_hash = "unknown"

    return {
        "cpu_model": cpu_model,
        "n_cores": str(psutil.cpu_count(logical=False) or "?"),
        "n_threads": str(psutil.cpu_count(logical=True) or "?"),
        "ram_gb": f"{psutil.virtual_memory().total / 1024**3:.1f}",
        "python_version": platform.python_version(),
        "mne_version": getattr(mne, "__version__", "unknown"),
        "mosaique_version": _get_mosaique_version(),
        "git_commit": git_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _get_mosaique_version() -> str:
    try:
        from importlib.metadata import version
        return version("mosaique")
    except Exception:
        return "dev"


# ---------------------------------------------------------------------------
# MNE spec slicing
# ---------------------------------------------------------------------------

def _get_mne_specs(
    group: str, n_features: int
) -> list[tuple[str, Callable[..., Any], dict[str, Any]]]:
    """Return the first *n_features* MNE specs for *group*."""
    if group == "simple":
        return MNE_SIMPLE_SPECS[:n_features]
    elif group == "tf_decomposition":
        return MNE_TF_SPECS[:n_features]
    elif group == "connectivity":
        return MNE_CONN_SPECS[:n_features]
    else:
        raise ValueError(f"Unknown group: {group}")


_MNE_RUNNERS: dict[str, Callable[..., None]] = {
    "simple": _mne_extract_simple,
    "tf_decomposition": _mne_extract_tf,
    "connectivity": _mne_extract_conn,
}


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    configs: list[BenchmarkConfig],
    edf_files: list[Path],
    repetitions: int,
    console: Console,
    warmup: bool = True,
) -> pl.DataFrame:
    """Execute all benchmark configurations and return results as a DataFrame."""
    metadata = collect_system_metadata()

    # Pre-load all epoch sets (not measured — shared across runs)
    console.print("[bold]Loading EDF files...[/bold]")
    all_epochs: list[mne.Epochs] = []
    for f in edf_files:
        all_epochs.append(load_and_epoch_edf(f))
    console.print(f"  Loaded {len(all_epochs)} files\n")

    warmup_count = 1 if warmup else 0
    total_runs = len(configs) * (repetitions + warmup_count)
    rows: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking", total=total_runs)

        for cfg in configs:
            epochs_subset = all_epochs[:cfg.n_files]

            # Build the run function
            run_func: Callable[[], None]
            if cfg.backend == "mne":
                specs = _get_mne_specs(cfg.feature_group, cfg.n_features)
                runner = _MNE_RUNNERS[cfg.feature_group]
                run_func = lambda _r=runner, _e=epochs_subset, _s=specs: _r(_e, _s)  # type: ignore[misc]
            else:
                run_func = lambda _e=epochs_subset, _g=cfg.feature_group, _f=cfg.n_features, _w=cfg.n_workers: _run_mosaique(_e, _g, _f, _w)  # type: ignore[misc]

            # Optional warm-up run (discarded)
            if warmup:
                gc.collect()
                measure_run(run_func)
                progress.advance(task)

            # Measured repetitions
            for rep in range(1, repetitions + 1):
                gc.collect()
                metrics = measure_run(run_func)

                row: dict[str, Any] = {
                    "backend": cfg.backend,
                    "n_files": cfg.n_files,
                    "feature_group": cfg.feature_group,
                    "n_features": cfg.n_features,
                    "feature_names": cfg.feature_names,
                    "n_workers": cfg.n_workers,
                    "repetition": rep,
                    "wall_s": metrics.wall_s,
                    "cpu_s": metrics.cpu_s,
                    "peak_rss_mb": metrics.peak_rss_mb,
                }
                row.update(metadata)
                rows.append(row)
                progress.advance(task)

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_summary(df: pl.DataFrame, console: Console) -> None:
    """Print a summary table with median wall times and speedup ratios."""
    # Compute medians per (backend, feature_group, n_files, n_workers) at max features
    max_feat = (
        df.group_by("feature_group")
        .agg(pl.col("n_features").max().alias("max_nf"))
    )
    df_max = df.join(max_feat, on="feature_group").filter(
        pl.col("n_features") == pl.col("max_nf")
    )

    mne_medians = (
        df_max.filter(pl.col("backend") == "mne")
        .group_by(["feature_group", "n_files"])
        .agg(pl.col("wall_s").median().alias("mne_wall_s"))
    )

    mosaique_medians = (
        df_max.filter(pl.col("backend") == "mosaique")
        .group_by(["feature_group", "n_files", "n_workers"])
        .agg(
            pl.col("wall_s").median().alias("mos_wall_s"),
            pl.col("wall_s").std().alias("mos_std"),
        )
    )

    # Best mosaique config per (group, n_files)
    best_mos = (
        mosaique_medians
        .sort("mos_wall_s")
        .group_by(["feature_group", "n_files"])
        .first()
    )

    combined = best_mos.join(mne_medians, on=["feature_group", "n_files"])
    combined = combined.with_columns(
        (pl.col("mne_wall_s") / pl.col("mos_wall_s")).alias("speedup")
    ).sort(["feature_group", "n_files"])

    table = Table(title="Benchmark Summary (all features, best worker config)")
    table.add_column("Group")
    table.add_column("N files", justify="right")
    table.add_column("Workers", justify="right")
    table.add_column("MNE (s)", justify="right")
    table.add_column("Mosaique (s)", justify="right")
    table.add_column("Speedup", justify="right")

    for row in combined.iter_rows(named=True):
        speedup = row["speedup"]
        color = "green" if speedup > 1 else "red"
        table.add_row(
            row["feature_group"],
            str(row["n_files"]),
            str(row["n_workers"]),
            f"{row['mne_wall_s']:.2f}",
            f"{row['mos_wall_s']:.2f}",
            f"[{color}]{speedup:.2f}x[/{color}]",
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark mosaique vs MNE-native feature extraction",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: 1 file, 1 worker, 1 repetition, no incremental",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(FEATURE_GROUPS.keys()),
        default=list(FEATURE_GROUPS.keys()),
        help="Feature groups to benchmark (default: all)",
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Worker counts for mosaique (default: 1 2 4)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Maximum number of EDF files to use (default: 5)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of measured repetitions per config (default: 3)",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Skip feature-count sweep (only benchmark max features per group)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: script/output/benchmark.parquet)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    quick_n_features: int | None = None
    warmup = True
    if args.quick:
        args.max_files = 1
        args.workers = [1]
        args.repetitions = 1
        args.no_incremental = True
        args.groups = ["simple"]  # simple group only — no slow CWT
        warmup = False            # skip warm-up to save time

    output_path = args.output or OUTPUT_DIR / "benchmark.parquet"

    console.print("[bold]Mosaique vs MNE Benchmark[/bold]\n")
    console.print(f"  Groups:        {args.groups}")
    console.print(f"  Max files:     {args.max_files}")
    console.print(f"  Workers:       {args.workers}")
    console.print(f"  Repetitions:   {args.repetitions}")
    console.print(f"  Incremental:   {not args.no_incremental}")
    console.print()

    edf_files = discover_edf_files(args.max_files)
    console.print(f"Found {len(edf_files)} EDF file(s)\n")

    configs = build_benchmark_grid(
        groups=args.groups,
        max_files=len(edf_files),
        workers=args.workers,
        incremental=not args.no_incremental,
        max_n_features=quick_n_features,
    )
    warmup_count = 1 if warmup else 0
    console.print(f"Total configurations: {len(configs)}")
    console.print(
        f"Total runs (incl. warm-up): {len(configs) * (args.repetitions + warmup_count)}\n"
    )

    df = run_benchmark(configs, edf_files, args.repetitions, console, warmup=warmup)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    console.print(f"\nResults saved to [bold]{output_path}[/bold]")

    print_summary(df, console)


if __name__ == "__main__":
    main()
