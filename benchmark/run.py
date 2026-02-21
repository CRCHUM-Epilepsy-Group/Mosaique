#!/usr/bin/env python
"""Benchmark: mosaique vs MNE-native feature extraction.

Runs both backends across a grid of (n_files × n_workers), collects
wall time / CPU time / peak RSS, and outputs a Parquet file + summary.

Usage
-----
    uv run benchmark/run.py --data-dir tests/test_data
    uv run benchmark/run.py --data-dir /big/dataset --max-files 50 --workers 1 4 8
    uv run benchmark/run.py --data-dir tests/test_data --quick
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

from mosaique import FeatureExtractor
from mosaique.config.types import ExtractionStep
from mosaique.features.univariate import line_length, spectral_entropy
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

GROUPS = ["simple", "tf_decomposition", "connectivity"]

TF_BANDS: list[tuple[float, float]] = [(4, 8), (8, 13), (13, 30), (30, 50)]
N_FREQS_PER_BAND = 20
N_CYCLES = 7.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def discover_edf_files(data_dir: Path, max_files: int | None) -> list[Path]:
    """Return up to *max_files* EDF paths from *data_dir*."""
    files = sorted(data_dir.rglob("*.edf"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No EDF files found in {data_dir}")
    return files


def load_epochs(edf_files: list[Path], console: Console) -> list[mne.Epochs]:
    """Load and preprocess EDF files into MNE Epochs (not measured)."""
    console.print("[bold]Loading EDF files...[/bold]")
    all_epochs: list[mne.Epochs] = []
    for f in edf_files:
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        raw.crop(tmax=min(120.0, raw.times[-1]))
        raw.filter(1.0, 50.0, verbose=False)
        epochs = mne.make_fixed_length_epochs(raw, duration=5.0, verbose=False)
        epochs.load_data()
        all_epochs.append(epochs)
    console.print(f"  Loaded {len(all_epochs)} file(s)\n")
    return all_epochs


# ---------------------------------------------------------------------------
# Measurement harness
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    wall_s: float
    cpu_s: float
    peak_rss_mb: float


def measure_run(func: Callable[[], None]) -> RunMetrics:
    """Execute *func* and return wall time, CPU time, and peak RSS."""
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

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    cpu_before = proc.cpu_times()
    wall_start = time.perf_counter()
    func()
    wall_end = time.perf_counter()
    cpu_after = proc.cpu_times()

    stop_event.set()
    monitor.join(timeout=2.0)

    return RunMetrics(
        wall_s=wall_end - wall_start,
        cpu_s=(cpu_after.user - cpu_before.user)
        + (cpu_after.system - cpu_before.system),
        peak_rss_mb=peak_rss / 1024**2,
    )


# ---------------------------------------------------------------------------
# System metadata
# ---------------------------------------------------------------------------
def collect_system_metadata() -> dict[str, str]:
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
        "git_commit": git_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
def build_file_counts(total: int) -> list[int]:
    """Return ~4 data points spread across [1, total]."""
    if total <= 4:
        return list(range(1, total + 1))
    points = {1, total // 4, total // 2, total}
    return sorted(points)


@dataclass
class BenchmarkConfig:
    backend: str  # "mne" or "mosaique"
    n_files: int
    feature_group: str
    n_workers: int  # always 1 for MNE


def build_grid(
    groups: list[str],
    file_counts: list[int],
    workers: list[int],
) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for group in groups:
        for n in file_counts:
            configs.append(BenchmarkConfig("mne", n, group, 1))
            for w in workers:
                configs.append(BenchmarkConfig("mosaique", n, group, w))
    return configs


# ---------------------------------------------------------------------------
# MNE backend: simple
# ---------------------------------------------------------------------------
def _mne_simple(epochs_list: list[mne.Epochs]) -> None:
    """MNE-style simple feature extraction: sequential loops."""
    for epochs in epochs_list:
        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]
        n_epochs, n_channels, _ = data.shape
        for i in range(n_epochs):
            for j in range(n_channels):
                line_length(data[i, j], sfreq=sfreq)
                spectral_entropy(data[i, j], sfreq=sfreq)


# ---------------------------------------------------------------------------
# Mosaique backend: simple
# ---------------------------------------------------------------------------
def _mosaique_simple(epochs_list: list[mne.Epochs], n_workers: int) -> None:
    features = {
        "simple": [
            ExtractionStep(name="line_length", function=line_length, params={}),
            ExtractionStep(
                name="spectral_entropy", function=spectral_entropy, params={}
            ),
        ]
    }
    transforms = {"simple": [ExtractionStep(name="simple", function=None, params={})]}
    for epochs in epochs_list:
        ext = FeatureExtractor(
            features,
            transforms,
            num_workers=n_workers,
            console=Console(quiet=True),
        )
        ext.extract_feature(epochs, eeg_id="bench")


# ---------------------------------------------------------------------------
# Backend dispatchers
# ---------------------------------------------------------------------------
_MNE_RUNNERS: dict[str, Callable[[list[mne.Epochs]], None]] = {
    "simple": _mne_simple,
}

_MOSAIQUE_RUNNERS: dict[str, Callable[[list[mne.Epochs], int], None]] = {
    "simple": _mosaique_simple,
}


def run_mne(epochs_list: list[mne.Epochs], group: str) -> None:
    _MNE_RUNNERS[group](epochs_list)


def run_mosaique(epochs_list: list[mne.Epochs], group: str, n_workers: int) -> None:
    _MOSAIQUE_RUNNERS[group](epochs_list, n_workers)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark(
    configs: list[BenchmarkConfig],
    all_epochs: list[mne.Epochs],
    reps: int,
    console: Console,
    warmup: bool = True,
    output_path: Path | None = None,
    resume: bool = True,
) -> pl.DataFrame:
    metadata = collect_system_metadata()

    # Resume from checkpoint
    existing_df: pl.DataFrame | None = None
    completed: set[tuple[Any, ...]] = set()
    if resume and output_path and output_path.exists():
        try:
            existing_df = pl.read_parquet(output_path)
            completed = set(
                existing_df.select(
                    ["backend", "n_files", "feature_group", "n_workers", "repetition"]
                ).iter_rows()
            )
            console.print(
                f"[bold]Resuming:[/bold] {len(completed)} run(s) already done\n"
            )
        except Exception as e:
            console.print(f"[yellow]Warning: checkpoint load failed ({e})[/yellow]\n")

    def _checkpoint(rows: list[dict[str, Any]]) -> None:
        if not output_path or not rows:
            return
        new_df = pl.DataFrame(rows)
        combined = (
            pl.concat([existing_df, new_df], how="diagonal_relaxed")
            if existing_df is not None
            else new_df
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(output_path)

    warmup_n = 1 if warmup else 0
    remaining = sum(
        len(missing) + (warmup_n if missing else 0)
        for cfg in configs
        if (
            missing := [
                r
                for r in range(1, reps + 1)
                if (cfg.backend, cfg.n_files, cfg.feature_group, cfg.n_workers, r)
                not in completed
            ]
        )
    )

    new_rows: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking", total=remaining)

        for cfg in configs:
            missing_reps = [
                r
                for r in range(1, reps + 1)
                if (cfg.backend, cfg.n_files, cfg.feature_group, cfg.n_workers, r)
                not in completed
            ]
            if not missing_reps:
                continue

            subset = all_epochs[: cfg.n_files]

            if cfg.backend == "mne":
                func = lambda _s=subset, _g=cfg.feature_group: run_mne(_s, _g)
            else:
                func = (
                    lambda _s=subset,
                    _g=cfg.feature_group,
                    _w=cfg.n_workers: run_mosaique(_s, _g, _w)
                )

            if warmup:
                gc.collect()
                measure_run(func)
                progress.advance(task)

            for rep in missing_reps:
                gc.collect()
                m = measure_run(func)
                row: dict[str, Any] = {
                    "backend": cfg.backend,
                    "n_files": cfg.n_files,
                    "feature_group": cfg.feature_group,
                    "n_workers": cfg.n_workers,
                    "repetition": rep,
                    "wall_s": m.wall_s,
                    "cpu_s": m.cpu_s,
                    "peak_rss_mb": m.peak_rss_mb,
                }
                row.update(metadata)
                new_rows.append(row)
                _checkpoint(new_rows)
                progress.advance(task)

    if existing_df is not None and new_rows:
        return pl.concat([existing_df, pl.DataFrame(new_rows)], how="diagonal_relaxed")
    if existing_df is not None:
        return existing_df
    return pl.DataFrame(new_rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(df: pl.DataFrame, console: Console) -> None:
    mne_medians = (
        df.filter(pl.col("backend") == "mne")
        .group_by(["feature_group", "n_files"])
        .agg(pl.col("wall_s").median().alias("mne_wall_s"))
    )
    mos_medians = (
        df.filter(pl.col("backend") == "mosaique")
        .group_by(["feature_group", "n_files", "n_workers"])
        .agg(pl.col("wall_s").median().alias("mos_wall_s"))
    )
    best_mos = (
        mos_medians.sort("mos_wall_s").group_by(["feature_group", "n_files"]).first()
    )
    combined = (
        best_mos.join(mne_medians, on=["feature_group", "n_files"])
        .with_columns((pl.col("mne_wall_s") / pl.col("mos_wall_s")).alias("speedup"))
        .sort(["feature_group", "n_files"])
    )

    table = Table(title="Benchmark Summary (best worker config)")
    table.add_column("Group")
    table.add_column("Files", justify="right")
    table.add_column("Workers", justify="right")
    table.add_column("MNE (s)", justify="right")
    table.add_column("Mosaique (s)", justify="right")
    table.add_column("Speedup", justify="right")

    for row in combined.iter_rows(named=True):
        s = row["speedup"]
        color = "green" if s > 1 else "red"
        table.add_row(
            row["feature_group"],
            str(row["n_files"]),
            str(row["n_workers"]),
            f"{row['mne_wall_s']:.2f}",
            f"{row['mos_wall_s']:.2f}",
            f"[{color}]{s:.2f}x[/{color}]",
        )
    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark mosaique vs MNE feature extraction"
    )
    p.add_argument("--data-dir", type=Path, required=True, help="EDF directory")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=None,
        help="Worker counts (default: 1 + half cores)",
    )
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--fresh", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    if args.workers is None:
        half = max(1, (psutil.cpu_count(logical=False) or 2) // 2)
        args.workers = sorted({1, half})

    warmup = True
    if args.quick:
        args.max_files = 1
        args.workers = [1]
        args.reps = 1
        warmup = False

    output_path = args.output or OUTPUT_DIR / "benchmark.parquet"

    console.print("[bold]Mosaique vs MNE Benchmark[/bold]\n")
    console.print(f"  Data dir:   {args.data_dir}")
    console.print(f"  Groups:     {args.groups}")
    console.print(f"  Workers:    {args.workers}")
    console.print(f"  Reps:       {args.reps}")
    console.print()

    edf_files = discover_edf_files(args.data_dir, args.max_files)
    console.print(f"Found {len(edf_files)} EDF file(s)\n")

    all_epochs = load_epochs(edf_files, console)
    file_counts = build_file_counts(len(all_epochs))
    configs = build_grid(args.groups, file_counts, args.workers)

    console.print(f"Configurations: {len(configs)}\n")

    df = run_benchmark(
        configs,
        all_epochs,
        args.reps,
        console,
        warmup=warmup,
        output_path=output_path,
        resume=not args.fresh,
    )
    console.print(f"\nResults: [bold]{output_path}[/bold]")
    print_summary(df, console)


if __name__ == "__main__":
    main()
