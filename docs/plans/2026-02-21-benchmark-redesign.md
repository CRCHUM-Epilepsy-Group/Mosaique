# Benchmark Suite Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current fake benchmark with a real mosaique-vs-MNE comparison that scales to large datasets.

**Architecture:** Single script `benchmark/run.py` that runs both backends (mosaique via FeatureExtractor, MNE via native APIs) across a grid of (n_files, n_workers), measuring wall time, CPU time, and peak RSS. Fixed feature subsets per group (2 each). Results saved as Parquet + rich terminal summary.

**Tech Stack:** mne, mne-connectivity, mosaique, polars, psutil, rich, argparse

---

### Task 1: Scaffold benchmark directory and add dependencies

**Files:**
- Create: `benchmark/run.py` (empty placeholder)
- Create: `benchmark/output/.gitkeep`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

**Step 1: Create benchmark directory structure**

```bash
mkdir -p benchmark/output
touch benchmark/output/.gitkeep
touch benchmark/run.py
```

**Step 2: Add mne-connectivity to optional deps in pyproject.toml**

In `pyproject.toml`, change the optional dependencies to:

```toml
[project.optional-dependencies]
mne = ["mne", "mne-connectivity"]
```

**Step 3: Add benchmark/output/ to .gitignore**

Append to `.gitignore`:

```
benchmark/output/*.parquet
```

**Step 4: Install updated deps**

```bash
uv sync --extra mne
```

Verify `mne_connectivity` is importable:

```bash
uv run python -c "import mne_connectivity; print(mne_connectivity.__version__)"
```

**Step 5: Commit**

```bash
git add benchmark/ pyproject.toml .gitignore uv.lock
git commit -m "scaffold benchmark directory and add mne-connectivity dep"
```

---

### Task 2: Write the measurement harness and CLI skeleton

**Files:**
- Modify: `benchmark/run.py`

This task ports the good parts from `script/benchmark.py` (the `measure_run` harness, system metadata collection, CLI parsing) and builds the new CLI interface. No backends yet — just the shell.

**Step 1: Write benchmark/run.py with harness and CLI**

```python
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
            cwd=PROJECT_ROOT, text=True,
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
    backend: str        # "mne" or "mosaique"
    n_files: int
    feature_group: str
    n_workers: int      # always 1 for MNE


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
# Backend runners (placeholder — filled in Tasks 3-5)
# ---------------------------------------------------------------------------
def run_mne(epochs_list: list[mne.Epochs], group: str) -> None:
    raise NotImplementedError(f"MNE runner for {group}")


def run_mosaique(
    epochs_list: list[mne.Epochs], group: str, n_workers: int
) -> None:
    raise NotImplementedError(f"Mosaique runner for {group}")


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
                existing_df
                .select(["backend", "n_files", "feature_group", "n_workers", "repetition"])
                .iter_rows()
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
                r for r in range(1, reps + 1)
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
                r for r in range(1, reps + 1)
                if (cfg.backend, cfg.n_files, cfg.feature_group, cfg.n_workers, r)
                not in completed
            ]
            if not missing_reps:
                continue

            subset = all_epochs[: cfg.n_files]

            if cfg.backend == "mne":
                func = lambda _s=subset, _g=cfg.feature_group: run_mne(_s, _g)
            else:
                func = lambda _s=subset, _g=cfg.feature_group, _w=cfg.n_workers: run_mosaique(_s, _g, _w)

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
        mos_medians.sort("mos_wall_s")
        .group_by(["feature_group", "n_files"])
        .first()
    )
    combined = (
        best_mos.join(mne_medians, on=["feature_group", "n_files"])
        .with_columns(
            (pl.col("mne_wall_s") / pl.col("mos_wall_s")).alias("speedup")
        )
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
    p.add_argument("--workers", nargs="+", type=int, default=None,
                   help="Worker counts (default: 1 + half cores)")
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
        configs, all_epochs, args.reps, console,
        warmup=warmup, output_path=output_path, resume=not args.fresh,
    )
    console.print(f"\nResults: [bold]{output_path}[/bold]")
    print_summary(df, console)


if __name__ == "__main__":
    main()
```

**Step 2: Verify the script runs (will fail at NotImplementedError)**

```bash
uv run benchmark/run.py --data-dir tests/test_data --quick
```

Expected: crashes with `NotImplementedError("MNE runner for simple")`. This confirms the harness works up to the point of calling backends.

**Step 3: Commit**

```bash
git add benchmark/run.py
git commit -m "add benchmark harness, CLI, measurement, and summary"
```

---

### Task 3: Implement the simple feature group (both backends)

**Files:**
- Modify: `benchmark/run.py`

The simple group extracts `line_length` and `spectral_entropy` directly from raw epoch data. MNE has no higher-level API for these, so both sides call the same feature functions — the difference is mosaique's orchestration and parallelism.

**Step 1: Add the simple group runners**

Add these imports at the top of `benchmark/run.py`:

```python
from mosaique import FeatureExtractor
from mosaique.config.types import ExtractionStep
from mosaique.features.univariate import line_length, spectral_entropy
```

Add the simple-group runners (replace the placeholder `run_mne` and `run_mosaique`):

```python
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
            ExtractionStep(name="spectral_entropy", function=spectral_entropy, params={}),
        ]
    }
    transforms = {
        "simple": [ExtractionStep(name="simple", function=None, params={})]
    }
    for epochs in epochs_list:
        ext = FeatureExtractor(
            features, transforms,
            num_workers=n_workers, console=Console(quiet=True),
        )
        ext.extract_feature(epochs, eeg_id="bench")
```

Update the dispatcher functions:

```python
_MNE_RUNNERS: dict[str, Callable[[list[mne.Epochs]], None]] = {
    "simple": _mne_simple,
}

_MOSAIQUE_RUNNERS: dict[str, Callable[[list[mne.Epochs], int], None]] = {
    "simple": _mosaique_simple,
}


def run_mne(epochs_list: list[mne.Epochs], group: str) -> None:
    _MNE_RUNNERS[group](epochs_list)


def run_mosaique(
    epochs_list: list[mne.Epochs], group: str, n_workers: int
) -> None:
    _MOSAIQUE_RUNNERS[group](epochs_list, n_workers)
```

**Step 2: Test the simple group**

```bash
uv run benchmark/run.py --data-dir tests/test_data --quick --groups simple
```

Expected: completes successfully, prints summary table with MNE and mosaique times for the simple group.

**Step 3: Commit**

```bash
git add benchmark/run.py
git commit -m "implement simple feature group benchmark (both backends)"
```

---

### Task 4: Implement the TF decomposition group (both backends)

**Files:**
- Modify: `benchmark/run.py`

MNE side uses `mne.time_frequency.tfr_array_morlet` for Morlet wavelet decomposition, then applies features to amplitude envelopes. Mosaique side uses `cwt_eeg` via the FeatureExtractor.

**Step 1: Add imports**

```python
from mosaique.features.univariate import sample_entropy
from mosaique.features.timefrequency import cwt_eeg
```

**Step 2: Add the TF runners**

```python
def _band_freqs(band: tuple[float, float]) -> np.ndarray:
    """Log-spaced frequencies within a band for MNE Morlet."""
    return np.logspace(np.log10(band[0]), np.log10(band[1]), N_FREQS_PER_BAND)


def _mne_tf(epochs_list: list[mne.Epochs]) -> None:
    """MNE-style TF feature extraction using tfr_array_morlet."""
    for epochs in epochs_list:
        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]
        n_epochs, n_channels, _ = data.shape

        for band in TF_BANDS:
            freqs = _band_freqs(band)
            tfr: np.ndarray = mne.time_frequency.tfr_array_morlet(
                data, sfreq, freqs,
                n_cycles=N_CYCLES, output="complex", verbose=False,
            )
            amplitude = np.abs(tfr).mean(axis=2)  # avg over freqs

            level = int(np.floor(np.log2(sfreq / (2 * band[1]))))
            ds = max(1, 2 * level)
            if ds > 1:
                amplitude = amplitude[..., ::ds]

            for i in range(n_epochs):
                for j in range(n_channels):
                    sample_entropy(amplitude[i, j], sfreq=sfreq, m=2, r=0.2)
                    line_length(amplitude[i, j], sfreq=sfreq)


def _mosaique_tf(epochs_list: list[mne.Epochs], n_workers: int) -> None:
    features = {
        "tf_decomposition": [
            ExtractionStep(
                name="sample_entropy", function=sample_entropy,
                params={"m": [2], "r": [0.2]},
            ),
            ExtractionStep(name="line_length", function=line_length, params={}),
        ]
    }
    transforms = {
        "tf_decomposition": [
            ExtractionStep(
                name="cwt", function=cwt_eeg,
                params={
                    "freqs": [TF_BANDS],
                    "skip_reconstr": [True],
                    "skip_complex": [True],
                },
            )
        ]
    }
    for epochs in epochs_list:
        ext = FeatureExtractor(
            features, transforms,
            num_workers=n_workers, console=Console(quiet=True),
        )
        ext.extract_feature(epochs, eeg_id="bench")
```

**Step 3: Register in dispatchers**

Add to `_MNE_RUNNERS`:
```python
"tf_decomposition": _mne_tf,
```

Add to `_MOSAIQUE_RUNNERS`:
```python
"tf_decomposition": _mosaique_tf,
```

**Step 4: Test the TF group**

```bash
uv run benchmark/run.py --data-dir tests/test_data --quick --groups tf_decomposition
```

Expected: completes, prints summary with times.

**Step 5: Commit**

```bash
git add benchmark/run.py
git commit -m "implement TF decomposition benchmark (both backends)"
```

---

### Task 5: Implement the connectivity group (both backends)

**Files:**
- Modify: `benchmark/run.py`

This is the most interesting group. MNE side uses `mne_connectivity.spectral_connectivity_epochs` with PLI method, then applies graph metrics. Mosaique uses its CWT-based PLI via `connectivity_from_coeff`.

Important: `spectral_connectivity_epochs` computes connectivity **across epochs** (one matrix per channel pair). To get per-epoch connectivity (matching mosaique), we call it once per epoch. This is realistic — it's what an MNE user needing epoch-level features would do.

**Step 1: Add imports**

```python
import mne_connectivity
from mosaique.features.connectivity import (
    average_clustering,
    global_efficiency,
    connectivity_from_coeff,
)
```

**Step 2: Add the connectivity runners**

```python
def _mne_connectivity(epochs_list: list[mne.Epochs]) -> None:
    """MNE-native connectivity: spectral_connectivity_epochs + graph metrics."""
    for epochs in epochs_list:
        n_channels = len(epochs.ch_names)
        n_epochs = len(epochs)

        for band in TF_BANDS:
            freqs = _band_freqs(band)

            # Call per-epoch to get epoch-level connectivity
            for ep_idx in range(n_epochs):
                single = epochs[ep_idx]
                con = mne_connectivity.spectral_connectivity_epochs(
                    single,
                    method="pli",
                    mode="cwt_morlet",
                    cwt_freqs=freqs,
                    cwt_n_cycles=N_CYCLES,
                    faverage=True,
                    verbose=False,
                )
                # con.get_data() shape: (n_connections, 1) with faverage=True
                # Reshape to (n_channels, n_channels) symmetric matrix
                pli_flat = con.get_data()[:, 0]
                pli_mat = np.zeros((n_channels, n_channels))
                idx = np.triu_indices(n_channels, k=1)
                pli_mat[idx] = pli_flat
                pli_mat += pli_mat.T

                average_clustering(pli_mat)
                global_efficiency(pli_mat)


def _mosaique_connectivity(
    epochs_list: list[mne.Epochs], n_workers: int
) -> None:
    features = {
        "connectivity": [
            ExtractionStep(
                name="average_clustering", function=average_clustering, params={},
            ),
            ExtractionStep(
                name="global_efficiency", function=global_efficiency, params={},
            ),
        ]
    }
    transforms = {
        "connectivity": [
            ExtractionStep(
                name="pli", function=connectivity_from_coeff,
                params={
                    "freqs": [TF_BANDS],
                    "wavelet": ["cmor1.5-1.0"],
                    "skip_reconstr": [True],
                },
            )
        ]
    }
    for epochs in epochs_list:
        ext = FeatureExtractor(
            features, transforms,
            num_workers=n_workers, console=Console(quiet=True),
        )
        ext.extract_feature(epochs, eeg_id="bench")
```

**Step 3: Register in dispatchers**

Add to `_MNE_RUNNERS`:
```python
"connectivity": _mne_connectivity,
```

Add to `_MOSAIQUE_RUNNERS`:
```python
"connectivity": _mosaique_connectivity,
```

**Step 4: Test connectivity group**

```bash
uv run benchmark/run.py --data-dir tests/test_data --quick --groups connectivity
```

Expected: completes, prints summary. MNE connectivity will likely be slow due to per-epoch calls.

**Step 5: Commit**

```bash
git add benchmark/run.py
git commit -m "implement connectivity benchmark with real mne_connectivity"
```

---

### Task 6: Full smoke test and cleanup

**Files:**
- Modify: `benchmark/run.py` (if any fixes needed)
- Delete: `script/benchmark.py`
- Delete: `script/output/benchmark.parquet`

**Step 1: Run full benchmark with test data**

```bash
uv run benchmark/run.py --data-dir tests/test_data --reps 1
```

Expected: runs all 3 groups, all file counts, prints a summary table. Should complete in a few minutes with the small test data.

**Step 2: Run quick mode**

```bash
uv run benchmark/run.py --data-dir tests/test_data --quick
```

Expected: finishes in under a minute.

**Step 3: Delete old benchmark**

```bash
rm script/benchmark.py
rm -f script/output/benchmark.parquet
```

**Step 4: Commit**

```bash
git add -A
git commit -m "remove old benchmark script, full suite working"
```

---

## Notes for the implementer

- The `mne_connectivity.spectral_connectivity_epochs` return format may need adjustment — verify `con.get_data()` shape and whether it uses upper triangle or full matrix ordering. Print shapes during implementation if unsure.
- The `connectivity_from_coeff` function used by mosaique's connectivity transform expects pre-computed CWT coefficients. When used as a transform function in `ExtractionStep`, the `ConnectivityTransform` handles CWT internally. Verify this works by checking the extract output has rows for each band.
- `mne-connectivity` may log warnings about single-epoch connectivity estimates. Suppress with `verbose=False`.
- Keep `script/features_config.yaml` and `script/feature_extraction_mne.py` — they're separate from the benchmark.
