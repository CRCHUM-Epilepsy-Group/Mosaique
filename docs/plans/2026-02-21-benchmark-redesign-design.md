# Benchmark Suite Redesign

## Problem

The current benchmark (`script/benchmark.py`) has several issues:

1. **Fake MNE baseline** — calls mosaique feature functions in manual loops instead of real MNE APIs. Connectivity uses raw numpy PLI, not `mne_connectivity`.
2. **Incremental feature sweep is noise** — sweeping 1..N features adds combinatorial explosion without answering a useful question.
3. **Tiny test data** — 5 short EDFs finish in seconds; noise dominates signal.
4. **Useless plots** — graphs don't communicate meaningful comparisons.
5. **Wrong location** — should be in a dedicated `benchmark/` directory.

## Goal

Answer the question: *Is mosaique more efficient than MNE for extracting a batch of EEG features from a dataset?*

Target environments:
- Hundreds of short EDF files (e.g. TUH abnormal corpus)
- ~10 very large EDF files
- Server with many CPU cores

## Design

### Structure

```
benchmark/
  run.py          # main benchmark script
  output/         # parquet results (gitignored)
```

Delete `script/benchmark.py` and old `script/output/` benchmark artifacts.

### CLI

```
uv run benchmark/run.py --data-dir /path/to/edfs
uv run benchmark/run.py --data-dir /path/to/edfs --max-files 50 --workers 1 4 8 --reps 3
uv run benchmark/run.py --data-dir tests/test_data --quick
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | required | Path to directory with `.edf` files |
| `--max-files` | all found | Cap on number of files |
| `--workers` | `[1, <half-cores>]` | Worker counts for mosaique |
| `--reps` | 3 | Measured repetitions |
| `--quick` | off | 1 file, 1 worker, 1 rep, no warmup |
| `--output` | `benchmark/output/benchmark.parquet` | Output path |
| `--fresh` | off | Ignore checkpoint |
| `--groups` | all | Filter to specific groups |

### Feature Subsets (fixed, hardcoded)

| Group | Features |
|-------|----------|
| Simple | `line_length`, `spectral_entropy` |
| TF decomposition | `sample_entropy`, `line_length` |
| Connectivity | `average_clustering`, `global_efficiency` |

Small enough to run fast, diverse enough to be representative.

### MNE-native implementations

Each group uses what a real MNE user would actually use:

- **Simple:** Loop over epochs/channels, call univariate functions directly. MNE has no higher-level API for these features, so the difference is mosaique's parallelism and orchestration.
- **TF:** `mne.time_frequency.tfr_array_morlet()` for decomposition, then apply features to amplitude envelopes.
- **Connectivity:** `mne_connectivity.spectral_connectivity_epochs()` with PLI method, then graph metrics on the resulting matrices.

### Sweep Grid

```
For each group in [simple, tf, connectivity]:
  For each n_files in [1, N/4, N/2, N]:
    MNE: 1 run (always sequential)
    Mosaique: 1 run per worker count
  x repetitions
```

With defaults (all files, 2 worker configs, 3 reps, 3 groups): ~72 measured runs.

### Measurement

Reuse the `measure_run` harness from the current script:
- Wall time (`time.perf_counter`)
- CPU time (user + system via `psutil`)
- Peak RSS including child processes (background thread at 100ms intervals)

### Preprocessing (shared, not measured)

- Load EDF -> crop to 120s -> filter 1-50 Hz -> 5s fixed epochs
- All epochs pre-loaded in memory before benchmarking starts

### Output

**Parquet columns:** `backend`, `n_files`, `feature_group`, `n_workers`, `repetition`, `wall_s`, `cpu_s`, `peak_rss_mb`, plus system metadata (`cpu_model`, `n_cores`, `ram_gb`, `git_commit`, `timestamp`).

**Terminal summary:** Rich table showing median wall time per (group, n_files) for MNE vs best mosaique config, with speedup ratio.

### Dependencies

- `mne-connectivity` added as a benchmark dependency (not a mosaique dependency)
