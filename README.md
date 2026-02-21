# Mosaïque

## Benchmarks

Compare mosaïque against MNE-native feature extraction:

```bash
uv run benchmark/run.py --data-dir path/to/edf/files
```

Options:

- `--workers 1 4 8` — number of parallel workers to test (default: 1 + half your cores)
- `--max-files 10` — limit the number of EDF files loaded
- `--quick` — fast smoke test (1 file, 1 worker, no warmup)

Results are saved to `benchmark/output/benchmark.parquet` and resumable across runs.
