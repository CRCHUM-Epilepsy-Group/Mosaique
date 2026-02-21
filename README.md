# Mosaïque


**Mosaïque** is a configuration-driven, parallel EEG feature extraction library. Features are extracted into a Polars DataFrame in long format.

The goals of this library are:
- Efficient extraction of large feature sets from EEG databases
- Simple interface: Features and parameters are defined in a configuration file
- Contains a large set of predefined EEG features
- Allows for user-defined features and transforms

## Quick start

### 1. Define a feature config

```yaml
# config.yaml
features:
  simple:
    - name: line_length
      function: univariate.line_length
    - name: spectral_entropy
      function: univariate.spectral_entropy
  tf_decomposition:
    - name: sample_entropy
      function: univariate.sample_entropy
      params:
        m: [2]
        r: [0.2]

transforms:
  simple:
    - name: simple
  tf_decomposition:
    - name: cwt
      function: timefrequency.cwt_eeg
      params:
        freqs:
          - - !!python/tuple [4, 8]
            - !!python/tuple [8, 13]
            - !!python/tuple [13, 30]
            - !!python/tuple [30, 50]
        skip_reconstr: [true]
        skip_complex: [true]
```

### 2. Run feature extraction

```python
import mne
from mosaique import FeatureExtractor, parse_featureextraction_config, resolve_pipeline

# --- Input: MNE Epochs from an EDF file ---
raw = mne.io.read_raw_edf("recording.edf", preload=True, verbose=False)
raw.filter(1.0, 50.0, verbose=False)
epochs = mne.make_fixed_length_epochs(raw, duration=5.0, verbose=False)
epochs.load_data()

# --- Alternative: numpy array (n_epochs, n_channels, n_times) ---
# import numpy as np
# data = np.random.randn(10, 19, 1280)

# Load config and extract
pipeline = parse_featureextraction_config("config.yaml")
features, transforms = resolve_pipeline(pipeline)

extractor = FeatureExtractor(features, transforms, num_workers=4)
df = extractor.extract_feature(epochs, eeg_id="subject_01")
# or: df = extractor.extract_feature(data, eeg_id="subject_01", sfreq=256.0)
df.write_parquet("features.parquet")
```

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
