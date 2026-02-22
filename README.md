# Mosaïque


**Mosaïque** is a configuration-driven, parallel EEG feature extraction library. Features are extracted into a Polars DataFrame in long format.

The goals of this library are:
- Efficient extraction of large feature sets from EEG databases
- Simple interface: Features and parameters are defined in a configuration file
- Contains a large set of predefined EEG features
- Allows for user-defined features and transforms

This pipeline was used in the following papers:
- [Improving diagnostic accuracy of routine EEG for epilepsy using deep learning](https://academic.oup.com/braincomms/advance-article/doi/10.1093/braincomms/fcaf319/8240832?utm_source=advanceaccess&utm_campaign=braincomms&utm_medium=email)
- [Development and validation of a deep survival model to predict time to seizure from routine electroencephalography](https://onlinelibrary.wiley.com/doi/10.1002/epi.70101?af=R)

## Quick start

### Installation

To install the package with uv:

``` bash
uv add git+https://github.com/CRCHUM-Epilepsy-Group/Mosaique.git
```

For Pip: 

``` bash
pip install git+https://github.com/CRCHUM-Epilepsy-Group/Mosaique.git
```

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
from mosaique import FeatureExtractor

# --- Input: MNE Epochs from an EDF file ---
raw = mne.io.read_raw_edf("recording.edf", preload=True, verbose=False)
raw.filter(1.0, 50.0, verbose=False)
epochs = mne.make_fixed_length_epochs(raw, duration=5.0, verbose=False)
epochs.load_data()

# --- Alternative: numpy array (n_epochs, n_channels, n_times) ---
# import numpy as np
# data = np.random.randn(10, 19, 1280)

extractor = FeatureExtractor("config.yaml", num_workers=4)
df = extractor.extract_feature(epochs, eeg_id="subject_01")
# or: df = extractor.extract_feature(data, eeg_id="subject_01", sfreq=256.0)
df.write_parquet("features.parquet")
```

Or in a single call:

```python
import mosaique

df = mosaique.extract("config.yaml", epochs, eeg_id="subject_01", num_workers=4)
```

## Configuration

### Minimal config

The simplest valid config — just a list of features under `features`, no `transforms` section needed:

```yaml
features:
  - univariate.line_length
  - name: sample_entropy
    function: univariate.sample_entropy
    params:
      m: [2, 3]
      r: [0.2]
```

### Full config structure

```yaml
features:
  <transform_key>:           # must match a key in transforms
    - name: <feature_name>   # label in the output DataFrame
      function: <dotted.path>
      params:
        <param>: [<value>, ...]  # list → grid expansion; scalar → single run

transforms:
  <transform_key>:
    - name: <transform_name>
      function: <dotted.path>   # omit for identity (simple) transform
      params:
        <param>: [<value>, ...]
```

`features` and `transforms` must have the same set of keys.  The Cartesian product of all param lists determines how many extraction tasks are generated.

### Feature reference

Features live in `mosaique.features`.  Reference them in config by their dotted path (e.g. `univariate.line_length`).

#### Univariate features (`univariate.*`)

| Name | Path | Key params | Description |
|------|------|-----------|-------------|
| Line length | `univariate.line_length` | — | Sum of absolute first differences |
| Shannon entropy | `univariate.shannon_entropy` | — | Histogram-based entropy |
| Sample entropy | `univariate.sample_entropy` | `m`, `r` | Template-matching entropy |
| Approximate entropy | `univariate.approximate_entropy` | `m`, `r` | Regularity measure |
| Spectral entropy | `univariate.spectral_entropy` | `sfreq` | Entropy of the PSD |
| Permutation entropy | `univariate.permutation_entropy` | `k` | Ordinal pattern entropy |
| Fuzzy entropy | `univariate.fuzzy_entropy` | `m`, `r`, `n` | Fuzzy membership entropy |
| Correlation dimension | `univariate.corr_dim` | `embed_dim` | Fractal dimension estimate |
| Peak alpha frequency | `univariate.peak_alpha` | `sfreq` | Frequency of PSD peak (8–13 Hz) |
| Hurst exponent | `univariate.hurst_exp` | `min_window`, `max_window` | Long-range dependence |
| Band power | `univariate.band_power` | `freqs`, `sfreq` | Power per frequency band |

#### Graph / connectivity features (`graph_metrics.*`)

Used after a `connectivity` transform.

| Name | Path | Description |
|------|------|-------------|
| Average clustering | `graph_metrics.average_clustering` | Mean clustering coefficient |
| Average node connectivity | `graph_metrics.average_node_connectivity` | Mean node connectivity |
| Average degree | `graph_metrics.average_degree` | Mean node degree |
| Global efficiency | `graph_metrics.global_efficiency` | Inverse average shortest path |
| Average shortest path length | `graph_metrics.average_shortest_path_length` | Mean shortest path |

### Transform reference

| Key | Description | Key params |
|-----|-------------|-----------|
| `simple` | Identity — no pre-processing, features applied directly to raw signal | — |
| `tf_decomposition` | Continuous Wavelet Transform decomposition into frequency bands | `freqs`, `skip_reconstr`, `skip_complex` |
| `connectivity` | Spectral connectivity matrix (PLI or correlation) between channels | `method`, `freqs` |

### Parameter grids

Scalar values and single-element lists are equivalent.  Lists create a Cartesian product of all parameters:

```yaml
params:
  m: [2, 3]   # combined with every value of r
  r: [0.1, 0.2]
# → 4 extraction tasks: (m=2, r=0.1), (m=2, r=0.2), (m=3, r=0.1), (m=3, r=0.2)
```

### Custom features

Write a function that accepts a 1-D NumPy array and `**kwargs`, then reference it by its full dotted import path:

```python
# my_features.py
def rms(x, **kwargs):
    """Root mean square amplitude."""
    return float(np.sqrt(np.mean(x ** 2)))
```

```yaml
features:
  simple:
    - name: rms
      function: my_features.rms
```

### Shorthand syntax

Bare dotted paths expand to `{name: <last_segment>, function: <path>}`:

```yaml
features:
  simple:
    - univariate.line_length            # equivalent to name: line_length, function: univariate.line_length
    - name: sampen
      function: univariate.sample_entropy
      params: {m: [2, 3]}
```

The `transforms` section can be omitted when all feature keys are `simple` — it is generated automatically.

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

