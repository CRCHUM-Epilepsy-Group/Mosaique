# Epoch Batching for Memory-Bounded Extraction

## Problem

The extraction pipeline holds CWT coefficients for ALL epochs in memory simultaneously. For a 36h EEG recording (~25,920 epochs at 5s), this requires ~124 GB per frequency band in complex128 — far exceeding the 32 GB RAM target.

Even moderate files (300 epochs) are tight on 32 GB.

## Goal

Process arbitrarily long EEG recordings with bounded memory by batching epochs through the full pipeline. Target: comfortable extraction on a 32 GB machine regardless of file length.

## Design

### Approach: batch at the extractor level

The extractor slices `EegData` into epoch batches before entering the transform loop. Each batch goes through ALL transforms (simple → tf_decomposition → connectivity), preserving the CWT cache within a batch. After each batch, the cache is cleared and the next batch starts.

Transforms and features are unaware of batching — they receive a smaller `EegData` and work exactly as before.

### Data flow

```
extract_feature(eeg, batch_size=128)
  → EegData (full recording)
  → for each batch of 128 epochs:
      → EegData.slice(start, end) → batch_eeg
      → for each transform group (simple, tf_decomposition, connectivity):
          → for each transform param combo:
              → transform.transform(batch_eeg) → transformed
              → for each feature:
                  → transform.extract_feature(transformed, ...) → DataFrame
                  → append to results list
          → clear _cached_coeffs
      → (next batch)
  → pl.concat(all batch results)
  → return single DataFrame
```

### Memory budget

The dominant cost is CWT coefficients: `batch_size × n_channels × n_scales × n_times × 16 bytes × n_bands`.

For typical values (20 channels, ~15 scales/band, 1000 timepoints, 4 bands):
- 128 epochs × ~19 MB/epoch ≈ 2.5 GB for CWT
- Leaves ~29 GB headroom for OS, raw EEG, connectivity matrices, output DataFrames

Default `batch_size=128`. Users can override.

### API change

`FeatureExtractor.__init__` gains one optional parameter:

```python
def __init__(self, ..., batch_size: int = 128):
```

No other API changes. Return type stays `pl.DataFrame`.

## Files to change

### 1. `src/mosaique/extraction/eegdata.py` — add `slice` method

Add a method to `EegData` that returns a new `EegData` for a contiguous range of epochs:

```python
def slice(self, start: int, end: int) -> "EegData":
```

Must slice: `data` (axis 0), `event_labels`, `timestamps`. Must preserve: `sfreq`, `ch_names`.

### 2. `src/mosaique/extraction/extractor.py` — add batch loop

**`__init__`**: Add `batch_size: int = 128` parameter, store as `self.batch_size`.

**`extract_feature`**: Restructure to wrap the existing transform iteration inside a batch loop.

Current structure (simplified):
```python
for transform_name, transform_group in self._transforms.items():
    for transform_params in self._transform_grid:
        transformed = transform.transform(eeg_data)
        features_df = self._extract_for_single_transform(transformed, ...)
        self._extracted_features.append(features_df)
```

New structure:
```python
for batch_start in range(0, n_epochs, self.batch_size):
    batch_end = min(batch_start + self.batch_size, n_epochs)
    batch_eeg = eeg_data.slice(batch_start, batch_end)

    for transform_name, transform_group in self._transforms.items():
        for transform_params in self._transform_grid:
            transformed = transform.transform(batch_eeg)
            features_df = self._extract_for_single_transform(transformed, ...)
            self._extracted_features.append(features_df)

    # Free CWT cache between batches
    self._cached_coeffs = {}
    self._cache_tag = ()
```

The progress bar should reflect total progress across all batches (not restart per batch).

### Not changed

- `extraction/transforms/` — transforms receive smaller EegData, no code changes
- `features/` — feature functions unchanged
- `config/` — no config changes
- Output format — still a single concatenated `pl.DataFrame`
