# Epoch Batching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Process arbitrarily long EEG recordings with bounded memory by batching epochs through the extraction pipeline.

**Architecture:** The extractor slices `EegData` into batches of 128 epochs (configurable). Each batch flows through all transforms and features, preserving the CWT cache within a batch. Cache is freed between batches. Transforms are unaware of batching.

**Tech Stack:** numpy, polars, existing mosaique extraction pipeline

**Design doc:** `docs/plans/2026-02-21-epoch-batching-design.md`

---

### Task 1: Add `EegData.slice()` method

**Files:**
- Modify: `src/mosaique/extraction/eegdata.py:35-145`
- Test: `tests/test_extractor.py`

**Step 1: Write the failing test**

Add to `tests/test_extractor.py`:

```python
from mosaique.extraction.eegdata import EegData

class TestEegDataSlice:
    def test_slice_returns_correct_epochs(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 3, 400))
        eeg = EegData.from_array(data, sfreq=200.0)
        sliced = eeg.slice(2, 5)

        assert sliced.data.shape == (3, 3, 400)
        np.testing.assert_array_equal(sliced.data, data[2:5])

    def test_slice_preserves_metadata(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 3, 400))
        ch_names = ["Fp1", "C3", "O1"]
        eeg = EegData.from_array(data, sfreq=200.0, ch_names=ch_names)
        sliced = eeg.slice(0, 4)

        assert sliced.sfreq == 200.0
        assert sliced.ch_names == ch_names
        assert len(sliced.event_labels) == 4
        assert len(sliced.timestamps) == 4

    def test_slice_event_labels_and_timestamps(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 2, 100))
        labels = ["a", "b", "c", "d", "e"]
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        eeg = EegData.from_array(
            data, sfreq=200.0, event_labels=labels, timestamps=timestamps
        )
        sliced = eeg.slice(1, 4)

        assert sliced.event_labels == ["b", "c", "d"]
        np.testing.assert_array_equal(sliced.timestamps, [1.0, 2.0, 3.0])
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extractor.py::TestEegDataSlice -v`
Expected: FAIL with `AttributeError: 'EegData' object has no attribute 'slice'`

**Step 3: Write minimal implementation**

Add this method to the `EegData` dataclass in `src/mosaique/extraction/eegdata.py` after the `from_array` classmethod:

```python
def slice(self, start: int, end: int) -> "EegData":
    """Return a new EegData containing epochs [start, end).

    Parameters
    ----------
    start : int
        First epoch index (inclusive).
    end : int
        Last epoch index (exclusive).

    Returns
    -------
    EegData
    """
    return EegData(
        data=self.data[start:end],
        sfreq=self.sfreq,
        ch_names=self.ch_names,
        event_labels=self.event_labels[start:end],
        timestamps=self.timestamps[start:end],
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extractor.py::TestEegDataSlice -v`
Expected: 3 passed

**Step 5: Commit**

```
git add src/mosaique/extraction/eegdata.py tests/test_extractor.py
git commit -m "add EegData.slice() for epoch batching"
```

---

### Task 2: Add `batch_size` parameter to `FeatureExtractor.__init__`

**Files:**
- Modify: `src/mosaique/extraction/extractor.py:77-115`
- Test: `tests/test_extractor.py`

**Step 1: Write the failing test**

Add to `tests/test_extractor.py`:

```python
class TestBatchSize:
    def test_default_batch_size(self, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            simple_features, simple_transforms, debug=True,
            console=Console(quiet=True),
        )
        assert extractor.batch_size == 128

    def test_custom_batch_size(self, simple_features, simple_transforms):
        extractor = FeatureExtractor(
            simple_features, simple_transforms, debug=True,
            batch_size=64,
            console=Console(quiet=True),
        )
        assert extractor.batch_size == 64
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extractor.py::TestBatchSize -v`
Expected: FAIL with `TypeError: FeatureExtractor.__init__() got an unexpected keyword argument 'batch_size'`

**Step 3: Write minimal implementation**

In `src/mosaique/extraction/extractor.py`, modify `__init__` (line 77):

Add `batch_size: int = 128` parameter after `num_workers`:

```python
def __init__(
    self,
    features: Mapping[str, list[ExtractionStep]],
    transforms: Mapping[str, list[ExtractionStep]],
    log_dir: str | Path | None = None,
    num_workers: int = 1,
    batch_size: int = 128,
    debug=False,
    console=Console(),
):
```

Add after `self.console = console` (line 115):

```python
self.batch_size = batch_size
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extractor.py::TestBatchSize -v`
Expected: 2 passed

**Step 5: Commit**

```
git add src/mosaique/extraction/extractor.py tests/test_extractor.py
git commit -m "add batch_size parameter to FeatureExtractor"
```

---

### Task 3: Wrap transform loop in batch loop

This is the core change. The existing transform iteration (lines 320-377 of `extractor.py`) moves inside a batch loop.

**Files:**
- Modify: `src/mosaique/extraction/extractor.py:309-385` (the `extract_feature` method body after EegData construction)
- Test: `tests/test_extractor.py`

**Step 1: Write the failing test**

Add to `tests/test_extractor.py`:

```python
class TestBatching:
    def test_batched_matches_unbatched(self, synthetic_array, simple_features, simple_transforms):
        """Batched extraction with batch_size=1 must produce same results as unbatched."""
        # "Unbatched" — batch_size larger than n_epochs (2 epochs)
        ext_full = FeatureExtractor(
            simple_features, simple_transforms, debug=True,
            batch_size=999,
            console=Console(quiet=True),
        )
        df_full = ext_full.extract_feature(synthetic_array, eeg_id="full", sfreq=200.0)

        # Batched — 1 epoch per batch
        ext_batched = FeatureExtractor(
            simple_features, simple_transforms, debug=True,
            batch_size=1,
            console=Console(quiet=True),
        )
        df_batched = ext_batched.extract_feature(synthetic_array, eeg_id="batched", sfreq=200.0)

        # Same shape
        assert df_full.shape == df_batched.shape

        # Same values (sort both identically to compare)
        sort_cols = ["epoch", "channel", "feature", "params"]
        df_full_sorted = df_full.sort(sort_cols)
        df_batched_sorted = df_batched.sort(sort_cols)
        np.testing.assert_allclose(
            df_full_sorted["value"].to_numpy(),
            df_batched_sorted["value"].to_numpy(),
            rtol=1e-10,
        )

    def test_batch_size_larger_than_epochs(self, synthetic_array, simple_features, simple_transforms):
        """When batch_size > n_epochs, should work identically to no batching."""
        extractor = FeatureExtractor(
            simple_features, simple_transforms, debug=True,
            batch_size=1000,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(synthetic_array, eeg_id="big_batch", sfreq=200.0)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extractor.py::TestBatching -v`
Expected: tests pass vacuously (the parameter exists but doesn't change behavior yet). That's fine — the test asserts correctness which will guard against regressions during the refactor.

**Step 3: Implement the batch loop**

Replace the body of `extract_feature` from line 309 (after `eeg_data` is constructed) through line 385 (return). The new structure:

```python
        self._cached_coeffs = {}
        self._cache_tag: tuple = ()
        self._init_logger(eeg_id)
        total_start = time.perf_counter()
        n_epochs = eeg_data.data.shape[0]
        log_str = (
            f"Starting extraction for {eeg_id} (number of epochs: {n_epochs})"
        )
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
                log_str = f"Starting transform: {transform_name}"
                self.logger.info("-" * 50)
                self.logger.info(log_str)
                self.logger.info("-" * 50)

                with Progress(console=self.console, transient=True) as progress:
                    n_features = len(self._make_param_grid(self._features[transform_name]))
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
                        self._transformed_eeg = self._curr_transform.transform(batch_eeg)
                        transform_time = time.perf_counter() - transform_start
                        progress.update(task_id, description=f"[batch {batch_idx + 1}/{n_batches}] Extracting ({transform_name})...")

                        self.logger.info(
                            f"Transform {transform_name}: {self._curr_transform_params} completed in {transform_time:.2f}s"
                        )

                        # 3. Extract all features for this transform
                        self._curr_features = self._features[transform_name]
                        features = self._extract_for_single_transform(
                            self._transformed_eeg, progress=progress, task_id=task_id
                        )
                        # 4. Add hyperparameters to dataframe
                        self._transformed_df = self._curr_transform.complete_df(features)

                        # 5. Append to list of dataframes
                        self._extracted_features.append(self._transformed_df)

                        # Transfer cached coeffs and tag back
                        self._cached_coeffs = self._curr_transform._cached_coeffs
                        self._cache_tag = self._curr_transform._cache_tag

                    transform_group_time = (time.perf_counter() - transform_group_start) / 60
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
```

Key differences from original:
- Outer `for batch_idx in range(n_batches)` loop wraps the transform iteration
- `eeg_data` → `batch_eeg` (sliced) passed to `transform.transform()`
- Progress bar description includes batch number
- `_cached_coeffs` and `_cache_tag` are cleared after each batch completes all transforms

**Step 4: Run ALL tests to verify nothing broke**

Run: `pytest tests/ -v`
Expected: All tests pass, including the new `TestBatching` tests

**Step 5: Commit**

```
git add src/mosaique/extraction/extractor.py tests/test_extractor.py
git commit -m "batch epochs in extract_feature for bounded memory"
```

---

### Task 4: Integration test with multi-epoch batching

Validates batching works end-to-end with more epochs than batch_size.

**Files:**
- Test: `tests/test_extractor.py`

**Step 1: Write the integration test**

Add to `tests/test_extractor.py`:

```python
class TestBatchingIntegration:
    def test_many_epochs_batched(self, simple_features, simple_transforms):
        """10 epochs with batch_size=3 → 4 batches (3+3+3+1)."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((10, 3, 400)) * 1e-6

        extractor = FeatureExtractor(
            simple_features, simple_transforms, debug=True,
            batch_size=3,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(data, eeg_id="many_epochs", sfreq=200.0)

        # 10 epochs × 3 channels × 3 features (linelength + sampen×2 params) = 90 rows
        assert len(df) == 90
        assert df["epoch"].n_unique() == 10

    def test_batch_size_one(self, simple_features, simple_transforms):
        """Extreme case: batch_size=1, one epoch per batch."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((4, 2, 200)) * 1e-6

        extractor = FeatureExtractor(
            simple_features, simple_transforms, debug=True,
            batch_size=1,
            console=Console(quiet=True),
        )
        df = extractor.extract_feature(data, eeg_id="single_batch", sfreq=200.0)

        # 4 epochs × 2 channels × 3 features = 24 rows
        assert len(df) == 24
```

**Step 2: Run tests**

Run: `pytest tests/test_extractor.py::TestBatchingIntegration -v`
Expected: 2 passed

**Step 3: Commit**

```
git add tests/test_extractor.py
git commit -m "add integration tests for epoch batching"
```

---

### Task 5: Update docstrings

**Files:**
- Modify: `src/mosaique/extraction/extractor.py` (docstrings for `__init__` and `extract_feature`)

**Step 1: Update `__init__` docstring**

Add `batch_size` to the Parameters section of the `__init__` docstring (after `num_workers`):

```
batch_size : int
    Number of epochs to process per batch.  Limits peak memory by
    running the full transform→feature pipeline on a subset of epochs
    at a time.  Default 128.
```

**Step 2: Update `extract_feature` docstring**

No parameter change needed (batch_size is on __init__). But add a note to the method docstring body:

```
When the recording contains more epochs than ``batch_size``, the data
is processed in batches to keep memory bounded.  Results are
concatenated into a single DataFrame.
```

**Step 3: Run tests to confirm nothing broke**

Run: `pytest tests/ -v`
Expected: All pass

**Step 4: Commit**

```
git add src/mosaique/extraction/extractor.py
git commit -m "document batch_size parameter"
```
