# Config Interface — Implementation Plan

**Goal:** Simplify the user-facing config interface and add comprehensive documentation.

## Branch 1: `config-interface` — Simplify config loading + FeatureExtractor init

### Commit 1: Accept config directly in `FeatureExtractor.__init__`

**`src/mosaique/extraction/extractor.py`**

Add a `config` parameter as the first positional arg. Dispatch on type:
- `str` ending in `.yaml`/`.yml` or `Path` → YAML file
- `str` (otherwise) → YAML string
- `dict` → raw config dict
- `None` (default) → fall back to existing `features`/`transforms` kwargs

Internally calls `parse_featureextraction_config` + `resolve_pipeline` when `config` is provided. Raise `ValueError` if both `config` and `features`/`transforms` are given, or if neither is given.

**`src/mosaique/config/loader.py`**

Update `parse_featureextraction_config` to also accept a YAML **string** (not just file path). Detect by checking: if it's a `str` and contains a newline or doesn't look like a file path, parse it as YAML directly.

**Tests:** Add tests for all input types (file, string, dict, backwards-compat kwargs).

### Commit 2: Auto-generate `simple` transform when `transforms` section is omitted

**`src/mosaique/config/types.py`** — `PipelineConfig`

Add a `model_validator(mode="before")` that:
1. If `transforms` key is missing entirely, auto-generate it: for each key in `features`, add a default `[{"name": "<key>"}]` transform entry — but **only** for `"simple"`. If `features` contains non-simple keys without a matching transform, raise a clear error.
2. If `features` is a **list** (not a dict), wrap it as `{"simple": <list>}` and auto-generate the simple transform.

This enables:
```yaml
# Minimal: just features, simple transform implied
features:
  - name: line_length
    function: univariate.line_length
```

**Tests:** Add tests for omitted transforms, list-of-features shorthand, and error on non-simple missing transforms.

### Commit 3: Shorthand feature syntax — string-only entries

**`src/mosaique/config/types.py`** — `ExtractionStepConfig`

Add a `model_validator(mode="before")` that converts a plain string to `{"name": <last_segment>, "function": <string>}`:

```yaml
features:
  simple:
    - univariate.line_length          # → name: "line_length", function: "univariate.line_length"
    - name: sample_entropy
      function: univariate.sample_entropy
      params: {m: [2, 3]}
```

**Tests:** Add tests for string shorthand in YAML and dict configs.

### Commit 4: Top-level `mosaique.extract()` convenience function

**`src/mosaique/__init__.py`**

```python
def extract(
    config: str | Path | dict,
    eeg: EpochsLike | np.ndarray,
    eeg_id: str,
    *,
    num_workers: int = 1,
    batch_size: int = 128,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
    **kwargs,
) -> pl.DataFrame:
    """One-call feature extraction."""
    extractor = FeatureExtractor(config, num_workers=num_workers, batch_size=batch_size)
    return extractor.extract_feature(eeg, eeg_id, sfreq=sfreq, ch_names=ch_names, **kwargs)
```

**Tests:** One test verifying `mosaique.extract()` produces the same result as the explicit workflow.

### Commit 5: Update `__init__.py` exports and docstrings

- Keep `parse_featureextraction_config` and `resolve_pipeline` in `__all__` for power users
- Add `extract` to `__all__`
- Update module docstring to show the simplified workflow

## Branch 2: `config-docs` — README documentation

### Commit 1: Add configuration documentation to README

Add a **"Configuration"** section to the README covering:

1. **Quick example** — minimal config (flat feature list, no transforms section)
2. **Full config structure** — `features` + `transforms` with all options
3. **Feature reference** — table of all built-in features with module path, params, and description
4. **Transform reference** — table of built-in transforms (`simple`, `tf_decomposition`, `connectivity`)
5. **Parameter grids** — how lists create Cartesian products, scalar params
6. **Custom features** — how to write and reference your own
7. **Shorthand syntax** — string-only entries, omitting transforms

Also update the Quick Start to use the simplified interface:
```python
extractor = FeatureExtractor("config.yaml", num_workers=4)
df = extractor.extract_feature(epochs, eeg_id="subject_01")
```

## Files to modify

| File | Changes |
|------|---------|
| `src/mosaique/extraction/extractor.py` | New `config` param in `__init__` |
| `src/mosaique/config/loader.py` | YAML string support in `parse_featureextraction_config` |
| `src/mosaique/config/types.py` | Auto-generate simple transform, string shorthand for features |
| `src/mosaique/__init__.py` | Add `extract()`, update exports and docstring |
| `tests/test_config.py` | Tests for all new config shapes |
| `tests/test_extractor.py` | Tests for `FeatureExtractor(config=...)` |
| `README.md` | Configuration docs, updated quick start |

## Verification

1. `uv run pytest tests/test_config.py tests/test_extractor.py -v` — all tests pass
2. Manually verify the simplified workflow works end-to-end with the example config
3. Review README renders correctly
