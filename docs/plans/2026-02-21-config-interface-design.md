# Config Interface Improvements

## Problem

The current user workflow requires 3 steps before extraction:

```python
pipeline = parse_featureextraction_config("config.yaml")
features, transforms = resolve_pipeline(pipeline)
extractor = FeatureExtractor(features, transforms, num_workers=4)
```

This leaks internal abstractions (`PipelineConfig`, the features/transforms split). The YAML config also requires verbose boilerplate for simple cases (explicit `simple` transform, params wrapped in lists, redundant `function: null`). The README has minimal documentation on how to define features.

## Goals

- Accept config directly in `FeatureExtractor` (no intermediate steps)
- Accept YAML string, file path, or dict as config input
- Auto-generate `simple` transform when `transforms` section is omitted
- Allow scalar params without wrapping in lists
- Shorthand string syntax for features with no params
- Top-level `mosaique.extract()` convenience function
- Comprehensive configuration documentation in README

## Design decisions

### Config dispatch in `FeatureExtractor.__init__`

Add `config` as the first positional arg. Dispatch on type:
- `str` ending in `.yaml`/`.yml` or `Path` → YAML file
- `str` (otherwise) → YAML string
- `dict` → raw config dict
- `None` (default) → fall back to existing `features`/`transforms` kwargs

```python
def __init__(
    self,
    config: str | Path | dict | None = None,
    *,
    features: Mapping[str, list[ExtractionStep]] | None = None,
    transforms: Mapping[str, list[ExtractionStep]] | None = None,
    ...
):
```

Raise `ValueError` if both `config` and `features`/`transforms` are given, or if neither.

### Auto-generate simple transform

`PipelineConfig` gets a `model_validator(mode="before")` that:
1. If `transforms` is missing entirely: for each key in `features`, add a default `[{"name": "<key>"}]` entry — but **only** for `"simple"`. Non-simple keys without matching transforms raise a clear error.
2. If `features` is a **list** (not a dict), wrap as `{"simple": <list>}` and auto-generate the simple transform.

### String shorthand for features

`ExtractionStepConfig` gets a `model_validator(mode="before")` converting a plain string to `{"name": <last_segment>, "function": <string>}`:

```yaml
features:
  simple:
    - univariate.line_length    # → name: "line_length", function: "univariate.line_length"
```

### Top-level `extract()` function

```python
df = mosaique.extract("config.yaml", epochs, eeg_id="subject_01", num_workers=4)
```

### YAML string support

`parse_featureextraction_config` detects YAML strings (contains newline or doesn't look like a file path) and parses them directly.

## What the simplified interface looks like

```yaml
# Minimal config — no transforms section, string shorthand
features:
  - univariate.line_length
  - univariate.spectral_entropy
  - name: sample_entropy
    function: univariate.sample_entropy
    params:
      m: [2, 3]
      r: 0.2
```

```python
extractor = FeatureExtractor("config.yaml", num_workers=4)
df = extractor.extract_feature(epochs, eeg_id="subject_01")

# Or one-liner:
df = mosaique.extract("config.yaml", epochs, eeg_id="subject_01", num_workers=4)
```
