# Feature Validation System

## Problem

When collaborators add new EEG features, there is no systematic way to verify that the feature is compatible with the project before merging. Current gaps:

- No CI pipeline: tests only run locally.
- The extractor silently swallows feature errors, producing zero rows with no visible error.
- No contract enforcement: return type and input shape compatibility with a transform type are only discovered at runtime.
- Manual test registration: contributors must edit `UNIVARIATE_FEATURES`/`CONNECTIVITY_FEATURES` lists by hand.

## Design

### 1. Feature Registry (`mosaique/features/registry.py`)

A `@register_feature` decorator that stores metadata about each built-in feature:

```python
from mosaique.features.registry import register_feature

@register_feature(transform="simple")
def my_new_feature(x, **kwargs):
    return float(...)
```

The decorator:
- Stores the function + metadata (compatible transform types) in a global `FEATURE_REGISTRY` dict.
- Validates at import time that the function accepts `**kwargs`.
- Tags the function with compatible transform type(s): `"simple"`, `"tf_decomposition"`, `"connectivity"`.

All existing built-in features get decorated. External user-defined features do not need the decorator.

### 2. Config-Time Validation

When `parse_featureextraction_config` resolves a built-in feature (one found in `FEATURE_REGISTRY`), it checks that the transform type declared in YAML matches the feature's registered compatible transforms. Raises a clear error on mismatch:

> Feature `line_length` is registered for `simple` but config assigns it to `connectivity`.

For external (unregistered) features, only the existing `**kwargs` warning applies.

### 3. Better Runtime Visibility

The extractor keeps its current lenient behavior (catch exceptions, skip broken features), but:
- Logs a **warning** (not just debug) per failed feature, with the feature name and exception.
- Prints a summary to the Rich console at the end of extraction: "N features failed: [list]. Check logs for details."

### 4. Auto-Generated Tests

`test_features.py` imports `FEATURE_REGISTRY` to build parametrized test lists dynamically:

```python
from mosaique.features.registry import FEATURE_REGISTRY

UNIVARIATE_FEATURES = [
    f.func for f in FEATURE_REGISTRY.values() if "simple" in f.transforms
]
CONNECTIVITY_FEATURES = [
    f.func for f in FEATURE_REGISTRY.values() if "connectivity" in f.transforms
]
```

Edge-case coverage (valid input, flat signal, NaN/Inf, short/empty, wrong dims, large values) is automatically applied to every registered feature.

A new **integration test** loads a minimal config with every registered feature, runs extraction on synthetic data, and asserts every feature produces at least one non-null row (catches the "silently skipped" scenario).

A CI check verifies that every public function in `mosaique/features/*.py` is decorated, so contributors cannot forget.

### 5. CI Pipeline (`.github/workflows/ci.yml`)

Runs on every PR to `main`:

1. **pytest** - full test suite including auto-generated feature tests and integration test.
2. **pyright** - static type checking for signature mismatches.
3. **ruff** - linting for consistent style.

## Scope

- Built-in features only. External user features are their own responsibility.
- No change to the public API. The decorator is additive.
- Runtime behavior stays lenient (no extraction crashes), but failures become visible.
