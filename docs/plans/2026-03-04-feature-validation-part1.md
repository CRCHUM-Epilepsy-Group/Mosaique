# Feature Validation System — Part 1: Registry & Tests

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Add a decorator-based feature registry with config-time validation, better runtime warnings, and auto-generated tests so that broken features are caught before extraction.

**Architecture:** A `@register_feature` decorator stores metadata in a global `FEATURE_REGISTRY`. The config loader validates registered features against their transform type. Tests auto-discover features from the registry.

**Tech Stack:** Python, pytest

---

### Task 1: Create the feature registry module

**Files:**
- Create: `src/mosaique/features/registry.py`
- Test: `tests/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_registry.py
"""Tests for the feature registry."""

import numpy as np
import pytest

from mosaique.features.registry import FEATURE_REGISTRY, register_feature


def test_register_feature_adds_to_registry():
    """A decorated function is added to the registry."""

    @register_feature(transform="simple")
    def dummy_feature(x, **kwargs):
        return float(np.mean(x))

    assert "dummy_feature" in FEATURE_REGISTRY
    entry = FEATURE_REGISTRY["dummy_feature"]
    assert entry.func is dummy_feature
    assert entry.transforms == {"simple"}
    del FEATURE_REGISTRY["dummy_feature"]


def test_register_feature_multiple_transforms():
    """A feature can be registered for multiple transforms."""

    @register_feature(transform=["simple", "tf_decomposition"])
    def multi_feature(x, **kwargs):
        return 0.0

    entry = FEATURE_REGISTRY["multi_feature"]
    assert entry.transforms == {"simple", "tf_decomposition"}
    del FEATURE_REGISTRY["multi_feature"]


def test_register_feature_rejects_missing_kwargs():
    """Functions without **kwargs are rejected at registration time."""
    with pytest.raises(TypeError, match="kwargs"):

        @register_feature(transform="simple")
        def bad_feature(x):
            return 0.0


def test_register_feature_preserves_function():
    """The decorator returns the original function unchanged."""

    @register_feature(transform="simple")
    def my_feat(x, **kwargs):
        return 1.0

    assert my_feat(np.array([1.0])) == 1.0
    del FEATURE_REGISTRY["my_feat"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mosaique.features.registry'`

**Step 3: Write minimal implementation**

```python
# src/mosaique/features/registry.py
"""Feature registry for built-in feature functions.

The ``@register_feature`` decorator stores metadata about each feature
(compatible transforms, the function itself) in a global registry.  This
registry is used by:

- The config loader to validate that a feature is wired to a compatible
  transform type.
- The test suite to auto-discover features for parametrized edge-case tests.
"""

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureEntry:
    """Metadata about a registered feature function."""

    func: Any
    transforms: frozenset[str]


# Global registry: function name -> FeatureEntry
FEATURE_REGISTRY: dict[str, FeatureEntry] = {}


def register_feature(
    transform: str | list[str],
) -> Any:
    """Decorator to register a feature function.

    Parameters
    ----------
    transform : str or list[str]
        Transform type(s) this feature is compatible with.
        One of ``"simple"``, ``"tf_decomposition"``, ``"connectivity"``.

    Raises
    ------
    TypeError
        If the function does not accept ``**kwargs``.

    Example
    -------
    ::

        @register_feature(transform="simple")
        def my_feature(x, sfreq=200, **kwargs):
            return float(np.mean(x))
    """
    if isinstance(transform, str):
        transforms = frozenset([transform])
    else:
        transforms = frozenset(transform)

    def decorator(func: Any) -> Any:
        sig = inspect.signature(func)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_keyword:
            raise TypeError(
                f"{func.__name__!r} must accept **kwargs to be registered "
                f"as a feature function"
            )

        FEATURE_REGISTRY[func.__name__] = FeatureEntry(
            func=func,
            transforms=transforms,
        )
        return func

    return decorator
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/mosaique/features/registry.py tests/test_registry.py
git commit -m "feat: add feature registry with @register_feature decorator"
```

---

### Task 2: Decorate all existing built-in features

**Files:**
- Modify: `src/mosaique/features/univariate.py`
- Modify: `src/mosaique/features/graph_metrics.py`
- Modify: `tests/test_registry.py`

**Step 1: Write the failing test**

Add to `tests/test_registry.py`:

```python
def test_all_univariate_features_registered():
    """All univariate features are in the registry."""
    import mosaique.features.univariate  # noqa: F401

    expected = {
        "approximate_entropy",
        "sample_entropy",
        "spectral_entropy",
        "permutation_entropy",
        "fuzzy_entropy",
        "corr_dim",
        "line_length",
        "peak_alpha",
        "hurst_exp",
        "band_power",
    }
    registered = {
        name
        for name, entry in FEATURE_REGISTRY.items()
        if "simple" in entry.transforms
    }
    assert expected <= registered


def test_all_connectivity_features_registered():
    """All connectivity features are in the registry."""
    import mosaique.features.graph_metrics  # noqa: F401

    expected = {
        "average_clustering",
        "average_node_connectivity",
        "average_degree",
        "global_efficiency",
        "average_shortest_path_length",
    }
    registered = {
        name
        for name, entry in FEATURE_REGISTRY.items()
        if "connectivity" in entry.transforms
    }
    assert expected <= registered
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py::test_all_univariate_features_registered tests/test_registry.py::test_all_connectivity_features_registered -v`
Expected: FAIL — features not yet in registry

**Step 3: Add decorators to univariate.py**

Add import at top of `src/mosaique/features/univariate.py`:

```python
from mosaique.features.registry import register_feature
```

Add `@register_feature(transform="simple")` above each of these functions:
- `approximate_entropy`
- `sample_entropy`
- `spectral_entropy`
- `permutation_entropy`
- `fuzzy_entropy`
- `corr_dim`
- `line_length`
- `peak_alpha`
- `hurst_exp`
- `band_power`

Do NOT decorate helpers: `_validate_signal`, `skip_nones`, `logarithmic_r`, `shannon_entropy`, `ordinal_distribution`, `rescaled_range`, `_multitaper_psd`.

**Step 4: Add decorators to graph_metrics.py**

Add import at top of `src/mosaique/features/graph_metrics.py`:

```python
from mosaique.features.registry import register_feature
```

Add `@register_feature(transform="connectivity")` above each of these functions:
- `average_clustering`
- `average_node_connectivity`
- `average_degree`
- `global_efficiency`
- `average_shortest_path_length`

Do NOT decorate `connected_threshold` or `binary_threshold`.

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (6 tests)

**Step 6: Run full test suite**

Run: `pytest -v`
Expected: All existing tests still pass.

**Step 7: Commit**

```bash
git add src/mosaique/features/univariate.py src/mosaique/features/graph_metrics.py tests/test_registry.py
git commit -m "feat: register all built-in features with @register_feature"
```

---

### Task 3: Config-time validation of transform compatibility

**Files:**
- Modify: `src/mosaique/config/loader.py`
- Modify: `tests/test_config.py`

**Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_resolve_pipeline_rejects_incompatible_transform():
    """A registered feature wired to the wrong transform raises ValueError."""
    from mosaique.config.loader import (
        parse_featureextraction_config,
        resolve_pipeline,
    )

    config = {
        "features": {
            "connectivity": [
                {"name": "line_length", "function": "univariate.line_length"},
            ]
        },
        "transforms": {
            "connectivity": [
                {
                    "name": "connectivity",
                    "function": "connectivity.cwt_spectral_connectivity",
                    "params": {
                        "freqs": [[(4, 8)]],
                        "method": "pli",
                    },
                }
            ],
        },
    }
    pipeline = parse_featureextraction_config(config)
    with pytest.raises(ValueError, match="registered for.*simple.*not.*connectivity"):
        resolve_pipeline(pipeline)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_resolve_pipeline_rejects_incompatible_transform -v`
Expected: FAIL — no ValueError raised

**Step 3: Add validation to `resolve_pipeline`**

In `src/mosaique/config/loader.py`, add import at top:

```python
from mosaique.features.registry import FEATURE_REGISTRY
```

In `resolve_pipeline`, in the feature resolution loop, after `load_feature_extraction_func(sc.function)`, add:

```python
func = load_feature_extraction_func(sc.function)

# Validate transform compatibility for registered features
if func is not None and func.__name__ in FEATURE_REGISTRY:
    entry = FEATURE_REGISTRY[func.__name__]
    if group_name not in entry.transforms:
        raise ValueError(
            f"Feature {sc.function!r} is registered for "
            f"{set(entry.transforms)} but config assigns it to "
            f"{group_name!r}; not compatible"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest -v`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/mosaique/config/loader.py tests/test_config.py
git commit -m "feat: validate feature-transform compatibility at config time"
```

---

### Task 4: Better runtime failure visibility

**Files:**
- Modify: `src/mosaique/extraction/extractor.py`
- Modify: `tests/test_extractor.py`

**Step 1: Write the failing test**

Add to `tests/test_extractor.py`:

```python
def test_failed_features_produce_warnings(synthetic_epochs, caplog):
    """When a feature fails at runtime, a warning is logged and a summary printed."""
    import logging

    def always_fails(x, **kwargs):
        raise RuntimeError("intentional failure")

    from mosaique.config.types import ExtractionStep

    features = {
        "simple": [
            ExtractionStep(
                name="always_fails",
                function=always_fails,
                params={},
            ),
        ]
    }
    transforms = {
        "simple": [
            ExtractionStep(name="simple", function=None, params={}),
        ]
    }

    extractor = FeatureExtractor(
        features=features, transforms=transforms, num_workers=1
    )

    with caplog.at_level(logging.WARNING):
        df = extractor.extract_feature(synthetic_epochs, eeg_id="test")

    assert any("always_fails" in record.message for record in caplog.records)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_extractor.py::test_failed_features_produce_warnings -v`
Expected: FAIL — currently uses `logger.error`, not `warning`

**Step 3: Modify extractor error handling**

In `src/mosaique/extraction/extractor.py`:

1. In `__init__`, add: `self._failed_features: list[str] = []`

2. In `_extract_for_single_transform`, change the except block (~line 221):

```python
            except Exception as e:
                self.logger.warning(
                    f"Feature {name} failed: {type(e).__name__}: {e}"
                )
                self.logger.debug(f"Traceback for {name}:\n{format_exc()}")
                self._failed_features.append(name)
                continue
```

3. At the end of `extract_feature`, before `return final_features_cleaned`:

```python
if self._failed_features:
    n = len(self._failed_features)
    names = ", ".join(sorted(set(self._failed_features)))
    self.console.print(
        f"[yellow]Warning: {n} feature extraction(s) failed: {names}. "
        f"Check logs for details.[/yellow]"
    )
    self._failed_features.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_extractor.py::test_failed_features_produce_warnings -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest -v`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/mosaique/extraction/extractor.py tests/test_extractor.py
git commit -m "feat: warn on failed features with summary at extraction end"
```

---

### Task 5: Auto-discover registered features in tests

**Files:**
- Modify: `tests/test_features.py`

**Step 1: Replace manual lists with registry-driven discovery**

Replace the imports and manual lists (lines 23-65) in `tests/test_features.py` with:

```python
# Force import to trigger decorators
import mosaique.features.univariate  # noqa: F401
import mosaique.features.graph_metrics  # noqa: F401

from mosaique.features.univariate import peak_alpha, band_power
from mosaique.features.registry import FEATURE_REGISTRY

# Auto-discover from registry
UNIVARIATE_FEATURES = [
    entry.func
    for entry in FEATURE_REGISTRY.values()
    if "simple" in entry.transforms
    and entry.func.__name__ not in ("band_power", "peak_alpha")
]

CONNECTIVITY_FEATURES = [
    entry.func
    for entry in FEATURE_REGISTRY.values()
    if "connectivity" in entry.transforms
]
```

**Step 2: Run test suite**

Run: `pytest tests/test_features.py -v`
Expected: PASS — same parametrized tests, now auto-discovered.

**Step 3: Commit**

```bash
git add tests/test_features.py
git commit -m "refactor: auto-discover features from registry in tests"
```

---

### Task 6: Integration test — all registered features produce output

**Files:**
- Modify: `tests/test_extractor.py`

**Step 1: Write the integration test**

Add to `tests/test_extractor.py`:

```python
def test_all_registered_simple_features_produce_output(synthetic_epochs):
    """Every registered 'simple' feature produces at least one row."""
    from mosaique.features.registry import FEATURE_REGISTRY
    from mosaique.config.types import ExtractionStep

    simple_features = [
        ExtractionStep(
            name=name,
            function=entry.func,
            params={},
        )
        for name, entry in FEATURE_REGISTRY.items()
        if "simple" in entry.transforms
        and name != "band_power"
    ]

    if not simple_features:
        pytest.skip("No simple features registered")

    features = {"simple": simple_features}
    transforms = {
        "simple": [ExtractionStep(name="simple", function=None, params={})],
    }

    extractor = FeatureExtractor(
        features=features, transforms=transforms, num_workers=1
    )
    df = extractor.extract_feature(synthetic_epochs, eeg_id="test")

    feature_names_in_output = set(df["feature"].unique().to_list())
    registered_names = {f.name for f in simple_features}
    missing = registered_names - feature_names_in_output
    assert not missing, f"These features produced no output (silently failed?): {missing}"
```

**Step 2: Run test**

Run: `pytest tests/test_extractor.py::test_all_registered_simple_features_produce_output -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_extractor.py
git commit -m "test: integration test for all registered features producing output"
```

---

### Task 7: CI check — all public feature functions are decorated

**Files:**
- Create: `tests/test_registry_completeness.py`

**Step 1: Write the test**

```python
# tests/test_registry_completeness.py
"""Ensure every public feature function in mosaique/features/ is registered."""

import inspect

import mosaique.features.univariate as univariate_mod
import mosaique.features.graph_metrics as graph_metrics_mod
from mosaique.features.registry import FEATURE_REGISTRY

_UNIVARIATE_HELPERS = {
    "_validate_signal",
    "skip_nones",
    "logarithmic_r",
    "shannon_entropy",
    "ordinal_distribution",
    "rescaled_range",
    "_multitaper_psd",
}

_GRAPH_METRICS_HELPERS = {
    "_validate_matrix",
    "_binary_adjacency",
    "connected_threshold",
    "binary_threshold",
}


def _public_functions(module, exclude):
    """Get all public callable names from a module, minus exclusions."""
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_")
        and name not in exclude
        and obj.__module__ == module.__name__
    }


def test_all_univariate_functions_registered():
    public = _public_functions(univariate_mod, _UNIVARIATE_HELPERS)
    registered = set(FEATURE_REGISTRY.keys())
    missing = public - registered
    assert not missing, (
        f"Univariate functions not registered: {missing}. "
        f"Add @register_feature decorator."
    )


def test_all_graph_metrics_functions_registered():
    public = _public_functions(graph_metrics_mod, _GRAPH_METRICS_HELPERS)
    registered = set(FEATURE_REGISTRY.keys())
    missing = public - registered
    assert not missing, (
        f"Graph metric functions not registered: {missing}. "
        f"Add @register_feature decorator."
    )
```

**Step 2: Run test**

Run: `pytest tests/test_registry_completeness.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_registry_completeness.py
git commit -m "test: ensure all public feature functions are registered"
```
