# CLAUDE.md

## Architecture

**Mosaïque** is a configuration-driven, parallel CPU-based EEG feature extraction library. The output is always a Polars DataFrame in long format.

Design principles:
- Simple user-facing interface
- Implement new features is easy
- Works with custom, user-defined features
- Minimal dependencies

Goal:
- Faster than MNE, more memory efficient (tested with @script/benchmark.py)
- "Smart" pipelining of feature extraction without redundant computations
- Large library of features

### Data flow

```
EEG input (MNE Epochs or np.ndarray)
  → EegData (normalized container, src/mosaique/extraction/eegdata.py)
  → FeatureExtractor (src/mosaique/extraction/extractor.py)
      for each PreExtractionTransform × param grid:
          apply transform → for each feature × param grid:
              extract → polars DataFrame row
  → concatenated polars DataFrame
```

### Key modules

| Module | Role |
|--------|------|
| `extraction/extractor.py` | Main orchestrator; manages param grid expansion, multiprocessing (pathos), progress (rich) |
| `extraction/eegdata.py` | `EegData` container + `EpochsLike` protocol bridging MNE and numpy |
| `extraction/transforms/` | Pre-extraction transforms: `simple` (identity), `tf_decomposition` (wavelet), `connectivity` (spectral connectivity matrices) |
| `features/univariate.py` | ~12 signal features (entropy, line length, Hurst exponent, peak alpha…) |
| `features/graph_metrics.py` | Graph metrics (clustering, efficiency, degree, path length) via numpy/scipy |
| `features/registry.py` | `@register_feature` decorator and `FEATURE_REGISTRY`; auto-discovery for tests and config-time validation |
| `features/timefrequency.py` | Wavelet helpers and `FrequencyBand` types |
| `config/types.py` | Pydantic models: `ExtractionStep`, `PipelineConfig` |
| `config/loader.py` | YAML parsing; resolves dotted string paths to callables |

### Configuration

Pipelines are defined in YAML or as dicts and loaded via `parse_featureextraction_config`. Each step specifies a transform type, a feature function (dotted import path), and a parameter grid — the Cartesian product generates all extraction tasks automatically.

See `script/features_config.yaml` for a working example.

### Extending

- **Custom transform**: subclass `PreExtractionTransform`, implement `transform()` and `extract_feature()`, register in `TRANSFORM_REGISTRY` (`extraction/transforms/__init__.py`).
- **Custom feature**: write a function matching `FeatureFunction` protocol; reference it by dotted path in YAML.
- **Registering a built-in feature**: decorate with `@register_feature(transform="simple")` from `mosaique.features.registry`. This auto-includes it in edge-case tests and enables config-time validation (mismatched transform types are caught before extraction).
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) runs pytest + pyright + ruff on every push to `main` and every PR to `main`.

## Git

- Do NOT add "Co-Authored-By" or "Generated with Claude Code" lines to commit messages

## Tooling

- Package manager: **uv** (`uv.lock` present)
- Testing: pytest (features are auto-discovered from `FEATURE_REGISTRY`)
- Linting: **ruff**
- Type checking: **pyright**
- Build backend: **hatchling**
- MNE is an optional dependency (`pip install mosaique[mne]`)
- Environment auto-activated via direnv (`.envrc`)
