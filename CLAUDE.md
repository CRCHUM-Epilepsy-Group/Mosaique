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
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) runs pytest + ruff on every push to `main` and every PR to `main`.

## Git

- Do NOT add "Co-Authored-By" or "Generated with Claude Code" lines to commit messages

## Tooling

- Package manager: **uv** (`uv.lock` present)
- Testing: pytest (features are auto-discovered from `FEATURE_REGISTRY`)
- Linting: **ruff**
- Build backend: **hatchling**
- MNE is an optional dependency (`pip install mosaique[mne]`)
- Environment auto-activated via direnv (`.envrc`)

# Worktree Addon — Python & Cleanup

Supplementary steps to run **after** the superpowers worktree skill completes worktree creation.

## Python Project Setup (direnv + uv)

**Critical for Python projects using direnv and/or uv.** Without these steps, the worktree silently uses the main repo's venv and editable install — tests run against the wrong source code.

### 1. direnv: Create worktree-local `.envrc`

direnv resolves paths relative to the `.envrc` file's location, NOT the CWD that triggered evaluation. Without a worktree-local `.envrc`, the main repo's `.envrc` activates the main repo's `.venv`.

```bash
if [ -f .envrc ]; then
  cp .envrc "$path/.envrc"
  direnv allow "$path"
fi
```

### 2. uv: Create worktree-local venv with correct editable install

`uv sync` generates `.pth` files pointing to the `src/` of wherever it's run. Must target the worktree to get `.pth` files pointing to `<worktree>/src/`.

```bash
if [ -f uv.lock ]; then
  # Check pyproject.toml for optional-dependencies and ask user which extras to include
  uv sync --project "$path"    # add --extra <name> as needed
fi
```

If the project doesn't use uv but has other Python tooling:
```bash
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
if [ -f pyproject.toml ] && command -v poetry &>/dev/null; then cd "$path" && poetry install; fi
```

### 3. Verify isolation

```bash
cd "$path"
python -c "import <package_name>; print(__import__('pathlib').Path(<package_name>.__file__).resolve())"
```

The printed path **must** be inside the worktree, not the main repo. If it points to the main repo, the setup failed — re-run `uv sync --project "$path"`.

## Worktree Cleanup Order

**This overrides the cleanup order from `finishing-a-development-branch` if they conflict.**

Always `cd` back to the main repo FIRST, then remove worktree BEFORE deleting branch. Getting this wrong is unrecoverable in the current shell.

```bash
# CORRECT order — three separate commands
cd <main-repo-root>              # step 1: leave the worktree
git worktree remove <path>       # step 2: remove worktree
git branch -d <branch>           # step 3: delete branch
```

**If you see:** `error: cannot delete branch '<name>' used by worktree at '<path>'`
→ You deleted the branch before removing the worktree. Run `git worktree remove <path>` first, then retry `git branch -d <branch>`.

**If you see:** shell CWD gone / all commands failing
→ You removed the worktree while still inside it. `cd <main-repo-root>` first.

**Never combine steps 2+3 on one line** — always run them sequentially and separately.
