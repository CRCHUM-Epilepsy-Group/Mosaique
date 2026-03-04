# Feature Validation System — Part 2: CI & Documentation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Add ruff/pyright tooling, a GitHub Actions CI pipeline, and update documentation to reflect the new registry and CI system.

**Architecture:** CI pipeline runs pytest + ruff + pyright on every PR. Documentation updated to guide contributors on using `@register_feature`.

**Tech Stack:** GitHub Actions, ruff, pyright, uv

**Prerequisite:** Part 1 must be completed first (registry, decorators, validation, tests).

---

### Task 1: Add ruff and pyright configuration

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add ruff and pyright to dev dependencies and configure**

In `pyproject.toml`, update `[dependency-groups]`:

```toml
[dependency-groups]
dev = [
    "psutil",
    "ruff>=0.9",
    "pyright>=1.1",
]
```

Add tool config:

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pyright]
venvPath = "."
venv = ".venv"
```

**Step 2: Install and run locally**

Run: `uv sync --group dev`
Run: `uv run ruff check src/ tests/`
Run: `uv run pyright src/mosaique/features/registry.py`
Expected: No blocking errors (fix any that appear).

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add ruff and pyright to dev dependencies"
```

---

### Task 2: Create GitHub Actions CI pipeline

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Write the CI config**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --extra mne --group dev

      - name: Lint with ruff
        run: uv run ruff check src/ tests/

      - name: Type check with pyright
        run: uv run pyright src/

      - name: Run tests
        run: uv run pytest -v --tb=short
```

**Step 2: Verify syntax**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: No errors.

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions pipeline with pytest, ruff, pyright"
```

---

### Task 3: Update documentation

**Files:**
- Modify: `src/mosaique/features/__init__.py`
- Modify: `CLAUDE.md`

**Step 1: Update features/__init__.py docstring**

Add to the "Writing a new feature function" section (after rule 4):

```
5. **Decorate with ``@register_feature``** to add the function to the
   built-in registry.  This enables automatic test coverage and
   config-time validation::

       from mosaique.features.registry import register_feature

       @register_feature(transform="simple")
       def my_feature(X, sfreq=200, **kwargs):
           return float(np.mean(X))
```

**Step 2: Update CLAUDE.md**

In the "Extending" section, add:

```markdown
- **Registering a feature**: decorate with `@register_feature(transform="simple")` from `mosaique.features.registry`. This auto-includes it in edge-case tests and enables config-time validation.
- **CI**: GitHub Actions runs pytest + ruff + pyright on every PR to `main`.
```

**Step 3: Commit**

```bash
git add src/mosaique/features/__init__.py CLAUDE.md
git commit -m "docs: document feature registry and CI pipeline"
```
