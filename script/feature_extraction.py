#!/usr/bin/env python
"""Example script: extract EEG features from EDF files using mosaique.

Loads EDF files from ``tests/test_data/``, epochs them into fixed-length
windows, runs the full feature extraction pipeline configured in
``features_config.yaml``, and prints the resulting feature DataFrame.
"""

import time
from pathlib import Path

import mne
import polars as pl
import psutil

from mosaique import FeatureExtractor, parse_featureextraction_config, resolve_pipeline

_PROC = psutil.Process()


def _snap() -> dict:
    """Capture a process resource snapshot."""
    cpu = _PROC.cpu_times()
    return {
        "wall": time.perf_counter(),
        "cpu_user": cpu.user,
        "cpu_sys": cpu.system,
        "rss_mb": _PROC.memory_info().rss / 1024**2,
    }


def _report(start: dict, end: dict, label: str) -> None:
    wall = end["wall"] - start["wall"]
    cpu = (end["cpu_user"] - start["cpu_user"]) + (end["cpu_sys"] - start["cpu_sys"])
    print(
        f"  [{label}] "
        f"wall={wall:.2f}s | cpu={cpu:.2f}s | "
        f"RAM start={start['rss_mb']:.1f} MB → end={end['rss_mb']:.1f} MB"
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "features_config.yaml"
DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
OUTPUT_DIR = SCRIPT_DIR / "output"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse the feature extraction config
    # ------------------------------------------------------------------
    pipeline = parse_featureextraction_config(CONFIG_FILE)
    features, frameworks = resolve_pipeline(pipeline)

    # ------------------------------------------------------------------
    # 2. Discover EDF files (limit to first 2 for benchmarking)
    # ------------------------------------------------------------------
    edf_files = sorted(DATA_DIR.rglob("*.edf"))[:2]
    if not edf_files:
        print(f"No EDF files found in {DATA_DIR}")
        return

    print(f"Processing {len(edf_files)} EDF file(s) from {DATA_DIR}\n")

    # ------------------------------------------------------------------
    # 3. Extract features for each file
    # ------------------------------------------------------------------
    all_features: list[pl.DataFrame] = []
    overall_start = _snap()

    for edf_path in edf_files:
        print(f"Processing: {edf_path.name}")
        file_start = _snap()

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.crop(tmax=min(120.0, raw.times[-1]))
        raw.filter(1.0, 50.0, verbose=False)
        epochs = mne.make_fixed_length_epochs(raw, duration=5.0, verbose=False)
        epochs.load_data()
        print(
            f"  {len(epochs)} epochs, {len(epochs.ch_names)} channels, "
            f"sfreq={epochs.info['sfreq']} Hz"
        )

        extractor = FeatureExtractor(
            features,
            frameworks,
            num_workers=4,
        )
        df = extractor.extract_feature(epochs, eeg_id=edf_path.stem)
        df = df.with_columns(file=pl.lit(edf_path.name))
        all_features.append(df)

        file_end = _snap()
        print(f"  Extracted {len(df)} feature rows")
        _report(file_start, file_end, edf_path.name)
        print()

    overall_end = _snap()

    # ------------------------------------------------------------------
    # 4. Combine and display results
    # ------------------------------------------------------------------
    result = pl.concat(all_features)
    print("=" * 60)
    print(f"Total feature rows: {len(result)}")
    print(f"Columns: {result.columns}")
    print("=" * 60)
    print(result.head(20))
    print()
    _report(overall_start, overall_end, "TOTAL")

    # ------------------------------------------------------------------
    # 5. Save to parquet
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "features.parquet"
    result.write_parquet(output_path)
    print(f"\nFeatures saved to {output_path}")


if __name__ == "__main__":
    main()
