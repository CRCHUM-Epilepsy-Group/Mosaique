#!/usr/bin/env python
"""Example script: extract EEG features from EDF files using mosaique.

Loads EDF files from ``tests/test_data/``, epochs them into fixed-length
windows, runs the full feature extraction pipeline configured in
``features_config.yaml``, and prints the resulting feature DataFrame.
"""

from pathlib import Path

import mne
import polars as pl

from mosaique import FeatureExtractor, parse_featureextraction_config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "features_config.yaml"
DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
OUTPUT_DIR = SCRIPT_DIR / "output"


def load_and_epoch_edf(
    edf_path: Path,
    epoch_duration: float = 2.0,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
) -> mne.Epochs:
    """Load an EDF file and segment it into fixed-length epochs.

    Parameters
    ----------
    edf_path : Path
        Path to the ``.edf`` file.
    epoch_duration : float
        Duration of each epoch in seconds.
    l_freq : float
        Lower bandpass frequency in Hz.
    h_freq : float
        Upper bandpass frequency in Hz.

    Returns
    -------
    mne.Epochs
        Epoched EEG data.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.filter(l_freq, h_freq, verbose=False)

    # Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, verbose=False)
    epochs.load_data()

    return epochs


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse the feature extraction config
    # ------------------------------------------------------------------
    features, frameworks = parse_featureextraction_config(CONFIG_FILE)

    # ------------------------------------------------------------------
    # 2. Discover EDF files
    # ------------------------------------------------------------------
    edf_files = sorted(DATA_DIR.glob("*.edf"))
    if not edf_files:
        print(f"No EDF files found in {DATA_DIR}")
        return

    print(f"Found {len(edf_files)} EDF file(s) in {DATA_DIR}\n")

    # ------------------------------------------------------------------
    # 3. Extract features for each file
    # ------------------------------------------------------------------
    all_features: list[pl.DataFrame] = []

    for edf_path in edf_files:
        print(f"Processing: {edf_path.name}")

        epochs = load_and_epoch_edf(edf_path)
        print(f"  {len(epochs)} epochs, {len(epochs.ch_names)} channels, "
              f"sfreq={epochs.info['sfreq']} Hz")

        extractor = FeatureExtractor(
            features,
            frameworks,
            num_workers=1,
        )
        df = extractor.extract_feature(epochs, eeg_id=edf_path.stem)
        df = df.with_columns(file=pl.lit(edf_path.name))
        all_features.append(df)

        print(f"  Extracted {len(df)} feature rows\n")

    # ------------------------------------------------------------------
    # 4. Combine and display results
    # ------------------------------------------------------------------
    result = pl.concat(all_features)
    print("=" * 60)
    print(f"Total feature rows: {len(result)}")
    print(f"Columns: {result.columns}")
    print("=" * 60)
    print(result.head(20))

    # ------------------------------------------------------------------
    # 5. Save to parquet
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "features.parquet"
    result.write_parquet(output_path)
    print(f"\nFeatures saved to {output_path}")


if __name__ == "__main__":
    main()
