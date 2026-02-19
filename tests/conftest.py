import numpy as np
import pytest
import mne


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def random_signal(rng):
    """500-sample random signal."""
    return rng.standard_normal(500)


@pytest.fixture
def flat_signal():
    """Constant signal — triggers zero-std edge cases."""
    return np.ones(500)


@pytest.fixture
def signal_with_nan(random_signal):
    """Random signal with NaN values scattered in."""
    sig = random_signal.copy()
    sig[10] = np.nan
    sig[200] = np.nan
    return sig


@pytest.fixture
def signal_with_inf(random_signal):
    """Random signal with +/- inf values."""
    sig = random_signal.copy()
    sig[10] = np.inf
    sig[200] = -np.inf
    return sig


@pytest.fixture
def short_signal():
    """Very short signal (10 samples) — may break embedding-based features."""
    return np.array([1.0, 0.5, -0.3, 0.8, -1.2, 0.1, 0.6, -0.4, 0.9, -0.7])


@pytest.fixture
def signal_2d(rng):
    """2D array (3 channels x 500 samples) — wrong shape for univariate functions."""
    return rng.standard_normal((3, 500))


@pytest.fixture
def synthetic_epochs():
    """Minimal MNE Epochs object for integration tests."""
    sfreq = 200
    n_epochs, n_channels, n_times = 2, 3, 400
    ch_names = ["Fp1", "C3", "O1"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_meas_date(0)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6
    events = np.array([[0, 0, 1], [400, 0, 1]])
    event_id = {"epoch": 1}
    return mne.EpochsArray(data, info, events=events, event_id=event_id)


@pytest.fixture
def minimal_config_file(tmp_path):
    """Write a minimal YAML config to a temp file and return its path."""
    config = tmp_path / "config.yaml"
    config.write_text("""\
features:
  simple:
    - name: linelength
      function: univariate.line_length
      params:
    - name: sampen
      function: univariate.sample_entropy
      params:
        m: [2, 3]
        r: [0.2]

frameworks:
  simple:
    - name: simple
      function: null
      params: null
""")
    return config
