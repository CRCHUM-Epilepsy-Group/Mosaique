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
def all_nan_signal():
    """Signal of all NaN values."""
    return np.full(500, np.nan)


@pytest.fixture
def all_inf_signal():
    """Signal of all inf values."""
    return np.full(500, np.inf)


@pytest.fixture
def large_value_signal(rng):
    """Signal with very large values (~1e15), simulating unscaled EEG."""
    return rng.standard_normal(500) * 1e15


@pytest.fixture
def empty_signal():
    """Empty signal — zero samples."""
    return np.array([])


@pytest.fixture
def single_sample_signal():
    """Single-sample signal."""
    return np.array([1.0])


@pytest.fixture
def list_signal(random_signal):
    """Python list version of random_signal (users passing lists)."""
    return random_signal.tolist()


@pytest.fixture
def connectivity_matrix():
    """Valid 3x3 symmetric connectivity matrix with values in [0, 1]."""
    mat = np.array([
        [0.0, 0.8, 0.3],
        [0.8, 0.0, 0.5],
        [0.3, 0.5, 0.0],
    ])
    return mat


@pytest.fixture
def zero_connectivity_matrix():
    """All-zeros 3x3 connectivity matrix."""
    return np.zeros((3, 3))


@pytest.fixture
def identity_connectivity_matrix():
    """3x3 identity connectivity matrix."""
    return np.eye(3)


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

transforms:
  simple:
    - name: simple
      function: null
      params: null
""")
    return config
