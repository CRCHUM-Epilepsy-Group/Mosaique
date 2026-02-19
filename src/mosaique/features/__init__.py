"""EEG feature functions.

This package contains the built-in feature and transform functions used by
:class:`~mosaique.extraction.extractor.FeatureExtractor`.

Modules
-------
``univariate``
    Scalar features for 1-D time-domain signals: entropy measures, line
    length, Hurst exponent, band power, etc.

``connectivity``
    Graph-theoretic features computed on channel-by-channel connectivity
    matrices: clustering coefficient, global efficiency, etc.

``timefrequency``
    Wavelet decomposition functions (DWT, WPD, CWT) that produce
    frequency-band-specific coefficients.

Writing a new feature function
------------------------------
A feature function is any callable with the following signature::

    def my_feature(X, sfreq=200, **kwargs):
        '''One-line description.'''
        return float(...)

Rules:

1. **First positional argument** is the input signal — a 1-D NumPy array
   for ``simple`` / ``tf_decomposition`` transforms, or a 2-D connectivity
   matrix ``(n_channels, n_channels)`` for ``connectivity`` transforms.
2. **Accept ``**kwargs``** so that extra parameters from the YAML config
   (and internal parameters like ``sfreq``) are forwarded without error.
3. **Return value** must be one of:

   - ``float`` — a single scalar per (epoch, channel).
   - ``dict[str, float]`` — multiple named values (e.g. one per frequency
     band, as in :func:`~mosaique.features.univariate.band_power`).
   - ``np.ndarray`` — one value per channel (for connectivity features).

4. Place the function in any importable module.  In the YAML config,
   reference it by its dotted path.  Paths under ``mosaique.features.*``
   can use the short form (e.g. ``univariate.sample_entropy``); external
   functions need their full module path (e.g.
   ``my_package.features.my_func``).

Example YAML entry::

    features:
      simple:
        - name: my_feature
          function: my_package.my_feature
          params:
            my_param: [0.3, 0.5, 0.7]
"""

from mosaique.features.timefrequency import FrequencyBand, WaveletCoefficients

__all__ = ["FrequencyBand", "WaveletCoefficients"]
