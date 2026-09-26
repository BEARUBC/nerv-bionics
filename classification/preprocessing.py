"""Shared preprocessing for every model in this package.

All three selected models consume the same epoch tensor, so the cleaning steps
live here once instead of being duplicated per model. Keeping one path also means
the online decoder in ``realtime.py`` can apply exactly what training applied,
which is where offline/online mismatches usually creep in.

Order matters: baseline correction before filtering (so the filter is not fed a
large DC step), then bandpass, then artifact rejection on the filtered signal.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

# 8-30 Hz spans mu and beta, the two rhythms that carry motor imagery ERD/ERS.
DEFAULT_BAND = (8.0, 30.0)

# Peak-to-peak threshold in microvolts. Anything larger is eye blink or movement
# artifact rather than cortical activity.
DEFAULT_REJECT_UV = 100.0

# Sub-bands for filter-bank style feature extraction, kept here so the band
# definitions stay in one place.
FILTER_BANK = ((4.0, 8.0), (8.0, 13.0), (13.0, 22.0), (22.0, 30.0))


def baseline_correct(X, n_baseline_samples=None):
    """Remove each channel's own offset so trials are comparable.

    Parameters
    ----------
    X : numpy.ndarray
        Epochs of shape (n_trials, n_channels, n_samples).
    n_baseline_samples : int, optional
        Number of leading samples to use as the baseline. When omitted the whole
        epoch mean is used, which is the right choice for windows that start
        after cue onset and therefore contain no pre-cue period.

    Returns
    -------
    numpy.ndarray
        Baseline-corrected copy of ``X``.
    """

    if n_baseline_samples:
        baseline = X[:, :, :n_baseline_samples].mean(axis=2, keepdims=True)
    else:
        baseline = X.mean(axis=2, keepdims=True)

    return X - baseline


def bandpass(X, fs, band=DEFAULT_BAND, order=4):
    """Apply a zero-phase Butterworth bandpass along the time axis.

    Parameters
    ----------
    X : numpy.ndarray
        Epochs of shape (n_trials, n_channels, n_samples).
    fs : float
        Sampling rate in Hz.
    band : tuple[float, float], optional
        Low and high cutoff in Hz.
    order : int, optional
        Butterworth order per direction.

    Returns
    -------
    numpy.ndarray
        Filtered copy of ``X``.
    """

    low, high = band
    sos = butter(order, [low / (fs / 2.0), high / (fs / 2.0)], btype='bandpass', output='sos')
    return sosfiltfilt(sos, X, axis=-1)


def artifact_mask(X, threshold_uv=DEFAULT_REJECT_UV):
    """Flag epochs whose peak-to-peak amplitude looks like an artifact.

    Parameters
    ----------
    X : numpy.ndarray
        Epochs of shape (n_trials, n_channels, n_samples).
    threshold_uv : float, optional
        Maximum allowed peak-to-peak amplitude in microvolts.

    Returns
    -------
    numpy.ndarray
        Boolean mask that is True for epochs worth keeping.
    """

    peak_to_peak = X.max(axis=2) - X.min(axis=2)
    return peak_to_peak.max(axis=1) <= threshold_uv


def prepare(X, fs, band=DEFAULT_BAND, n_baseline_samples=None, reject_uv=DEFAULT_REJECT_UV):
    """Run the full preprocessing chain and report what was dropped.

    Parameters
    ----------
    X : numpy.ndarray
        Raw epochs of shape (n_trials, n_channels, n_samples).
    fs : float
        Sampling rate in Hz.
    band : tuple[float, float] or None, optional
        Bandpass range, or None to skip filtering (used by filter-bank models
        that need to do their own per-band filtering).
    n_baseline_samples : int, optional
        Leading samples used for baseline correction.
    reject_uv : float or None, optional
        Peak-to-peak rejection threshold, or None to keep every epoch.

    Returns
    -------
    X_clean : numpy.ndarray
        Preprocessed epochs.
    keep : numpy.ndarray
        Boolean mask of retained epochs, so callers can filter their labels the
        same way.
    """

    X_clean = baseline_correct(X, n_baseline_samples)

    if band is not None:
        X_clean = bandpass(X_clean, fs, band)

    if reject_uv is None:
        keep = np.ones(X_clean.shape[0], dtype=bool)
    else:
        keep = artifact_mask(X_clean, reject_uv)

    return X_clean, keep


def prepare_trial(trial, fs, band=DEFAULT_BAND, n_baseline_samples=None):
    """Preprocess one epoch for online inference.

    Same chain as :func:`prepare` minus artifact rejection: online we still have
    to emit a decision, so a noisy window is classified rather than discarded.

    Parameters
    ----------
    trial : numpy.ndarray
        One epoch of shape (n_channels, n_samples).
    fs : float
        Sampling rate in Hz.
    band : tuple[float, float] or None, optional
        Bandpass range, or None to skip filtering.
    n_baseline_samples : int, optional
        Leading samples used for baseline correction.

    Returns
    -------
    numpy.ndarray
        Preprocessed epoch of shape (1, n_channels, n_samples), ready to hand to
        a model's ``predict``.
    """

    batch = np.asarray(trial, dtype=float)[np.newaxis, :, :]
    batch = baseline_correct(batch, n_baseline_samples)

    if band is not None:
        batch = bandpass(batch, fs, band)

    return batch
