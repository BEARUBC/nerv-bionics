"""Dataset loading for the classification layer.

The acquisition code replays a recording sample by sample. Model training needs
the opposite view: one array of fixed-length epochs cut around each motor
imagery cue. This module does that conversion straight from the BCI IV 2a
``.mat`` files that already live in ``acquisition/data/``.

The epoch window is a parameter on purpose. The earlier notebook benchmark
(commit c763fa8) used a pre-cut 0.8 s window, which caps accuracy well
below what the same models reach on a 0.5-3.5 s window, so being able to change
it is the single most useful knob here.
"""

from pathlib import Path

import numpy as np
import scipy.io

# Event codes used by the BCI Competition IV 2a recordings.
CUE_CODES = {
    769: 'left',
    770: 'right',
    771: 'foot',
    772: 'tongue',
}
TRIAL_START_CODE = 768
REJECTED_TRIAL_CODE = 1023

CLASSES = ('left', 'right', 'foot', 'tongue')

# Column layout of mat['s']: 22 EEG channels followed by 3 EOG channels.
N_EEG_CHANNELS = 22
EOG_COLUMNS = (22, 23, 24)

# 10-20 channel order for the 22 EEG columns, as documented for this dataset.
CHANNEL_NAMES = (
    'Fz',
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2',
    'POz',
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / 'acquisition' / 'data'


def interpolate_nans(signal: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in a continuous recording by linear interpolation.

    The raw dataset marks dropped samples as NaN. Covariance and filtering both
    propagate a single NaN across a whole trial, so the gaps are closed before
    anything else runs.

    Parameters
    ----------
    signal : numpy.ndarray
        Continuous recording of shape (samples, channels).

    Returns
    -------
    numpy.ndarray
        A copy with every NaN replaced by a linearly interpolated value.
    """

    filled = np.array(signal, dtype=float, copy=True)
    sample_index = np.arange(filled.shape[0])

    for channel in range(filled.shape[1]):
        column = filled[:, channel]
        bad = np.isnan(column)
        if not bad.any():
            continue
        if bad.all():
            filled[:, channel] = 0.0
            continue
        column[bad] = np.interp(sample_index[bad], sample_index[~bad], column[~bad])

    return filled


def load_recording(mat_path, drop_rejected=True):
    """Read one ``.mat`` recording as a continuous signal plus its cue list.

    Parameters
    ----------
    mat_path : str or pathlib.Path
        Path to a BCI IV 2a recording such as ``acquisition/data/A01T.mat``.
    drop_rejected : bool, optional
        Skip trials the dataset flags with the artifact code 1023.

    Returns
    -------
    signal : numpy.ndarray
        EEG of shape (n_samples, 22), EOG removed and NaN gaps filled.
    cues : list[tuple[int, str]]
        ``(sample_position, label)`` for each usable cue, in recording order.
    fs : float
        Sampling rate in Hz.
    n_rejected : int
        Number of cues skipped because their trial was flagged.
    """

    mat = scipy.io.loadmat(str(mat_path))

    fs = float(np.asarray(mat['SampleRate']).ravel()[0])
    signal = np.delete(np.asarray(mat['s'], dtype=float), EOG_COLUMNS, axis=1)
    signal = interpolate_nans(signal)[:, :N_EEG_CHANNELS]

    event_type = np.asarray(mat['EVENTTYP']).ravel().astype(int)
    event_pos = np.asarray(mat['EVENTPOS']).ravel().astype(int)

    def trial_start(i):
        for j in range(i, -1, -1):
            if event_type[j] == TRIAL_START_CODE:
                return int(event_pos[j])
        return None

    # A rejected trial is flagged on the 768 marker that precedes its cue, so
    # collect those start positions first and match cues back to them.
    rejected_starts = set()
    if drop_rejected:
        rejected_starts = {trial_start(i) for i, code in enumerate(event_type)
                           if code == REJECTED_TRIAL_CODE} - {None}

    cues = []
    n_rejected = 0
    for i, code in enumerate(event_type):
        if code not in CUE_CODES:
            continue
        if drop_rejected and trial_start(i) in rejected_starts:
            n_rejected += 1
            continue
        cues.append((int(event_pos[i]), CUE_CODES[code]))

    return signal, cues, fs, n_rejected


def cut_epochs(signal, cues, fs, tmin, tmax):
    """Cut one fixed window per cue out of a continuous signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Continuous EEG of shape (n_samples, n_channels).
    cues : list[tuple[int, str]]
        ``(sample_position, label)`` pairs, as returned by :func:`load_recording`.
    fs : float
        Sampling rate in Hz.
    tmin, tmax : float
        Window bounds in seconds relative to cue onset.

    Returns
    -------
    X : numpy.ndarray
        Epochs of shape (n_trials, n_channels, n_samples). Cues whose window runs
        off either end of the recording are skipped.
    y : numpy.ndarray
        Class labels as strings.
    """

    start_offset = int(round(tmin * fs))
    n_samples = int(round(tmax * fs)) - start_offset

    epochs, labels = [], []
    for position, label in cues:
        start = position + start_offset
        if start < 0 or start + n_samples > signal.shape[0]:
            continue
        epochs.append(signal[start:start + n_samples].T)
        labels.append(label)

    if not epochs:
        return np.empty((0, signal.shape[1], n_samples)), np.asarray(labels)
    return np.stack(epochs), np.asarray(labels)


def load_mat_epochs(mat_path, tmin=0.5, tmax=3.5, drop_rejected=True, verbose=True):
    """Cut labelled motor-imagery epochs out of one ``.mat`` recording.

    Parameters
    ----------
    mat_path : str or pathlib.Path
        Path to a BCI IV 2a recording such as ``acquisition/data/A01T.mat``.
    tmin : float, optional
        Epoch start in seconds relative to cue onset.
    tmax : float, optional
        Epoch end in seconds relative to cue onset.
    drop_rejected : bool, optional
        Skip trials the dataset flags with the artifact code 1023.
    verbose : bool, optional
        Print a one-line summary of what was loaded. Turn this off when loading
        in a loop, such as a window search, where it would bury the results.

    Returns
    -------
    X : numpy.ndarray
        Epochs of shape (n_trials, 22, n_samples).
    y : numpy.ndarray
        Class labels as strings, one of ``CLASSES``.
    fs : float
        Sampling rate in Hz.
    """

    mat_path = Path(mat_path)
    signal, cues, fs, n_rejected = load_recording(mat_path, drop_rejected)
    X, y = cut_epochs(signal, cues, fs, tmin, tmax)
    n_truncated = len(cues) - len(y)

    if not len(y):
        raise ValueError(f'No usable epochs found in {mat_path}')

    if verbose:
        print(
            f'{mat_path.name}: {X.shape[0]} epochs '
            f'({X.shape[1]} ch x {X.shape[2]} samples, {tmin}-{tmax}s @ {fs:g} Hz)'
            + (f', {n_rejected} rejected' if n_rejected else '')
            + (f', {n_truncated} truncated' if n_truncated else '')
        )

    return X, y, fs


def load_subjects(subject_ids, data_dir=None, tmin=0.5, tmax=3.5, drop_rejected=True, verbose=True):
    """Load several subjects and keep track of which epoch came from whom.

    Parameters
    ----------
    subject_ids : iterable of str
        Dataset stems such as ``['A01T', 'A04T']``.
    data_dir : str or pathlib.Path, optional
        Directory holding the ``.mat`` files. Defaults to ``acquisition/data``.
    tmin, tmax : float, optional
        Epoch window in seconds relative to cue onset.
    drop_rejected : bool, optional
        Skip artifact-flagged trials.
    verbose : bool, optional
        Print a line per recording loaded.

    Returns
    -------
    X : numpy.ndarray
        Epochs of shape (n_trials, 22, n_samples).
    y : numpy.ndarray
        Class labels.
    subjects : numpy.ndarray
        Subject id per epoch, aligned with ``X`` and ``y``.
    fs : float
        Sampling rate in Hz.
    """

    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    all_X, all_y, all_subjects = [], [], []
    fs = None

    for subject_id in subject_ids:
        X, y, subject_fs = load_mat_epochs(
            data_dir / f'{subject_id}.mat', tmin=tmin, tmax=tmax,
            drop_rejected=drop_rejected, verbose=verbose,
        )
        if fs is not None and subject_fs != fs:
            raise ValueError(f'Sampling rate mismatch: {subject_fs} != {fs}')
        fs = subject_fs
        all_X.append(X)
        all_y.append(y)
        all_subjects.append(np.full(X.shape[0], subject_id))

    return np.concatenate(all_X), np.concatenate(all_y), np.concatenate(all_subjects), fs


def available_subjects(data_dir=None):
    """List dataset stems present on disk.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory to scan. Defaults to ``acquisition/data``.

    Returns
    -------
    list[str]
        Sorted dataset stems, for example ``['A01T', 'A04T']``.
    """

    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return sorted(p.stem for p in data_dir.glob('*.mat'))
