"""Feature extraction helpers for the NERV EEG project.

This file takes selected cue trials, turns them into mu/beta bandpower
features, and saves two CSV files: one with trial-level features and one with
a class-level summary. I kept it separate from the playback code so it is
easier to follow, test, and change later.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


MU_BAND = (8.0, 12.0)
BETA_BAND = (13.0, 30.0)

# Based on the printed EEG channel order for the synthetic board:
# ['Fz', 'C3', 'Cz', 'C4', ...]
C3_COL = 1
C4_COL = 3

FEATURES_CSV_PATH = Path('data/train_features.csv')
SUMMARY_CSV_PATH = Path('data/train_feature_summary.csv')


def class_from_event_name(event_name: str) -> str:
    """Map a parsed event label to the corresponding motor-imagery class.

    Parameters
    ----------
    event_name : str
        Parsed event label such as 'Cue onset left (class 1) #3'.

    Returns
    -------
    str
        One of 'left', 'right', 'foot', 'tongue', or 'unknown'.
    """

    if 'Cue onset left (class 1)' in event_name:
        return 'left'
    if 'Cue onset right (class 2)' in event_name:
        return 'right'
    if 'Cue onset foot (class 3)' in event_name:
        return 'foot'
    if 'Cue onset tongue (class 4)' in event_name:
        return 'tongue'
    return 'unknown'


def bandpower(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """Compute absolute bandpower for a 1D EEG segment.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional EEG signal segment.
    fs : float
        Sampling rate in Hz.
    fmin : float
        Lower frequency bound of the band.
    fmax : float
        Upper frequency bound of the band.

    Returns
    -------
    float
        Integrated power inside the requested band, or NaN when the signal is
        too short or the band is empty.
    """

    if len(x) < 8:
        return np.nan

    nperseg = min(len(x), int(fs * 2))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return np.nan
    return float(np.trapz(psd[idx], freqs[idx]))


def build_feature_frame(desired_trials, csv, parsed_event_pos, frequency, source_name='unknown'):
    """Convert selected cue trials into a per-trial feature table.

    Parameters
    ----------
    desired_trials : list[int]
        Indices into parsed_event_pos that identify cue trials.
    csv : pandas.DataFrame
        BrainFlow-formatted EEG recording.
    parsed_event_pos : sequence
        Parsed event labels and sample positions from the dataset.
    frequency : float
        Sampling rate of the EEG recording.
    source_name : str, optional
        Dataset identifier written into the output table.

    Returns
    -------
    pandas.DataFrame | None, pandas.DataFrame | None
        Per-trial features and class summary, or (None, None) if no valid cue
        trials were selected.
    """

    rows = []

    for trial_idx in desired_trials:
        if trial_idx <= 0 or trial_idx >= len(parsed_event_pos) - 1:
            print(f"Skipping trial index {trial_idx}: out of valid event range.")
            continue

        event_name = parsed_event_pos[trial_idx][0]
        class_label = class_from_event_name(event_name)
        if class_label == 'unknown':
            print(f"Skipping trial index {trial_idx}: selected event is not a cue event ({event_name}).")
            continue

        start_pos = parsed_event_pos[trial_idx - 1][1]
        cue_pos = parsed_event_pos[trial_idx][1]
        end_pos = parsed_event_pos[trial_idx + 1][1]

        trial_df = csv.iloc[start_pos:end_pos]
        if trial_df.empty:
            print(f"Skipping trial index {trial_idx}: empty trial slice.")
            continue

        c3 = trial_df.iloc[:, C3_COL].to_numpy(dtype=float)
        c4 = trial_df.iloc[:, C4_COL].to_numpy(dtype=float)

        mu_c3 = bandpower(c3, frequency, MU_BAND[0], MU_BAND[1])
        mu_c4 = bandpower(c4, frequency, MU_BAND[0], MU_BAND[1])
        beta_c3 = bandpower(c3, frequency, BETA_BAND[0], BETA_BAND[1])
        beta_c4 = bandpower(c4, frequency, BETA_BAND[0], BETA_BAND[1])

        rows.append({
            'trial_id': trial_idx,
            'class_label': class_label,
            'start_sample': start_pos,
            'cue_sample': cue_pos,
            'end_sample': end_pos,
            'duration_s': round((end_pos - start_pos) / frequency, 3),
            'mu_power_C3': mu_c3,
            'mu_power_C4': mu_c4,
            'beta_power_C3': beta_c3,
            'beta_power_C4': beta_c4,
            'mu_asym_C3minusC4': mu_c3 - mu_c4,
            'beta_asym_C3minusC4': beta_c3 - beta_c4,
        })

    if not rows:
        print('\nNo cue trials selected for bandpower analysis.')
        return None, None

    feature_df = pd.DataFrame(rows)
    run_ts = datetime.now().isoformat(timespec='seconds')
    feature_df.insert(0, 'run_timestamp', run_ts)
    feature_df.insert(1, 'source_file', source_name)

    summary_df = (
        feature_df
        .groupby('class_label', as_index=False)
        .agg(
            n_trials=('trial_id', 'count'),
            mu_asym_mean=('mu_asym_C3minusC4', 'mean'),
            mu_asym_std=('mu_asym_C3minusC4', 'std'),
            beta_asym_mean=('beta_asym_C3minusC4', 'mean'),
            beta_asym_std=('beta_asym_C3minusC4', 'std'),
        )
        .fillna(0.0)
    )

    summary_df.insert(0, 'run_timestamp', run_ts)
    summary_df.insert(1, 'source_file', source_name)

    pd.set_option('display.max_columns', None)

    print('\nPer-trial mu/beta bandpower features:')
    print(feature_df.to_string(index=False, justify='left', float_format=lambda v: f"{v:.6f}"))
    print('\nClass summary (asymmetry features):')
    print(summary_df.to_string(index=False, justify='left', float_format=lambda v: f"{v:.6f}"))

    return feature_df, summary_df


def save_features(feature_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Append feature outputs to the configured CSV files.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Per-trial feature table.
    summary_df : pandas.DataFrame
        Per-class summary table.

    Returns
    -------
    None
        Both tables are appended to disk.
    """

    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    features_exists = FEATURES_CSV_PATH.exists()
    feature_df.to_csv(FEATURES_CSV_PATH, mode='a', header=not features_exists, index=False)

    summary_exists = SUMMARY_CSV_PATH.exists()
    summary_df.to_csv(SUMMARY_CSV_PATH, mode='a', header=not summary_exists, index=False)


def compute_trial_bandpower_features(desired_trials, csv, parsed_event_pos, frequency, source_name='unknown'):
    """Run feature extraction, print summaries, and persist the results.

    Parameters
    ----------
    desired_trials : list[int]
        Cue-trial indices selected by the user.
    csv : pandas.DataFrame
        BrainFlow-formatted EEG recording.
    parsed_event_pos : sequence
        Parsed event labels and sample positions.
    frequency : float
        Sampling rate of the EEG recording.
    source_name : str, optional
        Dataset identifier written into the output tables.

    Returns
    -------
    pandas.DataFrame | None, pandas.DataFrame | None
        The saved feature table and summary table, or (None, None) if no valid
        cue trials were selected.
    """

    feature_df, summary_df = build_feature_frame(desired_trials, csv, parsed_event_pos, frequency, source_name)
    if feature_df is None or summary_df is None:
        return None, None

    save_features(feature_df, summary_df)
    print(f"\nSaved per-trial features to: {FEATURES_CSV_PATH}")
    print(f"Saved class summary to: {SUMMARY_CSV_PATH}")
    return feature_df, summary_df
