# Acquisition Pipeline Spec

This document defines the current contract between the playback pipeline and the feature export pipeline.

## 1. `data_streaming.py`

### Responsibility
- Load the raw EEG `.mat` file.
- Convert it into a BrainFlow-compatible CSV.
- Parse event markers into human-readable trial labels.
- Let the user choose which trial indices to keep.
- Trim the selected trials into `data/stream.csv`.
- Replay `data/stream.csv` through BrainFlow and render the live graph.

### Inputs
- `data/<dataset>.mat`
- A list of selected trial indices entered at the prompt.

### Outputs
- `data/<dataset>.csv` when the source CSV does not already exist.
- `data/stream.csv` containing the selected trials for playback.
- Console output showing the event table and selected slice boundaries.

### Non-responsibilities
- No bandpower computation.
- No model training.
- No feature CSV export.

## 2. `feature_export.py`

### Responsibility
- Convert selected cue trials into per-trial bandpower features.
- Compute mu-band and beta-band power for C3 and C4.
- Compute left-right asymmetry features.
- Append trial-level features to `data/train_features.csv`.
- Append class-level summary statistics to `data/train_feature_summary.csv`.

### Inputs
- `data/<dataset>.csv`
- Parsed event positions from the dataset marker table.
- A list of selected trial indices.

### Outputs
- `data/train_features.csv`
- `data/train_feature_summary.csv`
- Console output showing the per-trial table and class summary.

### Non-responsibilities
- No live playback or BrainFlow graph rendering.
- No trial trimming for playback.
- No model fitting or evaluation.

## 3. Shared conventions

- Cue labels map to four classes:
  - `left`
  - `right`
  - `foot`
  - `tongue`
- Feature extraction uses EEG channels:
  - C3 = column 1
  - C4 = column 3
- Frequency bands:
  - mu = 8 to 12 Hz
  - beta = 13 to 30 Hz
- All feature outputs include metadata columns:
  - `run_timestamp`
  - `source_file`

## 4. `classification/` (model layer)

### Responsibility
- Cut fixed-length labelled epochs from the raw `.mat` recordings.
- Apply one preprocessing chain shared by training and inference.
- Fit and evaluate the three selected classifiers.
- Decode a live sample stream into class labels with confidence.

### Inputs
- `acquisition/data/<dataset>.mat`
- Epoch window bounds `tmin`/`tmax` in seconds relative to cue onset.

### Outputs
- `acquisition/data/model_scorecard.csv` from `python -m classification.benchmark`.
- Console scorecard covering accuracy, Cohen's kappa, macro F1, calibration time,
  single-window inference latency, model size, and cross-subject accuracy.
- `models/<subjects>.joblib` from `python -m classification.train`: the trained
  pipeline together with the sampling rate, window length and filter band it was
  trained under.
- `classification.realtime.Decision` and `classification.detector.Detection`
  objects during a live session.
- A local web UI on `127.0.0.1:8000` from `python -m classification.server`: a
  test bench at `/`, a live stream at `/live`, and a signal picker at `/identify`.

### Live-view guarantee
`classification.live.LiveSession` trains on one split of a recording and streams
only the held-out split, and `_frames_for()` takes the signal as its sole
argument. The true label is attached to the JSON response after the model has
produced its probabilities, so it reaches the browser and never the classifier.
Any change here must preserve both properties.

### Non-responsibilities
- No BrainFlow session management or graph rendering.
- No CSV conversion or trial trimming for playback.
- No mu/beta bandpower export; that stays in `feature_export.py`.

### Conventions
- Epoch window: 0.5 to 3.5 s after cue onset, 750 samples at 250 Hz.
- Channels: all 22 EEG columns, EOG columns 22 to 24 dropped.
- Bandpass: 8 to 30 Hz, zero-phase Butterworth, order 4.
- Artifact rejection: peak-to-peak above 100 microvolts, plus the dataset's own
  code 1023 flags.
- Classes: `left`, `right`, `foot`, `tongue`.

### Note on the feature convention in section 3
`feature_export.py` reads C3 and C4 from columns 1 and 3, which is the channel
order of the 8-channel synthetic BrainFlow board, not the 22-channel dataset
layout. The model layer therefore uses its own channel map
(`classification.data.CHANNEL_NAMES`) rather than those column indices.

## 5. Change tracking rule

If a change alters the inputs, outputs, or responsibilities above, update this file in the same change set.
