# NERV UBC Bionics

## Proposed Project Structure Diagram
📂 nerv-bionics/
│── 📂 src/
│   │── 📂 acquisition/           # Handles real-time data streaming
│   │── 📂 preprocessing/         # Segmentation, filtering.
│   │── 📂 feature_extraction/    # Extract CSP features.
│   │── 📂 classification/        # Machine learning models.
│   │── 📂 controller/             # Real-time BCI control. (binary keyboard control, continous control)

│   │── 📂 visualization /                    # displaying viz windows (ex: raw 8-channel data, processed data)
│   │── 📂 ui/                    # GUI and CLI
│   │   ├── gui.py                # PyQt-based GUI
│   │   ├── cli.py                # Command-line interface
│   │── main.py                   # Entry point, runs the system
│
│── 📂 models/                     # Trained ML models
│── 📂 config/                     # Config files (JSON)
│── 📂 utils/                      # Helper functions
│── 📂 tests/                      # Unit tests
│── requirements.txt                # Python dependencies
│── README.md                      # Documentation

## Datasets:
### Main datasets:
https://www.bbci.de/competition/iv/ 

### A CSV converted above dataset:
https://www.kaggle.com/datasets/aymanmostafa11/eeg-motor-imagery-bciciv-2a/data 

## Acquisition pipeline

The acquisition workflow is split into two modules to keep responsibilities separate:

- [acquisition/data_streaming.py](acquisition/data_streaming.py): dataset loading, event parsing, trial trimming, and live playback.
- [acquisition/feature_export.py](acquisition/feature_export.py): per-trial mu/beta feature extraction and CSV export.

The formal interface contract is documented in [acquisition/PIPELINE_SPEC.md](acquisition/PIPELINE_SPEC.md).

## Classification pipeline

The model layer lives in [classification/](classification/). It cuts labelled epochs
from the raw `.mat` recordings, applies one shared preprocessing chain, and exposes
the four models selected after benchmarking. Accuracy is on held-out test trials,
averaged over 20 random 50 / 25 / 25 train / check / test splits of both recordings:

| Rank | Model | Registry key | Accuracy |
|---|---|---|---|
| 1 | Riemannian ensemble (TS + LR with a filter bank), 0.5–4.5 s | `riemann_ensemble` | 0.694 |
| 2 | Riemannian tangent space + logistic regression | `riemann_ts_lr` | 0.661 |
| 3 | Riemannian MDM | `riemann_mdm` | 0.603 |
| 4 | CSP + LDA | `csp_lda` | 0.629 |

Ranks 2–4 are at the 0.5–3.5 s window. The ranking follows `models.py`, where MDM
sits above CSP + LDA because it calibrates faster and transfers better across
subjects.

Chance is 0.250 across four classes. CSP+SVM and EEGNet stay available in the
registry but were not selected, EEGNet came last on this dataset's two subjects,
which is a data-volume result worth revisiting once the other seven recordings are
added.

```bash
./run_ui.sh   # http://127.0.0.1:8000/live      Monitor: 22-channel EEG, live detector readouts
              # http://127.0.0.1:8000/identify  Identify: pick a held-out trial, watch it be named
              # http://127.0.0.1:8000           Review: run a test, read accuracy and precision
```

`run_ui.sh` picks an interpreter that has the dependencies. For the other tools,
use `python` (anaconda) rather than `python3` on this machine, `python3` resolves
to a Homebrew install without numpy:

```bash
python -m pip install -r requirements.txt

python -m classification.train  --subjects A01T --tune-window                # train + save
python -m classification.detect --model models/a01t.joblib --subject A04T    # run detection
python -m classification.ui     --model models/a01t.joblib --subject A01T --only tongue
python -m classification.benchmark                                           # re-score models
python -m classification.holdout                                             # 50/25/25 accuracy + precision
python -m pytest tests/ -v                                                   # 25 checks
```

The selection rationale, the measured scores on every axis, and the real-time usage
pattern are documented in [classification/README.md](classification/README.md).

## Key EEG nodes per movement

| Movement | Key nodes |
|---|---|
| Right hand | CP3, FC3, C1 (left side) |
| Left hand | CP4, C2 (right side) |
| Feet | CP3 + CP4 together |
| Tongue | FC3, CP3, Cz |
