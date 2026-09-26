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
the three models selected after benchmarking eight candidates:

| Rank | Model | Registry key | Accuracy | Calibrate |
|---|---|---|---|---|
| 1 | Riemannian tangent space + logistic regression | `riemann_ts_lr` | 0.691 | 0.19 s |
| 2 | Riemannian MDM | `riemann_mdm` | 0.614 | 0.18 s |
| 3 | CSP + LDA | `csp_lda` | 0.646 | 1.05 s |

Chance is 0.250 across four classes. CSP+SVM and EEGNet stay available in the
registry but were not selected — EEGNet came last on this dataset's two subjects,
which is a data-volume result worth revisiting once the other seven recordings are
added.

```bash
./run_ui.sh   # http://127.0.0.1:8000           test bench: pick settings, run a test
              # http://127.0.0.1:8000/live      live stream of unseen trials
              # http://127.0.0.1:8000/identify  pick a signal, watch it be identified
```

`run_ui.sh` picks an interpreter that has the dependencies. For the other tools,
use `python` (anaconda) rather than `python3` on this machine — `python3` resolves
to a Homebrew install without numpy:

```bash
python -m pip install -r requirements.txt

python -m classification.train  --subjects A01T --tune-window                # train + save
python -m classification.detect --model models/a01t.joblib --subject A04T    # run detection
python -m classification.ui     --model models/a01t.joblib --subject A01T --only tongue
python -m classification.benchmark                                           # re-score models
python -m pytest tests/ -v                                                   # 21 checks
```

The selection rationale, the measured scores on every axis, and the real-time usage
pattern are documented in [classification/README.md](classification/README.md).
