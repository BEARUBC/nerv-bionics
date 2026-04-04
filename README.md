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